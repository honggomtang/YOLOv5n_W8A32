#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef BARE_METAL
#include <stdio.h>
#endif

#include "utils/weights_loader.h"
#include "utils/image_loader.h"
#include "blocks/conv.h"
#include "blocks/c3.h"
#include "blocks/sppf.h"
#include "blocks/detect.h"
#include "blocks/decode.h"
#include "blocks/nms.h"
#include "operations/upsample.h"
#include "operations/concat.h"
#include "utils/feature_pool.h"
#ifdef BARE_METAL
#include "platform_config.h"
#include "xil_cache.h"
#include "xil_printf.h"
#include "utils/uart_dump.h"
#endif

#define W(name) weights_get_tensor_data(&weights, name)

#define INPUT_SIZE 640
#define NUM_CLASSES 80
#define CONF_THRESHOLD 0.25f
#define IOU_THRESHOLD 0.45f
#define MAX_DETECTIONS 300

#ifndef YOLO_VERBOSE
#define YOLO_VERBOSE 1
#endif

#if defined(BARE_METAL)
#define YOLO_LOG(...) xil_printf(__VA_ARGS__)
#elif YOLO_VERBOSE
#define YOLO_LOG(...) printf(__VA_ARGS__)
#else
#define YOLO_LOG(...) ((void)0)
#endif

static const float STRIDES[3] = {8.0f, 16.0f, 32.0f};
static const float ANCHORS[3][6] = {
    {10.0f, 13.0f, 16.0f, 30.0f, 33.0f, 23.0f},
    {30.0f, 61.0f, 62.0f, 45.0f, 59.0f, 119.0f},
    {116.0f, 90.0f, 156.0f, 198.0f, 373.0f, 326.0f}
};

static const char* const COCO_NAMES[NUM_CLASSES] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

int main(int argc, char* argv[]) {
#if defined(BARE_METAL)
    (void)argc;
    (void)argv;
#endif
#ifndef BARE_METAL
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
#endif

    YOLO_LOG("=== YOLOv5n Inference (Fused) ===\n\n");
    
    preprocessed_image_t img;
    weights_loader_t weights;

#ifdef BARE_METAL
    Xil_DCacheInvalidateRange((uintptr_t)WEIGHTS_DDR_BASE, (unsigned int)WEIGHTS_DDR_SIZE);
    Xil_DCacheInvalidateRange((uintptr_t)IMAGE_DDR_BASE, (unsigned int)IMAGE_DDR_SIZE);
    Xil_DCacheInvalidateRange((uintptr_t)FEATURE_POOL_BASE, (unsigned int)FEATURE_POOL_SIZE);
    Xil_DCacheInvalidateRange((uintptr_t)DETECT_HEAD_BASE, (unsigned int)DETECT_HEAD_SIZE);
    Xil_DCacheEnable();

    YOLO_LOG("Loading image from DDR 0x%08X...\n", (unsigned int)IMAGE_DDR_BASE);
    if (image_init_from_memory((uintptr_t)IMAGE_DDR_BASE, (size_t)IMAGE_DDR_SIZE, &img) != 0) {
        YOLO_LOG("ERROR: Failed to load image from DDR\n");
        return 1;
    }
    img.data = (float*)((uintptr_t)IMAGE_DDR_BASE + (uintptr_t)IMAGE_HEADER_SIZE);
    YOLO_LOG("Loading weights from DDR 0x%08X...\n", (unsigned int)WEIGHTS_DDR_BASE);
    if (weights_init_from_memory((uintptr_t)WEIGHTS_DDR_BASE, (size_t)WEIGHTS_DDR_SIZE, &weights) != 0) {
        YOLO_LOG("ERROR: Failed to load weights from DDR\n");
        image_free(&img);
        return 1;
    }
    {
        const float* bias24 = (const float*)W("model.24.m.0.bias");
        if (bias24) {
            uint32_t u0 = *(const uint32_t*)&bias24[0];
            uint32_t u4 = *(const uint32_t*)&bias24[4];
            YOLO_LOG("DBG model.24.m.0.bias @0x%08X [0]=0x%08X [4]=0x%08X\n",
                     (unsigned)(uintptr_t)bias24, (unsigned)u0, (unsigned)u4);
        }
    }
#else
    if (image_load_from_bin("data/input/preprocessed_image.bin", &img) != 0) {
        fprintf(stderr, "Failed to load image\n");
        return 1;
    }
    if (weights_load_from_file("assets/weights.bin", &weights) != 0) {
        fprintf(stderr, "Failed to load weights\n");
        image_free(&img);
        return 1;
    }
#endif
    YOLO_LOG("Image: %dx%d\n", img.w, img.h);
    YOLO_LOG("Weights: %d tensors\n\n", weights.num_tensors);

    feature_pool_init();
    const int n = 1;

    size_t sz_l0  = (size_t)(1 * 16  * 320 * 320 * sizeof(float));
    size_t sz_l1  = (size_t)(1 * 32  * 160 * 160 * sizeof(float));
    size_t sz_l2  = (size_t)(1 * 32  * 160 * 160 * sizeof(float));
    size_t sz_l3  = (size_t)(1 * 64  * 80  * 80  * sizeof(float));
    size_t sz_l4  = (size_t)(1 * 64  * 80  * 80  * sizeof(float));
    size_t sz_l5  = (size_t)(1 * 128 * 40  * 40  * sizeof(float));
    size_t sz_l6  = (size_t)(1 * 128 * 40  * 40  * sizeof(float));
    size_t sz_l7  = (size_t)(1 * 256 * 20  * 20  * sizeof(float));
    size_t sz_l8  = (size_t)(1 * 256 * 20  * 20  * sizeof(float));
    size_t sz_l9  = (size_t)(1 * 256 * 20  * 20  * sizeof(float));
    size_t sz_l10 = (size_t)(1 * 128 * 20  * 20  * sizeof(float));
    size_t sz_l11 = (size_t)(1 * 128 * 40  * 40  * sizeof(float));
    size_t sz_l12 = (size_t)(1 * 256 * 40  * 40  * sizeof(float));
    size_t sz_l13 = (size_t)(1 * 128 * 40  * 40  * sizeof(float));
    size_t sz_l14 = (size_t)(1 * 64  * 40  * 40  * sizeof(float));
    size_t sz_l15 = (size_t)(1 * 64  * 80  * 80  * sizeof(float));
    size_t sz_l16 = (size_t)(1 * 128 * 80  * 80  * sizeof(float));
    size_t sz_l17 = (size_t)(1 * 64  * 80  * 80  * sizeof(float));
    size_t sz_l18 = (size_t)(1 * 64  * 40  * 40  * sizeof(float));
    size_t sz_l19 = (size_t)(1 * 128 * 40  * 40  * sizeof(float));
    size_t sz_l20 = (size_t)(1 * 128 * 40  * 40  * sizeof(float));
    size_t sz_l21 = (size_t)(1 * 128 * 20  * 20  * sizeof(float));
    size_t sz_l22 = (size_t)(1 * 256 * 20  * 20  * sizeof(float));
    size_t sz_l23 = (size_t)(1 * 256 * 20  * 20  * sizeof(float));
    size_t sz_p3  = (size_t)(1 * 255 * 80  * 80  * sizeof(float));
    size_t sz_p4  = (size_t)(1 * 255 * 40  * 40  * sizeof(float));
    size_t sz_p5  = (size_t)(1 * 255 * 20  * 20  * sizeof(float));

    float* l0 = NULL, * l1 = NULL, * l2 = NULL, * l3 = NULL, * l4 = NULL;
    float* l5 = NULL, * l6 = NULL, * l7 = NULL, * l8 = NULL, * l9 = NULL;
    float* l10 = NULL, * l11 = NULL, * l12 = NULL, * l13 = NULL, * l14 = NULL;
    float* l15 = NULL, * l16 = NULL, * l17 = NULL, * l18 = NULL, * l19 = NULL;
    float* l20 = NULL, * l21 = NULL, * l22 = NULL, * l23 = NULL;
    float* p3 = NULL, * p4 = NULL, * p5 = NULL;

#define POOL_ALLOC(ptr, sz) do { \
    (ptr) = (float*)feature_pool_alloc(sz); \
    if (!(ptr)) { \
        YOLO_LOG("ERROR: Feature pool allocation failed\n"); \
        feature_pool_reset(); weights_free(&weights); image_free(&img); \
        return 1; \
    } \
} while(0)

#ifdef BARE_METAL
    Xil_DCacheInvalidateRange((uintptr_t)IMAGE_DDR_BASE, (unsigned int)IMAGE_DDR_SIZE);
    Xil_DCacheInvalidateRange((uintptr_t)WEIGHTS_DDR_BASE, (unsigned int)WEIGHTS_DDR_SIZE);
    {
        const float* pw = (const float*)W("model.0.conv.weight");
        float img0 = img.data ? img.data[0] : 0.0f;
        uint32_t u_img = *(const uint32_t*)(&img0);
        uint32_t u_w   = pw ? *(const uint32_t*)pw : 0u;
        YOLO_LOG("DBG img[0]=0x%08X w0[0]=0x%08X\n", (unsigned)u_img, (unsigned)u_w);
    }
#endif
    YOLO_LOG("Running inference...\n");
    YOLO_LOG("Backbone: ");

    // ===== Backbone =====
    // Layer 0: Conv 6x6 s2
    POOL_ALLOC(l0, sz_l0);
    conv_block_nchw_f32(img.data, n, 3, 640, 640,
        W("model.0.conv.weight"), 16, 6, 6, 2, 2, 2, 2,
        W("model.0.conv.bias"), l0, 320, 320);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l0, 16);
#endif

    // Layer 1: Conv 3x3 s2
    POOL_ALLOC(l1, sz_l1);
    conv_block_nchw_f32(l0, n, 16, 320, 320,
        W("model.1.conv.weight"), 32, 3, 3, 2, 2, 1, 1,
        W("model.1.conv.bias"), l1, 160, 160);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l1, 16);
#endif
    YOLO_LOG("L1 [0]=0x%08X\n", (unsigned)(*(const uint32_t*)&l1[0]));
    feature_pool_free(l0);

    // Layer 2: C3 (n=1)
#ifdef BARE_METAL
    { size_t largest = feature_pool_get_largest_free(); YOLO_LOG("  before L2 pool largest_free=%u\n", (unsigned)largest); }
#endif
    POOL_ALLOC(l2, sz_l2);
    const float* l2_cv1w[] = {W("model.2.m.0.cv1.conv.weight")};
    const float* l2_cv1b[] = {W("model.2.m.0.cv1.conv.bias")};
    const float* l2_cv2w[] = {W("model.2.m.0.cv2.conv.weight")};
    const float* l2_cv2b[] = {W("model.2.m.0.cv2.conv.bias")};
    c3_nchw_f32(l1, n, 32, 160, 160,
        W("model.2.cv1.conv.weight"), 16, W("model.2.cv1.conv.bias"),
        W("model.2.cv2.conv.weight"), 16, W("model.2.cv2.conv.bias"),
        W("model.2.cv3.conv.weight"), 32, W("model.2.cv3.conv.bias"),
        1, l2_cv1w, l2_cv1b, l2_cv2w, l2_cv2b, 1, l2);  // shortcut=1
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l2, 16);
#endif
    feature_pool_free(l1);

    // Layer 3: Conv 3x3 s2
    POOL_ALLOC(l3, sz_l3);
    conv_block_nchw_f32(l2, n, 32, 160, 160,
        W("model.3.conv.weight"), 64, 3, 3, 2, 2, 1, 1,
        W("model.3.conv.bias"), l3, 80, 80);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l3, 16);
#endif
    YOLO_LOG("L3 [0]=0x%08X\n", (unsigned)(*(const uint32_t*)&l3[0]));
    feature_pool_free(l2);

    // Layer 4: C3 (n=2)
    POOL_ALLOC(l4, sz_l4);
    const float* l4_cv1w[] = {W("model.4.m.0.cv1.conv.weight"), W("model.4.m.1.cv1.conv.weight")};
    const float* l4_cv1b[] = {W("model.4.m.0.cv1.conv.bias"), W("model.4.m.1.cv1.conv.bias")};
    const float* l4_cv2w[] = {W("model.4.m.0.cv2.conv.weight"), W("model.4.m.1.cv2.conv.weight")};
    const float* l4_cv2b[] = {W("model.4.m.0.cv2.conv.bias"), W("model.4.m.1.cv2.conv.bias")};
    c3_nchw_f32(l3, n, 64, 80, 80,
        W("model.4.cv1.conv.weight"), 32, W("model.4.cv1.conv.bias"),
        W("model.4.cv2.conv.weight"), 32, W("model.4.cv2.conv.bias"),
        W("model.4.cv3.conv.weight"), 64, W("model.4.cv3.conv.bias"),
        2, l4_cv1w, l4_cv1b, l4_cv2w, l4_cv2b, 1, l4);  // shortcut=1
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l4, 16);
#endif
    YOLO_LOG("L4 [0]=0x%08X\n", (unsigned)(*(const uint32_t*)&l4[0]));
    feature_pool_free(l3);

    // Layer 5: Conv 3x3 s2
    POOL_ALLOC(l5, sz_l5);
    conv_block_nchw_f32(l4, n, 64, 80, 80,
        W("model.5.conv.weight"), 128, 3, 3, 2, 2, 1, 1,
        W("model.5.conv.bias"), l5, 40, 40);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l5, 16);
#endif

    // Layer 6: C3 (n=3)
    POOL_ALLOC(l6, sz_l6);
    const float* l6_cv1w[] = {W("model.6.m.0.cv1.conv.weight"), W("model.6.m.1.cv1.conv.weight"), W("model.6.m.2.cv1.conv.weight")};
    const float* l6_cv1b[] = {W("model.6.m.0.cv1.conv.bias"), W("model.6.m.1.cv1.conv.bias"), W("model.6.m.2.cv1.conv.bias")};
    const float* l6_cv2w[] = {W("model.6.m.0.cv2.conv.weight"), W("model.6.m.1.cv2.conv.weight"), W("model.6.m.2.cv2.conv.weight")};
    const float* l6_cv2b[] = {W("model.6.m.0.cv2.conv.bias"), W("model.6.m.1.cv2.conv.bias"), W("model.6.m.2.cv2.conv.bias")};
    c3_nchw_f32(l5, n, 128, 40, 40,
        W("model.6.cv1.conv.weight"), 64, W("model.6.cv1.conv.bias"),
        W("model.6.cv2.conv.weight"), 64, W("model.6.cv2.conv.bias"),
        W("model.6.cv3.conv.weight"), 128, W("model.6.cv3.conv.bias"),
        3, l6_cv1w, l6_cv1b, l6_cv2w, l6_cv2b, 1, l6);  // shortcut=1
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l6, 16);
#endif
    YOLO_LOG("L6 [0]=0x%08X\n", (unsigned)(*(const uint32_t*)&l6[0]));
    feature_pool_free(l5);

    // Layer 7: Conv 3x3 s2
    POOL_ALLOC(l7, sz_l7);
    conv_block_nchw_f32(l6, n, 128, 40, 40,
        W("model.7.conv.weight"), 256, 3, 3, 2, 2, 1, 1,
        W("model.7.conv.bias"), l7, 20, 20);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l7, 16);
#endif
    YOLO_LOG("L7 [0]=0x%08X\n", (unsigned)(*(const uint32_t*)&l7[0]));

    // Layer 8: C3 (n=1)
    POOL_ALLOC(l8, sz_l8);
    const float* l8_cv1w[] = {W("model.8.m.0.cv1.conv.weight")};
    const float* l8_cv1b[] = {W("model.8.m.0.cv1.conv.bias")};
    const float* l8_cv2w[] = {W("model.8.m.0.cv2.conv.weight")};
    const float* l8_cv2b[] = {W("model.8.m.0.cv2.conv.bias")};
    c3_nchw_f32(l7, n, 256, 20, 20,
        W("model.8.cv1.conv.weight"), 128, W("model.8.cv1.conv.bias"),
        W("model.8.cv2.conv.weight"), 128, W("model.8.cv2.conv.bias"),
        W("model.8.cv3.conv.weight"), 256, W("model.8.cv3.conv.bias"),
        1, l8_cv1w, l8_cv1b, l8_cv2w, l8_cv2b, 1, l8);  // shortcut=1
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l8, 16);
#endif
    YOLO_LOG("L8 [0]=0x%08X\n", (unsigned)(*(const uint32_t*)&l8[0]));
    feature_pool_free(l7);

    // Layer 9: SPPF
    POOL_ALLOC(l9, sz_l9);
    sppf_nchw_f32(l8, n, 256, 20, 20,
        W("model.9.cv1.conv.weight"), 128, W("model.9.cv1.conv.bias"),
        W("model.9.cv2.conv.weight"), 256, W("model.9.cv2.conv.bias"),
        5, l9);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l9, 16);
#endif
    YOLO_LOG("L9 [0]=0x%08X\n", (unsigned)(*(const uint32_t*)&l9[0]));
    feature_pool_free(l8);

    // ===== Neck =====
    YOLO_LOG("\nNeck: ");
    // Layer 10: Conv 1x1
    POOL_ALLOC(l10, sz_l10);
    conv_block_nchw_f32(l9, n, 256, 20, 20,
        W("model.10.conv.weight"), 128, 1, 1, 1, 1, 0, 0,
        W("model.10.conv.bias"), l10, 20, 20);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l10, 16);
#endif
    YOLO_LOG("L10 [0]=0x%08X\n", (unsigned)(*(const uint32_t*)&l10[0]));
    feature_pool_free(l9);

    // Layer 11: Upsample
    POOL_ALLOC(l11, sz_l11);
    upsample_nearest2x_nchw_f32(l10, n, 128, 20, 20, l11);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l11, 16);
#endif
    YOLO_LOG("L11 [0]=0x%08X\n", (unsigned)(*(const uint32_t*)&l11[0]));

    // Layer 12: Concat (l11 + l6)
    POOL_ALLOC(l12, sz_l12);
    concat_nchw_f32(l11, 128, l6, 128, n, 40, 40, l12);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l12, 16);
#endif
    YOLO_LOG("L12 [0]=0x%08X\n", (unsigned)(*(const uint32_t*)&l12[0]));
    feature_pool_free(l11);
    feature_pool_free(l6);

    // Layer 13: C3 (n=1)
    POOL_ALLOC(l13, sz_l13);
    const float* l13_cv1w[] = {W("model.13.m.0.cv1.conv.weight")};
    const float* l13_cv1b[] = {W("model.13.m.0.cv1.conv.bias")};
    const float* l13_cv2w[] = {W("model.13.m.0.cv2.conv.weight")};
    const float* l13_cv2b[] = {W("model.13.m.0.cv2.conv.bias")};
    c3_nchw_f32(l12, n, 256, 40, 40,
        W("model.13.cv1.conv.weight"), 64, W("model.13.cv1.conv.bias"),
        W("model.13.cv2.conv.weight"), 64, W("model.13.cv2.conv.bias"),
        W("model.13.cv3.conv.weight"), 128, W("model.13.cv3.conv.bias"),
        1, l13_cv1w, l13_cv1b, l13_cv2w, l13_cv2b, 0, l13);  // shortcut=0 (head)
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l13, 16);
#endif
    YOLO_LOG("L13 [0]=0x%08X\n", (unsigned)(*(const uint32_t*)&l13[0]));
    feature_pool_free(l12);

    // Layer 14: Conv 1x1
    POOL_ALLOC(l14, sz_l14);
    conv_block_nchw_f32(l13, n, 128, 40, 40,
        W("model.14.conv.weight"), 64, 1, 1, 1, 1, 0, 0,
        W("model.14.conv.bias"), l14, 40, 40);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l14, 16);
#endif
    YOLO_LOG("L14 [0]=0x%08X\n", (unsigned)(*(const uint32_t*)&l14[0]));
    feature_pool_free(l13);

    // Layer 15: Upsample
    POOL_ALLOC(l15, sz_l15);
    upsample_nearest2x_nchw_f32(l14, n, 64, 40, 40, l15);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l15, 16);
#endif
    YOLO_LOG("L15 [0]=0x%08X\n", (unsigned)(*(const uint32_t*)&l15[0]));

    // Layer 16: Concat (l15 + l4)
    POOL_ALLOC(l16, sz_l16);
    concat_nchw_f32(l15, 64, l4, 64, n, 80, 80, l16);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l16, 16);
#endif
    YOLO_LOG("L16 [0]=0x%08X\n", (unsigned)(*(const uint32_t*)&l16[0]));
    feature_pool_free(l15);
    feature_pool_free(l4);

    // Layer 17: C3 (n=1) -> P3
    POOL_ALLOC(l17, sz_l17);
    const float* l17_cv1w[] = {W("model.17.m.0.cv1.conv.weight")};
    const float* l17_cv1b[] = {W("model.17.m.0.cv1.conv.bias")};
    const float* l17_cv2w[] = {W("model.17.m.0.cv2.conv.weight")};
    const float* l17_cv2b[] = {W("model.17.m.0.cv2.conv.bias")};
    c3_nchw_f32(l16, n, 128, 80, 80,
        W("model.17.cv1.conv.weight"), 32, W("model.17.cv1.conv.bias"),
        W("model.17.cv2.conv.weight"), 32, W("model.17.cv2.conv.bias"),
        W("model.17.cv3.conv.weight"), 64, W("model.17.cv3.conv.bias"),
        1, l17_cv1w, l17_cv1b, l17_cv2w, l17_cv2b, 0, l17);  // shortcut=0 (head)
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l17, 16);
#endif
    YOLO_LOG("L17 [0]=0x%08X\n", (unsigned)(*(const uint32_t*)&l17[0]));
    feature_pool_free(l16);

    // Layer 18: Conv 3x3 s2
    POOL_ALLOC(l18, sz_l18);
    conv_block_nchw_f32(l17, n, 64, 80, 80,
        W("model.18.conv.weight"), 64, 3, 3, 2, 2, 1, 1,
        W("model.18.conv.bias"), l18, 40, 40);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l18, 16);
#endif
    YOLO_LOG("L18 [0]=0x%08X\n", (unsigned)(*(const uint32_t*)&l18[0]));

    // Layer 19: Concat (l18 + l14)
    POOL_ALLOC(l19, sz_l19);
    concat_nchw_f32(l18, 64, l14, 64, n, 40, 40, l19);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l19, 16);
#endif
    YOLO_LOG("L19 [0]=0x%08X\n", (unsigned)(*(const uint32_t*)&l19[0]));
    feature_pool_free(l18);
    feature_pool_free(l14);

    // Layer 20: C3 (n=1) -> P4
    POOL_ALLOC(l20, sz_l20);
    const float* l20_cv1w[] = {W("model.20.m.0.cv1.conv.weight")};
    const float* l20_cv1b[] = {W("model.20.m.0.cv1.conv.bias")};
    const float* l20_cv2w[] = {W("model.20.m.0.cv2.conv.weight")};
    const float* l20_cv2b[] = {W("model.20.m.0.cv2.conv.bias")};
    c3_nchw_f32(l19, n, 128, 40, 40,
        W("model.20.cv1.conv.weight"), 64, W("model.20.cv1.conv.bias"),
        W("model.20.cv2.conv.weight"), 64, W("model.20.cv2.conv.bias"),
        W("model.20.cv3.conv.weight"), 128, W("model.20.cv3.conv.bias"),
        1, l20_cv1w, l20_cv1b, l20_cv2w, l20_cv2b, 0, l20);  // shortcut=0 (head)
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l20, 16);
#endif
    YOLO_LOG("L20 [0]=0x%08X\n", (unsigned)(*(const uint32_t*)&l20[0]));
    feature_pool_free(l19);

    // Layer 21: Conv 3x3 s2
    POOL_ALLOC(l21, sz_l21);
    conv_block_nchw_f32(l20, n, 128, 40, 40,
        W("model.21.conv.weight"), 128, 3, 3, 2, 2, 1, 1,
        W("model.21.conv.bias"), l21, 20, 20);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l21, 16);
#endif
    YOLO_LOG("L21 [0]=0x%08X\n", (unsigned)(*(const uint32_t*)&l21[0]));

    // Layer 22: Concat (l21 + l10)
    POOL_ALLOC(l22, sz_l22);
    concat_nchw_f32(l21, 128, l10, 128, n, 20, 20, l22);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l22, 16);
#endif
    YOLO_LOG("L22 [0]=0x%08X\n", (unsigned)(*(const uint32_t*)&l22[0]));
    feature_pool_free(l21);
    feature_pool_free(l10);

    // Layer 23: C3 (n=1) -> P5
    POOL_ALLOC(l23, sz_l23);
    const float* l23_cv1w[] = {W("model.23.m.0.cv1.conv.weight")};
    const float* l23_cv1b[] = {W("model.23.m.0.cv1.conv.bias")};
    const float* l23_cv2w[] = {W("model.23.m.0.cv2.conv.weight")};
    const float* l23_cv2b[] = {W("model.23.m.0.cv2.conv.bias")};
    c3_nchw_f32(l22, n, 256, 20, 20,
        W("model.23.cv1.conv.weight"), 128, W("model.23.cv1.conv.bias"),
        W("model.23.cv2.conv.weight"), 128, W("model.23.cv2.conv.bias"),
        W("model.23.cv3.conv.weight"), 256, W("model.23.cv3.conv.bias"),
        1, l23_cv1w, l23_cv1b, l23_cv2w, l23_cv2b, 0, l23);  // shortcut=0 (head)
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l23, 16);
#endif
    YOLO_LOG("L23 [0]=0x%08X\n", (unsigned)(*(const uint32_t*)&l23[0]));
    feature_pool_free(l22);

    // ===== Detect Head =====
    YOLO_LOG("\nHead: ");
#ifdef BARE_METAL
    (void)sz_p3;
    (void)sz_p4;
    (void)sz_p5;
    p3 = (float*)DETECT_HEAD_BASE;
    p4 = p3 + (255 * 80 * 80);
    p5 = p4 + (255 * 40 * 40);
#else
    POOL_ALLOC(p3, sz_p3);
    POOL_ALLOC(p4, sz_p4);
    POOL_ALLOC(p5, sz_p5);
#endif
#undef POOL_ALLOC
    detect_nchw_f32(
        l17, 64, 80, 80,
        l20, 128, 40, 40,
        l23, 256, 20, 20,
        W("model.24.m.0.weight"), W("model.24.m.0.bias"),
        W("model.24.m.1.weight"), W("model.24.m.1.bias"),
        W("model.24.m.2.weight"), W("model.24.m.2.bias"),
        p3, p4, p5);
    YOLO_LOG("Detect\n");
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)DETECT_HEAD_BASE, (unsigned int)DETECT_HEAD_SIZE);
    __sync_synchronize();
#endif
    feature_pool_free(l17);
    feature_pool_free(l20);
    feature_pool_free(l23);
#ifndef BARE_METAL
    feature_pool_free(p3);
    feature_pool_free(p4);
    feature_pool_free(p5);
#endif

    // ===== Decode =====
    detection_t* dets = malloc(MAX_DETECTIONS * sizeof(detection_t));
    int32_t num_dets = decode_nchw_f32(
        p3, 80, 80, p4, 40, 40, p5, 20, 20,
        NUM_CLASSES, CONF_THRESHOLD, INPUT_SIZE, STRIDES, ANCHORS,
        dets, MAX_DETECTIONS);

    YOLO_LOG("Decoded: %d detections\n", num_dets);
    if (p3) {
        int do_dbg = 0;
#ifdef BARE_METAL
        do_dbg = (num_dets == 0);
#else
        do_dbg = 1;
#endif
        if (do_dbg) {
            union { float f; uint32_t u; } u0 = { .f = p3[0] }, u1 = { .f = p3[1] }, u4 = { .f = p3[4 * 80 * 80] };
            YOLO_LOG("DBG p3[0]=0x%08X p3[1]=0x%08X p3[obj0]=0x%08X\n", (unsigned)u0.u, (unsigned)u1.u, (unsigned)u4.u);
        }
    }

    // Sort by confidence
    for (int i = 0; i < num_dets - 1; i++) {
        for (int j = i + 1; j < num_dets; j++) {
            if (dets[i].conf < dets[j].conf) {
                detection_t t = dets[i]; dets[i] = dets[j]; dets[j] = t;
            }
        }
    }

    // NMS
    detection_t* nms_dets = NULL;
    int32_t num_nms = 0;
    nms(dets, num_dets, &nms_dets, &num_nms, IOU_THRESHOLD, MAX_DETECTIONS);

    YOLO_LOG("After NMS: %d detections\n", num_nms);

    {
        uint8_t count = (uint8_t)(num_nms > 255 ? 255 : num_nms);
#ifdef BARE_METAL
        uint8_t* out = (uint8_t*)DETECTIONS_OUT_BASE;
        *out++ = count;
        for (int i = 0; i < count; i++) {
            hw_detection_t hw;
            hw.x = (uint16_t)(nms_dets[i].x * INPUT_SIZE);
            hw.y = (uint16_t)(nms_dets[i].y * INPUT_SIZE);
            hw.w = (uint16_t)(nms_dets[i].w * INPUT_SIZE);
            hw.h = (uint16_t)(nms_dets[i].h * INPUT_SIZE);
            hw.class_id = (uint8_t)nms_dets[i].cls_id;
            hw.confidence = (uint8_t)(nms_dets[i].conf * 255);
            hw.reserved[0] = 0;
            hw.reserved[1] = 0;
            memcpy(out, &hw, sizeof(hw_detection_t));
            out += sizeof(hw_detection_t);
        }
        YOLO_LOG("Sending %d detections to UART...\n", (int)count);
        yolo_uart_send_detections((const void*)((uint8_t*)DETECTIONS_OUT_BASE + 1), count);
        YOLO_LOG("Done. Results at DDR 0x%08X\n", (unsigned int)DETECTIONS_OUT_BASE);
        Xil_DCacheEnable();
#else
        FILE* f = fopen("data/output/detections.bin", "wb");
        if (f) {
            fwrite(&count, sizeof(uint8_t), 1, f);
            for (int i = 0; i < count; i++) {
                hw_detection_t hw;
                hw.x = (uint16_t)(nms_dets[i].x * INPUT_SIZE);
                hw.y = (uint16_t)(nms_dets[i].y * INPUT_SIZE);
                hw.w = (uint16_t)(nms_dets[i].w * INPUT_SIZE);
                hw.h = (uint16_t)(nms_dets[i].h * INPUT_SIZE);
                hw.class_id = (uint8_t)nms_dets[i].cls_id;
                hw.confidence = (uint8_t)(nms_dets[i].conf * 255);
                hw.reserved[0] = 0;
                hw.reserved[1] = 0;
                fwrite(&hw, sizeof(hw_detection_t), 1, f);
            }
            fclose(f);
            printf("Saved to data/output/detections.bin (%d bytes)\n",
                   1 + count * (int)sizeof(hw_detection_t));
        }
#endif
        YOLO_LOG("Summary: %d | ", (int)count);
        for (int i = 0; i < (int)count; i++) {
            int cls = nms_dets[i].cls_id;
            const char* name = (cls >= 0 && cls < NUM_CLASSES) ? COCO_NAMES[cls] : "?";
            int pct = (int)(nms_dets[i].conf * 100);
            int px = (int)(nms_dets[i].x * (float)INPUT_SIZE);
            int py = (int)(nms_dets[i].y * (float)INPUT_SIZE);
            YOLO_LOG("%s %d%% (%d,%d)%s", name, pct, px, py, (i < (int)count - 1) ? " | " : "");
        }
        YOLO_LOG("\n");
    }
    free(dets);
    if (nms_dets) free(nms_dets);
    feature_pool_reset();
    weights_free(&weights);
    image_free(&img);

    return 0;
}
