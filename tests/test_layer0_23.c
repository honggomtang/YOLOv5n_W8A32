#include <stdio.h>
#include <math.h>

#include "./test_vectors_layer0_23.h"
#include "../csrc/utils/weights_loader.h"
#include "../csrc/blocks/conv.h"
#include "../csrc/blocks/c3.h"
#include "../csrc/blocks/sppf.h"
#include "../csrc/operations/upsample.h"
#include "../csrc/operations/concat.h"

static float max_abs_diff(const float* a, const float* b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

// 헬퍼 매크로: 텐서 이름으로 데이터 가져오기
#define W(name) weights_get_tensor_data(&weights, name)

int main(void) {
    // .bin 파일에서 가중치 로드
    weights_loader_t weights;
    if (weights_load_from_file("assets/weights.bin", &weights) != 0) {
        fprintf(stderr, "Failed to load weights.bin\n");
        return 1;
    }
    // 입력
    const int n = TV_L0_23_X_N;
    const int c_in = TV_L0_23_X_C;
    const int h_in = TV_L0_23_X_H;
    const int w_in = TV_L0_23_X_W;

    // 중간 버퍼들 (각 레이어 출력)
    static float l0_out[TV_L0_N * TV_L0_C * TV_L0_H * TV_L0_W];
    static float l1_out[TV_L1_N * TV_L1_C * TV_L1_H * TV_L1_W];
    static float l2_out[TV_L2_N * TV_L2_C * TV_L2_H * TV_L2_W];
    static float l3_out[TV_L3_N * TV_L3_C * TV_L3_H * TV_L3_W];
    static float l4_out[TV_L4_N * TV_L4_C * TV_L4_H * TV_L4_W];
    static float l5_out[TV_L5_N * TV_L5_C * TV_L5_H * TV_L5_W];
    static float l6_out[TV_L6_N * TV_L6_C * TV_L6_H * TV_L6_W];
    static float l7_out[TV_L7_N * TV_L7_C * TV_L7_H * TV_L7_W];
    static float l8_out[TV_L8_N * TV_L8_C * TV_L8_H * TV_L8_W];
    static float l9_out[TV_L9_N * TV_L9_C * TV_L9_H * TV_L9_W];
    static float l10_out[TV_L10_N * TV_L10_C * TV_L10_H * TV_L10_W];
    static float l11_out[TV_L11_N * TV_L11_C * TV_L11_H * TV_L11_W];
    static float l12_out[TV_L12_N * TV_L12_C * TV_L12_H * TV_L12_W];
    static float l13_out[TV_L13_N * TV_L13_C * TV_L13_H * TV_L13_W];
    static float l14_out[TV_L14_N * TV_L14_C * TV_L14_H * TV_L14_W];
    static float l15_out[TV_L15_N * TV_L15_C * TV_L15_H * TV_L15_W];
    static float l16_out[TV_L16_N * TV_L16_C * TV_L16_H * TV_L16_W];
    static float l17_out[TV_L17_N * TV_L17_C * TV_L17_H * TV_L17_W];
    static float l18_out[TV_L18_N * TV_L18_C * TV_L18_H * TV_L18_W];
    static float l19_out[TV_L19_N * TV_L19_C * TV_L19_H * TV_L19_W];
    static float l20_out[TV_L20_N * TV_L20_C * TV_L20_H * TV_L20_W];
    static float l21_out[TV_L21_N * TV_L21_C * TV_L21_H * TV_L21_W];
    static float l22_out[TV_L22_N * TV_L22_C * TV_L22_H * TV_L22_W];
    static float l23_out[TV_L23_N * TV_L23_C * TV_L23_H * TV_L23_W];

    int all_ok = 1;

    // Layer 0~9는 test_layer0_9.c와 동일 (복사)
    // Layer 0: Conv (3->16, k=6, s=2, p=2)
    conv_block_nchw_f32(
        tv_l0_23_x, n, c_in, h_in, w_in,
        W("model.0.conv.weight"), 16, 6, 6,
        2, 2, 2, 2,
        W("model.0.bn.weight"), W("model.0.bn.bias"),
        W("model.0.bn.running_mean"), W("model.0.bn.running_var"),
        1e-3f,
        l0_out, TV_L0_H, TV_L0_W);
    {
        int elems = TV_L0_N * TV_L0_C * TV_L0_H * TV_L0_W;
        float diff = max_abs_diff(l0_out, tv_l0_out, elems);
        printf("Layer 0 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // Layer 1: Conv (16->32, k=3, s=2, p=1)
    conv_block_nchw_f32(
        l0_out, n, 16, TV_L0_H, TV_L0_W,
        W("model.1.conv.weight"), 32, 3, 3,
        2, 2, 1, 1,
        W("model.1.bn.weight"), W("model.1.bn.bias"),
        W("model.1.bn.running_mean"), W("model.1.bn.running_var"),
        1e-3f,
        l1_out, TV_L1_H, TV_L1_W);
    {
        int elems = TV_L1_N * TV_L1_C * TV_L1_H * TV_L1_W;
        float diff = max_abs_diff(l1_out, tv_l1_out, elems);
        printf("Layer 1 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // Layer 2: C3 (32->32, n_bottleneck=1)
    const float* l2_m_cv1_w[1];
    const float* l2_m_cv1_g[1];
    const float* l2_m_cv1_b[1];
    const float* l2_m_cv1_m[1];
    const float* l2_m_cv1_v[1];
    const float* l2_m_cv2_w[1];
    const float* l2_m_cv2_g[1];
    const float* l2_m_cv2_b[1];
    const float* l2_m_cv2_m[1];
    const float* l2_m_cv2_v[1];
    l2_m_cv1_w[0] = W("model.2.m.0.cv1.conv.weight");
    l2_m_cv1_g[0] = W("model.2.m.0.cv1.bn.weight");
    l2_m_cv1_b[0] = W("model.2.m.0.cv1.bn.bias");
    l2_m_cv1_m[0] = W("model.2.m.0.cv1.bn.running_mean");
    l2_m_cv1_v[0] = W("model.2.m.0.cv1.bn.running_var");
    l2_m_cv2_w[0] = W("model.2.m.0.cv2.conv.weight");
    l2_m_cv2_g[0] = W("model.2.m.0.cv2.bn.weight");
    l2_m_cv2_b[0] = W("model.2.m.0.cv2.bn.bias");
    l2_m_cv2_m[0] = W("model.2.m.0.cv2.bn.running_mean");
    l2_m_cv2_v[0] = W("model.2.m.0.cv2.bn.running_var");

    c3_nchw_f32(
        l1_out, n, 32, TV_L1_H, TV_L1_W,
        W("model.2.cv1.conv.weight"), 16,
        W("model.2.cv1.bn.weight"), W("model.2.cv1.bn.bias"),
        W("model.2.cv1.bn.running_mean"), W("model.2.cv1.bn.running_var"),
        W("model.2.cv2.conv.weight"), 16,
        W("model.2.cv2.bn.weight"), W("model.2.cv2.bn.bias"),
        W("model.2.cv2.bn.running_mean"), W("model.2.cv2.bn.running_var"),
        W("model.2.cv3.conv.weight"), 32,
        W("model.2.cv3.bn.weight"), W("model.2.cv3.bn.bias"),
        W("model.2.cv3.bn.running_mean"), W("model.2.cv3.bn.running_var"),
        1,
        l2_m_cv1_w, l2_m_cv1_g, l2_m_cv1_b, l2_m_cv1_m, l2_m_cv1_v,
        l2_m_cv2_w, l2_m_cv2_g, l2_m_cv2_b, l2_m_cv2_m, l2_m_cv2_v,
        1e-3f,
        l2_out);
    {
        int elems = TV_L2_N * TV_L2_C * TV_L2_H * TV_L2_W;
        float diff = max_abs_diff(l2_out, tv_l2_out, elems);
        printf("Layer 2 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // Layer 3: Conv (32->64, k=3, s=2, p=1)
    conv_block_nchw_f32(
        l2_out, n, 32, TV_L2_H, TV_L2_W,
        W("model.3.conv.weight"), 64, 3, 3,
        2, 2, 1, 1,
        W("model.3.bn.weight"), W("model.3.bn.bias"),
        W("model.3.bn.running_mean"), W("model.3.bn.running_var"),
        1e-3f,
        l3_out, TV_L3_H, TV_L3_W);
    {
        int elems = TV_L3_N * TV_L3_C * TV_L3_H * TV_L3_W;
        float diff = max_abs_diff(l3_out, tv_l3_out, elems);
        printf("Layer 3 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // Layer 4: C3 (64->64, n_bottleneck=2)
    const float* l4_m_cv1_w[2];
    l4_m_cv1_w[0] = W("model.4.m.0.cv1.conv.weight");
    l4_m_cv1_w[1] = W("model.4.m.1.cv1.conv.weight");
    const float* l4_m_cv1_g[2];
    l4_m_cv1_g[0] = W("model.4.m.0.cv1.bn.weight");
    l4_m_cv1_g[1] = W("model.4.m.1.cv1.bn.weight");
    const float* l4_m_cv1_b[2];
    l4_m_cv1_b[0] = W("model.4.m.0.cv1.bn.bias");
    l4_m_cv1_b[1] = W("model.4.m.1.cv1.bn.bias");
    const float* l4_m_cv1_m[2];
    l4_m_cv1_m[0] = W("model.4.m.0.cv1.bn.running_mean");
    l4_m_cv1_m[1] = W("model.4.m.1.cv1.bn.running_mean");
    const float* l4_m_cv1_v[2];
    l4_m_cv1_v[0] = W("model.4.m.0.cv1.bn.running_var");
    l4_m_cv1_v[1] = W("model.4.m.1.cv1.bn.running_var");
    const float* l4_m_cv2_w[2];
    l4_m_cv2_w[0] = W("model.4.m.0.cv2.conv.weight");
    l4_m_cv2_w[1] = W("model.4.m.1.cv2.conv.weight");
    const float* l4_m_cv2_g[2];
    l4_m_cv2_g[0] = W("model.4.m.0.cv2.bn.weight");
    l4_m_cv2_g[1] = W("model.4.m.1.cv2.bn.weight");
    const float* l4_m_cv2_b[2];
    l4_m_cv2_b[0] = W("model.4.m.0.cv2.bn.bias");
    l4_m_cv2_b[1] = W("model.4.m.1.cv2.bn.bias");
    const float* l4_m_cv2_m[2];
    l4_m_cv2_m[0] = W("model.4.m.0.cv2.bn.running_mean");
    l4_m_cv2_m[1] = W("model.4.m.1.cv2.bn.running_mean");
    const float* l4_m_cv2_v[2];
    l4_m_cv2_v[0] = W("model.4.m.0.cv2.bn.running_var");
    l4_m_cv2_v[1] = W("model.4.m.1.cv2.bn.running_var");

    c3_nchw_f32(
        l3_out, n, 64, TV_L3_H, TV_L3_W,
        W("model.4.cv1.conv.weight"), 32,
        W("model.4.cv1.bn.weight"), W("model.4.cv1.bn.bias"),
        W("model.4.cv1.bn.running_mean"), W("model.4.cv1.bn.running_var"),
        W("model.4.cv2.conv.weight"), 32,
        W("model.4.cv2.bn.weight"), W("model.4.cv2.bn.bias"),
        W("model.4.cv2.bn.running_mean"), W("model.4.cv2.bn.running_var"),
        W("model.4.cv3.conv.weight"), 64,
        W("model.4.cv3.bn.weight"), W("model.4.cv3.bn.bias"),
        W("model.4.cv3.bn.running_mean"), W("model.4.cv3.bn.running_var"),
        2,
        l4_m_cv1_w, l4_m_cv1_g, l4_m_cv1_b, l4_m_cv1_m, l4_m_cv1_v,
        l4_m_cv2_w, l4_m_cv2_g, l4_m_cv2_b, l4_m_cv2_m, l4_m_cv2_v,
        1e-3f,
        l4_out);
    {
        int elems = TV_L4_N * TV_L4_C * TV_L4_H * TV_L4_W;
        float diff = max_abs_diff(l4_out, tv_l4_out, elems);
        printf("Layer 4 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // Layer 5: Conv (64->128, k=3, s=2, p=1)
    conv_block_nchw_f32(
        l4_out, n, 64, TV_L4_H, TV_L4_W,
        W("model.5.conv.weight"), 128, 3, 3,
        2, 2, 1, 1,
        W("model.5.bn.weight"), W("model.5.bn.bias"),
        W("model.5.bn.running_mean"), W("model.5.bn.running_var"),
        1e-3f,
        l5_out, TV_L5_H, TV_L5_W);
    {
        int elems = TV_L5_N * TV_L5_C * TV_L5_H * TV_L5_W;
        float diff = max_abs_diff(l5_out, tv_l5_out, elems);
        printf("Layer 5 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // Layer 6: C3 (128->128, n_bottleneck=3)
    const float* l6_m_cv1_w[3];
    l6_m_cv1_w[0] = W("model.6.m.0.cv1.conv.weight");
    l6_m_cv1_w[1] = W("model.6.m.1.cv1.conv.weight");
    l6_m_cv1_w[2] = W("model.6.m.2.cv1.conv.weight");
    const float* l6_m_cv1_g[3];
    l6_m_cv1_g[0] = W("model.6.m.0.cv1.bn.weight");
    l6_m_cv1_g[1] = W("model.6.m.1.cv1.bn.weight");
    l6_m_cv1_g[2] = W("model.6.m.2.cv1.bn.weight");
    const float* l6_m_cv1_b[3];
    l6_m_cv1_b[0] = W("model.6.m.0.cv1.bn.bias");
    l6_m_cv1_b[1] = W("model.6.m.1.cv1.bn.bias");
    l6_m_cv1_b[2] = W("model.6.m.2.cv1.bn.bias");
    const float* l6_m_cv1_m[3];
    l6_m_cv1_m[0] = W("model.6.m.0.cv1.bn.running_mean");
    l6_m_cv1_m[1] = W("model.6.m.1.cv1.bn.running_mean");
    l6_m_cv1_m[2] = W("model.6.m.2.cv1.bn.running_mean");
    const float* l6_m_cv1_v[3];
    l6_m_cv1_v[0] = W("model.6.m.0.cv1.bn.running_var");
    l6_m_cv1_v[1] = W("model.6.m.1.cv1.bn.running_var");
    l6_m_cv1_v[2] = W("model.6.m.2.cv1.bn.running_var");
    const float* l6_m_cv2_w[3];
    l6_m_cv2_w[0] = W("model.6.m.0.cv2.conv.weight");
    l6_m_cv2_w[1] = W("model.6.m.1.cv2.conv.weight");
    l6_m_cv2_w[2] = W("model.6.m.2.cv2.conv.weight");
    const float* l6_m_cv2_g[3];
    l6_m_cv2_g[0] = W("model.6.m.0.cv2.bn.weight");
    l6_m_cv2_g[1] = W("model.6.m.1.cv2.bn.weight");
    l6_m_cv2_g[2] = W("model.6.m.2.cv2.bn.weight");
    const float* l6_m_cv2_b[3];
    l6_m_cv2_b[0] = W("model.6.m.0.cv2.bn.bias");
    l6_m_cv2_b[1] = W("model.6.m.1.cv2.bn.bias");
    l6_m_cv2_b[2] = W("model.6.m.2.cv2.bn.bias");
    const float* l6_m_cv2_m[3];
    l6_m_cv2_m[0] = W("model.6.m.0.cv2.bn.running_mean");
    l6_m_cv2_m[1] = W("model.6.m.1.cv2.bn.running_mean");
    l6_m_cv2_m[2] = W("model.6.m.2.cv2.bn.running_mean");
    const float* l6_m_cv2_v[3];
    l6_m_cv2_v[0] = W("model.6.m.0.cv2.bn.running_var");
    l6_m_cv2_v[1] = W("model.6.m.1.cv2.bn.running_var");
    l6_m_cv2_v[2] = W("model.6.m.2.cv2.bn.running_var");

    c3_nchw_f32(
        l5_out, n, 128, TV_L5_H, TV_L5_W,
        W("model.6.cv1.conv.weight"), 64,
        W("model.6.cv1.bn.weight"), W("model.6.cv1.bn.bias"),
        W("model.6.cv1.bn.running_mean"), W("model.6.cv1.bn.running_var"),
        W("model.6.cv2.conv.weight"), 64,
        W("model.6.cv2.bn.weight"), W("model.6.cv2.bn.bias"),
        W("model.6.cv2.bn.running_mean"), W("model.6.cv2.bn.running_var"),
        W("model.6.cv3.conv.weight"), 128,
        W("model.6.cv3.bn.weight"), W("model.6.cv3.bn.bias"),
        W("model.6.cv3.bn.running_mean"), W("model.6.cv3.bn.running_var"),
        3,
        l6_m_cv1_w, l6_m_cv1_g, l6_m_cv1_b, l6_m_cv1_m, l6_m_cv1_v,
        l6_m_cv2_w, l6_m_cv2_g, l6_m_cv2_b, l6_m_cv2_m, l6_m_cv2_v,
        1e-3f,
        l6_out);
    {
        int elems = TV_L6_N * TV_L6_C * TV_L6_H * TV_L6_W;
        float diff = max_abs_diff(l6_out, tv_l6_out, elems);
        printf("Layer 6 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // Layer 7: Conv (128->256, k=3, s=2, p=1)
    conv_block_nchw_f32(
        l6_out, n, 128, TV_L6_H, TV_L6_W,
        W("model.7.conv.weight"), 256, 3, 3,
        2, 2, 1, 1,
        W("model.7.bn.weight"), W("model.7.bn.bias"),
        W("model.7.bn.running_mean"), W("model.7.bn.running_var"),
        1e-3f,
        l7_out, TV_L7_H, TV_L7_W);
    {
        int elems = TV_L7_N * TV_L7_C * TV_L7_H * TV_L7_W;
        float diff = max_abs_diff(l7_out, tv_l7_out, elems);
        printf("Layer 7 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // Layer 8: C3 (256->256, n_bottleneck=1)
    const float* l8_m_cv1_w[1];
    l8_m_cv1_w[0] = W("model.8.m.0.cv1.conv.weight");
    const float* l8_m_cv1_g[1];
    l8_m_cv1_g[0] = W("model.8.m.0.cv1.bn.weight");
    const float* l8_m_cv1_b[1];
    l8_m_cv1_b[0] = W("model.8.m.0.cv1.bn.bias");
    const float* l8_m_cv1_m[1];
    l8_m_cv1_m[0] = W("model.8.m.0.cv1.bn.running_mean");
    const float* l8_m_cv1_v[1];
    l8_m_cv1_v[0] = W("model.8.m.0.cv1.bn.running_var");
    const float* l8_m_cv2_w[1];
    l8_m_cv2_w[0] = W("model.8.m.0.cv2.conv.weight");
    const float* l8_m_cv2_g[1];
    l8_m_cv2_g[0] = W("model.8.m.0.cv2.bn.weight");
    const float* l8_m_cv2_b[1];
    l8_m_cv2_b[0] = W("model.8.m.0.cv2.bn.bias");
    const float* l8_m_cv2_m[1];
    l8_m_cv2_m[0] = W("model.8.m.0.cv2.bn.running_mean");
    const float* l8_m_cv2_v[1];
    l8_m_cv2_v[0] = W("model.8.m.0.cv2.bn.running_var");

    c3_nchw_f32(
        l7_out, n, 256, TV_L7_H, TV_L7_W,
        W("model.8.cv1.conv.weight"), 128,
        W("model.8.cv1.bn.weight"), W("model.8.cv1.bn.bias"),
        W("model.8.cv1.bn.running_mean"), W("model.8.cv1.bn.running_var"),
        W("model.8.cv2.conv.weight"), 128,
        W("model.8.cv2.bn.weight"), W("model.8.cv2.bn.bias"),
        W("model.8.cv2.bn.running_mean"), W("model.8.cv2.bn.running_var"),
        W("model.8.cv3.conv.weight"), 256,
        W("model.8.cv3.bn.weight"), W("model.8.cv3.bn.bias"),
        W("model.8.cv3.bn.running_mean"), W("model.8.cv3.bn.running_var"),
        1,
        l8_m_cv1_w, l8_m_cv1_g, l8_m_cv1_b, l8_m_cv1_m, l8_m_cv1_v,
        l8_m_cv2_w, l8_m_cv2_g, l8_m_cv2_b, l8_m_cv2_m, l8_m_cv2_v,
        1e-3f,
        l8_out);
    {
        int elems = TV_L8_N * TV_L8_C * TV_L8_H * TV_L8_W;
        float diff = max_abs_diff(l8_out, tv_l8_out, elems);
        printf("Layer 8 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // Layer 9: SPPF (256->256)
    sppf_nchw_f32(
        l8_out, n, 256, TV_L8_H, TV_L8_W,
        W("model.9.cv1.conv.weight"), 128,
        W("model.9.cv1.bn.weight"), W("model.9.cv1.bn.bias"),
        W("model.9.cv1.bn.running_mean"), W("model.9.cv1.bn.running_var"),
        W("model.9.cv2.conv.weight"), 256,
        W("model.9.cv2.bn.weight"), W("model.9.cv2.bn.bias"),
        W("model.9.cv2.bn.running_mean"), W("model.9.cv2.bn.running_var"),
        5,
        1e-3f,
        l9_out);
    {
        int elems = TV_L9_N * TV_L9_C * TV_L9_H * TV_L9_W;
        float diff = max_abs_diff(l9_out, tv_l9_out, elems);
        printf("Layer 9 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // Layer 10: Conv (256->128, 1x1)
    conv_block_nchw_f32(
        l9_out, n, 256, TV_L9_H, TV_L9_W,
        W("model.10.conv.weight"), 128, 1, 1,
        1, 1, 0, 0,
        W("model.10.bn.weight"), W("model.10.bn.bias"),
        W("model.10.bn.running_mean"), W("model.10.bn.running_var"),
        1e-3f,
        l10_out, TV_L10_H, TV_L10_W);
    {
        int elems = TV_L10_N * TV_L10_C * TV_L10_H * TV_L10_W;
        float diff = max_abs_diff(l10_out, tv_l10_out, elems);
        printf("Layer 10 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // Layer 11: Upsample
    upsample_nearest2x_nchw_f32(
        l10_out, n, 128, TV_L10_H, TV_L10_W,
        l11_out);
    {
        int elems = TV_L11_N * TV_L11_C * TV_L11_H * TV_L11_W;
        float diff = max_abs_diff(l11_out, tv_l11_out, elems);
        printf("Layer 11 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // Layer 12: Concat (11, 6)
    concat_nchw_f32(
        l11_out, 128,
        l6_out, 128,
        n, TV_L11_H, TV_L11_W,
        l12_out);
    {
        int elems = TV_L12_N * TV_L12_C * TV_L12_H * TV_L12_W;
        float diff = max_abs_diff(l12_out, tv_l12_out, elems);
        printf("Layer 12 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // Layer 13: C3 (256->128, n_bottleneck=1, no-shortcut)
    const float* l13_m_cv1_w[1];
    l13_m_cv1_w[0] = W("model.13.m.0.cv1.conv.weight");
    const float* l13_m_cv1_g[1];
    l13_m_cv1_g[0] = W("model.13.m.0.cv1.bn.weight");
    const float* l13_m_cv1_b[1];
    l13_m_cv1_b[0] = W("model.13.m.0.cv1.bn.bias");
    const float* l13_m_cv1_m[1];
    l13_m_cv1_m[0] = W("model.13.m.0.cv1.bn.running_mean");
    const float* l13_m_cv1_v[1];
    l13_m_cv1_v[0] = W("model.13.m.0.cv1.bn.running_var");
    const float* l13_m_cv2_w[1];
    l13_m_cv2_w[0] = W("model.13.m.0.cv2.conv.weight");
    const float* l13_m_cv2_g[1];
    l13_m_cv2_g[0] = W("model.13.m.0.cv2.bn.weight");
    const float* l13_m_cv2_b[1];
    l13_m_cv2_b[0] = W("model.13.m.0.cv2.bn.bias");
    const float* l13_m_cv2_m[1];
    l13_m_cv2_m[0] = W("model.13.m.0.cv2.bn.running_mean");
    const float* l13_m_cv2_v[1];
    l13_m_cv2_v[0] = W("model.13.m.0.cv2.bn.running_var");

    c3_nchw_f32(
        l12_out, n, 256, TV_L12_H, TV_L12_W,
        W("model.13.cv1.conv.weight"), 128,
        W("model.13.cv1.bn.weight"), W("model.13.cv1.bn.bias"),
        W("model.13.cv1.bn.running_mean"), W("model.13.cv1.bn.running_var"),
        W("model.13.cv2.conv.weight"), 128,
        W("model.13.cv2.bn.weight"), W("model.13.cv2.bn.bias"),
        W("model.13.cv2.bn.running_mean"), W("model.13.cv2.bn.running_var"),
        W("model.13.cv3.conv.weight"), 128,
        W("model.13.cv3.bn.weight"), W("model.13.cv3.bn.bias"),
        W("model.13.cv3.bn.running_mean"), W("model.13.cv3.bn.running_var"),
        1,
        l13_m_cv1_w, l13_m_cv1_g, l13_m_cv1_b, l13_m_cv1_m, l13_m_cv1_v,
        l13_m_cv2_w, l13_m_cv2_g, l13_m_cv2_b, l13_m_cv2_m, l13_m_cv2_v,
        1e-3f,
        l13_out);
    {
        int elems = TV_L13_N * TV_L13_C * TV_L13_H * TV_L13_W;
        float diff = max_abs_diff(l13_out, tv_l13_out, elems);
        printf("Layer 13 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // Layer 14: Conv (128->64, 1x1)
    conv_block_nchw_f32(
        l13_out, n, 128, TV_L13_H, TV_L13_W,
        W("model.14.conv.weight"), 64, 1, 1,
        1, 1, 0, 0,
        W("model.14.bn.weight"), W("model.14.bn.bias"),
        W("model.14.bn.running_mean"), W("model.14.bn.running_var"),
        1e-3f,
        l14_out, TV_L14_H, TV_L14_W);
    {
        int elems = TV_L14_N * TV_L14_C * TV_L14_H * TV_L14_W;
        float diff = max_abs_diff(l14_out, tv_l14_out, elems);
        printf("Layer 14 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // Layer 15: Upsample
    upsample_nearest2x_nchw_f32(
        l14_out, n, 64, TV_L14_H, TV_L14_W,
        l15_out);
    {
        int elems = TV_L15_N * TV_L15_C * TV_L15_H * TV_L15_W;
        float diff = max_abs_diff(l15_out, tv_l15_out, elems);
        printf("Layer 15 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // Layer 16: Concat (15, 4)
    concat_nchw_f32(
        l15_out, 64,
        l4_out, 64,
        n, TV_L15_H, TV_L15_W,
        l16_out);
    {
        int elems = TV_L16_N * TV_L16_C * TV_L16_H * TV_L16_W;
        float diff = max_abs_diff(l16_out, tv_l16_out, elems);
        printf("Layer 16 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // Layer 17: C3 (128->64)
    const float* l17_m_cv1_w[1];
    l17_m_cv1_w[0] = W("model.17.m.0.cv1.conv.weight");
    const float* l17_m_cv1_g[1];
    l17_m_cv1_g[0] = W("model.17.m.0.cv1.bn.weight");
    const float* l17_m_cv1_b[1];
    l17_m_cv1_b[0] = W("model.17.m.0.cv1.bn.bias");
    const float* l17_m_cv1_m[1];
    l17_m_cv1_m[0] = W("model.17.m.0.cv1.bn.running_mean");
    const float* l17_m_cv1_v[1];
    l17_m_cv1_v[0] = W("model.17.m.0.cv1.bn.running_var");
    const float* l17_m_cv2_w[1];
    l17_m_cv2_w[0] = W("model.17.m.0.cv2.conv.weight");
    const float* l17_m_cv2_g[1];
    l17_m_cv2_g[0] = W("model.17.m.0.cv2.bn.weight");
    const float* l17_m_cv2_b[1];
    l17_m_cv2_b[0] = W("model.17.m.0.cv2.bn.bias");
    const float* l17_m_cv2_m[1];
    l17_m_cv2_m[0] = W("model.17.m.0.cv2.bn.running_mean");
    const float* l17_m_cv2_v[1];
    l17_m_cv2_v[0] = W("model.17.m.0.cv2.bn.running_var");

    c3_nchw_f32(
        l16_out, n, 128, TV_L17_H, TV_L17_W,
        W("model.17.cv1.conv.weight"), 64,
        W("model.17.cv1.bn.weight"), W("model.17.cv1.bn.bias"),
        W("model.17.cv1.bn.running_mean"), W("model.17.cv1.bn.running_var"),
        W("model.17.cv2.conv.weight"), 64,
        W("model.17.cv2.bn.weight"), W("model.17.cv2.bn.bias"),
        W("model.17.cv2.bn.running_mean"), W("model.17.cv2.bn.running_var"),
        W("model.17.cv3.conv.weight"), 64,
        W("model.17.cv3.bn.weight"), W("model.17.cv3.bn.bias"),
        W("model.17.cv3.bn.running_mean"), W("model.17.cv3.bn.running_var"),
        1,
        l17_m_cv1_w, l17_m_cv1_g, l17_m_cv1_b, l17_m_cv1_m, l17_m_cv1_v,
        l17_m_cv2_w, l17_m_cv2_g, l17_m_cv2_b, l17_m_cv2_m, l17_m_cv2_v,
        1e-3f,
        l17_out);
    {
        int elems = TV_L17_N * TV_L17_C * TV_L17_H * TV_L17_W;
        float diff = max_abs_diff(l17_out, tv_l17_out, elems);
        printf("Layer 17 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // Layer 18: Conv (64->64, k=3, s=2, p=1)
    conv_block_nchw_f32(
        l17_out, n, 64, TV_L17_H, TV_L17_W,
        W("model.18.conv.weight"), 64, 3, 3,
        2, 2, 1, 1,
        W("model.18.bn.weight"), W("model.18.bn.bias"),
        W("model.18.bn.running_mean"), W("model.18.bn.running_var"),
        1e-3f,
        l18_out, TV_L18_H, TV_L18_W);
    {
        int elems = TV_L18_N * TV_L18_C * TV_L18_H * TV_L18_W;
        float diff = max_abs_diff(l18_out, tv_l18_out, elems);
        printf("Layer 18 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // Layer 19: Concat (18, 14)
    concat_nchw_f32(
        l18_out, 64,
        l14_out, 64,
        n, TV_L18_H, TV_L18_W,
        l19_out);
    {
        int elems = TV_L19_N * TV_L19_C * TV_L19_H * TV_L19_W;
        float diff = max_abs_diff(l19_out, tv_l19_out, elems);
        printf("Layer 19 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // Layer 20: C3 (128->128)
    const float* l20_m_cv1_w[1];
    l20_m_cv1_w[0] = W("model.20.m.0.cv1.conv.weight");
    const float* l20_m_cv1_g[1];
    l20_m_cv1_g[0] = W("model.20.m.0.cv1.bn.weight");
    const float* l20_m_cv1_b[1];
    l20_m_cv1_b[0] = W("model.20.m.0.cv1.bn.bias");
    const float* l20_m_cv1_m[1];
    l20_m_cv1_m[0] = W("model.20.m.0.cv1.bn.running_mean");
    const float* l20_m_cv1_v[1];
    l20_m_cv1_v[0] = W("model.20.m.0.cv1.bn.running_var");
    const float* l20_m_cv2_w[1];
    l20_m_cv2_w[0] = W("model.20.m.0.cv2.conv.weight");
    const float* l20_m_cv2_g[1];
    l20_m_cv2_g[0] = W("model.20.m.0.cv2.bn.weight");
    const float* l20_m_cv2_b[1];
    l20_m_cv2_b[0] = W("model.20.m.0.cv2.bn.bias");
    const float* l20_m_cv2_m[1];
    l20_m_cv2_m[0] = W("model.20.m.0.cv2.bn.running_mean");
    const float* l20_m_cv2_v[1];
    l20_m_cv2_v[0] = W("model.20.m.0.cv2.bn.running_var");

    c3_nchw_f32(
        l19_out, n, 128, TV_L20_H, TV_L20_W,
        W("model.20.cv1.conv.weight"), 64,
        W("model.20.cv1.bn.weight"), W("model.20.cv1.bn.bias"),
        W("model.20.cv1.bn.running_mean"), W("model.20.cv1.bn.running_var"),
        W("model.20.cv2.conv.weight"), 64,
        W("model.20.cv2.bn.weight"), W("model.20.cv2.bn.bias"),
        W("model.20.cv2.bn.running_mean"), W("model.20.cv2.bn.running_var"),
        W("model.20.cv3.conv.weight"), 128,
        W("model.20.cv3.bn.weight"), W("model.20.cv3.bn.bias"),
        W("model.20.cv3.bn.running_mean"), W("model.20.cv3.bn.running_var"),
        1,
        l20_m_cv1_w, l20_m_cv1_g, l20_m_cv1_b, l20_m_cv1_m, l20_m_cv1_v,
        l20_m_cv2_w, l20_m_cv2_g, l20_m_cv2_b, l20_m_cv2_m, l20_m_cv2_v,
        1e-3f,
        l20_out);
    {
        int elems = TV_L20_N * TV_L20_C * TV_L20_H * TV_L20_W;
        float diff = max_abs_diff(l20_out, tv_l20_out, elems);
        printf("Layer 20 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // Layer 21: Conv (128->128, k=3, s=2, p=1)
    conv_block_nchw_f32(
        l20_out, n, 128, TV_L20_H, TV_L20_W,
        W("model.21.conv.weight"), 128, 3, 3,
        2, 2, 1, 1,
        W("model.21.bn.weight"), W("model.21.bn.bias"),
        W("model.21.bn.running_mean"), W("model.21.bn.running_var"),
        1e-3f,
        l21_out, TV_L21_H, TV_L21_W);
    {
        int elems = TV_L21_N * TV_L21_C * TV_L21_H * TV_L21_W;
        float diff = max_abs_diff(l21_out, tv_l21_out, elems);
        printf("Layer 21 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // Layer 22: Concat (21, 10)
    concat_nchw_f32(
        l21_out, 128,
        l10_out, 128,
        n, TV_L21_H, TV_L21_W,
        l22_out);
    {
        int elems = TV_L22_N * TV_L22_C * TV_L22_H * TV_L22_W;
        float diff = max_abs_diff(l22_out, tv_l22_out, elems);
        printf("Layer 22 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // Layer 23: C3 (256->256)
    const float* l23_m_cv1_w[1];
    l23_m_cv1_w[0] = W("model.23.m.0.cv1.conv.weight");
    const float* l23_m_cv1_g[1];
    l23_m_cv1_g[0] = W("model.23.m.0.cv1.bn.weight");
    const float* l23_m_cv1_b[1];
    l23_m_cv1_b[0] = W("model.23.m.0.cv1.bn.bias");
    const float* l23_m_cv1_m[1];
    l23_m_cv1_m[0] = W("model.23.m.0.cv1.bn.running_mean");
    const float* l23_m_cv1_v[1];
    l23_m_cv1_v[0] = W("model.23.m.0.cv1.bn.running_var");
    const float* l23_m_cv2_w[1];
    l23_m_cv2_w[0] = W("model.23.m.0.cv2.conv.weight");
    const float* l23_m_cv2_g[1];
    l23_m_cv2_g[0] = W("model.23.m.0.cv2.bn.weight");
    const float* l23_m_cv2_b[1];
    l23_m_cv2_b[0] = W("model.23.m.0.cv2.bn.bias");
    const float* l23_m_cv2_m[1];
    l23_m_cv2_m[0] = W("model.23.m.0.cv2.bn.running_mean");
    const float* l23_m_cv2_v[1];
    l23_m_cv2_v[0] = W("model.23.m.0.cv2.bn.running_var");

    c3_nchw_f32(
        l22_out, n, 256, TV_L22_H, TV_L22_W,
        W("model.23.cv1.conv.weight"), 128,
        W("model.23.cv1.bn.weight"), W("model.23.cv1.bn.bias"),
        W("model.23.cv1.bn.running_mean"), W("model.23.cv1.bn.running_var"),
        W("model.23.cv2.conv.weight"), 128,
        W("model.23.cv2.bn.weight"), W("model.23.cv2.bn.bias"),
        W("model.23.cv2.bn.running_mean"), W("model.23.cv2.bn.running_var"),
        W("model.23.cv3.conv.weight"), 256,
        W("model.23.cv3.bn.weight"), W("model.23.cv3.bn.bias"),
        W("model.23.cv3.bn.running_mean"), W("model.23.cv3.bn.running_var"),
        1,
        l23_m_cv1_w, l23_m_cv1_g, l23_m_cv1_b, l23_m_cv1_m, l23_m_cv1_v,
        l23_m_cv2_w, l23_m_cv2_g, l23_m_cv2_b, l23_m_cv2_m, l23_m_cv2_v,
        1e-3f,
        l23_out);
    {
        int elems = TV_L23_N * TV_L23_C * TV_L23_H * TV_L23_W;
        float diff = max_abs_diff(l23_out, tv_l23_out, elems);
        printf("Layer 23 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    weights_free(&weights);
    
    if (all_ok) {
        printf("\nAll layers OK\n");
        return 0;
    }
    printf("\nSome layers failed\n");
    return 1;
}
