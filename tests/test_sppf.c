#include <stdio.h>
#include <math.h>

#include "test_vectors_sppf.h"
#include "../csrc/utils/weights_loader.h"
#include "../csrc/blocks/sppf.h"

static float max_abs_diff(const float* a, const float* b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(void) {
    // .bin 파일에서 가중치 로드
    weights_loader_t weights;
    if (weights_load_from_file("assets/weights.bin", &weights) != 0) {
        fprintf(stderr, "Failed to load weights.bin\n");
        return 1;
    }
    
    // 필요한 텐서 가져오기
    const float* model_9_cv1_conv_weight = weights_get_tensor_data(&weights, "model.9.cv1.conv.weight");
    const float* model_9_cv1_bn_weight = weights_get_tensor_data(&weights, "model.9.cv1.bn.weight");
    const float* model_9_cv1_bn_bias = weights_get_tensor_data(&weights, "model.9.cv1.bn.bias");
    const float* model_9_cv1_bn_running_mean = weights_get_tensor_data(&weights, "model.9.cv1.bn.running_mean");
    const float* model_9_cv1_bn_running_var = weights_get_tensor_data(&weights, "model.9.cv1.bn.running_var");
    
    const float* model_9_cv2_conv_weight = weights_get_tensor_data(&weights, "model.9.cv2.conv.weight");
    const float* model_9_cv2_bn_weight = weights_get_tensor_data(&weights, "model.9.cv2.bn.weight");
    const float* model_9_cv2_bn_bias = weights_get_tensor_data(&weights, "model.9.cv2.bn.bias");
    const float* model_9_cv2_bn_running_mean = weights_get_tensor_data(&weights, "model.9.cv2.bn.running_mean");
    const float* model_9_cv2_bn_running_var = weights_get_tensor_data(&weights, "model.9.cv2.bn.running_var");
    
    if (!model_9_cv1_conv_weight || !model_9_cv2_conv_weight) {
        fprintf(stderr, "Failed to find required tensors\n");
        weights_free(&weights);
        return 1;
    }
    
    const int n = TV_SPPF_X_N;
    const int c_in = TV_SPPF_X_C;
    const int h = TV_SPPF_X_H;
    const int w = TV_SPPF_X_W;

    const int c_out = TV_SPPF_Y_C;
    const int h_out = TV_SPPF_Y_H;
    const int w_out = TV_SPPF_Y_W;

    static float y_out[TV_SPPF_X_N * TV_SPPF_Y_C * TV_SPPF_Y_H * TV_SPPF_Y_W];

    // YOLOv5n SPPF(Layer 9): cv1(256->128) + pool(k=5)x3 + concat + cv2(512->256)
    sppf_nchw_f32(
        tv_sppf_x, n, c_in, h, w,
        // cv1
        model_9_cv1_conv_weight, 128,
        model_9_cv1_bn_weight, model_9_cv1_bn_bias,
        model_9_cv1_bn_running_mean, model_9_cv1_bn_running_var,
        // cv2
        model_9_cv2_conv_weight, 256,
        model_9_cv2_bn_weight, model_9_cv2_bn_bias,
        model_9_cv2_bn_running_mean, model_9_cv2_bn_running_var,
        // pool
        5,
        1e-3f,
        y_out);

    const int elems = n * c_out * h_out * w_out;
    float diff = max_abs_diff(y_out, tv_sppf_y, elems);
    printf("sppf max_abs_diff = %g\n", diff);
    
    weights_free(&weights);

    if (diff < 1e-4f) {
        printf("OK\n");
        return 0;
    }
    printf("NG\n");
    return 1;
}
