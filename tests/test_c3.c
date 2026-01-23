#include <stdio.h>
#include <math.h>

#include "test_vectors_c3.h"
#include "../csrc/utils/weights_loader.h"
#include "../csrc/blocks/c3.h"

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
    const float* model_2_cv1_conv_weight = weights_get_tensor_data(&weights, "model.2.cv1.conv.weight");
    const float* model_2_cv1_bn_weight = weights_get_tensor_data(&weights, "model.2.cv1.bn.weight");
    const float* model_2_cv1_bn_bias = weights_get_tensor_data(&weights, "model.2.cv1.bn.bias");
    const float* model_2_cv1_bn_running_mean = weights_get_tensor_data(&weights, "model.2.cv1.bn.running_mean");
    const float* model_2_cv1_bn_running_var = weights_get_tensor_data(&weights, "model.2.cv1.bn.running_var");
    
    const float* model_2_cv2_conv_weight = weights_get_tensor_data(&weights, "model.2.cv2.conv.weight");
    const float* model_2_cv2_bn_weight = weights_get_tensor_data(&weights, "model.2.cv2.bn.weight");
    const float* model_2_cv2_bn_bias = weights_get_tensor_data(&weights, "model.2.cv2.bn.bias");
    const float* model_2_cv2_bn_running_mean = weights_get_tensor_data(&weights, "model.2.cv2.bn.running_mean");
    const float* model_2_cv2_bn_running_var = weights_get_tensor_data(&weights, "model.2.cv2.bn.running_var");
    
    const float* model_2_cv3_conv_weight = weights_get_tensor_data(&weights, "model.2.cv3.conv.weight");
    const float* model_2_cv3_bn_weight = weights_get_tensor_data(&weights, "model.2.cv3.bn.weight");
    const float* model_2_cv3_bn_bias = weights_get_tensor_data(&weights, "model.2.cv3.bn.bias");
    const float* model_2_cv3_bn_running_mean = weights_get_tensor_data(&weights, "model.2.cv3.bn.running_mean");
    const float* model_2_cv3_bn_running_var = weights_get_tensor_data(&weights, "model.2.cv3.bn.running_var");
    
    const float* model_2_m_0_cv1_conv_weight = weights_get_tensor_data(&weights, "model.2.m.0.cv1.conv.weight");
    const float* model_2_m_0_cv1_bn_weight = weights_get_tensor_data(&weights, "model.2.m.0.cv1.bn.weight");
    const float* model_2_m_0_cv1_bn_bias = weights_get_tensor_data(&weights, "model.2.m.0.cv1.bn.bias");
    const float* model_2_m_0_cv1_bn_running_mean = weights_get_tensor_data(&weights, "model.2.m.0.cv1.bn.running_mean");
    const float* model_2_m_0_cv1_bn_running_var = weights_get_tensor_data(&weights, "model.2.m.0.cv1.bn.running_var");
    
    const float* model_2_m_0_cv2_conv_weight = weights_get_tensor_data(&weights, "model.2.m.0.cv2.conv.weight");
    const float* model_2_m_0_cv2_bn_weight = weights_get_tensor_data(&weights, "model.2.m.0.cv2.bn.weight");
    const float* model_2_m_0_cv2_bn_bias = weights_get_tensor_data(&weights, "model.2.m.0.cv2.bn.bias");
    const float* model_2_m_0_cv2_bn_running_mean = weights_get_tensor_data(&weights, "model.2.m.0.cv2.bn.running_mean");
    const float* model_2_m_0_cv2_bn_running_var = weights_get_tensor_data(&weights, "model.2.m.0.cv2.bn.running_var");
    
    if (!model_2_cv1_conv_weight || !model_2_cv2_conv_weight || !model_2_cv3_conv_weight ||
        !model_2_m_0_cv1_conv_weight || !model_2_m_0_cv2_conv_weight) {
        fprintf(stderr, "Failed to find required tensors\n");
        weights_free(&weights);
        return 1;
    }
    
    // YOLOv5n Layer 2 (C3): 입력 (1, 32, 32, 32) → 출력 (1, 32, 32, 32)
    const int n = TV_C3_X_N;
    const int c_in = TV_C3_X_C;
    const int h = TV_C3_X_H;
    const int w = TV_C3_X_W;
    
    const int c_out = TV_C3_Y_C;
    const int h_out = TV_C3_Y_H;
    const int w_out = TV_C3_Y_W;
    
    static float y_out[1 * 32 * 32 * 32];
    
    // 디버그: 입력 통계
    float x_mean = 0.0f, x_std = 0.0f;
    for (int i = 0; i < n * c_in * h * w; i++) {
        x_mean += tv_c3_x[i];
    }
    x_mean /= (n * c_in * h * w);
    for (int i = 0; i < n * c_in * h * w; i++) {
        float d = tv_c3_x[i] - x_mean;
        x_std += d * d;
    }
    x_std = sqrtf(x_std / (n * c_in * h * w));
    printf("입력 통계: mean=%.6f, std=%.6f\n", x_mean, x_std);
    
    // C3 블록 실행
    const float* bn_cv1_w[1] = { model_2_m_0_cv1_conv_weight };
    const float* bn_cv1_g[1] = { model_2_m_0_cv1_bn_weight };
    const float* bn_cv1_b[1] = { model_2_m_0_cv1_bn_bias };
    const float* bn_cv1_m[1] = { model_2_m_0_cv1_bn_running_mean };
    const float* bn_cv1_v[1] = { model_2_m_0_cv1_bn_running_var };
    const float* bn_cv2_w[1] = { model_2_m_0_cv2_conv_weight };
    const float* bn_cv2_g[1] = { model_2_m_0_cv2_bn_weight };
    const float* bn_cv2_b[1] = { model_2_m_0_cv2_bn_bias };
    const float* bn_cv2_m[1] = { model_2_m_0_cv2_bn_running_mean };
    const float* bn_cv2_v[1] = { model_2_m_0_cv2_bn_running_var };

    c3_nchw_f32(
        tv_c3_x, n, c_in, h, w,
        // cv1: Conv(32→16, 1×1)
        model_2_cv1_conv_weight, 16,
        model_2_cv1_bn_weight, model_2_cv1_bn_bias,
        model_2_cv1_bn_running_mean, model_2_cv1_bn_running_var,
        // cv2: Conv(32→16, 1×1) - skip
        model_2_cv2_conv_weight, 16,
        model_2_cv2_bn_weight, model_2_cv2_bn_bias,
        model_2_cv2_bn_running_mean, model_2_cv2_bn_running_var,
        // cv3: Conv(32→32, 1×1)
        model_2_cv3_conv_weight, 32,
        model_2_cv3_bn_weight, model_2_cv3_bn_bias,
        model_2_cv3_bn_running_mean, model_2_cv3_bn_running_var,
        // bottleneck: n=1
        1,
        bn_cv1_w,
        bn_cv1_g, bn_cv1_b, bn_cv1_m, bn_cv1_v,
        bn_cv2_w,
        bn_cv2_g, bn_cv2_b, bn_cv2_m, bn_cv2_v,
        1e-3f,  // eps
        y_out);
    
    // 디버그: 출력 통계
    float y_mean = 0.0f, y_std = 0.0f;
    for (int i = 0; i < n * c_out * h_out * w_out; i++) {
        y_mean += y_out[i];
    }
    y_mean /= (n * c_out * h_out * w_out);
    for (int i = 0; i < n * c_out * h_out * w_out; i++) {
        float d = y_out[i] - y_mean;
        y_std += d * d;
    }
    y_std = sqrtf(y_std / (n * c_out * h_out * w_out));
    printf("C 출력 통계: mean=%.6f, std=%.6f\n", y_mean, y_std);
    
    // 파이썬 정답 통계
    float py_mean = 0.0f, py_std = 0.0f;
    for (int i = 0; i < n * c_out * h_out * w_out; i++) {
        py_mean += tv_c3_y[i];
    }
    py_mean /= (n * c_out * h_out * w_out);
    for (int i = 0; i < n * c_out * h_out * w_out; i++) {
        float d = tv_c3_y[i] - py_mean;
        py_std += d * d;
    }
    py_std = sqrtf(py_std / (n * c_out * h_out * w_out));
    printf("Python 정답 통계: mean=%.6f, std=%.6f\n", py_mean, py_std);
    
    const int elems = n * c_out * h_out * w_out;
    float diff = max_abs_diff(y_out, tv_c3_y, elems);
    printf("c3 max_abs_diff = %g\n", diff);
    
    weights_free(&weights);
    
    if (diff < 1e-4f) {
        printf("OK\n");
        return 0;
    }
    printf("NG\n");
    return 1;
}
