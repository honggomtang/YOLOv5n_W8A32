#ifndef DETECT_H
#define DETECT_H

#include <stdint.h>

void detect_nchw_f32(
    const float* p3, int32_t p3_c, int32_t p3_h, int32_t p3_w,
    const float* p4, int32_t p4_c, int32_t p4_h, int32_t p4_w,
    const float* p5, int32_t p5_c, int32_t p5_h, int32_t p5_w,
    const float* m0_w, const float* m0_b,
    const float* m1_w, const float* m1_b,
    const float* m2_w, const float* m2_b,
    float* p3_out, float* p4_out, float* p5_out);

#endif /* DETECT_H */
