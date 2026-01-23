#ifndef TEST_VECTORS_DECODE_H
#define TEST_VECTORS_DECODE_H

// 자동 생성됨 (Decode 검증)

#include "../csrc/blocks/decode.h"

#define TV_DECODE_NUM_DETECTIONS 2

static const detection_t tv_decode_detections[] = {
    { 2.51211792e-01f, 2.46330678e-01f, 9.82748687e-01f, 9.64104414e-01f, 1.35633545e-02f, 0 },
    { 2.46294677e-01f, 2.53429830e-01f, 9.82400775e-01f, 9.63196397e-01f, 1.00825168e-02f, 0 },
};

// cv2, cv3 입력 (decode 함수에 전달)
// P3 cv2: (64, 4, 4)
// P3 cv3: (80, 4, 4)
// P4 cv2: (64, 2, 2)
// P4 cv3: (80, 2, 2)
// P5 cv2: (64, 1, 1)
// P5 cv3: (80, 1, 1)

#define TV_DECODE_P3_CV2_C 64
#define TV_DECODE_P3_CV2_H 4
#define TV_DECODE_P3_CV2_W 4

#define TV_DECODE_P3_CV3_C 80
#define TV_DECODE_P3_CV3_H 4
#define TV_DECODE_P3_CV3_W 4

#define TV_DECODE_P4_CV2_C 64
#define TV_DECODE_P4_CV2_H 2
#define TV_DECODE_P4_CV2_W 2

#define TV_DECODE_P4_CV3_C 80
#define TV_DECODE_P4_CV3_H 2
#define TV_DECODE_P4_CV3_W 2

#define TV_DECODE_P5_CV2_C 64
#define TV_DECODE_P5_CV2_H 1
#define TV_DECODE_P5_CV2_W 1

#define TV_DECODE_P5_CV3_C 80
#define TV_DECODE_P5_CV3_H 1
#define TV_DECODE_P5_CV3_W 1

#define TV_DECODE_INPUT_SIZE 32
#define TV_DECODE_NUM_CLASSES 80
#define TV_DECODE_CONF_THRESHOLD 0.01f
#define TV_DECODE_STRIDE_0 8.0f
#define TV_DECODE_STRIDE_1 16.0f
#define TV_DECODE_STRIDE_2 32.0f

#endif // TEST_VECTORS_DECODE_H
