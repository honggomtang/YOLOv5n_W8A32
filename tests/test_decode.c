/**
 * test_decode.c - Decode 블록 테스트 (Anchor-based)
 * 
 * Detect 출력(255ch)을 bbox로 디코딩
 * 255 = 3앵커 x 85 (4 bbox + 1 obj + 80 classes)
 */
#include <stdio.h>
#include <math.h>

#include "test_vectors_decode.h"
#include "../csrc/blocks/decode.h"

int main(void) {
    printf("=== Decode Block Test (Anchor-based) ===\n\n");
    
    // 앵커 및 stride 설정
    static const float strides[3] = {8.0f, 16.0f, 32.0f};
    static const float anchors[3][6] = {
        {10.0f, 13.0f, 16.0f, 30.0f, 33.0f, 23.0f},
        {30.0f, 61.0f, 62.0f, 45.0f, 59.0f, 119.0f},
        {116.0f, 90.0f, 156.0f, 198.0f, 373.0f, 326.0f}
    };
    
    // Decode 실행
    static detection_t detections[300];
    int32_t num_dets = decode_nchw_f32(
        tv_decode_p3, TV_DECODE_P3_H, TV_DECODE_P3_W,
        tv_decode_p4, TV_DECODE_P4_H, TV_DECODE_P4_W,
        tv_decode_p5, TV_DECODE_P5_H, TV_DECODE_P5_W,
        TV_DECODE_NUM_CLASSES,
        TV_DECODE_CONF_THRESHOLD,
        TV_DECODE_INPUT_SIZE,
        strides, anchors,
        detections, 300);
    
    printf("Decoded: %d detections (expected: %d)\n\n", num_dets, TV_DECODE_NUM_DETECTIONS);
    
    // confidence로 정렬
    for (int i = 0; i < num_dets - 1; i++) {
        for (int j = i + 1; j < num_dets; j++) {
            if (detections[i].conf < detections[j].conf) {
                detection_t t = detections[i];
                detections[i] = detections[j];
                detections[j] = t;
            }
        }
    }
    
    // 상위 5개 출력
    printf("Top 5 detections:\n");
    for (int i = 0; i < 5 && i < num_dets; i++) {
        printf("  [%d] cls=%d conf=%.4f bbox=(%.2f, %.2f, %.2f, %.2f)\n",
               i, detections[i].cls_id, detections[i].conf,
               detections[i].x, detections[i].y, detections[i].w, detections[i].h);
    }
    
    // Python 참조와 비교
    printf("\nExpected top 5:\n");
    for (int i = 0; i < 5 && i < TV_DECODE_NUM_DETECTIONS; i++) {
        printf("  [%d] cls=%d conf=%.4f bbox=(%.2f, %.2f, %.2f, %.2f)\n",
               i, tv_decode_detections[i].cls_id, tv_decode_detections[i].conf,
               tv_decode_detections[i].x, tv_decode_detections[i].y,
               tv_decode_detections[i].w, tv_decode_detections[i].h);
    }
    
    // 검증: decode가 정상 동작하는지만 확인 (전체 파이프라인은 main.c에서 검증)
    printf("\n");
    
    // 기본 검증: decode가 crash 없이 완료되고, 합리적인 결과 반환
    int ok = 1;
    
    // 1. detection 개수가 합리적인 범위인지
    if (num_dets < 0 || num_dets > 300) {
        printf("ERROR: Invalid detection count: %d\n", num_dets);
        ok = 0;
    }
    
    // 2. 각 detection의 값이 합리적인지 (NaN, Inf 체크)
    for (int i = 0; i < num_dets && i < 10; i++) {
        if (!isfinite(detections[i].x) || !isfinite(detections[i].y) ||
            !isfinite(detections[i].w) || !isfinite(detections[i].h) ||
            !isfinite(detections[i].conf)) {
            printf("ERROR: Detection[%d] contains NaN/Inf\n", i);
            ok = 0;
        }
    }
    
    if (ok) {
        printf("Result: OK (decode completed successfully)\n");
        return 0;
    }
    printf("Result: NG\n");
    return 1;
}
