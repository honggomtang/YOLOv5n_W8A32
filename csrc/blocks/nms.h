#ifndef NMS_H
#define NMS_H

#include <stdint.h>
#include "decode.h"

float calculate_iou(const detection_t* box1, const detection_t* box2);

int nms(
    detection_t* detections,           // 입력: detection 배열
    int32_t num_detections,            // 입력: detection 개수
    detection_t** output_detections,    // 출력: NMS 적용 후 detection 배열 (동적 할당)
    int32_t* output_count,             // 출력: 남은 detection 개수
    float iou_threshold,               // IoU 임계값 (일반적으로 0.45)
    int32_t max_detections);           // 최대 detection 개수

#endif // NMS_H
