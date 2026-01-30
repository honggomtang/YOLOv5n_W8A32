/** UART 검출 결과 전송 (BARE_METAL). PC 스크립트로 시리얼 수신 → detections.bin */
#ifndef UART_DUMP_H
#define UART_DUMP_H

#include <stdint.h>

struct hw_detection_t;

void yolo_uart_send_detections(const void* hw_detections, uint8_t count);

#endif /* UART_DUMP_H */
