/**
 * 레이어 내부 연산(operation)별 시간 수집·출력.
 * main.c에서 set_layer() 후 각 블록 호출 → 블록 내부에서 begin(op)/end() →
 * 마지막에 print()로 "L0 conv2d 123 ms", "L0 silu 45 ms" 등 출력.
 */
#ifndef TIMING_H
#define TIMING_H

#include <stdint.h>

#ifdef BARE_METAL
#ifndef CPU_MHZ
#define CPU_MHZ 100
#endif
#endif

#define YOLO_TIMING_OP_MAX  16   /* op 이름 최대 길이 */
#define YOLO_TIMING_ENTRIES 512  /* 최대 기록 개수 */

/** 현재 레이어 설정 (0..23: L0..L23, 24: det, 25: dec, 26: nms) */
void yolo_timing_set_layer(int layer_id);

/** 연산 시작 (op 이름 등록, 시각 기록) */
void yolo_timing_begin(const char* op);

/** 연산 종료 (구간 시간 기록) */
void yolo_timing_end(void);

/**
 * 직전 레이어(현재 cursor)에서 수집된 operation들을 한 줄로 출력.
 * 예) "    conv2d 189.43, silu 9.70 ms"
 * main.c에서 LAYER_LOG 직후 호출하는 용도.
 */
void yolo_timing_print_layer_ops(int layer_id);

/** 버퍼 비우기 (다음 추론 전 호출) */
void yolo_timing_reset(void);

#endif /* TIMING_H */
