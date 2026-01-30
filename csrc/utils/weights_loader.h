#ifndef WEIGHTS_LOADER_H
#define WEIGHTS_LOADER_H

#include <stdint.h>
#include <stddef.h>

#define MAX_TENSOR_DIMS 8

typedef struct {
    char* name;              // 텐서 이름 (동적 할당)
    float* data;             // 텐서 데이터 (할당 또는 DDR 직참조)
    int32_t ndim;            // 차원 수
    int32_t shape[MAX_TENSOR_DIMS]; // shape 배열 (고정 크기, 정렬 문제 방지)
    size_t num_elements;     // 총 원소 개수
    unsigned char data_owned; // 1 = loader가 할당(해제 시 free), 0 = 외부(DDR) 참조
} tensor_info_t;

// 가중치 로더 구조체
typedef struct {
    tensor_info_t* tensors;
    int32_t num_tensors;
} weights_loader_t;

int weights_init_from_memory(uintptr_t base_addr, size_t size, weights_loader_t* loader);

int weights_load_from_file(const char* bin_path, weights_loader_t* loader);

// 특정 이름의 텐서 찾기
// 반환값: 텐서 포인터, 없으면 NULL
const tensor_info_t* weights_find_tensor(const weights_loader_t* loader, const char* name);

const float* weights_get_tensor_data(const weights_loader_t* loader, const char* name);

void weights_free(weights_loader_t* loader);

#endif // WEIGHTS_LOADER_H
