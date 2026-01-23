#ifndef WEIGHTS_LOADER_H
#define WEIGHTS_LOADER_H

#include <stdint.h>
#include <stddef.h>

// 텐서 정보 구조체
typedef struct {
    const char* name;     // 텐서 이름 (메모리 직접 접근 시 포인터만 저장)
    const float* data;    // 텐서 데이터 (메모리 직접 접근 시 읽기 전용)
    int32_t ndim;         // 차원 수
    const int32_t* shape; // shape 배열 (메모리 직접 접근 시 읽기 전용)
    size_t num_elements;  // 총 원소 개수
} tensor_info_t;

// 가중치 로더 구조체
typedef struct {
    tensor_info_t* tensors;
    int32_t num_tensors;
    // 메모리 직접 접근 모드인 경우, 메타데이터만 저장 (실제 데이터는 메모리 주소에 직접 매핑)
    uint8_t* metadata_buffer;  // 파일 시스템 모드에서만 사용
} weights_loader_t;

// ===== 임베디드용: 메모리 직접 접근 (파일 시스템 없음) =====
// Vitis에서 특정 메모리 주소(예: 0x81000000)에 .bin 데이터를 로드한 후,
// 해당 주소를 base_addr로 전달하여 초기화
// 반환값: 0 성공, -1 실패
int weights_init_from_memory(uintptr_t base_addr, weights_loader_t* loader);

// ===== 개발/테스트용: 파일 시스템에서 로드 =====
// 반환값: 0 성공, -1 실패
int weights_load_from_file(const char* bin_path, weights_loader_t* loader);

// 특정 이름의 텐서 찾기
// 반환값: 텐서 포인터, 없으면 NULL
const tensor_info_t* weights_find_tensor(const weights_loader_t* loader, const char* name);

// 특정 이름의 텐서 데이터 포인터 가져오기 (편의 함수)
// 반환값: 텐서 데이터 포인터, 없으면 NULL
const float* weights_get_tensor_data(const weights_loader_t* loader, const char* name);

// 가중치 로더 해제 (파일 시스템 모드에서만 필요)
void weights_free(weights_loader_t* loader);

#endif // WEIGHTS_LOADER_H
