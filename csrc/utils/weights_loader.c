#include "weights_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// 임베디드용: 메모리 직접 접근 (파일 시스템 없음)
// base_addr: .bin 데이터가 로드된 메모리 시작 주소
int weights_init_from_memory(uintptr_t base_addr, weights_loader_t* loader) {
    const uint8_t* ptr = (const uint8_t*)base_addr;
    
    // 텐서 개수 읽기
    uint32_t num_tensors;
    memcpy(&num_tensors, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    
    // 텐서 배열 할당 (메타데이터만)
    loader->tensors = (tensor_info_t*)calloc(num_tensors, sizeof(tensor_info_t));
    if (!loader->tensors) {
        return -1;
    }
    loader->num_tensors = num_tensors;
    loader->metadata_buffer = NULL;  // 메모리 직접 접근 모드에서는 할당 안 함
    
    // 각 텐서 메타데이터 파싱
    for (int i = 0; i < num_tensors; i++) {
        tensor_info_t* t = &loader->tensors[i];
        
        // 키 이름 길이
        uint32_t key_len;
        memcpy(&key_len, ptr, sizeof(uint32_t));
        ptr += sizeof(uint32_t);
        
        // 키 이름 포인터 저장 (메모리 내 문자열 직접 참조)
        t->name = (const char*)ptr;
        ptr += key_len;
        
        // shape 차원 수
        uint32_t ndim;
        memcpy(&ndim, ptr, sizeof(uint32_t));
        ptr += sizeof(uint32_t);
        t->ndim = ndim;
        
        // shape 배열 포인터 저장 (메모리 내 배열 직접 참조)
        t->shape = (const int32_t*)ptr;
        ptr += ndim * sizeof(uint32_t);
        
        // 원소 개수 계산
        t->num_elements = 1;
        for (int j = 0; j < ndim; j++) {
            t->num_elements *= t->shape[j];
        }
        
        // 데이터 포인터 저장 (메모리 내 데이터 직접 참조)
        t->data = (const float*)ptr;
        ptr += t->num_elements * sizeof(float);
    }
    
    return 0;
}

// 개발/테스트용: 파일 시스템에서 로드
int weights_load_from_file(const char* bin_path, weights_loader_t* loader) {
    FILE* f = fopen(bin_path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open weights file: %s\n", bin_path);
        return -1;
    }
    
    // 파일 크기 확인
    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    // 전체 파일을 메모리에 로드
    loader->metadata_buffer = (uint8_t*)malloc(file_size);
    if (!loader->metadata_buffer) {
        fclose(f);
        return -1;
    }
    
    if (fread(loader->metadata_buffer, 1, file_size, f) != file_size) {
        free(loader->metadata_buffer);
        loader->metadata_buffer = NULL;
        fclose(f);
        return -1;
    }
    fclose(f);
    
    // 메모리 직접 접근 방식과 동일하게 파싱
    const uint8_t* ptr = loader->metadata_buffer;
    
    // 텐서 개수 읽기
    uint32_t num_tensors;
    memcpy(&num_tensors, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    
    // 텐서 배열 할당
    loader->tensors = (tensor_info_t*)calloc(num_tensors, sizeof(tensor_info_t));
    if (!loader->tensors) {
        free(loader->metadata_buffer);
        loader->metadata_buffer = NULL;
        return -1;
    }
    loader->num_tensors = num_tensors;
    
    // 각 텐서 메타데이터 파싱
    for (int i = 0; i < num_tensors; i++) {
        tensor_info_t* t = &loader->tensors[i];
        
        // 키 이름 길이
        uint32_t key_len;
        memcpy(&key_len, ptr, sizeof(uint32_t));
        ptr += sizeof(uint32_t);
        
        // 키 이름 포인터 저장 (버퍼 내 문자열 직접 참조)
        t->name = (const char*)ptr;
        ptr += key_len;
        
        // shape 차원 수
        uint32_t ndim;
        memcpy(&ndim, ptr, sizeof(uint32_t));
        ptr += sizeof(uint32_t);
        t->ndim = ndim;
        
        // shape 배열 포인터 저장 (버퍼 내 배열 직접 참조)
        t->shape = (const int32_t*)ptr;
        ptr += ndim * sizeof(uint32_t);
        
        // 원소 개수 계산
        t->num_elements = 1;
        for (int j = 0; j < ndim; j++) {
            t->num_elements *= t->shape[j];
        }
        
        // 데이터 포인터 저장 (버퍼 내 데이터 직접 참조)
        t->data = (const float*)ptr;
        ptr += t->num_elements * sizeof(float);
    }
    
    return 0;
}

const tensor_info_t* weights_find_tensor(const weights_loader_t* loader, const char* name) {
    for (int i = 0; i < loader->num_tensors; i++) {
        // 키 이름은 null terminator 없이 저장되어 있으므로, 길이를 계산해서 비교
        const char* tensor_name = loader->tensors[i].name;
        size_t name_len = strlen(name);
        // 키 이름의 길이는 다음 필드(shape 차원 수)까지의 거리로 계산
        // 하지만 정확한 길이를 알 수 없으므로, strncmp 사용
        if (strncmp(tensor_name, name, name_len) == 0) {
            // 키 이름 뒤에 바로 다음 데이터가 오므로, 정확히 name_len 바이트인지 확인
            // (키 이름이 정확히 일치하는지 확인하기 위해 다음 문자가 숫자/문자가 아닌지 체크)
            char next_char = tensor_name[name_len];
            if (next_char == '\0' || next_char < 32 || next_char > 126) {
                // 다음 문자가 제어 문자이거나 범위 밖이면 키 이름이 끝난 것으로 간주
                return &loader->tensors[i];
            }
        }
    }
    return NULL;
}

const float* weights_get_tensor_data(const weights_loader_t* loader, const char* name) {
    const tensor_info_t* t = weights_find_tensor(loader, name);
    return t ? t->data : NULL;
}

void weights_free(weights_loader_t* loader) {
    if (!loader) return;
    
    // 파일 시스템 모드에서만 버퍼 해제
    if (loader->metadata_buffer) {
        free(loader->metadata_buffer);
        loader->metadata_buffer = NULL;
    }
    
    // 텐서 배열 해제 (메타데이터만, 실제 데이터는 버퍼/메모리에 있음)
    free(loader->tensors);
    loader->tensors = NULL;
    loader->num_tensors = 0;
}
