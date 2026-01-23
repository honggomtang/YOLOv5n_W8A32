#include "image_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// 임베디드용: 메모리 직접 접근 (파일 시스템 없음)
// base_addr: 이미지 데이터가 로드된 메모리 시작 주소
int image_init_from_memory(uintptr_t base_addr, preprocessed_image_t* img) {
    const uint8_t* ptr = (const uint8_t*)base_addr;
    
    // 헤더 읽기
    uint32_t original_w, original_h, size;
    float scale;
    uint32_t pad_x, pad_y;
    
    memcpy(&original_w, ptr, sizeof(uint32_t)); ptr += sizeof(uint32_t);
    memcpy(&original_h, ptr, sizeof(uint32_t)); ptr += sizeof(uint32_t);
    memcpy(&scale, ptr, sizeof(float)); ptr += sizeof(float);
    memcpy(&pad_x, ptr, sizeof(uint32_t)); ptr += sizeof(uint32_t);
    memcpy(&pad_y, ptr, sizeof(uint32_t)); ptr += sizeof(uint32_t);
    memcpy(&size, ptr, sizeof(uint32_t)); ptr += sizeof(uint32_t);
    
    img->original_w = original_w;
    img->original_h = original_h;
    img->scale = scale;
    img->pad_x = pad_x;
    img->pad_y = pad_y;
    img->c = 3;
    img->h = size;
    img->w = size;
    
    // 이미지 데이터 포인터 저장 (메모리 내 데이터 직접 참조)
    img->data = (const float*)ptr;
    img->buffer = NULL;  // 메모리 직접 접근 모드
    
    return 0;
}

// 개발/테스트용: 파일 시스템에서 로드
int image_load_from_bin(const char* bin_path, preprocessed_image_t* img) {
    FILE* f = fopen(bin_path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open image file: %s\n", bin_path);
        return -1;
    }
    
    // 파일 크기 확인
    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    // 전체 파일을 메모리에 로드
    uint8_t* buffer = (uint8_t*)malloc(file_size);
    if (!buffer) {
        fclose(f);
        return -1;
    }
    
    if (fread(buffer, 1, file_size, f) != file_size) {
        free(buffer);
        fclose(f);
        return -1;
    }
    fclose(f);
    
    // 메모리 직접 접근 방식과 동일하게 파싱
    const uint8_t* ptr = buffer;
    
    // 헤더 읽기
    uint32_t original_w, original_h, size;
    float scale;
    uint32_t pad_x, pad_y;
    
    memcpy(&original_w, ptr, sizeof(uint32_t)); ptr += sizeof(uint32_t);
    memcpy(&original_h, ptr, sizeof(uint32_t)); ptr += sizeof(uint32_t);
    memcpy(&scale, ptr, sizeof(float)); ptr += sizeof(float);
    memcpy(&pad_x, ptr, sizeof(uint32_t)); ptr += sizeof(uint32_t);
    memcpy(&pad_y, ptr, sizeof(uint32_t)); ptr += sizeof(uint32_t);
    memcpy(&size, ptr, sizeof(uint32_t)); ptr += sizeof(uint32_t);
    
    img->original_w = original_w;
    img->original_h = original_h;
    img->scale = scale;
    img->pad_x = pad_x;
    img->pad_y = pad_y;
    img->c = 3;
    img->h = size;
    img->w = size;
    
    // 이미지 데이터 포인터 저장 (버퍼 내 데이터 직접 참조)
    img->data = (const float*)ptr;
    img->buffer = buffer;  // 파일 시스템 모드: 버퍼 포인터 저장
    
    return 0;
}

void image_free(preprocessed_image_t* img) {
    if (!img) return;
    // 파일 시스템 모드에서만 버퍼 해제
    if (img->buffer) {
        free(img->buffer);
        img->buffer = NULL;
    }
    img->data = NULL;
}
