#include "image_loader.h"
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#ifndef BARE_METAL
#include <stdio.h>
#endif

static inline void safe_read(void* dest, const uint8_t** src, size_t size) {
    memcpy(dest, *src, size);
    *src += size;
}

static int parse_image_data(const uint8_t* ptr, size_t data_len, preprocessed_image_t* img, int zero_copy) {
    const uint8_t* curr = ptr;
    const uint8_t* end = ptr + data_len;

    if (curr + 24 > end) return -1;
    // 헤더 24B
    uint32_t original_w, original_h, size;
    float scale;
    uint32_t pad_x, pad_y;
    
    safe_read(&original_w, &curr, 4);
    safe_read(&original_h, &curr, 4);
    safe_read(&scale, &curr, 4);
    safe_read(&pad_x, &curr, 4);
    safe_read(&pad_y, &curr, 4);
    safe_read(&size, &curr, 4);
    
    img->original_w = (int32_t)original_w;
    img->original_h = (int32_t)original_h;
    img->scale = scale;
    img->pad_x = (int32_t)pad_x;
    img->pad_y = (int32_t)pad_y;
    img->c = 3;
    img->h = (int32_t)size;
    img->w = (int32_t)size;
    
    size_t data_bytes = 3 * (size_t)size * (size_t)size * sizeof(float);
    if (curr + data_bytes > end) return -1;

    if (zero_copy) {
        img->data = (float*)curr;
        img->data_owned = 0;
    } else {
        img->data = (float*)malloc(data_bytes);
        if (!img->data) return -1;
        safe_read(img->data, &curr, data_bytes);
        img->data_owned = 1;
    }
    return 0;
}

int image_init_from_memory(uintptr_t base_addr, size_t size, preprocessed_image_t* img) {
    if (!img || size < 24) return -1;
    return parse_image_data((const uint8_t*)base_addr, size, img, 1);
}

#ifndef BARE_METAL
int image_load_from_bin(const char* bin_path, preprocessed_image_t* img) {
    FILE* f = fopen(bin_path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open image file: %s\n", bin_path);
        return -1;
    }
    
    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
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
    
    int ret = parse_image_data(buffer, file_size, img, 0);
    free(buffer);
    
    return ret;
}
#endif

void image_free(preprocessed_image_t* img) {
    if (!img) return;
    if (img->data_owned && img->data) {
        free(img->data);
        img->data = NULL;
    }
}
