/**
 * 피처맵 풀 할당 (first-fit, 버퍼 재사용).
 * BARE_METAL: DDR FEATURE_POOL_BASE/SIZE. 호스트: malloc 한 번.
 */
#ifndef FEATURE_POOL_H
#define FEATURE_POOL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void feature_pool_init(void);
void* feature_pool_alloc(size_t size);
void feature_pool_free(void* ptr);
void feature_pool_reset(void);

size_t feature_pool_get_largest_free(void);

#ifdef __cplusplus
}
#endif

#endif /* FEATURE_POOL_H */
