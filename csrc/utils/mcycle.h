/**
 * 단계별 시간/사이클 측정
 * - BARE_METAL (MicroBlaze V / RISC-V): mcycle + mcycleh → 64비트 사이클
 * - 호스트: 고해상도 타이머 → 마이크로초 (출력 시 ms/sec)
 * 파일 I/O 없음.
 */
#ifndef MCYCLE_H
#define MCYCLE_H

#include <stdint.h>

#if defined(BARE_METAL)

/* -------- RISC-V: 64비트 mcycle -------- */
#define RISCV_CSR_MCYCLE   0xB00
#define RISCV_CSR_MCYCLEH  0xB80

static inline uint32_t mcycle_read_lo(void) {
    uint32_t v;
    __asm__ __volatile__("csrr %0, 0xB00" : "=r"(v) : : "memory");
    return v;
}

static inline uint32_t mcycle_read_hi(void) {
    uint32_t v;
    __asm__ __volatile__("csrr %0, 0xB80" : "=r"(v) : : "memory");
    return v;
}

/**
 * 64비트 사이클: (mcycleh << 32) | mcycle.
 * hi -> lo -> hi2 순으로 읽고, 항상 hi2와 lo를 사용하여 하위 32비트 롤오버
 * (0xFFFFFFFF -> 0x00000000) 찰나에 읽을 때 값이 뒤로 튀는 현상을 방지.
 */
static inline uint64_t mcycle_read64(void) {
    uint32_t hi  = mcycle_read_hi();
    uint32_t lo  = mcycle_read_lo();
    uint32_t hi2 = mcycle_read_hi();
    if (hi != hi2)
        lo = mcycle_read_lo();  /* 롤오버 발생 시 lo 재읽기 (hi2와 쌍 맞춤) */
    return ((uint64_t)hi2 << 32) | (uint64_t)lo;
}

static inline uint64_t mcycle_delta64(uint64_t start, uint64_t end) {
    return end - start;
}

#define timer_read64()   mcycle_read64()
#define timer_delta64(s, e)  mcycle_delta64(s, e)

#else

/* -------- 호스트: 고해상도 타이머 (마이크로초) -------- */
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
static inline uint64_t host_time_us(void) {
    static LARGE_INTEGER freq = { 0 };
    LARGE_INTEGER c;
    if (freq.QuadPart == 0)
        QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&c);
    return (uint64_t)((double)c.QuadPart * 1000000.0 / (double)freq.QuadPart);
}
#else
#include <sys/time.h>
static inline uint64_t host_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;
}
#endif

static inline uint64_t timer_delta64(uint64_t start, uint64_t end) {
    return end - start;
}

#define timer_read64()   host_time_us()

#endif /* BARE_METAL */

#endif /* MCYCLE_H */
