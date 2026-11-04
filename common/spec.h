/* spec.h — the portable workload spec shared by the CUDA and Metal runners.
 * Deterministic: same seed → same inputs → fp32 elementwise add is IEEE-exact
 * on both vendors, so result checksums must MATCH BITWISE. */
#ifndef SPEC_H
#define SPEC_H
#include <stdint.h>
#include <stdio.h>

#define SPEC_N 65536           /* elements */
#define SPEC_SEED 0xC0FFEEu    /* identical PRNG seed on both backends */

/* Portable deterministic LCG (identical sequence on any platform). */
static uint32_t spec_next(uint32_t *state) {
  *state = (*state * 1103515245u + 12345u) & 0x7fffffffu;
  return *state;
}
static void spec_fill(float *a, float *b) {
  uint32_t s = SPEC_SEED;
  for (int i = 0; i < SPEC_N; i++) {
    a[i] = (float)(spec_next(&s) % 20000) / 100.0f - 100.0f; /* [-100, 100) */
    b[i] = (float)(spec_next(&s) % 20000) / 100.0f - 100.0f;
  }
}
/* FNV-1a over the result bytes — the cross-vendor equality proof. */
static uint64_t spec_checksum(const float *c) {
  uint64_t h = 1469598103934665603ULL;
  const unsigned char *p = (const unsigned char *)c;
  for (size_t i = 0; i < SPEC_N * sizeof(float); i++) { h ^= p[i]; h *= 1099511628211ULL; }
  return h;
}
static void spec_json(const char *backend, uint64_t checksum, double ms,
                      const float *c) {
  printf("{\"backend\":\"%s\",\"n\":%d,\"checksum\":\"%016llx\",\"ms\":%.3f,"
         "\"first\":%.6f,\"mid\":%.6f,\"last\":%.6f}\n",
         backend, SPEC_N, (unsigned long long)checksum, ms, c[0],
         c[SPEC_N / 2], c[SPEC_N - 1]);
}
#endif
