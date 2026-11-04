/* gemm_ab.c — the Mac arm: the SAME GEMM spec via the shim (cuBLAS→MPS).
 * Build: make gemm_ab   Run: ./gemm_ab */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define N 512
#define SPEC_SEED 0xC0FFEEu
static uint32_t spec_next(uint32_t *s) { *s = (*s * 1103515245u + 12345u) & 0x7fffffffu; return *s; }

int main(void) {
  printf("main entered\n"); fflush(stdout);
  setvbuf(stdout, NULL, _IONBF, 0);
  void *h_cuda = dlopen("shim/libcuda_shim.dylib", RTLD_NOW | RTLD_GLOBAL);
  void *h = dlopen("shim/libcublas_shim.dylib", RTLD_NOW);
  if (!h_cuda || !h) { printf("FAIL dlopen: %s\n", dlerror()); return 1; }
  int (*cublasCreate)(uint64_t *) = dlsym(h, "cublasCreate");
  int (*cublasSgemm)(uint64_t, int, int, int, int, int, const float *,
                     const float *, int, const float *, int, const float *,
                     float *, int) = dlsym(h, "cublasSgemm");
  void *(*cudaMalloc)(void **, size_t) = dlsym(h_cuda, "cudaMalloc");
  int (*cuInit)(unsigned) = dlsym(h_cuda, "cuInit");
  if (!cuInit) { printf("FAIL: cuInit missing\n"); return 1; }
  int irc = cuInit(0);
  printf("cuInit -> %d\n", irc); fflush(stdout);

  static float a[N * N], b[N * N], c[N * N];
  uint32_t s = SPEC_SEED;
  for (int i = 0; i < N * N; i++) {
    a[i] = (float)(spec_next(&s) % 200) / 100.0f - 1.0f;
    b[i] = (float)(spec_next(&s) % 200) / 100.0f - 1.0f;
  }
  float alpha = 1.0f, beta = 0.0f;
  float *d_a, *d_b, *d_c;
  int mrc = (int)(uintptr_t)cudaMalloc((void **)&d_a, N * N * 4);
  if (mrc != 0) { printf("cudaMalloc failed: %d\n", mrc); return 1; }
  cudaMalloc((void **)&d_b, N * N * 4);
  cudaMalloc((void **)&d_c, N * N * 4);
  memcpy(d_a, a, N * N * 4);
  memcpy(d_b, b, N * N * 4);

  uint64_t handle = 0;
  cublasCreate(&handle);
  int rc = cublasSgemm(handle, 0 /*NN*/, 0, N, N, N, &alpha, d_a, N, d_b, N, &beta, d_c, N);

  uint64_t chk = 1469598103934665603ULL;
  const unsigned char *p = (const unsigned char *)d_c;
  for (size_t i = 0; i < N * N * 4; i++) { chk ^= p[i]; chk *= 1099511628211ULL; }
  printf("ELF C[0]=%.3f C[last]=%.3f\n", d_c[0], d_c[(size_t)N * N - 1]); fflush(stdout);
  printf("{\"backend\":\"metal-shim-mps\",\"n\":%d,\"checksum\":\"%016llx\",\"first\":%.6f,\"last\":%.6f}\n",
           N, (unsigned long long)chk, d_c[0], d_c[(size_t)N * N - 1]);
  return 0;
}
