// runner_blas.cu — the CUDA arm of the GEMM test: cuBLAS on the RTX 8000.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define N 512
#define SPEC_SEED 0xC0FFEEu
static uint32_t spec_next(uint32_t *s) { *s = (*s * 1103515245u + 12345u) & 0x7fffffffu; return *s; }

int main(void) {
  static float a[N * N], b[N * N], c[N * N];
  uint32_t s = SPEC_SEED;
  for (int i = 0; i < N * N; i++) {
    a[i] = (float)(spec_next(&s) % 200) / 100.0f - 1.0f;
    b[i] = (float)(spec_next(&s) % 200) / 100.0f - 1.0f;
  }
  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, N * N * 4);
  cudaMalloc(&d_b, N * N * 4);
  cudaMalloc(&d_c, N * N * 4);
  cudaMemcpy(d_a, a, N * N * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, N * N * 4, cudaMemcpyHostToDevice);

  cublasHandle_t h;
  cublasCreate(&h);
  float alpha = 1.0f, beta = 0.0f;
  cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_a, N, d_b, N, &beta, d_c, N);
  cudaDeviceSynchronize();
  cudaMemcpy(c, d_c, N * N * 4, cudaMemcpyDeviceToHost);

  uint64_t chk = 1469598103934665603ULL;
  const unsigned char *p = (const unsigned char *)c;
  for (size_t i = 0; i < N * N * 4; i++) { chk ^= p[i]; chk *= 1099511628211ULL; }
  printf("{\"backend\":\"cublas-rtx8000\",\"n\":%d,\"checksum\":\"%016llx\",\"first\":%.6f,\"last\":%.6f}\n",
         N, (unsigned long long)chk, c[0], c[(size_t)N * N - 1]);
  return 0;
}
