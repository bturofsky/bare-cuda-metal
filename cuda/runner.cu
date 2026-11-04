// runner.cu — the CUDA arm of the A/B test. Runs the spec workload on the
// RTX 8000 and emits the JSON record: checksum + timing + sample values.
// Build (bb23): nvcc -O2 -arch=sm_75 -o cuda_runner runner.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include "spec.h"

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main(void) {
  static float a[SPEC_N], b[SPEC_N], c[SPEC_N];
  spec_fill(a, b);

  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, SPEC_N * 4);
  cudaMalloc(&d_b, SPEC_N * 4);
  cudaMalloc(&d_c, SPEC_N * 4);

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaMemcpy(d_a, a, SPEC_N * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, SPEC_N * 4, cudaMemcpyHostToDevice);

  cudaEventRecord(t0);
  vectorAdd<<<(SPEC_N + 255) / 256, 256>>>(d_a, d_b, d_c, SPEC_N);
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  float gpu_ms = 0;
  cudaEventElapsedTime(&gpu_ms, t0, t1);

  cudaMemcpy(c, d_c, SPEC_N * 4, cudaMemcpyDeviceToHost);
  spec_json("cuda-rtx8000", spec_checksum(c), gpu_ms, c);
  return 0;
}
