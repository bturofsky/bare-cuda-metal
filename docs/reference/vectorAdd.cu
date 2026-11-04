#include <stdio.h>
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
int main() {
  const int n = 1024; float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, n*4); cudaMalloc(&d_b, n*4); cudaMalloc(&d_c, n*4);
  vectorAdd<<<1, 256>>>(d_a, d_b, d_c, n);
  cudaDeviceSynchronize();
  printf("done\n"); return 0;
}
