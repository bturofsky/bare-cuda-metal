/*
 * v2_loop.c — control-flow test: grid-stride loop kernel (backward branch,
 * forward exit branch, nctaid/mul.lo/add.s32) verified on the GPU.
 * grid 2 x 256 = 512 threads, n = 1024 → every thread runs the loop twice.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned long long CUdeviceptr;

int cuInit(unsigned int flags);
int cuMemAlloc(CUdeviceptr *dptr, size_t bytes);
int cuMemFree(CUdeviceptr dptr);
int cuMemcpyHtoD(CUdeviceptr dst, const void *src, size_t bytes);
int cuMemcpyDtoH(void *dst, CUdeviceptr src, size_t bytes);
int cuModuleLoadData(CUdeviceptr *module, const void *image);
int cuModuleGetFunction(CUdeviceptr *func, CUdeviceptr module, const char *name);
int cuLaunchKernel(CUdeviceptr func, unsigned gx, unsigned gy, unsigned gz,
                   unsigned bx, unsigned by, unsigned bz, unsigned sharedMem,
                   CUdeviceptr stream, void **params, void **extra);
int cuCtxSynchronize(void);

static char *read_file(const char *path, size_t *len) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *buf = malloc((size_t)sz + 1);
  size_t rd = fread(buf, 1, (size_t)sz, f);
  fclose(f);
  buf[rd] = 0;
  *len = rd;
  return buf;
}

#define N 1024

int main(void) {
  size_t ptx_len = 0;
  char *ptx = read_file("docs/reference/scaleLoop.ptx", &ptx_len);
  if (!ptx) { fprintf(stderr, "FAIL: cannot read docs/reference/scaleLoop.ptx\n"); return 2; }

  cuInit(0);

  float *hx = malloc(N * sizeof(float)), *hw = malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) hx[i] = (float)(i + 1);
  float s = 0.25f;

  CUdeviceptr dx = 0, module = 0, func = 0;
  int rc = cuModuleLoadData(&module, ptx);
  if (rc) { fprintf(stderr, "FAIL: cuModuleLoadData rc=%d\n", rc); return 1; }
  rc = cuModuleGetFunction(&func, module, "_Z9scaleLoopPffi");
  if (rc) { fprintf(stderr, "FAIL: cuModuleGetFunction rc=%d\n", rc); return 1; }
  cuMemAlloc(&dx, N * sizeof(float));
  cuMemcpyHtoD(dx, hx, N * sizeof(float));

  unsigned n = N;
  void *params[4] = { &dx, &s, &n, NULL };
  rc = cuLaunchKernel(func, 2, 1, 1, 256, 1, 1, 0, 0, params, NULL);
  if (rc) { fprintf(stderr, "FAIL: cuLaunchKernel rc=%d\n", rc); return 1; }
  cuCtxSynchronize();
  cuMemcpyDtoH(hx, dx, N * sizeof(float));

  int bad = 0;
  for (int i = 0; i < N; i++) {
    hw[i] = (float)(i + 1) * s;
    if (hx[i] != hw[i]) bad++;
  }
  if (bad == 0)
    printf("V2 LOOP OK — grid-stride loop kernel: %d/%d verified (512 threads x 2 iterations)\n", N, N);
  else
    printf("V2 LOOP FAILED — %d/%d mismatches\n", bad, N);

  cuMemFree(dx);
  free(hx); free(hw); free(ptx);
  return bad != 0;
}
