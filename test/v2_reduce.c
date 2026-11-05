/*
 * v2_reduce.c — shared-memory + atomic test:
 * blockSum: classic tree reduction over __shared__ (bar.sync, ld/st.shared,
 *   shr loop) — 1 block x 256 threads, 256 int-valued floats (all partials
 *   < 2^24, so any summation order is exact).
 * atomicCount: atom.global.add.u32 across 128 threads — counter must be 128.
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

#define N 256

int main(void) {
  size_t len = 0;
  char *ptx_sum = read_file("docs/reference/blockSum.ptx", &len);
  char *ptx_cnt = read_file("docs/reference/atomicCount.ptx", &len);
  if (!ptx_sum || !ptx_cnt) { fprintf(stderr, "FAIL: missing reference PTX\n"); return 2; }

  cuInit(0);
  int failures = 0;

  /* ---- blockSum ---- */
  {
    float *hin = malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) hin[i] = (float)(i + 1);
    CUdeviceptr d_in = 0, d_out = 0, module = 0, func = 0;
    cuModuleLoadData(&module, ptx_sum);
    cuModuleGetFunction(&func, module, "_Z8blockSumPPfS_i");
    cuMemAlloc(&d_in, N * sizeof(float));
    cuMemAlloc(&d_out, sizeof(float));
    cuMemcpyHtoD(d_in, hin, N * sizeof(float));
    unsigned n = N;
    void *params[4] = { &d_out, &d_in, &n, NULL };
    int rc = cuLaunchKernel(func, 1, 1, 1, 256, 1, 1, 0, 0, params, NULL);
    if (rc) { printf("blockSum: LAUNCH FAILED rc=%d\n", rc); failures++; }
    else {
      cuCtxSynchronize();
      float got = 0;
      cuMemcpyDtoH(&got, d_out, sizeof(float));
      float want = (float)(N * (N + 1) / 2); /* 32896 — exact in float */
      if (got == want)
        printf("blockSum: OK — tree reduction over __shared__: %g == %g\n", got, want);
      else { printf("blockSum: FAILED — got %g want %g\n", got, want); failures++; }
    }
    cuMemFree(d_in); cuMemFree(d_out);
    free(hin);
  }

  /* ---- atomicCount ---- */
  {
    CUdeviceptr d_cnt = 0, module = 0, func = 0;
    unsigned zero = 0;
    cuModuleLoadData(&module, ptx_cnt);
    cuModuleGetFunction(&func, module, "_Z11atomicCountPi");
    cuMemAlloc(&d_cnt, sizeof(unsigned));
    cuMemcpyHtoD(d_cnt, &zero, sizeof(unsigned));
    void *params[2] = { &d_cnt, NULL };
    int rc = cuLaunchKernel(func, 4, 1, 1, 32, 1, 1, 0, 0, params, NULL);
    if (rc) { printf("atomicCount: LAUNCH FAILED rc=%d\n", rc); failures++; }
    else {
      cuCtxSynchronize();
      unsigned got = 0;
      cuMemcpyDtoH(&got, d_cnt, sizeof(unsigned));
      if (got == 128)
        printf("atomicCount: OK — atom.global.add.u32 across 128 threads: %u == 128\n", got);
      else { printf("atomicCount: FAILED — counter %u want 128\n", got); failures++; }
    }
    cuMemFree(d_cnt);
  }

  if (failures == 0)
    printf("V2 REDUCE OK — shared memory, barriers, and atomics verified on the GPU\n");
  else
    printf("V2 REDUCE FAILED — %d case(s)\n", failures);
  free(ptx_sum); free(ptx_cnt);
  return failures != 0;
}
