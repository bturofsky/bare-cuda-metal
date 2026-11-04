/*
 * v2_launch.c — end-to-end test of the kernel path:
 * cuModuleLoadData(raw PTX) → cuModuleGetFunction → cuLaunchKernel on Metal,
 * with results verified against the CPU.
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
  char *ptx = read_file("docs/reference/vectorAdd.ptx", &ptx_len);
  if (!ptx) { fprintf(stderr, "FAIL: cannot read docs/reference/vectorAdd.ptx\n"); return 2; }
  printf("loaded PTX (%zu bytes)\n", ptx_len);

  cuInit(0);

  float *ha = malloc(N * sizeof(float));
  float *hb = malloc(N * sizeof(float));
  float *hc = malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) { ha[i] = (float)i; hb[i] = (float)(2 * i); }

  CUdeviceptr da = 0, db = 0, dc = 0, module = 0, func = 0;
  int rc = 0;
  if ((rc = cuMemAlloc(&da, N * sizeof(float))) || (rc = cuMemAlloc(&db, N * sizeof(float))) ||
      (rc = cuMemAlloc(&dc, N * sizeof(float)))) {
    fprintf(stderr, "FAIL: cuMemAlloc rc=%d\n", rc);
    return 1;
  }
  cuMemcpyHtoD(da, ha, N * sizeof(float));
  cuMemcpyHtoD(db, hb, N * sizeof(float));

  if ((rc = cuModuleLoadData(&module, ptx))) {
    fprintf(stderr, "FAIL: cuModuleLoadData rc=%d\n", rc);
    return 1;
  }
  if ((rc = cuModuleGetFunction(&func, module, "_Z9vectorAddPKfS0_Pfi"))) {
    fprintf(stderr, "FAIL: cuModuleGetFunction rc=%d\n", rc);
    return 1;
  }

  /* grid 4 blocks x 256 threads = 1024 threads, i = ctaid*ntid + tid */
  unsigned n = N;
  void *params[5] = { &da, &db, &dc, &n, NULL };
  rc = cuLaunchKernel(func, 4, 1, 1, 256, 1, 1, 0, 0, params, NULL);
  if (rc) { fprintf(stderr, "FAIL: cuLaunchKernel rc=%d\n", rc); return 1; }
  cuCtxSynchronize();

  cuMemcpyDtoH(hc, dc, N * sizeof(float));

  int bad = 0;
  for (int i = 0; i < N; i++) {
    float want = ha[i] + hb[i];
    if (hc[i] != want) {
      bad++;
      if (bad <= 3) fprintf(stderr, "mismatch @%d: got %f want %f\n", i, hc[i], want);
    }
  }
  if (bad == 0)
    printf("V2 LAUNCH OK — %d/%d elements computed on the GPU match the CPU\n", N, N);
  else
    printf("V2 LAUNCH FAILED — %d/%d mismatches\n", bad, N);

  cuMemFree(da); cuMemFree(db); cuMemFree(dc);
  free(ha); free(hb); free(hc); free(ptx);
  return bad != 0;
}
