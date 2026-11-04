/*
 * v2_launch.c — end-to-end test of the kernel path:
 * cuModuleLoadData(raw PTX) → cuModuleGetFunction → cuLaunchKernel on Metal,
 * with results verified against the CPU.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned long long CUdeviceptr;

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

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

  double t0 = now_ms();
  rc = cuLaunchKernel(func, 4, 1, 1, 256, 1, 1, 0, 0, params, NULL);
  double t1 = now_ms();
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

  /* Second launch: same kernel, fresh data — exercises the pipeline cache
     and proves buffers rebind correctly (not stale from launch 1). */
  for (int i = 0; i < N; i++) { ha[i] = 3.0f * i + 1.0f; hb[i] = -2.0f * i; }
  cuMemcpyHtoD(da, ha, N * sizeof(float));
  cuMemcpyHtoD(db, hb, N * sizeof(float));
  double t2 = now_ms();
  rc = cuLaunchKernel(func, 4, 1, 1, 256, 1, 1, 0, 0, params, NULL);
  double t3 = now_ms();
  if (rc) { fprintf(stderr, "FAIL: cached cuLaunchKernel rc=%d\n", rc); return 1; }
  cuCtxSynchronize();
  cuMemcpyDtoH(hc, dc, N * sizeof(float));
  for (int i = 0; i < N; i++) {
    float want = ha[i] + hb[i];
    if (hc[i] != want) {
      bad++;
      if (bad <= 6) fprintf(stderr, "cached mismatch @%d: got %f want %f\n", i, hc[i], want);
    }
  }

  if (bad == 0)
    printf("V2 LAUNCH OK — 2x%d/%d GPU elements match CPU; "
           "launch1 %.0f ms (compile+dispatch), launch2 %.3f ms (cached)\n",
           N, N, t1 - t0, t3 - t2);
  else
    printf("V2 LAUNCH FAILED — %d mismatches across both launches\n", bad);

  cuMemFree(da); cuMemFree(db); cuMemFree(dc);
  free(ha); free(hb); free(hc); free(ptx);
  return bad != 0;
}
