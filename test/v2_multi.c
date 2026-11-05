/*
 * v2_multi.c — multi-kernel module test: one PTX image with two kernels
 * (vectorAdd + in-place scale), two function handles from one module,
 * interleaved launches, float-scalar param binding, and a cache-hit
 * relaunch to prove cached param types rebind correctly.
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

static int verify(const float *hc, const float *want) {
  int bad = 0;
  for (int i = 0; i < N; i++)
    if (hc[i] != want[i]) bad++;
  return bad;
}

int main(void) {
  size_t ptx_len = 0;
  char *ptx = read_file("docs/reference/vectorAdd_scale.ptx", &ptx_len);
  if (!ptx) { fprintf(stderr, "FAIL: cannot read docs/reference/vectorAdd_scale.ptx\n"); return 2; }

  cuInit(0);

  float *ha = malloc(N * sizeof(float)), *hb = malloc(N * sizeof(float)),
        *hc = malloc(N * sizeof(float)), *hx = malloc(N * sizeof(float)),
        *want = malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) { ha[i] = (float)i; hb[i] = (float)(2 * i); }

  CUdeviceptr da = 0, db = 0, dc = 0, dx = 0, module = 0, f_add = 0, f_scale = 0;
  int rc = cuModuleLoadData(&module, ptx);
  if (rc) { fprintf(stderr, "FAIL: cuModuleLoadData rc=%d\n", rc); return 1; }
  rc = cuModuleGetFunction(&f_add, module, "_Z9vectorAddPKfS0_Pfi");
  if (rc) { fprintf(stderr, "FAIL: GetFunction vectorAdd rc=%d\n", rc); return 1; }
  rc = cuModuleGetFunction(&f_scale, module, "_Z5scalePffi");
  if (rc) { fprintf(stderr, "FAIL: GetFunction scale rc=%d\n", rc); return 1; }

  cuMemAlloc(&da, N * sizeof(float));
  cuMemAlloc(&db, N * sizeof(float));
  cuMemAlloc(&dc, N * sizeof(float));
  cuMemAlloc(&dx, N * sizeof(float));
  cuMemcpyHtoD(da, ha, N * sizeof(float));
  cuMemcpyHtoD(db, hb, N * sizeof(float));

  unsigned n = N;
  int bad = 0;

  /* launch 1: vectorAdd (compile path, types from mangled name) */
  void *p1[5] = { &da, &db, &dc, &n, NULL };
  rc = cuLaunchKernel(f_add, 4, 1, 1, 256, 1, 1, 0, 0, p1, NULL);
  if (rc) { fprintf(stderr, "FAIL: vectorAdd launch rc=%d\n", rc); return 1; }
  cuCtxSynchronize();
  cuMemcpyDtoH(hc, dc, N * sizeof(float));
  for (int i = 0; i < N; i++) want[i] = ha[i] + hb[i];
  bad += verify(hc, want);

  /* launch 2: scale (same image, second kernel — float scalar binding) */
  float s = 3.5f;
  cuMemcpyHtoD(dx, ha, N * sizeof(float));
  void *p2[4] = { &dx, &s, &n, NULL };
  rc = cuLaunchKernel(f_scale, 4, 1, 1, 256, 1, 1, 0, 0, p2, NULL);
  if (rc) { fprintf(stderr, "FAIL: scale launch rc=%d\n", rc); return 1; }
  cuCtxSynchronize();
  cuMemcpyDtoH(hx, dx, N * sizeof(float));
  for (int i = 0; i < N; i++) want[i] = ha[i] * s;
  bad += verify(hx, want);

  /* launch 3: vectorAdd again (cache-hit path — types from the cache) */
  rc = cuLaunchKernel(f_add, 4, 1, 1, 256, 1, 1, 0, 0, p1, NULL);
  if (rc) { fprintf(stderr, "FAIL: cached vectorAdd launch rc=%d\n", rc); return 1; }
  cuCtxSynchronize();
  cuMemcpyDtoH(hc, dc, N * sizeof(float));
  for (int i = 0; i < N; i++) want[i] = ha[i] + hb[i];
  bad += verify(hc, want);

  if (bad == 0)
    printf("V2 MULTI OK — vectorAdd + scale (f32 scalar) + cached relaunch: %dx%d verified\n", 3, N);
  else
    printf("V2 MULTI FAILED — %d mismatches\n", bad);

  cuMemFree(da); cuMemFree(db); cuMemFree(dc); cuMemFree(dx);
  free(ha); free(hb); free(hc); free(hx); free(want); free(ptx);
  return bad != 0;
}
