/*
 * v2_realfat.c — REAL nvcc fatbin acceptance: docs/reference/vectorAdd.real.fatbin
 * was produced by CUDA 12.4 nvcc on the compute host from our own
 * vectorAdd.cu (-gencode arch=compute_86,code=compute_86), then extracted
 * with objcopy --only-section=.nv_fatbin. It contains one LZ4-compressed
 * PTX entry with nvcc's custom framing (possible uncompressed footer).
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
  size_t fat_len = 0;
  char *fatbin = read_file("docs/reference/vectorAdd.real.fatbin", &fat_len);
  if (!fatbin) { fprintf(stderr, "FAIL: missing vectorAdd.real.fatbin\n"); return 2; }

  cuInit(0);

  float *ha = malloc(N * sizeof(float)), *hb = malloc(N * sizeof(float)),
        *hc = malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) { ha[i] = (float)i + 0.5f; hb[i] = (float)(2 * i); }

  CUdeviceptr da = 0, db = 0, dc = 0, module = 0, func = 0;
  int rc = cuModuleLoadData(&module, fatbin);
  if (rc) { fprintf(stderr, "FAIL: real-fatbin cuModuleLoadData rc=%d\n", rc); return 1; }
  rc = cuModuleGetFunction(&func, module, "_Z9vectorAddPKfS0_Pfi");
  if (rc) { fprintf(stderr, "FAIL: GetFunction rc=%d\n", rc); return 1; }

  cuMemAlloc(&da, N * sizeof(float));
  cuMemAlloc(&db, N * sizeof(float));
  cuMemAlloc(&dc, N * sizeof(float));
  cuMemcpyHtoD(da, ha, N * sizeof(float));
  cuMemcpyHtoD(db, hb, N * sizeof(float));

  unsigned n = N;
  void *params[5] = { &da, &db, &dc, &n, NULL };
  rc = cuLaunchKernel(func, 4, 1, 1, 256, 1, 1, 0, 0, params, NULL);
  if (rc) { fprintf(stderr, "FAIL: launch rc=%d\n", rc); return 1; }
  cuCtxSynchronize();
  cuMemcpyDtoH(hc, dc, N * sizeof(float));

  int bad = 0;
  for (int i = 0; i < N; i++)
    if (hc[i] != ha[i] + hb[i]) bad++;

  if (bad == 0)
    printf("V2 REALFAT OK — real nvcc 12.4 compressed-PTX fatbin launched: %d/%d verified\n", N, N);
  else
    printf("V2 REALFAT FAILED — %d/%d mismatches\n", bad, N);

  cuMemFree(da); cuMemFree(db); cuMemFree(dc);
  free(ha); free(hb); free(hc); free(fatbin);
  return bad != 0;
}
