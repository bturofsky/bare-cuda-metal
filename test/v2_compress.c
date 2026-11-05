/*
 * v2_compress.c — compressed-entry completeness: launches the same kernel
 * through a zstd-compressed container (FFLAG_ZSTD 0x8000, libzstd frame) and
 * a deflate-compressed container (flag 0x1000, zlib). Fixtures crafted from
 * the reference PTX with the documented entry layout.
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

static int run_case(const char *label, const char *path) {
  size_t flen = 0;
  char *fatbin = read_file(path, &flen);
  if (!fatbin) { printf("%s: MISSING %s\n", label, path); return 1; }
  cuInit(0);
  float *ha = malloc(N * sizeof(float)), *hb = malloc(N * sizeof(float)),
        *hc = malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) { ha[i] = 0.5f * (float)i; hb[i] = (float)i; }
  CUdeviceptr da = 0, db = 0, dc = 0, module = 0, func = 0;
  int rc = cuModuleLoadData(&module, fatbin);
  if (rc) { printf("%s: LOAD FAILED rc=%d\n", label, rc); return 1; }
  cuModuleGetFunction(&func, module, "_Z9vectorAddPKfS0_Pfi");
  cuMemAlloc(&da, N * sizeof(float));
  cuMemAlloc(&db, N * sizeof(float));
  cuMemAlloc(&dc, N * sizeof(float));
  cuMemcpyHtoD(da, ha, N * sizeof(float));
  cuMemcpyHtoD(db, hb, N * sizeof(float));
  unsigned n = N;
  void *params[5] = { &da, &db, &dc, &n, NULL };
  rc = cuLaunchKernel(func, 4, 1, 1, 256, 1, 1, 0, 0, params, NULL);
  if (rc) { printf("%s: LAUNCH FAILED rc=%d\n", label, rc); return 1; }
  cuCtxSynchronize();
  cuMemcpyDtoH(hc, dc, N * sizeof(float));
  int bad = 0;
  for (int i = 0; i < N; i++)
    if (hc[i] != ha[i] + hb[i]) bad++;
  if (bad == 0) printf("%s: OK — %d/%d verified (%zu B container)\n", label, N, N, flen);
  else printf("%s: FAILED — %d mismatches\n", label, bad);
  cuMemFree(da); cuMemFree(db); cuMemFree(dc);
  free(ha); free(hb); free(hc); free(fatbin);
  return bad != 0;
}

int main(void) {
  int failures = 0;
  failures += run_case("zstd   ", "docs/reference/vectorAdd.zstd.fatbin");
  failures += run_case("deflate", "docs/reference/vectorAdd.deflate.fatbin");
  if (failures == 0)
    printf("V2 COMPRESS OK — zstd + deflate entries verified through launch\n");
  else
    printf("V2 COMPRESS FAILED — %d case(s)\n", failures);
  return failures != 0;
}
