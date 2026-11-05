/*
 * v2_section.c — self-hosted fatbin: the REAL nvcc fatbin is embedded in
 * this binary's own __NV_CUDA,__nv_fatbin section (the layout NVIDIA's
 * fatBinaryCtl.h defines for Mach-O). cuModuleLoadData receives the
 * section pointer with NO size — the shim must resolve the exact extent
 * by walking loaded Mach-O images with getsectiondata().
 * The section array is generated at build time from the committed fixture.
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

#include "nv_fatbin_section.h"

#define N 1024

int main(void) {
  cuInit(0);

  float *ha = malloc(N * sizeof(float)), *hb = malloc(N * sizeof(float)),
        *hc = malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) { ha[i] = (float)i; hb[i] = (float)(3 * i); }

  CUdeviceptr da = 0, db = 0, dc = 0, module = 0, func = 0;
  int rc = cuModuleLoadData(&module, g_nv_fatbin); /* pointer into our own image */
  if (rc) { fprintf(stderr, "FAIL: section cuModuleLoadData rc=%d\n", rc); return 1; }
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
    printf("V2 SECTION OK — self-hosted __NV_CUDA,__nv_fatbin extent resolved, %d/%d verified\n", N, N);
  else
    printf("V2 SECTION FAILED — %d/%d mismatches\n", bad, N);

  cuMemFree(da); cuMemFree(db); cuMemFree(dc);
  free(ha); free(hb); free(hc);
  return bad != 0;
}
