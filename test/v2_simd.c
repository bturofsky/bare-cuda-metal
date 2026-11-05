/*
 * v2_simd.c — warp shuffle (simd_shuffle) and selp (ternary + 0f bit-pattern
 * immediates) through the translator, verified on the GPU.
 * shflTest: one warp, out[i] = lane-0 broadcast of in → out[i] == in[0].
 * selpTest: out[i] = (i >= n) ? 0.0f : in[i].
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

int main(void) {
  size_t len = 0;
  char *ptx = read_file("docs/reference/shfl_sel.ptx", &len);
  if (!ptx) { fprintf(stderr, "FAIL: missing shfl_sel.ptx\n"); return 2; }

  cuInit(0);
  int failures = 0;

  /* ---- shflTest: 1 warp, lane-0 broadcast ---- */
  {
    const int N = 32;
    float *hin = malloc(N * sizeof(float)), *hout = malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) hin[i] = 100.0f + (float)i;
    CUdeviceptr d_in = 0, d_out = 0, module = 0, func = 0;
    cuModuleLoadData(&module, ptx);
    cuModuleGetFunction(&func, module, "_Z8shflTestPfS_i");
    cuMemAlloc(&d_in, N * sizeof(float));
    cuMemAlloc(&d_out, N * sizeof(float));
    cuMemcpyHtoD(d_in, hin, N * sizeof(float));
    unsigned n = N;
    void *params[4] = { &d_out, &d_in, &n, NULL };
    int rc = cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, 0, params, NULL);
    if (rc) { printf("shflTest: LAUNCH FAILED rc=%d\n", rc); failures++; }
    else {
      cuCtxSynchronize();
      cuMemcpyDtoH(hout, d_out, N * sizeof(float));
      int bad = 0;
      for (int i = 0; i < N; i++)
        if (hout[i] != hin[0]) bad++;
      if (bad == 0)
        printf("shflTest: OK — simd_shuffle lane-0 broadcast across 32 lanes\n");
      else { printf("shflTest: FAILED — %d lanes wrong\n", bad); failures++; }
    }
    cuMemFree(d_in); cuMemFree(d_out);
    free(hin); free(hout);
  }

  /* ---- selpTest: out[i] = (i >= n) ? 0 : in[i] ---- */
  {
    const int N = 16;
    float *hin = malloc(N * sizeof(float)), *hout = malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) hin[i] = -3.0f - (float)i;
    CUdeviceptr d_in = 0, d_out = 0, module = 0, func = 0;
    cuModuleLoadData(&module, ptx);
    cuModuleGetFunction(&func, module, "_Z8selpTestPfS_i");
    cuMemAlloc(&d_in, N * sizeof(float));
    cuMemAlloc(&d_out, N * sizeof(float));
    cuMemcpyHtoD(d_in, hin, N * sizeof(float));
    unsigned n = 8;
    void *params[4] = { &d_out, &d_in, &n, NULL };
    int rc = cuLaunchKernel(func, 1, 1, 1, 16, 1, 1, 0, 0, params, NULL);
    if (rc) { printf("selpTest: LAUNCH FAILED rc=%d\n", rc); failures++; }
    else {
      cuCtxSynchronize();
      cuMemcpyDtoH(hout, d_out, N * sizeof(float));
      int bad = 0;
      for (int i = 0; i < N; i++) {
        float want = (i >= 8) ? 0.0f : hin[i];
        if (hout[i] != want) bad++;
      }
      if (bad == 0)
        printf("selpTest: OK — selp.f32 with 0f00000000 immediate\n");
      else { printf("selpTest: FAILED — %d lanes wrong\n", bad); failures++; }
    }
    cuMemFree(d_in); cuMemFree(d_out);
    free(hin); free(hout);
  }

  if (failures == 0)
    printf("V2 SIMD OK — shfl + selp verified on the GPU\n");
  else
    printf("V2 SIMD FAILED — %d case(s)\n", failures);
  free(ptx);
  return failures != 0;
}
