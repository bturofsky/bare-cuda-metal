/*
 * v2_pool.c — multi-__shared__ pool test: two shared arrays (256 floats each)
 * allocated in one pooled threadgroup allocation with per-var base pointers.
 * dualSum stores in[i] to sA and 2*in[i] to sB, barriers, then thread 0
 * publishes sA[0] and sB[31] — exercising pool interior addressing.
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

#define W 32

int main(void) {
  size_t len = 0;
  char *ptx = read_file("docs/reference/dualSum.ptx", &len);
  if (!ptx) { fprintf(stderr, "FAIL: missing dualSum.ptx\n"); return 2; }
  cuInit(0);

  float *hin = malloc(W * sizeof(float)), *hout = malloc(2 * sizeof(float));
  for (int i = 0; i < W; i++) hin[i] = 5.0f + (float)i;

  CUdeviceptr d_in = 0, d_out = 0, module = 0, func = 0;
  int rc = cuModuleLoadData(&module, ptx);
  if (rc) { fprintf(stderr, "FAIL: load rc=%d\n", rc); return 1; }
  rc = cuModuleGetFunction(&func, module, "_Z7dualSumPf00");
  if (rc) { fprintf(stderr, "FAIL: getfunction rc=%d\n", rc); return 1; }
  cuMemAlloc(&d_in, W * sizeof(float));
  cuMemAlloc(&d_out, 2 * sizeof(float));
  cuMemcpyHtoD(d_in, hin, W * sizeof(float));

  void *params[3] = { &d_out, &d_in, NULL };
  rc = cuLaunchKernel(func, 1, 1, 1, W, 1, 1, 0, 0, params, NULL);
  if (rc) { fprintf(stderr, "FAIL: launch rc=%d\n", rc); return 1; }
  cuCtxSynchronize();
  cuMemcpyDtoH(hout, d_out, 2 * sizeof(float));

  int bad = 0;
  if (hout[0] != hin[0]) bad++;
  if (hout[1] != 2.0f * hin[W - 1]) bad++;

  if (bad == 0)
    printf("V2 POOL OK — two __shared__ vars in one pool: out={%g, %g} as expected\n",
           hout[0], hout[1]);
  else
    printf("V2 POOL FAILED — got {%g, %g}\n", hout[0], hout[1]);

  cuMemFree(d_in); cuMemFree(d_out);
  free(hin); free(hout); free(ptx);
  return bad != 0;
}
