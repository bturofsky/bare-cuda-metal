/*
 * perf_cuda.c — the TRANSLATED arm of the performance gate: the same
 * workload as perf_native.m (262,144 fp32 adds, 1024x256 grid) launched
 * through the shim's CUDA path (real nvcc PTX -> our MSL -> AOT metallib),
 * timed on GPU clocks via cuEvent pairs. Fastest of 50 after warm-up.
 * Link: perf_cuda.c shim/libcuda_shim.dylib
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned long long CUdeviceptr;
int cuInit(unsigned int);
int cuMemAlloc(CUdeviceptr *, size_t);
int cuMemFree(CUdeviceptr);
int cuMemcpyHtoD(CUdeviceptr, const void *, size_t);
int cuModuleLoadData(CUdeviceptr *, const void *);
int cuModuleGetFunction(CUdeviceptr *, CUdeviceptr, const char *);
int cuLaunchKernel(CUdeviceptr, unsigned, unsigned, unsigned, unsigned, unsigned,
                   unsigned, unsigned, CUdeviceptr, void **, void **);
int cuCtxSynchronize(void);
int cuStreamCreate(CUdeviceptr *, unsigned);
int cuStreamSynchronize(CUdeviceptr);
float cuEventElapsedTime(CUdeviceptr, CUdeviceptr);
int cuEventCreate(CUdeviceptr *, unsigned);
int cuEventRecord(CUdeviceptr, CUdeviceptr);
int cuEventSynchronize(CUdeviceptr);
int cuEventDestroy(CUdeviceptr);

static char *read_file(const char *path, size_t *len) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  char *buf = malloc((size_t)sz + 1);
  size_t rd = fread(buf, 1, (size_t)sz, f);
  fclose(f); buf[rd] = 0; *len = rd;
  return buf;
}

#define N 262144u

int main(void) {
  size_t ptx_len = 0;
  char *ptx = read_file("docs/reference/vectorAdd.ptx", &ptx_len);
  if (!ptx) { fprintf(stderr, "FAIL: no fixture PTX\n"); return 2; }
  cuInit(0);

  CUdeviceptr da = 0, db = 0, dc = 0, module = 0, func = 0;
  if (cuMemAlloc(&da, N * 4) || cuMemAlloc(&db, N * 4) || cuMemAlloc(&dc, N * 4)) return 1;
  float *ha = malloc(N * 4), *hb = malloc(N * 4);
  for (unsigned i = 0; i < N; i++) { ha[i] = 1.0f; hb[i] = 2.0f; }
  cuMemcpyHtoD(da, ha, N * 4);
  cuMemcpyHtoD(db, hb, N * 4);

  if (cuModuleLoadData(&module, ptx)) return 1;
  if (cuModuleGetFunction(&func, module, "_Z9vectorAddPKfS0_Pfi")) return 1;

  uint32_t n = N;
  void *params[5] = { &da, &db, &dc, &n, NULL };
  CUdeviceptr stream = 0;
  if (cuStreamCreate(&stream, 0)) return 1;
  CUdeviceptr ev0 = 0, ev1 = 0;
  if (cuEventCreate(&ev0, 0) || cuEventCreate(&ev1, 0)) return 1;

  cuEventRecord(ev0, stream);
  cuLaunchKernel(func, 1024, 1, 1, 256, 1, 1, 0, stream, params, NULL);
  cuEventRecord(ev1, stream);
  cuStreamSynchronize(stream);
  cuEventSynchronize(ev1);
  cuEventElapsedTime(ev0, ev1);

  double best = 1e9;
  for (int it = 0; it < 50; it++) {
    cuEventRecord(ev0, stream);
    int rc = cuLaunchKernel(func, 1024, 1, 1, 256, 1, 1, 0, stream, params, NULL);
    cuEventRecord(ev1, stream);
    cuStreamSynchronize(stream);
    cuEventSynchronize(ev1);
    if (rc) { fprintf(stderr, "FAIL launch rc=%d\n", rc); return 1; }
    double ms = cuEventElapsedTime(ev0, ev1);
    if (ms > 0 && ms < best) best = ms;
  }
  printf("{\"backend\":\"bare-cuda-translated\",\"elements\":%u,"
         "\"iterations\":50,\"best_gpu_ms\":%.6f}\n", N, best);
  cuEventDestroy(ev0); cuEventDestroy(ev1);
  cuMemFree(da); cuMemFree(db); cuMemFree(dc);
  free(ha); free(hb); free(ptx);
  return 0;
}
