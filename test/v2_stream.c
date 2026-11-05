/*
 * v2_stream.c — async stream + event test: launch the grid-stride loop
 * kernel into a named stream (async commit), synchronize the stream, verify,
 * and time the kernel with events recorded on that stream.
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
int cuStreamCreate(CUdeviceptr *stream, unsigned flags);
int cuStreamDestroy(CUdeviceptr stream);
int cuStreamSynchronize(CUdeviceptr stream);
int cuEventCreate(CUdeviceptr *event, unsigned flags);
int cuEventRecord(CUdeviceptr event, CUdeviceptr stream);
int cuEventSynchronize(CUdeviceptr event);
int cuEventDestroy(CUdeviceptr event);
float cuEventElapsedTime(CUdeviceptr start, CUdeviceptr end);

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
  size_t len = 0;
  char *ptx = read_file("docs/reference/scaleLoop.ptx", &len);
  if (!ptx) { fprintf(stderr, "FAIL: missing scaleLoop.ptx\n"); return 2; }

  cuInit(0);

  CUdeviceptr stream = 0, e0 = 0, e1 = 0, dx = 0, module = 0, func = 0;
  int rc = cuStreamCreate(&stream, 0);
  if (rc) { fprintf(stderr, "FAIL: cuStreamCreate rc=%d\n", rc); return 1; }
  cuEventCreate(&e0, 0);
  cuEventCreate(&e1, 0);

  float *hx = malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) hx[i] = (float)(i + 1);
  float s = 2.0f;

  cuModuleLoadData(&module, ptx);
  cuModuleGetFunction(&func, module, "_Z9scaleLoopPffi");
  cuMemAlloc(&dx, N * sizeof(float));
  cuMemcpyHtoD(dx, hx, N * sizeof(float));

  unsigned n = N;
  void *params[4] = { &dx, &s, &n, NULL };

  /* first launch populates the stream; record e0 while its buffer is still
     the stream's last (StreamSynchronize consumes the slot) */
  rc = cuLaunchKernel(func, 4, 1, 1, 256, 1, 1, 0, stream, params, NULL);
  if (rc) { fprintf(stderr, "FAIL: async launch rc=%d\n", rc); return 1; }
  cuEventRecord(e0, stream);
  cuStreamSynchronize(stream);

  /* fresh input so the timed launch's expectation stays x*s */
  cuMemcpyHtoD(dx, hx, N * sizeof(float));
  rc = cuLaunchKernel(func, 4, 1, 1, 256, 1, 1, 0, stream, params, NULL);
  if (rc) { fprintf(stderr, "FAIL: async launch rc=%d\n", rc); return 1; }
  cuEventRecord(e1, stream);

  rc = cuStreamSynchronize(stream);
  if (rc) { fprintf(stderr, "FAIL: cuStreamSynchronize rc=%d\n", rc); return 1; }
  cuEventSynchronize(e1);

  cuMemcpyDtoH(hx, dx, N * sizeof(float));
  int bad = 0;
  for (int i = 0; i < N; i++)
    if (hx[i] != (float)(i + 1) * s) bad++;

  float ms = cuEventElapsedTime(e0, e1);

  if (bad == 0)
    printf("V2 STREAM OK — async commit + stream sync verified, %d/%d correct, kernel time %.3f ms\n",
           N, N, ms);
  else
    printf("V2 STREAM FAILED — %d/%d mismatches\n", bad, N);

  cuEventDestroy(e0); cuEventDestroy(e1);
  cuStreamDestroy(stream);
  cuMemFree(dx);
  free(hx); free(ptx);
  return bad != 0;
}
