/*
 * v2_shfl.c — ISA-exact shuffle arithmetic, verified on the GPU:
 * warpSum:  butterfly reduction (bfly c=31, deltas 16..1) — every lane of a
 *   32-thread warp ends holding the warp total.
 * warpScan: inclusive scan via the ISA's own idiom — shfl.up with clamp 0x0,
 *   predicate destination (d|p), and @p-guarded adds.
 * sel64Off: selp.u64 picks a byte offset (16 or 32) consumed by pointer
 *   arithmetic — 64-bit select through the address path.
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

static float want_sum(int i, const float *in) { (void)i; (void)in; return 32.0f; }
static float want_scan(int i, const float *in) { (void)in; return (float)(i + 1); }
static float want_sel(int i, const float *in) { return (i >= 16) ? in[4] : in[8]; }

static int run_n(const char *ptx, const char *name, const float *in, float *out,
                 unsigned n, float (*want)(int i, const float *in));

static int run(const char *ptx, const char *name, const float *in, float *out,
               float (*want)(int i, const float *in)) {
  return run_n(ptx, name, in, out, W, want);
}

static int run_n(const char *ptx, const char *name, const float *in, float *out,
                 unsigned n, float (*want)(int i, const float *in)) {
  CUdeviceptr d_in = 0, d_out = 0, module = 0, func = 0;
  cuModuleLoadData(&module, ptx);
  cuModuleGetFunction(&func, module, name);
  cuMemAlloc(&d_in, W * sizeof(float));
  cuMemAlloc(&d_out, W * sizeof(float));
  cuMemcpyHtoD(d_in, in, W * sizeof(float));
  void *params[4] = { &d_out, &d_in, &n, NULL };
  int rc = cuLaunchKernel(func, 1, 1, 1, W, 1, 1, 0, 0, params, NULL);
  if (rc) { printf("%s: LAUNCH FAILED rc=%d\n", name, rc); return 1; }
  cuCtxSynchronize();
  cuMemcpyDtoH(out, d_out, W * sizeof(float));
  int bad = 0;
  for (int i = 0; i < W; i++)
    if (out[i] != want(i, in)) bad++;
  cuMemFree(d_in); cuMemFree(d_out);
  return bad;
}

int main(void) {
  size_t len = 0;
  char *ptx = read_file("docs/reference/shfl_reduce.ptx", &len);
  if (!ptx) { fprintf(stderr, "FAIL: missing shfl_reduce.ptx\n"); return 2; }
  cuInit(0);

  float *hin = malloc(W * sizeof(float)), *hout = malloc(W * sizeof(float));
  for (int i = 0; i < W; i++) hin[i] = 1.0f;
  int failures = 0;

  {
    int bad = run(ptx, "_Z7warpSumPfS_i", hin, hout, want_sum);
    if (bad == 0) printf("warpSum: OK — butterfly reduction, all 32 lanes hold 32.0\n");
    else { printf("warpSum: FAILED — %d lanes wrong\n", bad); failures++; }
  }
  {
    int bad = run(ptx, "_Z8warpScanPfS_i", hin, hout, want_scan);
    if (bad == 0) printf("warpScan: OK — inclusive scan (shfl.up clamp 0x0 + @p adds)\n");
    else {
      printf("warpScan: FAILED — %d lanes wrong; got:", bad);
      for (int i = 0; i < 8; i++) printf(" %g", hout[i]);
      printf("\n");
      failures++;
    }
  }
  {
    for (int i = 0; i < W; i++) hin[i] = 10.0f + (float)i;
    int bad = run_n(ptx, "_Z7sel64OffPfS_i", hin, hout, 16, want_sel);
    if (bad == 0) printf("sel64Off: OK — selp.u64 offset consumed by address math\n");
    else { printf("sel64Off: FAILED — %d lanes wrong\n", bad); failures++; }
  }

  if (failures == 0)
    printf("V2 SHFL OK — ISA-exact shuffles + selp.u64 verified on the GPU\n");
  else
    printf("V2 SHFL FAILED — %d case(s)\n", failures);
  free(hin); free(hout); free(ptx);
  return failures != 0;
}
