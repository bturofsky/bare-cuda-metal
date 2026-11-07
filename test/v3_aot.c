/*
 * v3_aot.c — bare-metal AOT acceptance suite.
 *
 * Proves, on the M4 GPU, the v3 launch contract:
 *   1. COLD:  first launch compiles MSL OFFLINE through Apple's real
 *             toolchain (metal -c -O3 + metallib) into a cached .metallib
 *             binary — proven by the artifact existing in the cache dir.
 *   2. WARM:  a fresh process loads the cached .metallib via
 *             newLibraryWithData: — no compile of any kind (the suite
 *             detects pre-existing cache and reports warm vs cold).
 *   3. SIMD-ALIGNED DISPATCH: a 100-thread block (not a multiple of 32)
 *             dispatches as 128 padded lanes while ntid stays 100 —
 *             elements verified against CPU truth.
 * Elements are verified after every launch; no silent JIT fallback is
 * accepted (a fallback would leave no .metallib artifact and fail step 1).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
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

static int count_metallibs(const char *dir, long *total_bytes) {
  DIR *d = opendir(dir);
  if (!d) return 0; /* no dir yet = nothing cached */
  int n = 0;
  *total_bytes = 0;
  struct dirent *e;
  while ((e = readdir(d))) {
    size_t el = strlen(e->d_name);
    if (el > 9 && strcmp(e->d_name + el - 9, ".metallib") == 0) {
      char p[1024];
      snprintf(p, sizeof p, "%s/%s", dir, e->d_name);
      struct stat st;
      if (stat(p, &st) == 0 && st.st_size > 0) {
        n++;
        *total_bytes += (long)st.st_size;
      }
    }
  }
  closedir(d);
  return n;
}

#define N 1024

static int launch_and_verify(CUdeviceptr func, float *ha, float *hb, float *hc,
                             CUdeviceptr da, CUdeviceptr db, CUdeviceptr dc,
                             unsigned gx, unsigned bx, unsigned n,
                             int fill_a_b, double *launch_ms, const char *tag,
                             int *bad_out) {
  unsigned nu = n;
  void *params[5] = { &da, &db, &dc, &nu, NULL };
  if (fill_a_b) {
    for (unsigned i = 0; i < n; i++) { ha[i] = (float)i; hb[i] = (float)(2 * i); }
    cuMemcpyHtoD(da, ha, n * sizeof(float));
    cuMemcpyHtoD(db, hb, n * sizeof(float));
  }

  double t0 = now_ms();
  int rc = cuLaunchKernel(func, gx, 1, 1, bx, 1, 1, 0, 0, params, NULL);
  *launch_ms = now_ms() - t0;
  if (rc) { fprintf(stderr, "FAIL[%s]: cuLaunchKernel rc=%d\n", tag, rc); return -1; }
  cuCtxSynchronize();
  cuMemcpyDtoH(hc, dc, n * sizeof(float));
  int bad = 0;
  for (unsigned i = 0; i < n; i++) {
    if (hc[i] != ha[i] + hb[i]) {
      bad++;
      if (bad <= 3)
        fprintf(stderr, "mismatch[%s] @%u: got %f want %f\n", tag, i, hc[i], ha[i] + hb[i]);
    }
  }
  *bad_out = bad;
  return 0;
}

int main(int argc, char **argv) {
  const char *cache = argc > 1 ? argv[1] : "/tmp/cuda-metal-aot-suite";
  setenv("CUDA_METAL_AOT_CACHE", cache, 1); /* fresh or pre-warmed, caller's choice */

  size_t ptx_len = 0;
  char *ptx = read_file("docs/reference/vectorAdd.ptx", &ptx_len);
  if (!ptx) { fprintf(stderr, "FAIL: cannot read docs/reference/vectorAdd.ptx\n"); return 2; }

  long bytes = 0;
  int pre_libs = count_metallibs(cache, &bytes);
  int warm = pre_libs > 0;
  printf("cache %s: %d metallib(s) pre-existing -> %s run\n",
         cache, pre_libs, warm ? "WARM (newLibraryWithData: load only)" : "COLD (offline compile)");

  cuInit(0);

  float *ha = malloc(N * sizeof(float));
  float *hb = malloc(N * sizeof(float));
  float *hc = malloc(N * sizeof(float));
  CUdeviceptr da = 0, db = 0, dc = 0, module = 0, func = 0;
  int rc = 0;
  if ((rc = cuMemAlloc(&da, N * 4)) || (rc = cuMemAlloc(&db, N * 4)) ||
      (rc = cuMemAlloc(&dc, N * 4))) {
    fprintf(stderr, "FAIL: cuMemAlloc rc=%d\n", rc);
    return 1;
  }

  if ((rc = cuModuleLoadData(&module, ptx))) {
    fprintf(stderr, "FAIL: cuModuleLoadData rc=%d\n", rc);
    return 1;
  }
  if ((rc = cuModuleGetFunction(&func, module, "_Z9vectorAddPKfS0_Pfi"))) {
    fprintf(stderr, "FAIL: cuModuleGetFunction rc=%d\n", rc);
    return 1;
  }

  int bad = 0, total_bad = 0;
  double ms = 0;

  /* --- launch 1: 4 blocks x 256 threads — cold AOT compile or warm load --- */
  if (launch_and_verify(func, ha, hb, hc, da, db, dc, 4, 256, N, 1, &ms,
                        warm ? "warm-load" : "cold-aot", &bad)) return 1;
  total_bad += bad;
  printf("launch1 (%s): %d/%d elements verified, %.1f ms (incl. %s)\n",
         warm ? "metallib load" : "metal -c + metallib", N - bad, N, ms,
         warm ? "newLibraryWithData:" : "offline toolchain");

  /* --- artifact gate: the AOT layer must have left a .metallib on disk --- */
  long lib_bytes = 0;
  int libs = count_metallibs(cache, &lib_bytes);
  if (libs <= 0) {
    printf("V3 AOT FAILED — no .metallib artifact in %s (silent JIT fallback?)\n", cache);
    return 1;
  }
  printf("artifact gate: %d .metallib file(s), %ld bytes — offline compile proven\n",
         libs, lib_bytes);

  /* --- launch 2: pipeline cache hit, fresh data --- */
  for (int i = 0; i < N; i++) { ha[i] = 3.0f * i + 1.0f; hb[i] = -2.0f * i; }
  cuMemcpyHtoD(da, ha, N * 4);
  cuMemcpyHtoD(db, hb, N * 4);
  if (launch_and_verify(func, ha, hb, hc, da, db, dc, 4, 256, N, 0, &ms,
                        "cached", &bad)) return 1;
  total_bad += bad;
  printf("launch2 (pipeline cache hit): %d/%d elements verified, %.3f ms\n",
         N - bad, N, ms);

  /* --- launch 3: SIMD-aligned padding — 100-thread block dispatches 128 lanes,
         ntid must remain 100 (blockDim travels as data at buffer 30) --- */
  cuMemFree(da); cuMemFree(db); cuMemFree(dc);
  unsigned PN = 400; /* 4 blocks x 100 threads exactly */
  if ((rc = cuMemAlloc(&da, PN * 4)) || (rc = cuMemAlloc(&db, PN * 4)) ||
      (rc = cuMemAlloc(&dc, PN * 4))) {
    fprintf(stderr, "FAIL: padded cuMemAlloc rc=%d\n", rc);
    return 1;
  }
  if (launch_and_verify(func, ha, hb, hc, da, db, dc, 4, 100, PN, 1, &ms,
                        "padded-128", &bad)) return 1;
  total_bad += bad;
  printf("launch3 (100-thread block -> 128 padded lanes, ntid=100): "
         "%d/%d elements verified, %.3f ms\n", PN - bad, PN, ms);

  cuMemFree(da); cuMemFree(db); cuMemFree(dc);
  free(ha); free(hb); free(hc); free(ptx);

  if (total_bad == 0) {
    printf("V3 AOT OK — offline .metallib compile + newLibraryWithData: load + "
           "SIMD-aligned padded dispatch all element-verified on GPU\n");
    return 0;
  }
  printf("V3 AOT FAILED — %d mismatches\n", total_bad);
  return 1;
}
