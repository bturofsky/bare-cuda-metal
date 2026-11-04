/*
 * libcuda_shim.c — v0 CUDA API interposition skeleton for Apple Silicon.
 *
 * Build as shim/libcuda_shim.dylib. Applications that dlopen("libcuda.dylib")
 * (or run under DYLD_INSERT_LIBRARIES=shim/libcuda_shim.dylib) get this
 * library instead of NVIDIA's. Every call is logged (the coverage record that
 * sizes the v2 PTX->MSL subset) and returns plausible codes so execution
 * proceeds far enough to enumerate what the app really needs.
 *
 * v0 scope: logging + a one-Apple-GPU device surface. No Metal yet — v1 wires
 * MTLDevice/MTLBuffer/MTLCommandQueue behind these same symbols.
 */
#include <dlfcn.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* ---- CUDA result codes (subset of cuda.h / driver_types.h) -------------- */
enum {
  CUDA_SUCCESS = 0,
  CUDA_ERROR_NOT_INITIALIZED = 3,
  CUDA_ERROR_INVALID_DEVICE = 101,
  CUDA_ERROR_INVALID_VALUE = 1,
  CUDA_ERROR_OUT_OF_MEMORY = 2,
  CUDA_ERROR_NOT_SUPPORTED = 801,
};

#define CU_DEVICE_CPU 0x7F  /* cudaCpuDeviceId — unused here, documented */

/* ---- logging ------------------------------------------------------------ */
static FILE *g_log = NULL;
static pthread_mutex_t g_log_lock = PTHREAD_MUTEX_INITIALIZER;
static long g_call_count = 0;

static void shim_log_open(void) {
  const char *path = getenv("CUDA_METAL_SHIM_LOG");
  if (!path) path = "/tmp/cuda-metal-shim.log";
  g_log = fopen(path, "a");
  if (!g_log) g_log = stderr;
  setvbuf(g_log, NULL, _IOLBF, 0);
  fprintf(g_log, "==== cuda-metal shim v0 attached (pid %d) ====\n", getpid());
}

static void slog(const char *fn, const char *fmt, ...) {
  pthread_mutex_lock(&g_log_lock);
  if (!g_log) shim_log_open();
  g_call_count++;
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  fprintf(g_log, "[%ld.%03ld] #%ld %s(", (long)ts.tv_sec, ts.tv_nsec / 1000000,
          g_call_count, fn);
  if (fmt && fmt[0]) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(g_log, fmt, ap);
    va_end(ap);
  }
  fprintf(g_log, ")\n");
  pthread_mutex_unlock(&g_log_lock);
}

/* ---- init / device ------------------------------------------------------ */
static int g_initialized = 0;

int cuInit(unsigned int flags) {
  slog("cuInit", "flags=0x%x", flags);
  g_initialized = 1;
  shim_log_open(); /* idempotent */
  fprintf(g_log ? g_log : stderr,
          "[cuda-metal] v0: intercepting CUDA for Apple-Silicon Metal (phase plan: README.md)\n");
  return CUDA_SUCCESS;
}

int cuDriverGetVersion(int *version) {
  slog("cuDriverGetVersion", "");
  /* We advertise a CUDA 12.x driver so tooling proceeds; the shim is not
     version-complete and v1 will gate features by real capability. */
  if (version) *version = 12040;
  return CUDA_SUCCESS;
}

int cuDeviceGetCount(int *count) {
  slog("cuDeviceGetCount", "");
  if (!g_initialized) { if (count) *count = 0; return CUDA_ERROR_NOT_INITIALIZED; }
  if (count) *count = 1; /* the one Apple GPU */
  return CUDA_SUCCESS;
}

int cuDeviceGet(int *device, int ordinal) {
  slog("cuDeviceGet", "ordinal=%d", ordinal);
  if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;
  if (ordinal != 0) return CUDA_ERROR_INVALID_DEVICE;
  if (device) *device = 0;
  return CUDA_SUCCESS;
}

int cuDeviceGetName(char *name, int len, int device) {
  slog("cuDeviceGetName", "len=%d device=%d", len, device);
  if (!name || len <= 0) return CUDA_ERROR_INVALID_VALUE;
  const char *n = "Apple Silicon (via cuda-metal v0)";
  snprintf(name, (size_t)len, "%s", n);
  return CUDA_SUCCESS;
}

int cuDeviceTotalMem(size_t *bytes, int device) {
  slog("cuDeviceTotalMem", "device=%d", device);
  /* v0: report 0; v1 reads actual Metal heap size via MTLDevice. */
  if (bytes) *bytes = 0;
  return CUDA_SUCCESS;
}

int cuDeviceComputeCapability(int *major, int *minor, int device) {
  slog("cuDeviceComputeCapability", "device=%d", device);
  /* Advertise 8.6-ish so apps take the Ampere code paths, the closest public
     feature set to what we plan to support in v2. */
  if (major) *major = 8;
  if (minor) *minor = 6;
  return CUDA_SUCCESS;
}

/* ---- context ------------------------------------------------------------ */
int cuCtxCreate(int *ctx, unsigned flags, int device) {
  slog("cuCtxCreate", "flags=0x%x device=%d", flags, device);
  if (ctx) *ctx = 1; /* opaque handle; v1 maps to MTLCommandQueue */
  return CUDA_SUCCESS;
}
int cuCtxSetCurrent(int ctx) { slog("cuCtxSetCurrent", "ctx=%d", ctx); return CUDA_SUCCESS; }
int cuCtxGetCurrent(int *ctx) { slog("cuCtxGetCurrent", ""); if (ctx) *ctx = 1; return CUDA_SUCCESS; }
int cuCtxSynchronize(void) { slog("cuCtxSynchronize", ""); return CUDA_SUCCESS; }
int cuCtxDestroy(int ctx) { slog("cuCtxDestroy", "ctx=%d", ctx); return CUDA_SUCCESS; }

/* ---- memory ------------------------------------------------------------- */
int cuMemAlloc(int *dptr, size_t bytes) {
  slog("cuMemAlloc", "bytes=%zu", bytes);
  /* v0: hand out a fake device pointer (an offset counter in v0.1); v1: a
     real MTLBuffer allocation. Apps that dereference on the CPU will fault —
     logged, expected, and the first thing v1 fixes. */
  static uintptr_t fake = 0x1000;
  if (dptr) *dptr = (int)(fake);
  fake += ((bytes + 255) & ~255) + 256;
  return CUDA_SUCCESS;
}

int cuMemAllocPitch(int *dptr, size_t *pitch, size_t width, size_t height, unsigned int elementSize) {
  slog("cuMemAllocPitch", "width=%zu height=%zu elem=%u", width, height, elementSize);
  size_t p = ((width + 511) / 512) * 512;
  if (pitch) *pitch = p;
  return cuMemAlloc(dptr, p * height);
}

int cuMemFree(int dptr) { slog("cuMemFree", "dptr=0x%x", dptr); return CUDA_SUCCESS; }

int cuMemHostAlloc(void **pp, size_t bytes, unsigned flags) {
  slog("cuMemHostAlloc", "bytes=%zu", bytes);
  /* Host memory is real memory — give the app a real allocation so CPU-side
     paths work even in v0. */
  if (pp) *pp = malloc(bytes ? bytes : 1);
  return *pp ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
}
int cuMemFreeHost(void *p) { slog("cuMemFreeHost", ""); free(p); return CUDA_SUCCESS; }

int cuMemcpy(int dst, int src, size_t bytes) {
  slog("cuMemcpy", "dst=0x%x src=0x%x bytes=%zu", dst, src, bytes);
  return CUDA_SUCCESS; /* v0: no-op; nothing dereferenceable yet */
}
int cuMemcpyHtoD(int dst, const void *src, size_t bytes) {
  slog("cuMemcpyHtoD", "dst=0x%x bytes=%zu", dst, bytes);
  return CUDA_SUCCESS;
}
int cuMemcpyDtoH(void *dst, int src, size_t bytes) {
  slog("cuMemcpyDtoH", "src=0x%x bytes=%zu", src, bytes);
  return CUDA_SUCCESS;
}

/* ---- streams / events --------------------------------------------------- */
int cuStreamCreate(int *stream, unsigned flags) {
  slog("cuStreamCreate", "flags=0x%x", flags);
  if (stream) *stream = 2;
  return CUDA_SUCCESS;
}
int cuStreamDestroy(int stream) { slog("cuStreamDestroy", "stream=%d", stream); return CUDA_SUCCESS; }
int cuStreamSynchronize(int stream) { slog("cuStreamSynchronize", "stream=%d", stream); return CUDA_SUCCESS; }
int cuEventCreate(int *event, unsigned flags) {
  slog("cuEventCreate", "flags=0x%x", flags);
  if (event) *event = 3;
  return CUDA_SUCCESS;
}
int cuEventRecord(int event, int stream) { slog("cuEventRecord", "event=%d stream=%d", event, stream); return CUDA_SUCCESS; }
int cuEventSynchronize(int event) { slog("cuEventSynchronize", "event=%d", event); return CUDA_SUCCESS; }
int cuEventDestroy(int event) { slog("cuEventDestroy", "event=%d", event); return CUDA_SUCCESS; }
float cuEventElapsedTime(int start, int end) {
  slog("cuEventElapsedTime", "start=%d end=%d", start, end);
  return 0.0f;
}

/* ---- module / kernel (the v2 frontier) ---------------------------------- */
int cuModuleLoad(int *module, const char *fname) {
  slog("cuModuleLoad", "fname=%s", fname ? fname : "(null)");
  /* THE frontier. Real apps embed fatbinaries (PTX + SASS). v2 parses PTX
     from the fatbin container and translates the PTX text to MSL. v0 loads a
     phantom module so the app reaches cuModuleGetFunction and we learn the
     kernel names — the coverage record's most valuable column. */
  if (module) *module = 10;
  return CUDA_SUCCESS;
}
int cuModuleLoadData(int *module, const void *image) {
  const unsigned char *img = image;
  size_t probe = 0;
  char magic[5] = {0};
  if (img) {
    /* fatbin magic per cudaFatBinaryHeader: 0xBA55ED50 little-endian */
    memcpy(magic, img, 4);
    while (probe < 32 && img[probe]) probe++;
  }
  slog("cuModuleLoadData", "module=10 magic=%s imgbytes_dumped_to_log", magic);
  if (g_log && img) {
    /* Dump the first 512 bytes — enough to identify the fatbin container
       version and whether PTX text is present (the v2 feasibility input). */
    fprintf(g_log, "[cuda-metal] fatbin probe:");
    for (size_t i = 0; i < 512; i++) {
      if (i % 32 == 0) fprintf(g_log, "\n  %04zx ", i);
      unsigned char c = ((const unsigned char *)img)[i];
      fprintf(g_log, "%02x", c);
    }
    fprintf(g_log, "\n");
  }
  if (module) *module = 10;
  return CUDA_SUCCESS;
}
int cuModuleUnload(int module) { slog("cuModuleUnload", "module=%d", module); return CUDA_SUCCESS; }

int cuModuleGetFunction(int *func, int module, const char *name) {
  slog("cuModuleGetFunction", "module=%d name=%s  << KERNEL NAME — the v2 worklist",
       module, name ? name : "(null)");
  /* This line is why v0 exists: every (kernel, arg-signature) the real app
     uses lands in the log. That list IS the v2 translation backlog. */
  if (func) *func = 20;
  return CUDA_SUCCESS;
}

int cuLaunchKernel(int func, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                   unsigned by, unsigned bz, unsigned sharedMem, int stream,
                   void **params, void **extra) {
  slog("cuLaunchKernel",
       "func=%d grid=%ux%ux%u block=%ux%ux%u shared=%u stream=%d params=%p  "
       "<< v2: PTX->MSL translate + MTLComputePipeline dispatch",
       func, gx, gy, gz, bx, by, bz, sharedMem, stream, (void *)params);
  /* v0: NOT_SUPPORTED so callers that check errors branch sensibly instead of
     reading garbage from unimplemented kernels. */
  return CUDA_ERROR_NOT_SUPPORTED;
}

/* ---- runtime-API aliases (cudart statically links these names) ---------- */
int cudaGetDeviceCount(int *count) {
  slog("cudaGetDeviceCount", "");
  return cuDeviceGetCount(count);
}
int cudaGetDeviceProperties(void *props, int device) {
  slog("cudaGetDeviceProperties", "device=%d", device);
  /* v0: zero-filled props; v1 fills the fields apps actually branch on
     (name, major/minor, multiProcessorCount, sharedMemPerBlock, warpSize=32). */
  if (props) memset(props, 0, 4096 < 0 ? 4096 : 4096);
  return CUDA_SUCCESS;
}
int cudaMalloc(void **devPtr, size_t bytes) {
  int d = 0;
  slog("cudaMalloc", "bytes=%zu", bytes);
  int r = cuMemAlloc(&d, bytes);
  if (devPtr) *(uintptr_t *)devPtr = (uintptr_t)d;
  return r;
}
int cudaFree(void *p) { slog("cudaFree", ""); return CUDA_SUCCESS; }
int cudaMemcpy(void *dst, const void *src, size_t bytes, int kind) {
  slog("cudaMemcpy", "bytes=%zu kind=%d", bytes, kind);
  /* host<->host kinds can be honest in v0 */
  if (kind == 0 /*cudaMemcpyHostToHost*/ && dst && src) memcpy(dst, src, bytes);
  return CUDA_SUCCESS;
}
int cudaDeviceSynchronize(void) { slog("cudaDeviceSynchronize", ""); return CUDA_SUCCESS; }
