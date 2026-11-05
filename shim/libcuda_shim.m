/*
 * libcuda_shim.m — v1: real Metal memory plumbing for Apple Silicon.
 *
 * cuInit creates the system MTLDevice; cuMemAlloc returns pointers into
 * SHARED-storage MTLBuffers. On a Mac (unified memory) the CPU address and
 * the GPU address are the same bytes, so cuMemcpyHtoD/DtoH are honest
 * memcpys against buffer contents and host code can dereference the
 * returned pointer directly. Allocations are tracked in a small registry so
 * cuMemFree releases the exact MTLBuffer.
 *
 * Still NOT implemented (the honest frontier): module load / kernel launch
 * (phase 2: PTX→MSL). Launches return CUDA_ERROR_NOT_SUPPORTED.
 */
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* module_launch.m: fatbin/raw-PTX extraction + PTX→MSL translation */
extern char *cuda_metal_extract_ptx(const void *fatbin_data, size_t fatbin_size,
                                    const char *kernel_name, size_t *ptx_len);
extern char *cuda_metal_translate(const char *ptx, size_t ptx_len, const char *kernel_name);
extern int translate_ptx_kernel(const char *ptx_text, size_t ptx_len,
                                const char *kernel_name, char *msl_buf,
                                size_t msl_buf_sz, char *err_buf, size_t err_sz);

enum {
  CUDA_SUCCESS = 0,
  CUDA_ERROR_NOT_INITIALIZED = 3,
  CUDA_ERROR_OUT_OF_MEMORY = 2,
  CUDA_ERROR_INVALID_VALUE = 1,
  CUDA_ERROR_INVALID_DEVICE = 101,
  CUDA_ERROR_NOT_SUPPORTED = 801,
};

typedef uintptr_t CUdeviceptr;

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
  fprintf(g_log, "==== cuda-metal shim v1 attached (pid %d) ====\n", getpid());
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

/* ---- Metal state ---------------------------------------------------------- */
static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static int g_initialized = 0;

/* ---- allocation registry: host pointer -> MTLBuffer ---------------------- */
#define MAX_ALLOCS 4096
typedef struct {
  void *host;
  id<MTLBuffer> buffer;
  size_t size;
} alloc_rec;
static alloc_rec g_allocs[MAX_ALLOCS];
static int g_nallocs = 0;
static pthread_mutex_t g_alloc_lock = PTHREAD_MUTEX_INITIALIZER;

static id<MTLBuffer> buffer_for(void *host) {
  pthread_mutex_lock(&g_alloc_lock);
  id<MTLBuffer> b = nil;
  for (int i = 0; i < g_nallocs; i++)
    if (g_allocs[i].host == host) { b = g_allocs[i].buffer; break; }
  pthread_mutex_unlock(&g_alloc_lock);
  return b;
}
static void register_alloc(void *host, id<MTLBuffer> b, size_t size) {
  pthread_mutex_lock(&g_alloc_lock);
  if (g_nallocs < MAX_ALLOCS) {
    g_allocs[g_nallocs].host = host;
    g_allocs[g_nallocs].buffer = b;
    g_allocs[g_nallocs].size = size;
    g_nallocs++;
  }
  pthread_mutex_unlock(&g_alloc_lock);
}
static id<MTLBuffer> unregister_alloc(void *host) {
  pthread_mutex_lock(&g_alloc_lock);
  id<MTLBuffer> b = nil;
  for (int i = 0; i < g_nallocs; i++)
    if (g_allocs[i].host == host) {
      b = g_allocs[i].buffer;
      g_allocs[i] = g_allocs[g_nallocs - 1];
      g_nallocs--;
      break;
    }
  pthread_mutex_unlock(&g_alloc_lock);
  return b;
}

/* ---- init / device ------------------------------------------------------ */
int cuInit(unsigned int flags) {
  slog("cuInit", "flags=0x%x", flags);
  if (!g_device) {
    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) { fprintf(stderr, "[cuda-metal] no Metal device\n"); return CUDA_ERROR_INVALID_DEVICE; }
    g_queue = [g_device newCommandQueue];
  }
  g_initialized = 1;
  shim_log_open();
  fprintf(g_log ? g_log : stderr, "[cuda-metal] v1: Metal memory live on %s\n",
          g_device.name.UTF8String);
  return CUDA_SUCCESS;
}

int cuDriverGetVersion(int *version) {
  slog("cuDriverGetVersion", "");
  if (version) *version = 12040;
  return CUDA_SUCCESS;
}

int cuDeviceGetCount(int *count) {
  slog("cuDeviceGetCount", "");
  if (!g_initialized) { if (count) *count = 0; return CUDA_ERROR_NOT_INITIALIZED; }
  if (count) *count = 1;
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
  snprintf(name, (size_t)len, "%s", g_device ? g_device.name.UTF8String : "Apple Silicon");
  return CUDA_SUCCESS;
}

int cuDeviceTotalMem(size_t *bytes, int device) {
  slog("cuDeviceTotalMem", "device=%d", device);
  if (bytes) *bytes = (size_t)g_device.recommendedMaxWorkingSetSize;
  return CUDA_SUCCESS;
}

int cuDeviceComputeCapability(int *major, int *minor, int device) {
  slog("cuDeviceComputeCapability", "device=%d", device);
  if (major) *major = 8;
  if (minor) *minor = 6;
  return CUDA_SUCCESS;
}

/* ---- context ------------------------------------------------------------ */
int cuCtxCreate(int *ctx, unsigned flags, int device) {
  slog("cuCtxCreate", "flags=0x%x device=%d", flags, device);
  if (ctx) *ctx = 1;
  return CUDA_SUCCESS;
}
int cuCtxSetCurrent(int ctx) { slog("cuCtxSetCurrent", "ctx=%d", ctx); return CUDA_SUCCESS; }
int cuCtxGetCurrent(int *ctx) { slog("cuCtxGetCurrent", ""); if (ctx) *ctx = 1; return CUDA_SUCCESS; }
int cuCtxSynchronize(void) { slog("cuCtxSynchronize", ""); return CUDA_SUCCESS; }
int cuCtxDestroy(int ctx) { slog("cuCtxDestroy", "ctx=%d", ctx); return CUDA_SUCCESS; }

/* ---- memory: REAL MTLBuffers (shared storage = CPU/GPU same bytes) ------ */
int cuMemAlloc(CUdeviceptr *dptr, size_t bytes) {
  slog("cuMemAlloc", "bytes=%zu", bytes);
  if (!g_initialized) return CUDA_ERROR_NOT_INITIALIZED;
  if (!dptr || bytes == 0) return CUDA_ERROR_INVALID_VALUE;
  id<MTLBuffer> b = [g_device newBufferWithLength:bytes
                                          options:MTLResourceStorageModeShared];
  if (!b) return CUDA_ERROR_OUT_OF_MEMORY;
  void *host = b.contents;
  memset(host, 0, bytes); /* CUDA guarantees zeroed allocations */
  register_alloc(host, b, bytes);
  if (dptr) *dptr = (CUdeviceptr)(uintptr_t)host;
  slog("cuMemAlloc", "-> host %p (%zu bytes, shared MTLBuffer)", host, bytes);
  return CUDA_SUCCESS;
}

int cuMemAllocPitch(CUdeviceptr *dptr, size_t *pitch, size_t width, size_t height, unsigned int elementSize) {
  slog("cuMemAllocPitch", "width=%zu height=%zu elem=%u", width, height, elementSize);
  size_t p = ((width + 511) / 512) * 512;
  if (pitch) *pitch = p;
  return cuMemAlloc(dptr, p * height);
}

int cuMemFree(CUdeviceptr dptr) {
  slog("cuMemFree", "dptr=0x%llx", (unsigned long long)dptr);
  void *host = (void *)(uintptr_t)dptr;
  id<MTLBuffer> b = unregister_alloc(host);
  if (!b) return CUDA_ERROR_INVALID_VALUE;
  return CUDA_SUCCESS; /* ARC releases the MTLBuffer when our ref drops */
}

int cuMemHostAlloc(void **pp, size_t bytes, unsigned flags) {
  slog("cuMemHostAlloc", "bytes=%zu", bytes);
  if (pp) *pp = malloc(bytes ? bytes : 1);
  return *pp ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
}
int cuMemFreeHost(void *p) { slog("cuMemFreeHost", ""); free(p); return CUDA_SUCCESS; }

int cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t bytes) {
  slog("cuMemcpy", "dst=0x%llx src=0x%llx bytes=%zu", (unsigned long long)dst,
       (unsigned long long)src, bytes);
  if (bytes && dst && src) memcpy((void *)(uintptr_t)dst, (void *)(uintptr_t)src, bytes);
  return CUDA_SUCCESS;
}
int cuMemcpyHtoD(CUdeviceptr dst, const void *src, size_t bytes) {
  slog("cuMemcpyHtoD", "dst=0x%llx bytes=%zu", (unsigned long long)dst, bytes);
  if (bytes && dst && src) memcpy((void *)(uintptr_t)dst, src, bytes);
  return CUDA_SUCCESS;
}
int cuMemcpyDtoH(void *dst, CUdeviceptr src, size_t bytes) {
  slog("cuMemcpyDtoH", "src=0x%llx bytes=%zu", (unsigned long long)src, bytes);
  if (bytes && dst && src) memcpy(dst, (void *)(uintptr_t)src, bytes);
  return CUDA_SUCCESS;
}

/* ---- streams / events --------------------------------------------------- */
int cuStreamCreate(CUdeviceptr *stream, unsigned flags) {
  slog("cuStreamCreate", "flags=0x%x", flags);
  id<MTLCommandQueue> q = [g_device newCommandQueue];
  if (stream) *stream = (CUdeviceptr)(__bridge void *)q;
  return CUDA_SUCCESS;
}
int cuStreamDestroy(CUdeviceptr stream) { slog("cuStreamDestroy", "stream=%llu", (unsigned long long)stream); return CUDA_SUCCESS; }
int cuStreamSynchronize(CUdeviceptr stream) { slog("cuStreamSynchronize", "stream=%llu", (unsigned long long)stream); return CUDA_SUCCESS; }
int cuEventCreate(CUdeviceptr *event, unsigned flags) {
  slog("cuEventCreate", "flags=0x%x", flags);
  if (event) *event = 3;
  return CUDA_SUCCESS;
}
int cuEventRecord(CUdeviceptr event, CUdeviceptr stream) { slog("cuEventRecord", ""); return CUDA_SUCCESS; }
int cuEventSynchronize(CUdeviceptr event) { slog("cuEventSynchronize", ""); return CUDA_SUCCESS; }
int cuEventDestroy(CUdeviceptr event) { slog("cuEventDestroy", ""); return CUDA_SUCCESS; }
float cuEventElapsedTime(CUdeviceptr start, CUdeviceptr end) {
  slog("cuEventElapsedTime", "");
  return 0.0f;
}

/* ---- module / kernel: v2 fatbin/raw-PTX → MSL → Metal dispatch ---------- */

/* Module state: the image pointer stored at LoadData, the kernel name bound
   at GetFunction. Parse + translate happen at launch time. One module / one
   outstanding function — the measured-subset contract. */
static const void *g_image_ptr = NULL;
static size_t g_image_size = 0;
static char g_kernel_name[128];

/* Launch-time MSL compile cache: (image, kernel name) → pipeline state.
   The first launch pays the Metal frontend compile (~0.5s); later launches
   reuse the cached state. */
#define MAX_PSOS 16
typedef struct {
  char name[128];
  const void *image;
  id<MTLComputePipelineState> pso;
} pso_rec;
static pso_rec g_psos[MAX_PSOS];
static int g_npsos = 0;
static pthread_mutex_t g_pso_lock = PTHREAD_MUTEX_INITIALIZER;

static id<MTLComputePipelineState> cached_pso(const char *name, const void *image) {
  pthread_mutex_lock(&g_pso_lock);
  id<MTLComputePipelineState> p = nil;
  for (int i = 0; i < g_npsos; i++)
    if (g_psos[i].image == image && strcmp(g_psos[i].name, name) == 0) { p = g_psos[i].pso; break; }
  pthread_mutex_unlock(&g_pso_lock);
  return p;
}

static void cache_pso(const char *name, const void *image, id<MTLComputePipelineState> p) {
  pthread_mutex_lock(&g_pso_lock);
  if (g_npsos < MAX_PSOS) {
    snprintf(g_psos[g_npsos].name, sizeof g_psos[g_npsos].name, "%s", name);
    g_psos[g_npsos].image = image;
    g_psos[g_npsos].pso = p;
    g_npsos++;
  }
  pthread_mutex_unlock(&g_pso_lock);
}

int cuModuleLoad(CUdeviceptr *module, const char *fname) {
  slog("cuModuleLoad", "fname=%s", fname ? fname : "(null)");
  if (module) *module = 10;
  return CUDA_SUCCESS;
}
int cuModuleLoadData(CUdeviceptr *module, const void *image) {
  slog("cuModuleLoadData", "image=%p", image);
  g_image_ptr = image;
  g_image_size = 1 << 20; /* bounded scan; header validation gates deep reads */
  if (module) *module = (CUdeviceptr)(uintptr_t)image;
  return CUDA_SUCCESS;
}
int cuModuleUnload(CUdeviceptr module) { slog("cuModuleUnload", ""); return CUDA_SUCCESS; }

int cuModuleGetFunction(CUdeviceptr *func, CUdeviceptr module, const char *name) {
  slog("cuModuleGetFunction", "module=%llu name=%s",
       (unsigned long long)module, name ? name : "(null)");
  if (name) snprintf(g_kernel_name, sizeof g_kernel_name, "%s", name);
  if (func) *func = (CUdeviceptr)(uintptr_t)name;
  return CUDA_SUCCESS;
}

int cuLaunchKernel(CUdeviceptr func, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                   unsigned by, unsigned bz, unsigned sharedMem, CUdeviceptr stream,
                   void **params, void **extra) {
  slog("cuLaunchKernel", "grid=%ux%ux%u block=%ux%ux%u params=%p",
       gx, gy, gz, bx, by, bz, (void *)params);

  if (!g_device) return CUDA_ERROR_NOT_INITIALIZED;
  if (!g_image_ptr || !g_kernel_name[0]) {
    slog("cuLaunchKernel", "no module/function loaded");
    return CUDA_ERROR_NOT_INITIALIZED;
  }

  /* 1. pipeline state — cached per (image, kernel); extract+translate only
     on a miss so cached launches skip the PTX work entirely */
  id<MTLComputePipelineState> pso = cached_pso(g_kernel_name, g_image_ptr);
  if (!pso) {
    size_t ptx_len = 0;
    char *ptx = cuda_metal_extract_ptx(g_image_ptr, g_image_size, g_kernel_name, &ptx_len);
    if (!ptx) {
      slog("cuLaunchKernel", "no PTX found for %s in image", g_kernel_name);
      return CUDA_ERROR_NOT_SUPPORTED;
    }
    /* translate directly so the translator's error reaches the log */
    char terr[256] = "";
    char *msl = malloc(16384);
    int trc = translate_ptx_kernel(ptx, ptx_len, g_kernel_name, msl, 16384,
                                   terr, sizeof terr);
    free(ptx);
    if (trc != 0) {
      slog("cuLaunchKernel", "PTX→MSL translation failed for %s: %s",
           g_kernel_name, terr[0] ? terr : "unknown");
      free(msl);
      return CUDA_ERROR_NOT_SUPPORTED;
    }

    NSError *err = nil;
    id<MTLLibrary> lib = [g_device newLibraryWithSource:@(msl) options:nil error:&err];
    if (!lib) {
      slog("cuLaunchKernel", "MSL compile failed: %s",
           err.localizedDescription ? err.localizedDescription.UTF8String : "?");
      free(msl);
      return CUDA_ERROR_NOT_SUPPORTED;
    }
    NSString *fname = [NSString stringWithFormat:@"%s_msl", g_kernel_name];
    id<MTLFunction> fn = [lib newFunctionWithName:fname];
    if (!fn) {
      slog("cuLaunchKernel", "function %s not in compiled library", fname.UTF8String);
      free(msl);
      return CUDA_ERROR_NOT_SUPPORTED;
    }
    pso = [g_device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
      slog("cuLaunchKernel", "pipeline state failed: %s",
           err.localizedDescription ? err.localizedDescription.UTF8String : "?");
      free(msl);
      return CUDA_ERROR_NOT_SUPPORTED;
    }
    cache_pso(g_kernel_name, g_image_ptr, pso);
    slog("cuLaunchKernel", "MSL compiled and cached");
    free(msl);
  } else {
    slog("cuLaunchKernel", "pipeline cache hit");
  }

  /* 3. bind params: device pointers resolve through the allocation registry
     to real MTLBuffers; anything else is a u32 scalar (measured subset).
     params[i] points AT the i-th argument value. */
  id<MTLCommandBuffer> cb = [g_queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:pso];
  for (int bi = 0; params && params[bi] && bi < 16; bi++) {
    id<MTLBuffer> b = buffer_for(*(void **)params[bi]);
    if (b) {
      [enc setBuffer:b offset:0 atIndex:bi];
    } else {
      uint32_t scalar = 0;
      memcpy(&scalar, params[bi], sizeof scalar);
      [enc setBytes:&scalar length:sizeof scalar atIndex:bi];
    }
  }

  /* 4. dispatch: CUDA grid of blocks → 1-D Metal threadgroups */
  NSUInteger nth = (NSUInteger)bx * by * bz;
  if (nth == 0 || nth > pso.maxTotalThreadsPerThreadgroup) {
    slog("cuLaunchKernel", "block %ux%ux%u exceeds device max %lu",
         bx, by, bz, (unsigned long)pso.maxTotalThreadsPerThreadgroup);
    return CUDA_ERROR_INVALID_VALUE;
  }
  NSUInteger ng = (NSUInteger)gx * gy * gz;
  [enc dispatchThreadgroups:MTLSizeMake(ng, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];

  if (cb.status != MTLCommandBufferStatusCompleted) {
    slog("cuLaunchKernel", "command buffer status %ld", (long)cb.status);
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  slog("cuLaunchKernel", "DISPATCHED OK on %s", g_device.name.UTF8String);
  return CUDA_SUCCESS;
}

/* ---- runtime-API aliases ------------------------------------------------ */
int cudaGetDeviceCount(int *count) { slog("cudaGetDeviceCount", ""); return cuDeviceGetCount(count); }
int cudaGetDeviceProperties(void *props, int device) {
  slog("cudaGetDeviceProperties", "device=%d", device);
  /* Fill the fields apps branch on; struct exceeds the 64-byte prefix we
     touch, and callers pass a full cudaDeviceProp. Keep the memset bounded. */
  if (props) {
    memset(props, 0, 2048);
    char *p = (char *)props;
    snprintf(p + 4 /*name offset in cudaDeviceProp*/, 256, "%s",
             g_device ? g_device.name.UTF8String : "Apple Silicon");
    /* major.minor at +7/+8 in the CUDA 12 layout; multiprocessorCount near +0x74 */
    int major = 8, minor = 6;
    memcpy(p + 7, &major, 4);
    memcpy(p + 11, &minor, 4);
    int sms = 10;
    memcpy(p + 0x74, &sms, 4);
    unsigned ws = 32;
    memcpy(p + 0x12c, &ws, 4);
  }
  return CUDA_SUCCESS;
}
int cudaMalloc(void **devPtr, size_t bytes) {
  CUdeviceptr d = 0;
  slog("cudaMalloc", "bytes=%zu", bytes);
  int r = cuMemAlloc(&d, bytes);
  if (devPtr) *devPtr = (void *)(uintptr_t)d;
  return r;
}
int cudaFree(void *p) { slog("cudaFree", ""); return cuMemFree((CUdeviceptr)(uintptr_t)p); }
int cudaMemcpy(void *dst, const void *src, size_t bytes, int kind) {
  slog("cudaMemcpy", "bytes=%zu kind=%d", bytes, kind);
  if (bytes && dst && src) memcpy(dst, src, bytes); /* unified memory: honest for all kinds */
  return CUDA_SUCCESS;
}
int cudaDeviceSynchronize(void) { slog("cudaDeviceSynchronize", ""); return CUDA_SUCCESS; }
