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
#import <mach-o/dyld.h>
#import <mach-o/getsect.h>

/* module_launch.m: fatbin/raw-PTX extraction + PTX→MSL translation */
extern char *cuda_metal_extract_ptx(const void *fatbin_data, size_t fatbin_size,
                                    const char *kernel_name, size_t *ptx_len);
extern char *cuda_metal_translate(const char *ptx, size_t ptx_len, const char *kernel_name);
extern int translate_ptx_kernel(const char *ptx_text, size_t ptx_len,
                                const char *kernel_name, char *msl_buf,
                                size_t msl_buf_sz, char *err_buf, size_t err_sz);
extern int ptx_param_types(const char *mangled, const char *ptx_text, size_t ptx_len,
                           char *types, int max_types);

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

/* ---- streams / events: real queues, async commit, GPU timestamps -------- */

/* Streams keep strong queue references (the old __bridge cast leaked the
   queue the moment the local went out of scope). Named streams commit
   asynchronously; cuStreamSynchronize waits on the stream's last buffer.
   The default stream (0) keeps the legacy synchronous path. */
#define MAX_STREAMS 8
#define MAX_EVENTS 8
typedef struct {
  int used;
  id<MTLCommandQueue> q;
  id<MTLCommandBuffer> last_cb;
} stream_rec;
static stream_rec g_streams[MAX_STREAMS];
typedef struct {
  int used;
  id<MTLCommandBuffer> cb;
} event_rec;
static event_rec g_events[MAX_EVENTS];

static id<MTLCommandQueue> queue_for(CUdeviceptr h) {
  if (h == 0) return g_queue;
  int i = (int)h - 1;
  if (i < 0 || i >= MAX_STREAMS || !g_streams[i].used) return nil;
  return g_streams[i].q;
}
static id<MTLCommandBuffer> *last_cb_slot(CUdeviceptr h) {
  if (h == 0) return NULL;
  int i = (int)h - 1;
  if (i < 0 || i >= MAX_STREAMS || !g_streams[i].used) return NULL;
  return &g_streams[i].last_cb;
}

int cuStreamCreate(CUdeviceptr *stream, unsigned flags) {
  slog("cuStreamCreate", "flags=0x%x", flags);
  if (!g_device) return CUDA_ERROR_NOT_INITIALIZED;
  int slot = -1;
  for (int i = 0; i < MAX_STREAMS; i++)
    if (!g_streams[i].used) { slot = i; break; }
  if (slot < 0) return CUDA_ERROR_OUT_OF_MEMORY;
  g_streams[slot].q = [g_device newCommandQueue];
  g_streams[slot].last_cb = nil;
  g_streams[slot].used = 1;
  if (stream) *stream = (CUdeviceptr)(slot + 1);
  return CUDA_SUCCESS;
}
int cuStreamDestroy(CUdeviceptr stream) {
  int i = (int)stream - 1;
  if (i >= 0 && i < MAX_STREAMS && g_streams[i].used) {
    g_streams[i].used = 0;
    g_streams[i].q = nil;
    g_streams[i].last_cb = nil;
  }
  slog("cuStreamDestroy", "stream=%llu", (unsigned long long)stream);
  return CUDA_SUCCESS;
}
int cuStreamSynchronize(CUdeviceptr stream) {
  int rc = CUDA_SUCCESS;
  id<MTLCommandBuffer> *slot = last_cb_slot(stream);
  if (slot && *slot) {
    [*slot waitUntilCompleted];
    if ((*slot).status != MTLCommandBufferStatusCompleted) {
      slog("cuStreamSynchronize", "stream=%llu status %ld",
           (unsigned long long)stream, (long)(*slot).status);
      rc = CUDA_ERROR_NOT_SUPPORTED;
    }
    *slot = nil;
  }
  slog("cuStreamSynchronize", "stream=%llu", (unsigned long long)stream);
  return rc;
}

int cuEventCreate(CUdeviceptr *event, unsigned flags) {
  slog("cuEventCreate", "flags=0x%x", flags);
  int slot = -1;
  for (int i = 0; i < MAX_EVENTS; i++)
    if (!g_events[i].used) { slot = i; break; }
  if (slot < 0) return CUDA_ERROR_OUT_OF_MEMORY;
  g_events[slot].used = 1;
  g_events[slot].cb = nil;
  if (event) *event = (CUdeviceptr)(slot + 1);
  return CUDA_SUCCESS;
}
int cuEventRecord(CUdeviceptr event, CUdeviceptr stream) {
  int ei = (int)event - 1;
  if (ei < 0 || ei >= MAX_EVENTS || !g_events[ei].used) return CUDA_ERROR_INVALID_VALUE;
  id<MTLCommandBuffer> *slot = last_cb_slot(stream);
  g_events[ei].cb = slot ? *slot : nil;
  slog("cuEventRecord", "event=%llu stream=%llu cb=%s",
       (unsigned long long)event, (unsigned long long)stream, slot ? "bound" : "default");
  return CUDA_SUCCESS;
}
int cuEventSynchronize(CUdeviceptr event) {
  int ei = (int)event - 1;
  if (ei < 0 || ei >= MAX_EVENTS || !g_events[ei].used) return CUDA_ERROR_INVALID_VALUE;
  if (g_events[ei].cb) [g_events[ei].cb waitUntilCompleted];
  slog("cuEventSynchronize", "event=%llu", (unsigned long long)event);
  return CUDA_SUCCESS;
}
int cuEventDestroy(CUdeviceptr event) {
  int ei = (int)event - 1;
  if (ei >= 0 && ei < MAX_EVENTS) {
    g_events[ei].used = 0;
    g_events[ei].cb = nil;
  }
  slog("cuEventDestroy", "event=%llu", (unsigned long long)event);
  return CUDA_SUCCESS;
}
float cuEventElapsedTime(CUdeviceptr start, CUdeviceptr end) {
  int si = (int)start - 1, ei = (int)end - 1;
  float ms = 0.0f;
  if (si >= 0 && si < MAX_EVENTS && ei >= 0 && ei < MAX_EVENTS &&
      g_events[si].cb && g_events[ei].cb) {
    id<MTLCommandBuffer> s = g_events[si].cb, e = g_events[ei].cb;
    [s waitUntilCompleted];
    [e waitUntilCompleted];
    if (s.GPUEndTime > 0 && e.GPUStartTime > 0)
      ms = (float)((e.GPUStartTime - s.GPUEndTime) * 1000.0);
    slog("cuEventElapsedTime", "gpu ts: start_end=%f end_start=%f", s.GPUEndTime, e.GPUStartTime);
  }
  slog("cuEventElapsedTime", "start=%llu end=%llu -> %f ms",
       (unsigned long long)start, (unsigned long long)end, ms);
  return ms;
}

/* ---- module / kernel: v2 fatbin/raw-PTX → MSL → Metal dispatch ---------- */

/* Resolve the true extent of a fatbin pointer: when it lies inside a
   __NV_CUDA,__nv_fatbin section of any loaded Mach-O image (the layout
   NVIDIA's fatBinaryCtl.h defines for Darwin), return the exact section
   extent. Falls back to the caller's bound otherwise. */
static size_t resolve_image_size(const void *image, size_t fallback) {
  if (!image) return fallback;
  for (uint32_t i = 0; i < _dyld_image_count(); i++) {
    const struct mach_header_64 *hdr =
        (const struct mach_header_64 *)_dyld_get_image_header(i);
    if (!hdr) continue;
    unsigned long size = 0;
    char *data = getsectiondata(hdr, "__NV_CUDA", "__nv_fatbin", &size);
    if (data && size > 0 && (const char *)image >= data &&
        (const char *)image < data + size)
      return (size_t)(data + size - (const char *)image);
  }
  return fallback;
}

/* Module and function tables: a module holds its image pointer; a function
   handle (index+1) binds one kernel name to one module. */
#define MAX_MODULES 8
#define MAX_FUNCS 32
typedef struct { const void *image; size_t size; } mod_rec;
static mod_rec g_mods[MAX_MODULES];
static int g_nmods = 0;
typedef struct { int mod; char name[128]; } func_rec;
static func_rec g_funcs[MAX_FUNCS];
static int g_nfuncs = 0;

/* Launch-time MSL compile cache: (image, kernel name) → pipeline state +
   the decoded param types used for binding. The first launch pays the
   Metal frontend compile (~0.5s); later launches reuse the cached state. */
#define MAX_PSOS 16
#define MAX_PTYPES 16
typedef struct {
  char name[128];
  const void *image;
  id<MTLComputePipelineState> pso;
  char ptypes[MAX_PTYPES];
  int nptypes;
} pso_rec;
static pso_rec g_psos[MAX_PSOS];
static int g_npsos = 0;
static pthread_mutex_t g_pso_lock = PTHREAD_MUTEX_INITIALIZER;

static id<MTLComputePipelineState> cached_pso(const char *name, const void *image,
                                              char *types_out, int *n_out) {
  pthread_mutex_lock(&g_pso_lock);
  id<MTLComputePipelineState> p = nil;
  for (int i = 0; i < g_npsos; i++)
    if (g_psos[i].image == image && strcmp(g_psos[i].name, name) == 0) {
      p = g_psos[i].pso;
      if (types_out && n_out) {
        memcpy(types_out, g_psos[i].ptypes, (size_t)g_psos[i].nptypes);
        *n_out = g_psos[i].nptypes;
      }
      break;
    }
  pthread_mutex_unlock(&g_pso_lock);
  return p;
}

static void cache_pso(const char *name, const void *image, id<MTLComputePipelineState> p,
                      const char *types, int ntypes) {
  pthread_mutex_lock(&g_pso_lock);
  if (g_npsos < MAX_PSOS) {
    snprintf(g_psos[g_npsos].name, sizeof g_psos[g_npsos].name, "%s", name);
    g_psos[g_npsos].image = image;
    g_psos[g_npsos].pso = p;
    if (types && ntypes > 0) {
      memcpy(g_psos[g_npsos].ptypes, types, (size_t)ntypes);
      g_psos[g_npsos].nptypes = ntypes;
    }
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
  int mi = -1;
  for (int i = 0; i < g_nmods; i++)
    if (g_mods[i].image == image) { mi = i; break; }
  if (mi < 0) {
    if (g_nmods >= MAX_MODULES) {
      slog("cuModuleLoadData", "module table full");
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    g_mods[g_nmods].image = image;
    g_mods[g_nmods].size = resolve_image_size(image, 1 << 20); /* exact when the image is a Mach-O __NV_CUDA,__nv_fatbin section; else bounded scan */
    mi = g_nmods++;
  }
  if (module) *module = (CUdeviceptr)(mi + 1); /* handle: index+1, 0 invalid */
  return CUDA_SUCCESS;
}
int cuModuleUnload(CUdeviceptr module) { slog("cuModuleUnload", ""); return CUDA_SUCCESS; }

int cuModuleGetFunction(CUdeviceptr *func, CUdeviceptr module, const char *name) {
  int mi = (int)module - 1;
  slog("cuModuleGetFunction", "module=%d name=%s", mi, name ? name : "(null)");
  if (mi < 0 || mi >= g_nmods) return CUDA_ERROR_INVALID_VALUE;
  if (!name || !name[0]) return CUDA_ERROR_INVALID_VALUE;
  if (g_nfuncs >= MAX_FUNCS) {
    slog("cuModuleGetFunction", "function table full");
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  g_funcs[g_nfuncs].mod = mi;
  snprintf(g_funcs[g_nfuncs].name, sizeof g_funcs[g_nfuncs].name, "%s", name);
  if (func) *func = (CUdeviceptr)(g_nfuncs + 1);
  g_nfuncs++;
  return CUDA_SUCCESS;
}

int cuLaunchKernel(CUdeviceptr func, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                   unsigned by, unsigned bz, unsigned sharedMem, CUdeviceptr stream,
                   void **params, void **extra) {
  int fidx = (int)func - 1;
  if (!g_device) return CUDA_ERROR_NOT_INITIALIZED;
  if (fidx < 0 || fidx >= g_nfuncs) {
    slog("cuLaunchKernel", "invalid function handle %llu", (unsigned long long)func);
    return CUDA_ERROR_INVALID_VALUE;
  }
  int mi = g_funcs[fidx].mod;
  const void *image = g_mods[mi].image;
  size_t image_size = g_mods[mi].size;
  const char *kname = g_funcs[fidx].name;
  slog("cuLaunchKernel", "%s grid=%ux%ux%u block=%ux%ux%u params=%p",
       kname, gx, gy, gz, bx, by, bz, (void *)params);

  /* 1. pipeline state — cached per (image, kernel); extract+translate only
     on a miss so cached launches skip the PTX work entirely */
  char ptypes[MAX_PTYPES];
  int nptypes = 0;
  id<MTLComputePipelineState> pso = cached_pso(kname, image, ptypes, &nptypes);
  if (!pso) {
    size_t ptx_len = 0;
    char *ptx = cuda_metal_extract_ptx(image, image_size, kname, &ptx_len);
    if (!ptx) {
      slog("cuLaunchKernel", "no PTX found for %s in image", kname);
      return CUDA_ERROR_NOT_SUPPORTED;
    }
    nptypes = ptx_param_types(kname, ptx, ptx_len, ptypes, MAX_PTYPES);
    if (nptypes <= 0) {
      nptypes = 0;
      slog("cuLaunchKernel", "param types unavailable — heuristic binding");
    }
    /* translate directly so the translator's error reaches the log */
    char terr[256] = "";
    char *msl = malloc(16384);
    int trc = translate_ptx_kernel(ptx, ptx_len, kname, msl, 16384,
                                   terr, sizeof terr);
    free(ptx);
    if (trc != 0) {
      slog("cuLaunchKernel", "PTX→MSL translation failed for %s: %s",
           kname, terr[0] ? terr : "unknown");
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
    NSString *fname = [NSString stringWithFormat:@"%s_msl", kname];
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
    cache_pso(kname, image, pso, ptypes, nptypes);
    slog("cuLaunchKernel", "MSL compiled and cached");
    free(msl);
  } else {
    slog("cuLaunchKernel", "pipeline cache hit");
  }

  /* 3. bind params: types come from the mangled name (exact) or the PTX
     .param list (u64 ambiguous). 'p': registry lookup → real MTLBuffer,
     else the raw 8 bytes. 'l'/'d': 8-byte scalar. 'u'/'i'/'f': exactly 4
     bytes (never reads past a 4-byte argument slot). */
  id<MTLCommandQueue> q = queue_for(stream);
  if (!q) {
    slog("cuLaunchKernel", "invalid stream %llu", (unsigned long long)stream);
    return CUDA_ERROR_INVALID_VALUE;
  }
  id<MTLCommandBuffer> cb = [q commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:pso];
  for (int bi = 0; params && params[bi] && bi < 16; bi++) {
    char t = bi < nptypes ? ptypes[bi] : '?';
    if (t == 'p' || t == '?') {
      void *val = *(void **)params[bi];
      id<MTLBuffer> b = buffer_for(val);
      if (b) {
        [enc setBuffer:b offset:0 atIndex:bi];
      } else {
        /* pointer outside the registry: bind the 8 raw bytes (unified
           memory makes the host pointer a usable device address) */
        [enc setBytes:params[bi] length:8 atIndex:bi];
      }
    } else if (t == 'l' || t == 'd') {
      [enc setBytes:params[bi] length:8 atIndex:bi];
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

  /* Default stream: synchronous (wait + status check). Named streams:
     async commit — cuStreamSynchronize owns the wait. */
  id<MTLCommandBuffer> *slot = last_cb_slot(stream);
  if (slot) {
    *slot = cb;
    slog("cuLaunchKernel", "committed async on %s", g_device.name.UTF8String);
    return CUDA_SUCCESS;
  }
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
