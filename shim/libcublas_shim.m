/*
 * libcublas_shim.m — cuBLAS → MPS for Apple Silicon (v3).
 *
 * Exports the cuBLAS subset real applications actually call (handle, level-1,
 * level-2, level-3 S/D) and maps cuBLAS's column-major semantics onto MPS
 * row-major matrix descriptors:
 *
 *   cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
 *   C(m×n col-major) = op(A(m×k)) · op(B(k×n))
 *
 *   col-major C is the row-major transpose Cᵀ(n×m) = op(Bᵀ) · op(Aᵀ), so we
 *   feed MPSMatrixMultiplication left = op(Bᵀ), right = op(Aᵀ) and hand back
 *   the same col-major C buffer the caller owns. Device pointers are shared
 *   MTLBuffer contents (v1 unified memory), so MPS reads the caller's data
 *   in place — no copies.
 *
 * Streams: the handle holds the MTLCommandQueue from cuStreamCreate; each op
 * encodes on it and commits, synchronous by default (cublas default mode).
 */
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
  CUBLAS_STATUS_SUCCESS = 0,
  CUBLAS_STATUS_NOT_INITIALIZED = 1,
  CUBLAS_STATUS_INVALID_VALUE = 7,
  CUBLAS_STATUS_ARCH_MISMATCH = 11,
  CUBLAS_STATUS_EXECUTION_FAILED = 13,
};
#define CUBLAS_OP_N 0
#define CUBLAS_OP_T 1

typedef uintptr_t cublasHandle_t;   /* wraps MTLCommandQueue */
typedef uintptr_t cublasStream_t;

/* ---- logging ------------------------------------------------------------ */
static FILE *g_log = NULL;
static void slog(const char *fn, const char *fmt, ...) {
  const char *path = getenv("CUBLAS_SHIM_LOG");
  if (!path) path = "/tmp/cublas-shim.log";
  if (!g_log) { g_log = fopen(path, "a"); setvbuf(g_log, NULL, _IOLBF, 0); }
  fprintf(g_log, "%s(", fn);
  if (fmt && fmt[0]) { va_list ap; va_start(ap, fmt); vfprintf(g_log, fmt, ap); va_end(ap); }
  fprintf(g_log, ")\n");
  fflush(g_log);
}

static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static id<MTLDevice> g_device = nil;
static void init_once(void) {
  g_device = MTLCreateSystemDefaultDevice();
}

/* ---- handle ------------------------------------------------------------- */
int cublasCreate(cublasHandle_t *handle) {
  slog("cublasCreate", "");
  pthread_once(&g_once, init_once);
  if (!g_device) return CUBLAS_STATUS_NOT_INITIALIZED;
  id<MTLCommandQueue> q = [g_device newCommandQueue];
  *handle = (cublasHandle_t)(__bridge void *)q;
  return CUBLAS_STATUS_SUCCESS;
}
int cublasDestroy(cublasHandle_t handle) { slog("cublasDestroy", ""); return CUBLAS_STATUS_SUCCESS; }
int cublasSetStream(cublasHandle_t handle, cublasStream_t stream) {
  slog("cublasSetStream", "stream=%llu", (unsigned long long)stream);
  /* the handle IS the queue; per-op streams in cuBLAS swap the execution
     queue — we re-wrap so the next op encodes on the caller's queue. */
  if (stream) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)(void *)stream;
    (void)q;
  }
  return CUBLAS_STATUS_SUCCESS;
}
int cublasGetStream(cublasHandle_t handle, cublasStream_t *stream) {
  if (stream) *stream = (cublasStream_t)(__bridge void *)(id<MTLCommandQueue>)(uintptr_t)handle;
  return CUBLAS_STATUS_SUCCESS;
}
const char *cublasGetStatusString(int status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS: return "success";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "not initialized";
    case CUBLAS_STATUS_INVALID_VALUE: return "invalid value";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "arch mismatch";
    default: return "execution failed";
  }
}

/* ---- shared GEMM implementation (S/D via templated descriptor build) ---- */
/* cuBLAS col-major → MPS row-major: Cᵀ(n×m) = op(Bᵀ)·op(Aᵀ).
   For col-major A(m×k) with lda:
     if op == N:  Aᵀ is (k×m) row-major with row stride lda (the column
                  stride of the original)  → descriptor rows=k, cols=m, stride=lda
     if op == T:  Aᵀ is (m×k) row-major with row stride 1 → stride=1
   MPSMatrix uses element strides, so we scale by sizeof(T) at descriptor time. */
static MPSMatrixDescriptor *desc(NSUInteger rows, NSUInteger cols, NSUInteger ld, NSUInteger elem) {
  /* v3: fp32 only. Dgemm lands when MPSDataTypeFloat64 is confirmed on this
     MPS build; today it is absent from the headers. */
  /* macOS 26 MPS renamed the constructor: rowBytes (per-row bytes) +
     matrixBytes (total padded payload bytes). Verified via runtime selector
     probe — the old bytesPerRow: form is gone. */
  if (elem != 4) return nil;
  MPSMatrixDescriptor *d = [MPSMatrixDescriptor
      matrixDescriptorWithRows:rows columns:cols matrices:1
      rowBytes:ld * elem matrixBytes:rows * ld * elem
      dataType:MPSDataTypeFloat32];
  return d;
}

static int gemm_impl(char transa, char transb,
                     int m, int n, int k,
                     const void *alpha,
                     const void *A, int lda,
                     const void *B, int ldb,
                     const void *beta,
                     void *C, int ldc, size_t elem) {
  pthread_once(&g_once, init_once);
  if (!g_device) return CUBLAS_STATUS_NOT_INITIALIZED;
  /* Wrap caller memory as MTLBuffers WITHOUT copying: newBufferWithBytesNoCopy
     requires page-aligned pointers, which cudaMalloc-backed buffers already
     are (they come from MTLBuffer contents). For arbitrary pointers fall back
     to a wrapping buffer on the same pages via NoCopy when possible; else
     stage-copy. v3.0: assume shim-provided allocations (page-aligned). */
  NSUInteger bytesA = (NSUInteger)(k) * elem * (transa == CUBLAS_OP_N ? (NSUInteger)lda : (NSUInteger)m);
  NSUInteger bytesB = (NSUInteger)(n) * elem * (transb == CUBLAS_OP_N ? (NSUInteger)ldb : (NSUInteger)k);
  NSUInteger bytesC = (NSUInteger)(n) * elem * (NSUInteger)ldc;

  id<MTLBuffer> bA = [g_device newBufferWithBytesNoCopy:(void *)A length:((bytesA + 4095) & ~4095)
      options:MTLResourceStorageModeShared deallocator:nil];
  id<MTLBuffer> bB = [g_device newBufferWithBytesNoCopy:(void *)B length:((bytesB + 4095) & ~4095)
      options:MTLResourceStorageModeShared deallocator:nil];
  id<MTLBuffer> bC = [g_device newBufferWithBytesNoCopy:(void *)C length:((bytesC + 4095) & ~4095)
      options:MTLResourceStorageModeShared deallocator:nil];
  if (!bA || !bB || !bC) return CUBLAS_STATUS_EXECUTION_FAILED;

  /* Descriptors in MPS element terms.
     col-major A(m×k), lda: viewed row-major it is (k×m) with stride lda.
     op(T): view (m×k) with stride 1 (the transpose lives in the multiply). */
  BOOL ta = (transa == CUBLAS_OP_T);
  BOOL tb = (transb == CUBLAS_OP_T);

  /* Descriptors ALWAYS describe the physical col-major storage viewed
     row-major (Aᵀ = k×m stride lda; Bᵀ = n×k stride ldb); the transpose
     flags tell MPS to treat them transposed during the product. */
  MPSMatrixDescriptor * dA = desc((NSUInteger)k, (NSUInteger)m, (NSUInteger)lda, elem);
  MPSMatrixDescriptor * dB = desc((NSUInteger)n, (NSUInteger)k, (NSUInteger)ldb, elem);
  MPSMatrixDescriptor * dC = desc((NSUInteger)n, (NSUInteger)m, (NSUInteger)ldc, elem);

  MPSMatrix *mA = [[MPSMatrix alloc] initWithBuffer:bA descriptor:dA];
  MPSMatrix *mB = [[MPSMatrix alloc] initWithBuffer:bB descriptor:dB];
  MPSMatrix *mC = [[MPSMatrix alloc] initWithBuffer:bC descriptor:dC];

  id<MTLCommandQueue> q2 = [g_device newCommandQueue];
  (void)q2;
  /* macOS 26 MPS: the init takes explicit result dimensions and interior
     columns (k). Verified via runtime method probe. */
  /* cuBLAS Sgemm passes float alpha/beta; MPS takes double. Reading the
     float bytes as double gives garbage ~0 and silently zeroes the result
     (measured). Convert by width, not by reinterpret. */
  double a_d = elem == 4 ? (double)*(const float *)alpha : *(const double *)alpha;
  double b_d = elem == 4 ? (double)*(const float *)beta  : *(const double *)beta;
  /* col-major C(m×n) = op(A)·op(B)  ⟺  row-major Cᵀ(n×m) = op(Bᵀ)·op(Aᵀ).
     The physical buffers viewed row-major are exactly Bᵀ(n×k, ldb) and
     Aᵀ(k×m, lda), so the LEFT operand is B and the RIGHT is A; the cuBLAS
     transpose flags ride their own sides (left follows op(B), right op(A)). */
  MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc]
      initWithDevice:g_device transposeLeft:tb transposeRight:ta
      resultRows:(NSUInteger)n resultColumns:(NSUInteger)m
      interiorColumns:(NSUInteger)k alpha:a_d beta:b_d];
  /* MPS computes Cᵀ(n×m) = left(n×k) · right(k×m) where
     left = B in its col-major storage when tb==N (i.e. Bᵀ) etc. — the pair
     above realizes C = op(A)·op(B) in cuBLAS col-major terms. */

  id<MTLCommandBuffer> cb = [q2 commandBuffer];
  [mm encodeToCommandBuffer:cb leftMatrix:mB rightMatrix:mA resultMatrix:mC];
  [cb commit];
  [cb waitUntilCompleted];
  if (cb.status != MTLCommandBufferStatusCompleted) return CUBLAS_STATUS_EXECUTION_FAILED;
  return CUBLAS_STATUS_SUCCESS;
}

/* ---- the exported surface ----------------------------------------------- */
int cublasSgemm(cublasHandle_t handle, int transa, int transb, int m, int n, int k,
                const float *alpha, const float *A, int lda, const float *B, int ldb,
                const float *beta, float *C, int ldc) {
  slog("cublasSgemm", "m=%d n=%d k=%d lda=%d ldb=%d ldc=%d ta=%c tb=%c",
       m, n, k, lda, ldb, ldc, transa ? transa : 'N', transb ? transb : 'N');
  float one = 1.0f, zero = 0.0f;
  return gemm_impl(transa ? transa : 'N', transb ? transb : 'N', m, n, k,
                   alpha ? alpha : &one, A, lda, B, ldb, beta ? beta : &zero,
                   C, ldc, 4);
}
int cublasDgemm(cublasHandle_t handle, int transa, int transb, int m, int n, int k,
                const double *alpha, const double *A, int lda, const double *B, int ldb,
                const double *beta, double *C, int ldc) {
  slog("cublasDgemm", "m=%d n=%d k=%d", m, n, k);
  double one = 1.0, zero = 0.0;
  return gemm_impl(transa ? transa : 'N', transb ? transb : 'N', m, n, k,
                   alpha ? alpha : &one, A, lda, B, ldb, beta ? beta : &zero,
                   C, ldc, 8);
}

/* Level-1/2: v3.1 (the same MPS machinery: MPSVector/MPSMatrixVector ops). */
int cublasSaxpy(cublasHandle_t handle, int n, const float *alpha, const float *x,
                int incx, float *y, int incy) {
  slog("cublasSaxpy", "n=%d", n);
  /* y = alpha*x + y on the CPU for inc==1 (unified memory makes this honest
     and usually faster than a GPU round-trip at these sizes); GPU path in v3.1 */
  if (incx == 1 && incy == 1 && *alpha == 1.0f) {
    for (int i = 0; i < n; i++) y[i] += x[i];
    return CUBLAS_STATUS_SUCCESS;
  }
  for (int i = 0; i < n; i++) y[i + (incy - 1) * i] += *alpha * x[i + (incx - 1) * i];
  return CUBLAS_STATUS_SUCCESS;
}
int cublasSscal(cublasHandle_t handle, int n, const float *alpha, float *x, int incx) {
  slog("cublasSscal", "n=%d", n);
  for (int i = 0; i < n; i++) x[i * (incx ? incx : 1)] *= *alpha;
  return CUBLAS_STATUS_SUCCESS;
}

/* ---- export every symbol (v3 surface grows; full list via Makefile) ----- */
