// perf_native.m — the NATIVE arm of the performance gate (family F5).
// Hand-written MSL vectorAdd compiled directly through Apple's runtime —
// the baseline our translated path is compared against. GPU clocks only,
// warm-up dispatch, fastest of 50 iterations (the CuMetal-adopted gate
// methodology, upgraded from wall clock to GPU timestamps).
// Build: clang -O2 -Werror -framework Metal -framework Foundation -o perf_native perf_native.m
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <time.h>

static const char kMSL[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void vectorAdd(device const float* a [[buffer(0)]],\n"
    "                      device const float* b [[buffer(1)]],\n"
    "                      device float* c [[buffer(2)]],\n"
    "                      constant uint& n [[buffer(3)]],\n"
    "                      uint i [[thread_position_in_grid]])\n"
    "{ if (i < n) c[i] = a[i] + b[i]; }\n";

static double gpu_ms(id<MTLCommandQueue> q, id<MTLComputePipelineState> pso,
                     id<MTLBuffer> a, id<MTLBuffer> b, id<MTLBuffer> c,
                     id<MTLBuffer> n, NSUInteger groups, NSUInteger nth) {
  id<MTLCommandBuffer> cb = [q commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:pso];
  [enc setBuffer:a offset:0 atIndex:0];
  [enc setBuffer:b offset:0 atIndex:1];
  [enc setBuffer:c offset:0 atIndex:2];
  [enc setBuffer:n offset:0 atIndex:3];
  [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  return (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
}

int main(void) {
  @autoreleasepool {
    const NSUInteger N = 262144, GROUPS = 1024, THREADS = 256; /* CuMetal-gate geometry */
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { printf("ERROR no device\n"); return 1; }
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@(kMSL) options:nil error:&err];
    if (!lib) { printf("ERROR compile: %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"vectorAdd"];
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) { printf("ERROR pso: %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];

    id<MTLBuffer> a = [dev newBufferWithLength:N * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> b = [dev newBufferWithLength:N * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> c = [dev newBufferWithLength:N * 4 options:MTLResourceStorageModeShared];
    uint32_t n = (uint32_t)N;
    id<MTLBuffer> bn = [dev newBufferWithBytes:&n length:4 options:MTLResourceStorageModeShared];
    for (NSUInteger i = 0; i < N; i++) { ((float *)a.contents)[i] = 1.0f; ((float *)b.contents)[i] = 2.0f; }

    double best = 1e9;
    gpu_ms(q, pso, a, b, c, bn, GROUPS, THREADS); /* warm-up */
    for (int it = 0; it < 50; it++) {
      double ms = gpu_ms(q, pso, a, b, c, bn, GROUPS, THREADS);
      if (ms > 0 && ms < best) best = ms;
    }
    printf("{\"backend\":\"native-metal\",\"device\":\"%s\",\"elements\":%lu,"
           "\"iterations\":50,\"best_gpu_ms\":%.6f}\n",
           dev.name.UTF8String, (unsigned long)N, best);
    return 0;
  }
}
