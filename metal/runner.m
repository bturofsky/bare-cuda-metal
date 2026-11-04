// runner.m — the Metal arm of the A/B test. Compiles the v2 target MSL
// (docs/reference/vectorAdd.msl, inlined here verbatim minus comments) at
// runtime on the Mac's Apple GPU and emits the JSON record.
// Build (Mac): clang -O2 -framework Metal -framework Foundation -o metal_runner runner.m
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include "../common/spec.h"

static double now_ms(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static const char *kMSL =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void vectorAdd(device const float* a [[buffer(0)]],\n"
    "                      device const float* b [[buffer(1)]],\n"
    "                      device float*       c [[buffer(2)]],\n"
    "                      constant uint&      n [[buffer(3)]],\n"
    "                      uint3 tid [[thread_position_in_threadgroup]],\n"
    "                      uint3 bid [[threadgroup_position_in_grid]],\n"
    "                      uint3 bdim [[threads_per_threadgroup]])\n"
    "{\n"
    "    uint r1 = bid.x * bdim.x + tid.x;\n"
    "    if (r1 >= n) return;\n"
    "    c[r1] = a[r1] + b[r1];\n"
    "}\n";

int main(void) {
  static float a[SPEC_N], b[SPEC_N], c[SPEC_N];
  spec_fill(a, b);

  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) { printf("ERROR no Metal device\n"); return 1; }
  fprintf(stderr, "stage: device ok (%s)\n", device.name.UTF8String); fflush(stdout);
  NSError *err = nil;
  id<MTLLibrary> lib = [device newLibraryWithSource:@(kMSL) options:nil error:&err];
  if (!lib) { printf("ERROR compile: %s\n", err.localizedDescription.UTF8String); return 1; }
  fprintf(stderr, "stage: msl compiled\n"); fflush(stdout);
  id<MTLFunction> fn = [lib newFunctionWithName:@"vectorAdd"];
  id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
  if (!pso) { printf("ERROR pso: %s\n", err.localizedDescription.UTF8String); return 1; }
  fprintf(stderr, "stage: pso ok\n"); fflush(stdout);

  id<MTLCommandQueue> q = [device newCommandQueue];
  id<MTLBuffer> ba = [device newBufferWithBytes:a length:SPEC_N*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> bb = [device newBufferWithBytes:b length:SPEC_N*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> bc = [device newBufferWithLength:SPEC_N*4 options:MTLResourceStorageModeShared];
  uint32_t n = SPEC_N;
  id<MTLBuffer> bn = [device newBufferWithBytes:&n length:4 options:MTLResourceStorageModeShared];

  fprintf(stderr, "stage: buffers ok\n"); fflush(stdout);
  double t0 = now_ms();
  id<MTLCommandBuffer> cb = [q commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:pso];
  [enc setBuffer:ba offset:0 atIndex:0];
  [enc setBuffer:bb offset:0 atIndex:1];
  [enc setBuffer:bc offset:0 atIndex:2];
  [enc setBuffer:bn offset:0 atIndex:3];
  NSUInteger groups = (SPEC_N + 255) / 256;
  [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [enc endEncoding];
  [cb commit];
  fprintf(stderr, "stage: dispatch queued\n"); fflush(stdout);
  [cb waitUntilCompleted];
  double t1 = now_ms();
  if (cb.status != MTLCommandBufferStatusCompleted) { printf("ERROR cb status %ld\n", (long)cb.status); return 1; }

  double gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;  // the GPU's own clock
  memcpy(c, bc.contents, SPEC_N * 4);
  fprintf(stderr, "wall_ms (single dispatch incl. launch overhead): %.3f\n", t1 - t0);
  fprintf(stderr, "gpu_ms  (GPU work only, GPU clock):                %.3f\n", gpu_ms);
  spec_json("metal-applesilicon", spec_checksum(c), gpu_ms, c);
  return 0;
}
