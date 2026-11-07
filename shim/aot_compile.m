/*
 * aot_compile.m — the bare-metal AOT layer (v3).
 *
 * MSL from the PTX→MSL translator is compiled OFFLINE by Apple's real
 * toolchain — `metal -c -O3 -std=metal3.0` to AIR, then `metallib` into a
 * standalone .metallib binary — and cached on disk keyed by an FNV-1a64
 * hash of the MSL text. At launch the binary is loaded with
 * newLibraryWithData: (no runtime source compile at all). This kills the
 * ~0.5 s first-launch Metal frontend JIT: the first process pays one
 * offline compile, every later process loads machine-ready AIR bytes.
 *
 * Failure contract: any AOT step failing (toolchain missing, compile
 * error, unreadable cache) returns nil and the caller falls back to the
 * v2 runtime JIT path. Never a wrong result — same MSL, same semantics.
 *
 * Env:
 *   CUDA_METAL_AOT=0          disable (JIT-only, v2 behavior)
 *   CUDA_METAL_AOT_CACHE=dir  cache dir (default ~/.cache/cuda-metal)
 */
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define AOT_SALT "cuda-metal-aot-v1" /* bump to invalidate older caches */

static uint64_t fnv1a64(const void *data, size_t n) {
  const unsigned char *p = (const unsigned char *)data;
  uint64_t h = 0xcbf29ce484222325ULL;
  for (size_t i = 0; i < n; i++) {
    h ^= p[i];
    h *= 0x100000001b3ULL;
  }
  return h;
}

static int aot_enabled(void) {
  const char *v = getenv("CUDA_METAL_AOT");
  return v && strcmp(v, "0") == 0 ? 0 : 1;
}

static NSString *aot_cache_dir(void) {
  const char *env = getenv("CUDA_METAL_AOT_CACHE");
  NSString *dir = env && env[0] ? @(env)
                                : [@"~/.cache/cuda-metal" stringByExpandingTildeInPath];
  return dir;
}

/* The metallib path for an MSL text: <cache>/<hash>_<name>.metallib.
   Returns nil when AOT is disabled. *fresh_out is set when the caller must
   compile (no cached binary exists). */
static NSString *aot_path_for(const char *kernel_name, const char *msl, int *fresh_out) {
  if (!aot_enabled()) return nil;
  size_t msl_len = strlen(msl);
  uint64_t h = fnv1a64(AOT_SALT, strlen(AOT_SALT));
  h = fnv1a64(msl, msl_len) ^ (h << 1); /* chain salt + text */
  char hash[17];
  snprintf(hash, sizeof hash, "%016llx", (unsigned long long)h);

  char safe[64];
  size_t k = 0;
  for (const char *p = kernel_name; *p && k < sizeof safe - 1; p++)
    if ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
        (*p >= '0' && *p <= '9') || *p == '_')
      safe[k++] = *p;
  safe[k] = 0;

  NSString *dir = aot_cache_dir();
  [[NSFileManager defaultManager] createDirectoryAtPath:dir
                            withIntermediateDirectories:YES
                                            attributes:nil
                                                 error:nil];
  if (fresh_out) *fresh_out = 1;
  NSString *path = [dir stringByAppendingFormat:@"/%s_%s.metallib", hash, safe];
  if ([[NSFileManager defaultManager] fileExistsAtPath:path] &&
      [[NSFileManager defaultManager] attributesOfItemAtPath:path error:nil].fileSize > 0)
    if (fresh_out) *fresh_out = 0;
  return path;
}

/* Run /usr/bin/xcrun <launch> <args...>. Returns 0 on exit status 0. */
static int run_xcrun(NSArray *args) {
  NSTask *t = [[NSTask alloc] init];
  t.launchPath = @"/usr/bin/xcrun";
  t.arguments = args;
  t.standardOutput = [NSPipe pipe];
  t.standardError = [NSPipe pipe];
  @try {
    [t launch];
    [t waitUntilExit];
  } @catch (NSException *e) {
    return -1;
  }
  return t.terminationStatus == 0 ? 0 : -1;
}

/* Load a .metallib binary and return its library (nil on any failure). */
static id<MTLLibrary> load_metallib(id<MTLDevice> dev, NSString *path, NSError **err) {
  NSData *bin = [NSData dataWithContentsOfFile:path];
  if (!bin) return nil;
  dispatch_data_t dd = dispatch_data_create(bin.bytes, bin.length,
                                            dispatch_get_global_queue(0, 0), ^{
                                              (void)bin; /* retain for the data's lifetime */
                                            });
  return [dev newLibraryWithData:dd error:err];
}

/* Public entry: AOT library for this MSL, compiling through the offline
   toolchain on a miss. kernel_name selects the function inside the library
   (the translator emits <name>_msl). Logs the path taken into logbuf. */
id<MTLLibrary> aot_library_for_msl(id<MTLDevice> dev, const char *kernel_name,
                                   const char *msl, char *logbuf, size_t logcap) {
  if (logbuf && logcap) logbuf[0] = 0;
  if (!aot_enabled()) return nil;

  int fresh = 0;
  NSString *path = aot_path_for(kernel_name, msl, &fresh);
  if (!path) return nil;

  if (!fresh) {
    NSError *err = nil;
    id<MTLLibrary> lib = load_metallib(dev, path, &err);
    if (lib) {
      if (logbuf) snprintf(logbuf, logcap, "AOT metallib loaded from cache (%s)",
                           path.fileSystemRepresentation);
      return lib;
    }
    /* corrupt/stale binary: fall through and recompile */
    if (logbuf) snprintf(logbuf, logcap, "AOT cache unreadable (%s), recompiling",
                         err.localizedDescription.UTF8String ?: "?");
  }

  NSString *base = [path stringByDeletingPathExtension];
  NSString *msl_path = [base stringByAppendingPathExtension:@"metal"];
  NSString *air_path = [base stringByAppendingPathExtension:@"air"];
  NSString *tmp_lib = [base stringByAppendingPathExtension:@"metallib.tmp"];

  if (![@(msl) writeToFile:msl_path atomically:YES encoding:NSUTF8StringEncoding error:nil]) {
    if (logbuf && !logbuf[0]) snprintf(logbuf, logcap, "AOT msl write failed");
    return nil;
  }

  int rc = run_xcrun(@[@"metal", @"-c", @"-O3", @"-std=metal3.0",
                       msl_path, @"-o", air_path]);
  if (rc != 0) {
    if (logbuf && !logbuf[0]) snprintf(logbuf, logcap, "AOT metal -c failed (rc=%d)", rc);
    return nil;
  }
  rc = run_xcrun(@[@"metallib", air_path, @"-o", tmp_lib]);
  if (rc != 0) {
    if (logbuf && !logbuf[0]) snprintf(logbuf, logcap, "AOT metallib link failed (rc=%d)", rc);
    return nil;
  }
  /* atomic publish: readers either see the old bytes or the complete new lib */
  [[NSFileManager defaultManager] removeItemAtPath:path error:nil];
  if (![[NSFileManager defaultManager] moveItemAtPath:tmp_lib toPath:path error:nil]) {
    if (logbuf && !logbuf[0]) snprintf(logbuf, logcap, "AOT publish failed");
    return nil;
  }
  [[NSFileManager defaultManager] removeItemAtPath:air_path error:nil];
  [[NSFileManager defaultManager] removeItemAtPath:msl_path error:nil];

  NSError *err = nil;
  id<MTLLibrary> lib = load_metallib(dev, path, &err);
  if (lib && logbuf)
    snprintf(logbuf, logcap, "AOT compiled offline: metal -c -O3 + metallib -> %s",
             path.fileSystemRepresentation);
  else if (!lib && logbuf)
    snprintf(logbuf, logcap, "AOT load-after-compile failed: %s",
             err.localizedDescription.UTF8String ?: "?");
  return lib;
}

/* Suite support: report the cache path the current MSL would use. */
int aot_probe_path(const char *kernel_name, const char *msl, char *out, size_t cap) {
  int fresh = 0;
  NSString *path = aot_path_for(kernel_name, msl, &fresh);
  if (!path) return -1;
  snprintf(out, cap, "%s", path.fileSystemRepresentation);
  return fresh;
}
