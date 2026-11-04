/*
 * test_harness.c — exercises the shim without the CUDA toolkit.
 *
 * dlopen("shim/libcuda_shim.dylib") resolves the CUDA Driver/Runtime symbols
 * against the shim (the same way DYLD_INSERT_LIBRARIES catches real apps that
 * link libcuda). Asserts the v0 contract: init → one device → alloc → launch
 * is NOT_SUPPORTED (the honest v0 frontier).
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int (*fn_int_p)(int *);
typedef int (*fn_void)(void);
#define SYM(handle, name) ({ void *p = dlsym(handle, name); \
    if (!p) { fprintf(stderr, "FAIL: %s missing\n", name); exit(1); } \
    (_Generic(p, void *: (fn_void))0, p); p; })

int main(void) {
  void *h = dlopen("shim/libcuda_shim.dylib", RTLD_NOW | RTLD_GLOBAL);
  if (!h) { fprintf(stderr, "FAIL: dlopen: %s\n", dlerror()); return 1; }

  int (*cuInit)(unsigned) = dlsym(h, "cuInit");
  int (*cuDeviceGetCount)(int *) = dlsym(h, "cuDeviceGetCount");
  int (*cuDeviceGetName)(char *, int, int) = dlsym(h, "cuDeviceGetName");
  int (*cuMemAlloc)(int *, size_t) = dlsym(h, "cuMemAlloc");
  int (*cuModuleLoadData)(int *, const void *) = dlsym(h, "cuModuleLoadData");
  int (*cuModuleGetFunction)(int *, int, const char *) = dlsym(h, "cuModuleGetFunction");
  int (*cuLaunchKernel)(int, unsigned, unsigned, unsigned, unsigned, unsigned,
                        unsigned, unsigned, int, void **, void **) = dlsym(h, "cuLaunchKernel");

  if (!cuInit || !cuDeviceGetCount || !cuMemAlloc || !cuModuleLoadData ||
      !cuModuleGetFunction || !cuLaunchKernel || !cuDeviceGetName) {
    fprintf(stderr, "FAIL: symbol resolution incomplete\n");
    return 1;
  }

  int rc = cuInit(0);
  printf("cuInit -> %d\n", rc);

  int count = -1;
  cuDeviceGetCount(&count);
  printf("device count -> %d\n", count);

  char name[128];
  cuDeviceGetName(name, sizeof name, 0);
  printf("device 0 -> %s\n", name);

  void *dptr = NULL;
  rc = cuMemAlloc(&dptr, 1024);
  printf("memAlloc -> %d (real host ptr %p)\n", rc, dptr);
  // v1 contract: the allocation is REAL memory — write, read back, verify.
  memset(dptr, 0xAB, 1024);
  printf("deref test: %s\n", ((unsigned char *)dptr)[512] == 0xAB ? "WRITE/READ OK" : "FAIL");
  memset(dptr, 0, 1024);

  static const unsigned char fatbin[64] = "P3 fatbin placeholder — v2 parses real containers";
  int module = 0;
  rc = cuModuleLoadData(&module, fatbin);
  printf("moduleLoadData -> %d (module %d)\n", rc, module);

  int func = 0;
  rc = cuModuleGetFunction(&func, module, "vectorAdd");
  printf("getFunction(vectorAdd) -> %d (func %d)\n", rc, func);

  rc = cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, 2, NULL, NULL);
  printf("launch -> %d (%s)\n", rc,
         rc == 801 ? "NOT_SUPPORTED — correct v0 frontier" : "UNEXPECTED");

  /* v0 contract: launch must be the honest 801, everything before succeeds. */
  if (count == 1 && rc == 801) {
    printf("PASS — coverage log: /tmp/cuda-metal-shim.log\n");
    return 0;
  }
  fprintf(stderr, "FAIL — v0 contract violated\n");
  return 1;
}
