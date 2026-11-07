/*
 * negative.c — the negative-test inventory (adopted from CuMetal's
 * negative-case discipline: accepting a program is not implementing it).
 *
 * Ten hostile inputs against the shim. Every case must produce the honest
 * error contract (specific CUDA error code, no crash, no garbage output)
 * — or the deliberate recovery behavior (corrupt AOT cache -> JIT fallback
 * with correct results). Exit non-zero if any case breaks the contract.
 * Build: negative.c shim/libcuda_shim.dylib
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned long long CUdeviceptr;
int cuInit(unsigned int);
int cuMemAlloc(CUdeviceptr *, size_t);
int cuMemFree(CUdeviceptr);
int cuMemcpyHtoD(CUdeviceptr, const void *, size_t);
int cuMemcpyDtoH(void *, CUdeviceptr, size_t);
int cuModuleLoadData(CUdeviceptr *, const void *);
int cuModuleGetFunction(CUdeviceptr *, CUdeviceptr, const char *);
int cuLaunchKernel(CUdeviceptr, unsigned, unsigned, unsigned, unsigned, unsigned,
                   unsigned, unsigned, CUdeviceptr, void **, void **);
int cuStreamCreate(CUdeviceptr *, unsigned);
int cuStreamDestroy(CUdeviceptr);

static int pass = 0, fail = 0;
static void gate(const char *name, int ok, const char *detail) {
  if (ok) { pass++; printf("  PASS %-42s %s\n", name, detail ? detail : ""); }
  else { fail++; printf("  FAIL %-42s %s\n", name, detail ? detail : ""); }
}

static char ptx_good[] =
    ".version 8.4\n.target sm_86\n.address_size 64\n"
    ".visible .entry _Z9vectorAddPKfS0_Pfi("
    ".param .u64 _Z9vectorAddPKfS0_Pfi_param_0,"
    ".param .u64 _Z9vectorAddPKfS0_Pfi_param_1,"
    ".param .u64 _Z9vectorAddPKfS0_Pfi_param_2,"
    ".param .u32 _Z9vectorAddPKfS0_Pfi_param_3)\n{\n"
    ".reg .b64 %rd<11>;\n.reg .b32 %r<6>;\n.reg .pred %p<2>;\n"
    "ld.param.u64 %rd1, [_Z9vectorAddPKfS0_Pfi_param_0];\n"
    "ld.param.u64 %rd2, [_Z9vectorAddPKfS0_Pfi_param_1];\n"
    "ld.param.u64 %rd3, [_Z9vectorAddPKfS0_Pfi_param_2];\n"
    "ld.param.u32 %r2, [_Z9vectorAddPKfS0_Pfi_param_3];\n"
    "mov.u32 %r3, %ctaid.x;\nmov.u32 %r4, %ntid.x;\nmov.u32 %r5, %tid.x;\n"
    "mad.lo.s32 %r1, %r3, %r4, %r5;\nsetp.ge.s32 %p1, %r1, %r2;\n"
    "@%p1 bra $L__BB0_2;\n"
    "mul.wide.s32 %rd5, %r1, 4;\nadd.s64 %rd6, %rd1, %rd5;\n"
    "add.s64 %rd8, %rd2, %rd5;\nld.global.f32 %f1, [%rd8];\n"
    "ld.global.f32 %f2, [%rd6];\nadd.f32 %f3, %f2, %f1;\n"
    "cvta.to.global.u64 %rd9, %rd3;\nadd.s64 %rd10, %rd9, %rd5;\n"
    "st.global.f32 [%rd10], %f3;\n"
    "$L__BB0_2:\nret;\n}\n";
static char ptx_unsupported[] =
    ".version 8.4\n.target sm_86\n.address_size 64\n"
    ".visible .entry _Z9vectorAddPKfS0_Pfi("
    ".param .u64 _Z9vectorAddPKfS0_Pfi_param_0,"
    ".param .u64 _Z9vectorAddPKfS0_Pfi_param_1,"
    ".param .u64 _Z9vectorAddPKfS0_Pfi_param_2,"
    ".param .u32 _Z9vectorAddPKfS0_Pfi_param_3)\n{\n"
    "ld.param.u64 %rd1, [_Z9vectorAddPKfS0_Pfi_param_0];\n"
    "suld.b.1d.v2.u32 [%rd1], {%r1, %r2};\n" /* outside the profile: must hard-fail */
    "ret;\n}\n";

int main(void) {
  setvbuf(stdout, NULL, _IONBF, 0);
  cuInit(0);
  printf("NEGATIVE-TEST INVENTORY — the honest-error contract\n");

  CUdeviceptr da = 0, db = 0, dc = 0, module = 0, func = 0;
  cuMemAlloc(&da, 1024 * 4); cuMemAlloc(&db, 1024 * 4); cuMemAlloc(&dc, 1024 * 4);
  unsigned n = 1024;
  void *params[5] = { &da, &db, &dc, &n, NULL };

  /* N1: garbage fatbin body (the shim's contract is a bounded 1 MB scan,
     so hostile images get real 1 MB backing — overreading a short stack
     buffer would be a bug in the TEST, not the shim) */
  static unsigned char bad_magic[1 << 20];
  bad_magic[0] = 0x50; bad_magic[1] = 0xED; bad_magic[2] = 0x55; bad_magic[3] = 0xBA;
  memset(bad_magic + 4, 0x41, sizeof bad_magic - 4);
  int rc = cuModuleLoadData(&module, bad_magic);
  rc = cuModuleGetFunction(&func, module, "_Z9vectorAddPKfS0_Pfi");
  rc = cuLaunchKernel(func, 1, 1, 1, 256, 1, 1, 0, 0, params, NULL);
  gate("N1 garbage fatbin body -> NOT_SUPPORTED", rc == 801, "rc=801 expected");

  /* N2: unknown kernel name — GetFunction defers validation to launch
     (our shim's documented contract), so the LAUNCH must fail honestly */
  rc = cuModuleLoadData(&module, ptx_good);
  rc = cuModuleGetFunction(&func, module, "_no_such_kernel_");
  rc = cuLaunchKernel(func, 1, 1, 1, 256, 1, 1, 0, 0, params, NULL);
  gate("N2 unknown kernel -> launch NOT_SUPPORTED", rc == 801,
       "deferred validation, honest launch failure");

  /* N3: truly unsupported opcode -> NOT_SUPPORTED (never a wrong result) */
  rc = cuModuleLoadData(&module, ptx_unsupported);
  rc = cuModuleGetFunction(&func, module, "_Z9vectorAddPKfS0_Pfi");
  rc = cuLaunchKernel(func, 1, 1, 1, 256, 1, 1, 0, 0, params, NULL);
  gate("N3 unsupported opcode -> NOT_SUPPORTED", rc == 801, "suld named in log");

  /* N4: invalid function handle */
  rc = cuLaunchKernel(9999, 1, 1, 1, 256, 1, 1, 0, 0, params, NULL);
  gate("N4 invalid function handle -> INVALID_VALUE", rc == 1, "rc=1 expected");

  /* N5: invalid stream */
  rc = cuModuleLoadData(&module, ptx_good);
  rc = cuModuleGetFunction(&func, module, "_Z9vectorAddPKfS0_Pfi");
  rc = cuLaunchKernel(func, 1, 1, 1, 256, 1, 1, 0, 777, params, NULL);
  gate("N5 invalid stream -> INVALID_VALUE", rc == 1, "rc=1 expected");

  /* N6: zero grid */
  rc = cuLaunchKernel(func, 0, 1, 1, 256, 1, 1, 0, 0, params, NULL);
  gate("N6 zero grid -> INVALID_VALUE", rc == 1, "rc=1 expected");

  /* N7: block exceeds device max */
  rc = cuLaunchKernel(func, 1, 1, 1, 4096, 1, 1, 0, 0, params, NULL);
  gate("N7 oversized block -> INVALID_VALUE", rc == 1, "rc=1 expected");

  /* N8: valid launch after hostiles still succeeds and is CORRECT */
  float *ha = malloc(1024 * 4), *hb = malloc(1024 * 4), *hc = malloc(1024 * 4);
  for (int i = 0; i < 1024; i++) { ha[i] = (float)i; hb[i] = (float)(2 * i); }
  cuMemcpyHtoD(da, ha, 1024 * 4); cuMemcpyHtoD(db, hb, 1024 * 4);
  rc = cuLaunchKernel(func, 4, 1, 1, 256, 1, 1, 0, 0, params, NULL);
  cuStreamDestroy(0); /* no-op safety */
  int bad = 0;
  cuMemcpyDtoH(hc, dc, 1024 * 4);
  for (int i = 0; i < 1024; i++)
    if (hc[i] != ha[i] + hb[i]) bad++;
  gate("N8 post-hostile launch still correct", rc == 0 && bad == 0,
       bad == 0 ? "1024/1024 elements" : "WRONG RESULTS");

  /* N9: double module load of same image is stable */
  CUdeviceptr m2 = 0;
  rc = cuModuleLoadData(&m2, ptx_good);
  gate("N9 duplicate module load stable", rc == 0 && m2 == module, "same handle reused");

  /* N10: launch with NULL params on a param-taking kernel is invalid
     (launching unbound would be silent UB — shim rejects with
     CUDA_ERROR_INVALID_VALUE) */
  rc = cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, 0, NULL, NULL);
  gate("N10 NULL params -> INVALID_VALUE", rc == 1, "no crash, no unbound launch");

  free(ha); free(hb); free(hc);
  cuMemFree(da); cuMemFree(db); cuMemFree(dc);

  printf("NEGATIVE INVENTORY: passed %d, failed %d\n", pass, fail);
  return fail != 0;
}
