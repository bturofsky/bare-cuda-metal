/*
 * v2_fatbin.c — container-path test: builds real fatbin images in memory
 * (uncompressed PTX, LZ4-compressed PTX with genuine matches, and a
 * two-entry container with a leading ELF entry) and drives each through
 * cuModuleLoadData → GetFunction → cuLaunchKernel, verifying results.
 *
 * The LZ4 encoder is a naive-window greedy matcher: a real LZ4 block stream
 * (matches + extended lengths), so the shim's decoder runs its match-copy
 * path, not just literals. Note: no nvcc on arm64 macOS, so these crafted
 * containers are the deterministic acceptance input for the parser; they
 * follow the same layout rules the shim validates.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef unsigned long long CUdeviceptr;

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

/* ---- minimal LZ4 block compressor (valid matches, extended lengths) ----- */
static size_t lz4_compress_simple(const uint8_t *in, size_t n, uint8_t *out) {
  size_t ip = 0, op = 0;
  while (ip < n) {
    size_t lit_start = ip, lit_len = 0, m_off = 0, m_len = 0;
    while (ip + lit_len < n) {
      size_t p = ip + lit_len;
      size_t max_off = p < 65536 ? p : 65536;
      for (size_t off = 1; off <= max_off; off++) {
        size_t l = 0;
        while (p + l < n && in[p + l] == in[p - off + l] && l < 65535u + 15u) l++;
        if (l >= 4 && l > m_len) { m_len = l; m_off = off; if (l >= 128) break; }
      }
      if (m_len >= 4) break;
      lit_len++;
    }
    unsigned ml = m_len ? (unsigned)(m_len - 4) : 0;
    unsigned ll = (unsigned)lit_len;
    out[op++] = (uint8_t)((ll < 15 ? ll << 4 : 0xF0) | (ml < 15 ? ml : 0x0F));
    if (ll >= 15) { unsigned r = ll - 15; while (r >= 255) { out[op++] = 255; r -= 255; } out[op++] = (uint8_t)r; }
    memcpy(out + op, in + lit_start, lit_len);
    op += lit_len; ip += lit_len;
    if (m_len) {
      out[op++] = (uint8_t)(m_off & 0xff);
      out[op++] = (uint8_t)(m_off >> 8);
      if (ml >= 15) { unsigned r = ml - 15; while (r >= 255) { out[op++] = 255; r -= 255; } out[op++] = (uint8_t)r; }
      ip += m_len;
    } else break;
  }
  return op;
}

/* ---- fatbin container construction -------------------------------------- */
#define FATBIN_MAGIC 0xBA55ED50u
#define FFLAG_LZ4 0x2000ULL

typedef struct {
  uint32_t magic;
  uint16_t version, header_size;
  uint64_t size;
} fat_header_t;

typedef struct {
  uint16_t kind, version;
  uint32_t header_size, payload_size, unknown0, compressed_size, unknown1,
      unknown2, sm_version, bit_width, unknown3;
  uint64_t flags, unknown5, uncompressed_payload;
} fat_entry_t;

/* use_lz4: compress the PTX payload; elf_first: prepend a dummy ELF entry */
static uint8_t *build_fatbin(const uint8_t *ptx, size_t ptx_len, int use_lz4,
                             int elf_first, size_t *out_len) {
  uint8_t *payload;
  size_t paylen;
  uint64_t flags = 0;
  uint64_t uncomp = 0;
  if (use_lz4) {
    uint8_t *comp = malloc(ptx_len * 2 + 64);
    paylen = lz4_compress_simple(ptx, ptx_len, comp);
    payload = comp;
    flags = FFLAG_LZ4;
    uncomp = ptx_len;
  } else {
    payload = malloc(ptx_len);
    memcpy(payload, ptx, ptx_len);
    paylen = ptx_len;
  }

  size_t n_entries = 1 + (elf_first ? 1 : 0);
  size_t entries_bytes = n_entries * sizeof(fat_entry_t) + (elf_first ? 32 : 0) + paylen;
  size_t total = sizeof(fat_header_t) + entries_bytes;
  uint8_t *buf = calloc(1, total);
  fat_header_t fh = {.magic = FATBIN_MAGIC, .version = 1,
                     .header_size = (uint16_t)sizeof(fat_header_t),
                     .size = entries_bytes};
  memcpy(buf, &fh, sizeof fh);
  size_t o = sizeof fh;
  if (elf_first) {
    fat_entry_t elf = {.kind = 2 /* ELF */, .version = 1,
                       .header_size = (uint32_t)sizeof(fat_entry_t),
                       .payload_size = 32, .sm_version = 86, .bit_width = 64};
    memcpy(buf + o, &elf, sizeof elf);
    o += sizeof elf;
    memset(buf + o, 0xCC, 32); /* dummy ELF payload */
    o += 32;
  }
  fat_entry_t pe = {.kind = 1 /* PTX */, .version = 1,
                    .header_size = (uint32_t)sizeof(fat_entry_t),
                    .payload_size = (uint32_t)paylen, .compressed_size = (uint32_t)paylen,
                    .sm_version = 86, .bit_width = 64, .flags = flags,
                    .uncompressed_payload = uncomp};
  memcpy(buf + o, &pe, sizeof pe);
  o += sizeof pe;
  memcpy(buf + o, payload, paylen);
  o += paylen;
  free(payload);
  *out_len = o;
  return buf;
}

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

#define N 1024

static int run_case(const char *label, const void *image, float *ha, float *hb, float *hc,
                    CUdeviceptr da, CUdeviceptr db, CUdeviceptr dc) {
  CUdeviceptr module = 0, func = 0;
  int rc = cuModuleLoadData(&module, image);
  if (rc) { printf("%s: LOAD FAILED rc=%d\n", label, rc); return 1; }
  rc = cuModuleGetFunction(&func, module, "_Z9vectorAddPKfS0_Pfi");
  if (rc) { printf("%s: GETFUNCTION FAILED rc=%d\n", label, rc); return 1; }
  unsigned n = N;
  void *params[5] = { &da, &db, &dc, &n, NULL };
  rc = cuLaunchKernel(func, 4, 1, 1, 256, 1, 1, 0, 0, params, NULL);
  if (rc) { printf("%s: LAUNCH FAILED rc=%d\n", label, rc); return 1; }
  cuCtxSynchronize();
  cuMemcpyDtoH(hc, dc, N * sizeof(float));
  int bad = 0;
  for (int i = 0; i < N; i++)
    if (hc[i] != ha[i] + hb[i]) bad++;
  if (bad == 0) printf("%s: OK — %d/%d verified\n", label, N, N);
  else printf("%s: FAILED — %d mismatches\n", label, bad);
  return bad != 0;
}

int main(void) {
  size_t ptx_len = 0;
  char *ptx = read_file("docs/reference/vectorAdd.ptx", &ptx_len);
  if (!ptx) { fprintf(stderr, "FAIL: cannot read docs/reference/vectorAdd.ptx\n"); return 2; }

  cuInit(0);
  float *ha = malloc(N * sizeof(float)), *hb = malloc(N * sizeof(float)),
        *hc = malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) { ha[i] = (float)i; hb[i] = (float)(2 * i); }
  CUdeviceptr da = 0, db = 0, dc = 0;
  cuMemAlloc(&da, N * sizeof(float));
  cuMemAlloc(&db, N * sizeof(float));
  cuMemAlloc(&dc, N * sizeof(float));
  cuMemcpyHtoD(da, ha, N * sizeof(float));
  cuMemcpyHtoD(db, hb, N * sizeof(float));

  int failures = 0;
  size_t len = 0;
  uint8_t *raw = build_fatbin((const uint8_t *)ptx, ptx_len, 0, 0, &len);
  printf("containers: raw=%zu B (ptx %zu B)\n", len, ptx_len);
  failures += run_case("fatbin/raw     ", raw, ha, hb, hc, da, db, dc);

  uint8_t *lz4c = build_fatbin((const uint8_t *)ptx, ptx_len, 1, 0, &len);
  printf("            lz4 =%zu B\n", len);
  failures += run_case("fatbin/lz4    ", lz4c, ha, hb, hc, da, db, dc);

  uint8_t *mixed = build_fatbin((const uint8_t *)ptx, ptx_len, 0, 1, &len);
  printf("            elf1=%zu B\n", len);
  failures += run_case("fatbin/elf+ptx", mixed, ha, hb, hc, da, db, dc);

  if (failures == 0)
    printf("V2 FATBIN OK — container parse + LZ4 decode verified through launch\n");
  else
    printf("V2 FATBIN FAILED — %d case(s)\n", failures);

  cuMemFree(da); cuMemFree(db); cuMemFree(dc);
  free(ha); free(hb); free(hc); free(ptx); free(raw); free(lz4c); free(mixed);
  return failures != 0;
}
