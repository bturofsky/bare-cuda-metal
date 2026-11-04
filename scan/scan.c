/*
 * scan.c — the portability scanner (cuda-metal's champion tool).
 *
 * Takes ANY CUDA binary, locates its embedded fatbin containers, extracts
 * every entry (PTX text incl. LZ4/ZSTD/deflate-compressed, ELF cubins), names
 * the kernels, counts the PTX opcode census, and reports the translation-
 * effort profile as JSON: the "will it run on Metal and how hard" answer.
 *
 * Header layout per ZLUDA dark_api (MIT; verified byte-for-byte against
 * CUDA 12.4 containers on bb23: ELF entry payload @+72, PTX @+88, sm_version
 * field, flags 0x2000=LZ4 / 0x8000=ZSTD / 0x1000=deflate).
 *
 * Build: make scan/scan   Usage: ./scan/scan <cuda-binary>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <zlib.h>
#include <lz4.h>
#ifdef HAVE_ZSTD
#include <zstd.h>
#endif

#define FATBIN_MAGIC 0xBA55ED50u
#define FATBIN_PTX 1
#define FATBIN_ELF 2
#define FFLAG_IS64 0x1ULL
#define FFLAG_DEFLATE 0x1000ULL
#define FFLAG_LZ4 0x2000ULL
#define FFLAG_ZSTD 0x8000ULL

typedef struct {
  uint32_t magic;
  uint16_t version;
  uint16_t header_size; /* 16 */
  uint64_t size;
} fat_header;

typedef struct {
  uint16_t kind;         /* 1 PTX, 2 ELF */
  uint16_t version;      /* 0x0101 */
  uint32_t header_size;  /* 72 ELF / 88 PTX — payload at +this */
  uint32_t payload_size;
  uint32_t unknown0;
  uint32_t compressed_size;
  uint32_t unknown1;
  uint32_t unknown2;
  uint32_t sm_version;   /* decimal sm number */
  uint32_t bit_width;
  uint32_t unknown3;
  uint64_t flags;
  uint64_t unknown5;
  uint64_t uncompressed_payload;
} fat_entry;

typedef struct { const unsigned char *data; size_t size; } blob;

/* ---- zlib inflate (deflate-flagged entries) ----------------------------- */
static unsigned char *inflate_zlib(const unsigned char *in, size_t inlen, size_t *outlen) {
  uLongf cap = inlen * 12 + 65536;
  unsigned char *out = malloc(cap);
  if (!out) return NULL;
  for (int tries = 0; tries < 6; tries++) {
    uLongf outn = cap;
    int zr = uncompress(out, &outn, in, (uLong)inlen);
    if (zr == Z_OK) { *outlen = outn; return out; }
    if (zr != Z_BUF_ERROR) break;
    cap *= 2;
    unsigned char *bigger = realloc(out, cap);
    if (!bigger) break;
    out = bigger;
  }
  free(out);
  return NULL;
}

/* ---- LZ4 block decoder (self-contained; the fatbin PTX compression) ----- */
static unsigned char *lz4_block_decompress(const unsigned char *in, size_t inlen,
                                           size_t expected, size_t *outlen) {
  /* liblz4's proven safe decoder (the fatbin payload is an LZ4 block whose
     bound we know from the entry's uncompressed_size field). */
  size_t cap = expected ? expected + 8 : inlen * 8 + 1024;
  unsigned char *out = malloc(cap);
  if (!out) return NULL;
  int r = LZ4_decompress_safe((const char *)in, (char *)out, (int)inlen, (int)cap);
  if (r < 0) { free(out); return NULL; }
  *outlen = (size_t)r;
  return out;
}

#ifdef HAVE_ZSTD
static unsigned char *zstd_decompress(const unsigned char *in, size_t inlen, size_t *outlen) {
  unsigned long long bound = ZSTD_getFrameContentSize(in, inlen);
  if (bound == ZSTD_CONTENTSIZE_ERROR || bound == ZSTD_CONTENTSIZE_UNKNOWN) return NULL;
  unsigned char *out = malloc((size_t)bound + 1);
  if (!out) return NULL;
  size_t r = ZSTD_decompress(out, (size_t)bound + 1, in, inlen);
  if (ZSTD_isError(r)) { free(out); return NULL; }
  out[r] = 0;
  *outlen = r;
  return out;
}
#endif

/* ---- opcode census ------------------------------------------------------- */
#define MAXOPS 256
typedef struct { char op[32]; long count; } opcount;
static opcount g_ops[MAXOPS];
static int g_nops = 0;

static void census(const char *ptx, size_t len) {
  char word[64];
  size_t i = 0;
  int at_stmt_start = 1;
  while (i < len) {
    char c = ptx[i];
    if (c == '\n' || c == ';') { at_stmt_start = 1; i++; continue; }
    if (c == ' ' || c == '\t' || c == '\r' || c == ',') { i++; continue; }
    if (!at_stmt_start) {
      while (i < len && ptx[i] != ' ' && ptx[i] != '\t' && ptx[i] != '\n' &&
             ptx[i] != ';' && ptx[i] != ',') i++;
      continue;
    }
    size_t w = 0;
    while (i < len && w < 63 && ptx[i] != ' ' && ptx[i] != '\t' && ptx[i] != '\n' &&
           ptx[i] != ';' && ptx[i] != ',') word[w++] = ptx[i++];
    word[w] = 0;
    at_stmt_start = 0;
    if (!w) continue;
    if (word[0] == '.' || (w > 1 && word[w - 1] == ':')) continue;
    int is_op = strchr(word, '.') != NULL;
    if (!is_op && (!strcmp(word, "bra") || !strcmp(word, "ret") ||
                   !strcmp(word, "call") || !strcmp(word, "nop"))) is_op = 1;
    if (!is_op) continue;
    int found = 0;
    for (int k = 0; k < g_nops; k++)
      if (!strcmp(g_ops[k].op, word)) { g_ops[k].count++; found = 1; break; }
    if (!found && g_nops < MAXOPS) {
      strncpy(g_ops[g_nops].op, word, 31);
      g_ops[g_nops].op[31] = 0;
      g_ops[g_nops].count = 1;
      g_nops++;
    }
  }
}

/* ---- kernel names -------------------------------------------------------- */
#define MAXK 512
static char g_kernels[MAXK][128];
static int g_nkernels = 0;
static void find_kernels(const char *ptx, size_t len) {
  const char *needle = ".visible .entry ";
  const char *p = ptx;
  while ((size_t)(p - ptx) < len &&
         (p = memmem(p, len - (size_t)(p - ptx), needle, strlen(needle)))) {
    const char *start = p + strlen(needle);
    const char *end = start;
    while ((size_t)(end - ptx) < len && *end && *end != '(' && *end != '\n') end++;
    size_t n = (size_t)(end - start);
    if (n > 0 && n < 127 && g_nkernels < MAXK) {
      memcpy(g_kernels[g_nkernels], start, n);
      g_kernels[g_nkernels][n] = 0;
      g_nkernels++;
    }
    p = end;
  }
}

static long g_entries_total = 0, g_ptx_entries = 0, g_elf_entries = 0, g_compressed = 0;

static const char *kind_name(uint16_t k) {
  return k == FATBIN_PTX ? "ptx" : (k == FATBIN_ELF ? "cubin" : "other");
}

static void scan_fatbin(blob b, size_t off) {
  fat_header fh;
  memcpy(&fh, b.data + off, sizeof fh);
  if (fh.magic != FATBIN_MAGIC) return;
  size_t p = off + sizeof(fat_header);
  size_t end = off + sizeof(fat_header) + fh.size;
  if (end > b.size) end = b.size;
  printf("{\"fatbin_offset\":%zu,\"version\":%u,\"entries\":[\n", off, fh.version);
  int first = 1;
  while (p + sizeof(fat_entry) <= end) {
    fat_entry fe;
    memcpy(&fe, b.data + p, sizeof fe);
    if (fe.header_size < 16 || fe.header_size > 256 ||
        fe.payload_size == 0 || fe.payload_size > b.size) break;
    const unsigned char *payload = b.data + p + fe.header_size;
    size_t payload_len = fe.payload_size;
    unsigned char *inflated = NULL;
    char comp[16] = "none";
    int usable = 1;
    if (fe.flags & FFLAG_LZ4) {
      snprintf(comp, sizeof comp, "lz4");
      size_t clen = fe.compressed_size ? fe.compressed_size : payload_len;
      inflated = lz4_block_decompress(payload, clen,
                                      (size_t)fe.uncompressed_payload, &payload_len);
      usable = inflated != NULL;
    } else if (fe.flags & FFLAG_ZSTD) {
      snprintf(comp, sizeof comp, "zstd");
#ifdef HAVE_ZSTD
      inflated = zstd_decompress(payload, payload_len, &payload_len);
      usable = inflated != NULL;
#else
      usable = 0;
#endif
    } else if (fe.flags & FFLAG_DEFLATE) {
      snprintf(comp, sizeof comp, "deflate");
      inflated = inflate_zlib(payload, payload_len, &payload_len);
      usable = inflated != NULL;
    }
    const char *kind = kind_name(fe.kind);
    char arch[32] = "?";
    if (fe.sm_version) snprintf(arch, sizeof arch, "sm_%u", fe.sm_version);

    if (!first) printf(",\n");
    first = 0;
    printf("  {\"kind\":\"%s\",\"payload\":%zu,\"compression\":\"%s\",\"arch\":\"%s\","
           "\"uncompressed_size\":%llu,\"flags\":\"0x%llx\"",
           kind, (size_t)fe.payload_size, comp, arch,
           (unsigned long long)fe.uncompressed_payload, (unsigned long long)fe.flags);
    if (fe.kind == FATBIN_PTX) {
      g_ptx_entries++;
      if (usable) {
        find_kernels((const char *)(inflated ? inflated : payload), payload_len);
        census((const char *)(inflated ? inflated : payload), payload_len);
        printf(",\"ptx_bytes\":%zu,\"census\":true", payload_len);
        if (g_nkernels > 0) {
          printf(",\"kernels\":[");
          for (int k = 0; k < g_nkernels; k++)
            printf("%s\"%s\"", k ? "," : "", g_kernels[k]);
          printf("]");
        }
      } else {
        printf(",\"census\":false,\"reason\":\"decompressor unsupported for this codec\"");
        g_compressed++;
      }
    } else {
      g_elf_entries++;
      if (usable && g_nkernels > 0) {
        printf(",\"kernels\":[");
        for (int k = 0; k < g_nkernels; k++)
          printf("%s\"%s\"", k ? "," : "", g_kernels[k]);
        printf("]");
        g_nkernels = 0;
      } else {
        g_nkernels = 0;
      }
    }
    free(inflated);
    g_entries_total++;
    printf(" }");
    p += fe.header_size + fe.payload_size;
  }
  printf("]}");
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <cuda-binary>\n", argv[0]); return 64; }
  FILE *f = fopen(argv[1], "rb");
  if (!f) { perror(argv[1]); return 1; }
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);
  unsigned char *data = malloc(size);
  if (!data || fread(data, 1, size, f) != (size_t)size) { fprintf(stderr, "read failed\n"); return 1; }
  fclose(f);
  blob b = { data, (size_t)size };

  printf("{\"binary\":\"%s\",\"file_size\":%ld,\n", argv[1], size);
  printf(" \"fatbins\":[");
  int nfound = 0;
  for (size_t i = 0; i + sizeof(fat_header) <= b.size; i++) {
    uint32_t magic;
    memcpy(&magic, b.data + i, 4);
    if (magic == FATBIN_MAGIC) {
      fat_header fh;
      memcpy(&fh, b.data + i, sizeof fh);
      if (fh.version == 1 && fh.header_size == sizeof(fat_header) &&
          fh.size > 0 && i + sizeof(fat_header) + fh.size <= b.size) {
        if (nfound > 0) printf(",");
        scan_fatbin(b, i);
        nfound++;
        i += sizeof(fat_header) + fh.size - 1;
      }
    }
  }
  printf("],\n");
  printf(" \"fatbins_found\": %d,\n", nfound);
  printf(" \"entries_total\": %ld, \"ptx_entries\": %ld, \"elf_entries\": %ld, \"compressed_ptx\": %ld,\n",
         g_entries_total, g_ptx_entries, g_elf_entries, g_compressed);
  printf(" \"distinct_kernels\": %d,\n", g_nkernels);
  printf(" \"opcode_census\": {\n");
  for (int k = 0; k < g_nops; k++)
    printf("    \"%s\": %ld%s\n", g_ops[k].op, g_ops[k].count,
           k + 1 < g_nops ? "," : "");
  printf("  }\n}\n");
  return 0;
}
