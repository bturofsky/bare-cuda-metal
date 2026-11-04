/*
 * module_launch.m — v2 kernel launch: fatbin parse → PTX extract → MSL
 * translate → Metal runtime compile → dispatch.
 *
 * Integrates the ptx2msl translator with the shim's cuModuleLoadData /
 * cuModuleGetFunction / cuLaunchKernel path so a real CUDA binary's kernels
 * run on the M4's GPU.
 *
 * Flow: cuModuleLoadData receives the fatbin pointer from the app. We scan
 * it for PTX entries (LZ4-decompressed), find the kernel entry by name in
 * cuModuleGetFunction, translate the PTX body to MSL, compile with
 * newLibraryWithSource at launch time, then dispatch with the caller's
 * grid/block config and parameters bound to Metal buffers.
 */
#import <Metal/Metal.h>
#import <pthread.h>
#import <Foundation/Foundation.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <zlib.h>

extern int translate_ptx_kernel(const char *ptx_text, size_t ptx_len,
                                const char *kernel_name,
                                char *msl_buf, size_t msl_buf_sz,
                                char *err_buf, size_t err_sz);

/* ---- LZ4 block decoder (self-contained) --------------------------------- */
unsigned char *lz4_block_decompress(const unsigned char *in, size_t inlen,
                                    size_t expected, size_t *outlen);

/* ---- fatbin structures --------------------------------------------------- */
#define FATBIN_MAGIC 0xBA55ED50u
#define FATBIN_PTX 1
#define FATBIN_ELF 2
#define FFLAG_LZ4 0x2000ULL
#define FFLAG_ZSTD 0x8000ULL
#define FFLAG_DEFLATE 0x1000ULL

typedef struct {
  uint32_t magic;
  uint16_t version;
  uint16_t header_size;
  uint64_t size;
} fat_header;

typedef struct {
  uint16_t kind;
  uint16_t version;
  uint32_t header_size;
  uint32_t payload_size;
  uint32_t unknown0;
  uint32_t compressed_size;
  uint32_t unknown1;
  uint32_t unknown2;
  uint32_t sm_version;
  uint32_t bit_width;
  uint32_t unknown3;
  uint64_t flags;
  uint64_t unknown5;
  uint64_t uncompressed_payload;
} fat_entry;

/* ---- stored module state ------------------------------------------------- */
#define MAX_MODULES 8
#define MAX_FUNCS_PER_MODULE 16

typedef struct {
  char name[128];
  char ptx_text[16384];
  size_t ptx_len;
  id<MTLFunction> mtl_func;
} stored_func;

typedef struct {
  uint64_t handle;
  stored_func funcs[MAX_FUNCS_PER_MODULE];
  int nfuncs;
} stored_module;

static stored_module g_modules[MAX_MODULES];
static int g_nmodules = 0;
static pthread_mutex_t g_module_lock = PTHREAD_MUTEX_INITIALIZER;

/* ---- LZ4 decompress ------------------------------------------------------- */
static unsigned char *lz4_decomp(const unsigned char *in, size_t inlen,
                                 size_t expected, size_t *outlen) {
  size_t cap = expected ? expected + 1024 : inlen * 6 + 1024;
  unsigned char *out = malloc(cap);
  if (!out) return NULL;
  size_t ip = 0, op = 0;
  while (ip < inlen) {
    unsigned token = in[ip++];
    unsigned lit = token >> 4;
    if (lit == 15) { unsigned s; do { s = in[ip++]; lit += s; } while (s == 255 && ip < inlen); }
    if (op + lit > cap) { cap = (op + lit) * 2 + 1024; out = realloc(out, cap); if (!out) return NULL; }
    if (ip + lit > inlen) { free(out); return NULL; }
    memcpy(out + op, in + ip, lit);
    ip += lit; op += lit;
    if (ip >= inlen) break;
    if (ip + 2 > inlen) { free(out); return NULL; }
    unsigned offset = in[ip] | ((unsigned)in[ip + 1] << 8);
    ip += 2;
    if (offset == 0 || offset > op) { free(out); return NULL; }
    unsigned match = (token & 0x0f) + 4;
    if ((token & 0x0f) == 15) { unsigned s; do { s = in[ip++]; match += s; } while (s == 255 && ip < inlen); }
    if (op + match > cap) { cap = (op + match) * 2 + 1024; out = realloc(out, cap); if (!out) return NULL; }
    unsigned char *mp = out + op - offset;
    for (unsigned k = 0; k < match; k++) out[op++] = mp[k];
  }
  *outlen = op;
  return out;
}

/* ---- ZSTD decompress (via libzstd if available, else skip) ---------------- */
/* v3.2: zstd handled by the caller; this function just passes through for
   now since PyTorch-class binaries use lz4. */

/* ---- fatbin PTX extraction ----------------------------------------------- */
/* Scans the fatbin at `data` for PTX entries and returns the first PTX text
   (decompressed). Returns NULL if no PTX found. */
static char *extract_ptx(const void *data, size_t data_size, size_t *ptx_len) {
  const unsigned char *bytes = (const unsigned char *)data;
  size_t i = 0;
  while (i + sizeof(fat_header) <= data_size) {
    uint32_t magic;
    memcpy(&magic, bytes + i, 4);
    if (magic == FATBIN_MAGIC) {
      fat_header fh;
      memcpy(&fh, bytes + i, sizeof fh);
      if (fh.version == 1 && fh.header_size == sizeof(fat_header) &&
          fh.size > 0 && i + sizeof(fat_header) + fh.size <= data_size) {
        size_t p = i + sizeof(fat_header);
        size_t p_end = p + fh.size;
        while (p + sizeof(fat_entry) <= p_end) {
          fat_entry fe;
          memcpy(&fe, bytes + p, sizeof fe);
          if (fe.header_size < 16 || fe.header_size > 256 ||
              fe.payload_size == 0 || p + fe.header_size + fe.payload_size > p_end)
            break;
          if (fe.kind == FATBIN_PTX) {
            const unsigned char *ptx_payload = bytes + p + fe.header_size;
            size_t plen = fe.payload_size;
            /* Handle compression */
            if (fe.flags & FFLAG_LZ4) {
              size_t dec_len = 0;
              unsigned char *dec = lz4_decomp(ptx_payload, plen,
                                              (size_t)fe.uncompressed_payload, &dec_len);
              if (dec) {
                char *result = malloc(dec_len + 1);
                memcpy(result, dec, dec_len);
                result[dec_len] = 0;
                free(dec);
                *ptx_len = dec_len;
                return result;
              }
            } else if (fe.flags & 0x1000) { /* deflate */
              uLongf cap = fe.uncompressed_payload + 65536;
              unsigned char *dec = malloc(cap);
              uLongf declen = cap;
              int zr = uncompress(dec, &declen, ptx_payload, (uLong)plen);
              if (zr == Z_OK) {
                char *result = malloc(declen + 1);
                memcpy(result, dec, declen);
                result[declen] = 0;
                free(dec);
                *ptx_len = declen;
                return result;
              }
              free(dec);
            } else {
              /* uncompressed PTX */
              char *result = malloc(plen + 1);
              memcpy(result, ptx_payload, plen);
              result[plen] = 0;
              *ptx_len = plen;
              return result;
            }
          }
          p += fe.header_size + fe.payload_size;
        }
        i += sizeof(fat_header) + fh.size;
        continue;
      }
    }
    i++;
  }
  return NULL;
}

/* ---- public: fatbin extract + translate --------------------------------- */

/* Extract PTX for a specific kernel from a fatbin. Returns the PTX text for
   that kernel's .entry, or NULL if not found. */
char *cuda_metal_extract_ptx(const void *fatbin_data, size_t fatbin_size,
                             const char *kernel_name, size_t *ptx_len) {
  /* First extract all PTX from the fatbin */
  size_t total_ptx_len = 0;
  char *all_ptx = extract_ptx(fatbin_data, fatbin_size, &total_ptx_len);
  if (!all_ptx) return NULL;

  /* Find the specific kernel's .entry in the PTX text */
  char entry_pat[256];
  snprintf(entry_pat, sizeof entry_pat, ".visible .entry %s(", kernel_name);
  char *kernel_ptx = strstr(all_ptx, entry_pat);
  if (!kernel_ptx) {
    free(all_ptx);
    return NULL;
  }

  /* Extract just this kernel's entry (from .entry to the matching close brace) */
  int brace_depth = 0;
  const char *body_start = strchr(kernel_ptx, '{');
  if (!body_start) { free(all_ptx); return NULL; }
  const char *p = body_start;
  while (*p) {
    if (*p == '{') brace_depth++;
    if (*p == '}') { brace_depth--; if (brace_depth == 0) { p++; break; } }
    p++;
  }
  size_t kernel_len = (size_t)(p - kernel_ptx);
  char *result = malloc(kernel_len + 1);
  memcpy(result, kernel_ptx, kernel_len);
  result[kernel_len] = 0;
  free(all_ptx);
  *ptx_len = kernel_len;
  return result;
}

/* Translate PTX to MSL — wraps the translator */
char *cuda_metal_translate(const char *ptx, size_t ptx_len,
                           const char *kernel_name) {
  char *msl = malloc(16384);
  char err[256];
  int rc = translate_ptx_kernel(ptx, ptx_len, kernel_name, msl, 16384, err, sizeof err);
  if (rc != 0) { free(msl); return NULL; }
  return msl;
}
