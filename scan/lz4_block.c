/* lz4_block.c — a self-contained LZ4 BLOCK decoder (the format the fatbin
 * compressed-PTX payloads use). No library dependency: the algorithm is the
 * published LZ4 block format (literals + back-references). */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

unsigned char *lz4_block_decompress(const unsigned char *in, size_t inlen,
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
    if (ip >= inlen) break; /* the terminal literal sequence */
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
