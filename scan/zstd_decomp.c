/* zstd_decomp.c — ZSTD payload decompression for fatbin PTX entries.
 * Implementation approach ported from ZLUDA dark_api (MIT) — the flags say
 * zstd/lz4/deflate; the decompressors are the standard ones. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <zstd.h>

unsigned char *zstd_decompress(const unsigned char *in, size_t inlen, size_t *outlen) {
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
