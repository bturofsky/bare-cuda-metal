/* Standalone: translate a PTX file to MSL. Usage: ptx2msl <ptx_file> <kernel_name> */
#include <stdio.h>
#include <stdlib.h>
extern int translate_ptx_kernel(const char *ptx, size_t ptx_len, const char *kernel_name,
                                char *msl_buf, size_t msl_buf_size, char *error, size_t err_size);
int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s <ptx_file> <kernel_name>\n", argv[0]); return 64; }
  FILE *f = fopen(argv[1], "rb");
  if (!f) { perror(argv[1]); return 1; }
  fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  char *ptx = malloc(sz + 1); fread(ptx, 1, sz, f); ptx[sz] = 0; fclose(f);
  char msl[16384], err[256];
  int rc = translate_ptx_kernel(ptx, sz, argv[2], msl, sizeof(msl), err, sizeof(err));
  if (rc != 0) { fprintf(stderr, "translate error: %s\n", err); return 1; }
  fputs(msl, stdout);
  free(ptx);
  return 0;
}
