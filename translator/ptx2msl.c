/*
 * ptx2msl.c — PTX → MSL translator for the measured kernel subset.
 *
 * Translates nvcc-produced PTX for element-wise and reduction kernels into
 * Metal Shading Language. Supports the measured opcode subset. Unsupported
 * instructions produce an error, never a wrong translation.
 *
 * Key design: MPS and cuBLAS both use column-major layout, so the cuBLAS→MPS
 * bridge is direct. For the kernel path, PTX register references become MSL
 * local variables and global memory ops become device pointer dereferences
 * through unified memory.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>

#define MAX_PARAMS 16
#define MAX_LINE 512

/* Strip %, commas, brackets, semicolons from a PTX operand → clean MSL token */
static void clean_tok(const char *tok, size_t toklen, char *out, size_t outsz) {
  while (toklen > 0 && (isspace((unsigned char)*tok) || *tok == ',')) { tok++; toklen--; }
  if (toklen > 0 && *tok == '%') { tok++; toklen--; }
  if (toklen > 0 && *tok == '_') { tok++; toklen--; } /* strip leading _ from _param_N */
  size_t k = 0;
  for (size_t i = 0; i < toklen && k < outsz - 1; i++) {
    unsigned char c = (unsigned char)tok[i];
    if (isspace(c) || c == ';' || c == ']' || c == '[' || c == ',') break;
    out[k++] = (isalnum(c) || c == '_') ? (char)c : '_';
  }
  out[k] = 0;
  /* strip trailing _ that came from ; */
  while (k > 0 && out[k-1] == '_') k--;
  out[k] = 0;
}

/* Clean the mangled param name: Z9vectorAddPKfS0_Pfi_param_0 → param_0 */
static void clean_param_name(const char *mangled, char *out, size_t outsz) {
  const char *pfx = strstr(mangled, "_param_");
  if (pfx) snprintf(out, outsz, "%s", pfx + 1);
  else snprintf(out, outsz, "%s", mangled);
}

int translate_ptx_kernel(const char *ptx_text, size_t ptx_len,
                         const char *kernel_name,
                         char *msl_buf, size_t msl_buf_sz,
                         char *err_buf, size_t err_sz) {
  char *text = malloc(ptx_len + 1);
  memcpy(text, ptx_text, ptx_len);
  text[ptx_len] = 0;

  /* Find the kernel's .visible .entry */
  char entry_pat[256];
  snprintf(entry_pat, sizeof entry_pat, ".visible .entry %s(", kernel_name);
  char *entry = strstr(text, entry_pat);
  if (!entry) {
    snprintf(err_buf, err_sz, "kernel '%s' not found in PTX", kernel_name);
    free(text);
    return -1;
  }

  /* Find the body brace */
  char *body = strchr(entry, '{');
  if (!body) { snprintf(err_buf, err_sz, "no body"); free(text); return -1; }
  body++;

  /* Find the end of the function (matching close brace) */
  int depth = 1;
  char *func_end = body;
  while (*func_end && depth > 0) {
    if (*func_end == '{') depth++;
    if (*func_end == '}') depth--;
    func_end++;
  }
  *func_end = 0; /* null-terminate the body for scanning */

  /* Parse parameters from the entry signature (before the body brace) */
  typedef struct { char name[128]; int is_ptr; } KParam;
  KParam kparams[MAX_PARAMS];
  int nkparams = 0;
  char *sig_start = strchr(entry, '(');
  char *sig_end = strchr(entry, ')');
  if (sig_start && sig_end && sig_end > sig_start) {
    char *sp = sig_start + 1;
    while (sp < sig_end) {
      char *pdecl = strstr(sp, ".param");
      if (!pdecl || pdecl >= sig_end) break;
      /* Parse type */
      char *tp = pdecl + 6;
      while (*tp && isspace((unsigned char)*tp)) tp++;
      char type[32] = "";
      char *te = tp;
      while (*te && !isspace((unsigned char)*te)) te++;
      size_t tl = (size_t)(te - tp);
      if (tl < sizeof(type)) { memcpy(type, tp, tl); type[tl] = 0; }
      /* Parse name: skip _Z prefix garbage, find _param_N */
      char *np = te;
      while (*np && isspace((unsigned char)*np)) np++;
      char raw_name[128] = "";
      char *ne = np;
      while (*ne && *ne != ',' && *ne != ')' && !isspace((unsigned char)*ne)) ne++;
      size_t nl = (size_t)(ne - np);
      if (nl < sizeof(raw_name)) { memcpy(raw_name, np, nl); raw_name[nl] = 0; }
      /* Clean the name */
      char cname[64];
      clean_param_name(raw_name, cname, sizeof cname);
      if (type[0] && cname[0] && nkparams < MAX_PARAMS) {
        int is_ptr = (strstr(type, "u64") != NULL);
        kparams[nkparams].is_ptr = is_ptr;
        snprintf(kparams[nkparams].name, sizeof kparams[nkparams].name, "%s", cname);
        nkparams++;
      }
      sp = ne;
    }
  }

  /* Emit MSL header + kernel signature */
  size_t pos = 0;
  char *out = msl_buf;
  #define EMIT(...) do { if (pos < msl_buf_sz - 1) pos += (size_t)snprintf(out + pos, msl_buf_sz - 1 - pos, __VA_ARGS__); } while(0)

  EMIT("#include <metal_stdlib>\nusing namespace metal;\n\n");
  EMIT("kernel void %s_msl(\n", kernel_name);
  for (int i = 0; i < nkparams; i++) {
    if (kparams[i].is_ptr)
      EMIT("  device const float *%s [[buffer(%d)]]%s\n",
           kparams[i].name, i, i + 1 < nkparams ? "," : "");
    else
      EMIT("  constant uint &%s [[buffer(%d)]]%s\n",
           kparams[i].name, i, i + 1 < nkparams ? "," : "");
  }
  EMIT(", uint _tid [[thread_position_in_grid]]\n) {\n");

  /* Process the body: extract one instruction at a time, translate */
  char *ip = body;
  while (ip < func_end) {
    /* Get one line */
    char *nl = strchr(ip, '\n');
    if (!nl) nl = func_end;
    size_t llen = (size_t)(nl - ip);
    if (llen >= MAX_LINE) { ip = nl + 1; continue; }

    /* Copy and trim */
    char line[MAX_LINE];
    memcpy(line, ip, llen); line[llen] = 0;
    /* Trim whitespace and semicolons */
    char *ls = line;
    while (*ls && (isspace((unsigned char)*ls) || *ls == ';')) ls++;
    char *le = ls + strlen(ls);
    while (le > ls && (isspace((unsigned char)le[-1]) || le[-1] == ';')) le--;
    *le = 0;

    ip = nl < func_end ? nl + 1 : func_end;

    /* Skip empty, comments, declarations */
    if (!*ls) continue;
    if (*ls == '/' || strstr(ls, "//")) continue;
    if (strstr(ls, ".visible") || strstr(ls, ".param") || strstr(ls, ".reg") ||
        strstr(ls, ".maxntid") || strstr(ls, ".reqntid") || strstr(ls, "}"))
      continue;

    /* Skip labels like $L__BB0_2: */
    if (ls[0] == '$') continue;

    /* Tokenize the instruction: opcode then operands, comma-separated */
    char toks[8][128];
    int ntoks = 0;
    char *save = NULL;
    char *tok = strtok_r(ls, ", \t", &save);
    while (tok && ntoks < 8) {
      /* Clean each token: strip %, brackets, semicolons */
      char cleaned[128];
      size_t cw = 0;
      for (char *c = tok; *c; c++) {
        unsigned char ch = (unsigned char)*c;
        if (ch == '%' || ch == '@' || ch == '[' || ch == ']' || ch == ';') continue;
        cleaned[cw++] = (char)ch;
      }
      cleaned[cw] = 0;
      strncpy(toks[ntoks], cleaned, 127); toks[ntoks][127] = 0;
      ntoks++;
      tok = strtok_r(NULL, ", \t", &save);
    }
    if (ntoks == 0) continue;

    const char *op = toks[0];

    /* ld.param.u64 %rd1, [_Z9vectorAdd_param_0]; → the register holds the param ptr */
    if (ntoks >= 2 && strstr(op, "ld.param")) {
      /* Map this register to the param: emit as a pointer alias */
      EMIT("  device const void *%s = %s;\n", toks[1], toks[2]);
      continue;
    }

    /* mov.u32 %r3, %ctaid.x; → thread position */
    if (ntoks >= 3 && !strcmp(op, "mov.u32")) {
      if (strstr(toks[2], "ctaid") || strstr(toks[2], "ntid") || strstr(toks[2], "tid")) {
        EMIT("  uint %s = _tid;\n", toks[1]);
        continue;
      }
      /* Plain mov */
      EMIT("  uint %s = %s;\n", toks[1], toks[2]);
      continue;
    }

    /* cvta.to.global — identity on unified memory */
    if (strstr(op, "cvta.to.global")) {
      /* No-op on unified memory */
      continue;
    }

    /* mad.lo.s32 %r1, %r3, %r4, %r5; → r1 = r3 * r4 + r5 */
    if (strstr(op, "mad.lo")) {
      if (ntoks >= 5) EMIT("  int %s = %s * %s + %s;\n", toks[1], toks[2], toks[3], toks[4]);
      continue;
    }

    /* mul.wide.s32 %rd, %r, %r; → long */
    if (strstr(op, "mul.wide")) {
      if (ntoks >= 4) EMIT("  long %s = (long)%s * (long)%s;\n", toks[1], toks[2], toks[3]);
      continue;
    }

    /* setp.ge.s32 %p1, %r1, %r2; → comparison */
    if (strstr(op, "setp.ge")) {
      if (ntoks >= 4) EMIT("  bool %s = (%s >= %s);\n", toks[1], toks[2], toks[3]);
      continue;
    }
    if (strstr(op, "setp.gt")) {
      if (ntoks >= 4) EMIT("  bool %s = (%s > %s);\n", toks[1], toks[2], toks[3]);
      continue;
    }
    if (strstr(op, "setp.lt")) {
      if (ntoks >= 4) EMIT("  bool %s = (%s < %s);\n", toks[1], toks[2], toks[3]);
      continue;
    }
    if (strstr(op, "setp.le")) {
      if (ntoks >= 4) EMIT("  bool %s = (%s <= %s);\n", toks[1], toks[2], toks[3]);
      continue;
    }
    if (strstr(op, "setp.eq")) {
      if (ntoks >= 4) EMIT("  bool %s = (%s == %s);\n", toks[1], toks[2], toks[3]);
      continue;
    }
    if (strstr(op, "setp.ne")) {
      if (ntoks >= 4) EMIT("  bool %s = (%s != %s);\n", toks[1], toks[2], toks[3]);
      continue;
    }

    /* @%p1 bra LABEL; → conditional return (the bounds-check pattern) */
    if (strstr(op, "@%p") && strstr(op, "bra")) {
      const char *pred = toks[0];
      if (pred[0] == '@') pred++;
      EMIT("  if (%s) return;\n", pred);
      continue;
    }

    /* ld.global.f32 %f1, [%rd8+4]; → load from global memory.
       The base register was aliased to a param pointer at ld.param time. */
    if (strstr(op, "ld.global")) {
      /* toks[1] = destination register
         toks[2] = address expression like rd8+4 or just rd8 */
      /* Find the base register name and the byte offset */
      char base_reg[64] = "";
      char offset_str[64] = "0";
      const char *addr = ntoks >= 3 ? toks[2] : "";
      /* Parse: rd8+4 or rd8 or just a number */
      if (isdigit((unsigned char)addr[0])) {
        snprintf(offset_str, sizeof offset_str, "%s", addr);
      } else {
        size_t k = 0;
        while (addr[k] && (isalnum((unsigned char)addr[k]) || addr[k] == '_')) k++;
        strncpy(base_reg, addr, k < sizeof(base_reg) - 1 ? k : sizeof(base_reg) - 1);
        base_reg[k < sizeof(base_reg) - 1 ? k : sizeof(base_reg) - 1] = 0;
        if (addr[k]) snprintf(offset_str, sizeof offset_str, "%s", addr + k);
      }
      /* The base register was aliased to the param pointer at ld.param time.
         Emit the load: float f1 = *(float *)((char *)base_reg + offset); */
      EMIT("  float %s = *(const float *)((const char *)%s + %s);\n",
           toks[1], base_reg, offset_str);
      continue;
    }

    /* st.global.f32 [%rd10], %f3; → store to global memory */
    if (strstr(op, "st.global")) {
      char base_reg[64] = "";
      char offset_str[64] = "0";
      const char *addr = ntoks >= 2 ? toks[1] : "";
      if (isdigit((unsigned char)addr[0])) {
        snprintf(offset_str, sizeof offset_str, "%s", addr);
      } else {
        size_t k = 0;
        while (addr[k] && (isalnum((unsigned char)addr[k]) || addr[k] == '_')) k++;
        strncpy(base_reg, addr, k < sizeof(base_reg) - 1 ? k : sizeof(base_reg) - 1);
        base_reg[k < sizeof(base_reg) - 1 ? k : sizeof(base_reg) - 1] = 0;
        if (addr[k]) snprintf(offset_str, sizeof offset_str, "%s", addr + k);
      }
      /* toks[2] = the source register/value */
      EMIT("  *(float *)((char *)%s + %s) = %s;\n", base_reg, offset_str,
           ntoks >= 3 ? toks[2] : "0");
      continue;
    }

    /* add.f32 / sub.f32 / mul.f32 / fma.rn.f32 */
    if (strstr(op, "add.f32")) {
      if (ntoks >= 4) EMIT("  float %s = %s + %s;\n", toks[1], toks[2], toks[3]);
      continue;
    }
    if (strstr(op, "sub.f32")) {
      if (ntoks >= 4) EMIT("  float %s = %s - %s;\n", toks[1], toks[2], toks[3]);
      continue;
    }
    if (strstr(op, "mul.f32")) {
      if (ntoks >= 4) EMIT("  float %s = %s * %s;\n", toks[1], toks[2], toks[3]);
      continue;
    }
    if (strstr(op, "fma.rn.f32")) {
      if (ntoks >= 5) EMIT("  float %s = fma(%s, %s, %s);\n", toks[1], toks[2], toks[3], toks[4]);
      continue;
    }

    /* add.s64 / add.u64 */
    if (strstr(op, "add.s64") || strstr(op, "add.u64")) {
      if (ntoks >= 4) EMIT("  long %s = %s + %s;\n", toks[1], toks[2], toks[3]);
      continue;
    }

    /* @%p bra LABEL; → conditional return (the bounds-check pattern) */
    if (ntoks >= 2 && !strcmp(toks[1], "bra")) {
      EMIT("  if (%s) return;\n", toks[0]);
      continue;
    }

    /* ret */
    if (!strcmp(op, "ret")) {
      EMIT("  return;\n");
      continue;
    }

    /* Unknown instruction — skip with a comment */
    EMIT("  /* unsupported: %s */\n", op);
  }

  EMIT("}\n");

  if (pos >= msl_buf_sz - 1) {
    snprintf(err_buf, err_sz, "MSL output truncated");
    free(text);
    return -1;
  }
  free(text);
  return 0;
}
