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

/* Clean the mangled param name: Z9vectorAddPKfS0_Pfi_param_0 → param_0 */
static void clean_param_name(const char *mangled, char *out, size_t outsz) {
  const char *pfx = strstr(mangled, "_param_");
  if (pfx) snprintf(out, outsz, "%s", pfx + 1);
  else snprintf(out, outsz, "%s", mangled);
}

/* Clean a PTX label ($L__BB0_2:) into a legal MSL identifier (L__BB0_2) */
static void label_clean(const char *tok, char *out, size_t outsz) {
  size_t k = 0;
  for (; *tok && k + 1 < outsz; tok++) {
    if (*tok == '$' || *tok == '%' || *tok == '@') continue;
    if (*tok == ':' || *tok == ';' || *tok == ',') break;
    out[k++] = (isalnum((unsigned char)*tok) || *tok == '_') ? *tok : '_';
  }
  out[k] = 0;
}

/* Opcode family match on the FIRST token: prefix + suffix allowed
   (ld.global matches ld.global.f32). Never use strstr on opcodes —
   "atom.global.add.u32" contains "add.u32". */
static int op_is(const char *op, const char *name) {
  return strncmp(op, name, strlen(name)) == 0;
}

/* PTX operand → MSL operand: float immediates 0f<8hex> are bit patterns
   (0f00000000 = 0.0f); MSL materializes them with as_type<float>. */
static void ptx_val(const char *tok, char *out, size_t outsz) {
  if (tok[0] == '0' && tok[1] == 'f' && strlen(tok) == 10)
    snprintf(out, outsz, "as_type<float>(0x%su)", tok + 2);
  else
    snprintf(out, outsz, "%s", tok);
}

/* Registers known to hold device pointers (ld.param.u64 / cvta chains) */
static int is_ptr_reg(char regs[][32], int n, const char *r) {
  for (int i = 0; i < n; i++) if (!strcmp(regs[i], r)) return 1;
  return 0;
}
static void mark_ptr_reg(char regs[][32], int *n, const char *r) {
  if (!is_ptr_reg(regs, *n, r) && *n < 64) snprintf(regs[(*n)++], 32, "%s", r);
}

/* ---- param type decoding for launch-time binding ------------------------ */

/* The PTX .param list cannot distinguish a .u64 device pointer from a .u64
   scalar — but the kernel's Itanium mangled name can. Decode it; fall back
   to the PTX types when the name is not a decodable mangle. */

static char classify_type(const char *t) {
  if (strchr(t, 'P')) return 'p';
  char c = t[strlen(t) - 1];
  switch (c) {
    case 'f': return 'f';
    case 'd': case 'e': return 'd'; /* double, long double */
    case 'l': case 'm': return 'l'; /* long, unsigned long/size_t */
    case 'i': case 'j': case 's': case 't': case 'c': case 'a':
    case 'h': case 'b': case 'w': return 'u';
    default: return '?';
  }
}

/* Parse one mangled type; appends completed composites to the substitution
   table per Itanium rules (substitutions and builtins do not join it). */
static int mangle_type(const char **pp, char table[][24], int *nt,
                       char *out, size_t outsz) {
  const char *p = *pp;
  if (!*p) return -1;
  if (*p == 'K' || *p == 'V' || *p == 'r') {
    (*pp)++;
    char inner[24];
    if (mangle_type(pp, table, nt, inner, sizeof inner) < 0) return -1;
    snprintf(out, outsz, "%c%s", *p == 'r' ? 'K' : *p, inner);
    if (*nt < 32) snprintf(table[(*nt)++], 24, "%s", out);
    return 0;
  }
  if (*p == 'P') {
    (*pp)++;
    char inner[24];
    if (mangle_type(pp, table, nt, inner, sizeof inner) < 0) return -1;
    snprintf(out, outsz, "P%s", inner);
    if (*nt < 32) snprintf(table[(*nt)++], 24, "%s", out);
    return 0;
  }
  if (*p == 'S') {
    const char *q = p + 1;
    long idx;
    if (*q == '_') { idx = 0; q++; }
    else if (isalnum((unsigned char)*q)) {
      idx = strtoul(q, (char **)&q, 36) + 1;
      if (*q != '_') return -1;
      q++;
    } else return -1;
    if (idx < 0 || idx >= *nt) return -1;
    snprintf(out, outsz, "%s", table[idx]);
    *pp = q;
    return 0;
  }
  if (strchr("fvailjbmbscth", *p)) {
    snprintf(out, outsz, "%c", *p);
    (*pp)++;
    return 0;
  }
  return -1; /* templates, arrays, names: unsupported */
}

/* Fill types[] with per-param class chars ('p','u','i','f','d','l').
   Returns the param count, or -1 if the kernel cannot be classified. */
int ptx_param_types(const char *mangled, const char *ptx_text, size_t ptx_len,
                    char *types, int max_types) {
  /* Count the PTX .param decls for cross-checking */
  int ptx_count = 0;
  if (ptx_text && ptx_len) {
    char *text = malloc(ptx_len + 1);
    if (text) {
      memcpy(text, ptx_text, ptx_len);
      text[ptx_len] = 0;
      char *p = text;
      while ((p = strstr(p, ".param"))) { ptx_count++; p += 6; }
      free(text);
    }
  }

  /* 1. mangled-name decode */
  int n = 0;
  if (mangled && strncmp(mangled, "_Z", 2) == 0) {
    const char *p = mangled + 2;
    if (isdigit((unsigned char)*p)) { /* skip <length><name> */
      char *e;
      long len = strtol(p, &e, 10);
      p = e + (len > 0 ? len : 0);
      char table[32][24];
      int nt = 0;
      while (*p && n < max_types) {
        char tstr[24];
        if (mangle_type(&p, table, &nt, tstr, sizeof tstr) < 0) { n = -1; break; }
        char cls = classify_type(tstr);
        if (cls == '?') { n = -1; break; }
        types[n++] = cls;
      }
      if (n >= 0 && *p) n = -1; /* trailing garbage: not a pure param mangle */
    } else n = -1;
  } else n = -1;

  /* 2. accept only when the count matches the PTX signature */
  if (n > 0 && (ptx_count == 0 || n == ptx_count)) return n;

  /* 3. fallback: PTX .param types (u64 ambiguous → treated as pointer;
     the shim's registry miss still yields an 8-byte scalar bind) */
  n = 0;
  if (ptx_text && ptx_len) {
    char *text = malloc(ptx_len + 1);
    if (!text) return -1;
    memcpy(text, ptx_text, ptx_len);
    text[ptx_len] = 0;
    char *p = text;
    while ((p = strstr(p, ".param")) && n < max_types) {
      p += 6;
      while (*p && isspace((unsigned char)*p)) p++;
      char t[16] = "";
      int k = 0;
      while (*p && !isspace((unsigned char)*p) && k < 15) t[k++] = *p++;
      t[k] = 0;
      char cls = '?';
      if (strstr(t, "u64") || strstr(t, "b64")) cls = 'p';
      else if (strstr(t, "s64")) cls = 'l';
      else if (strstr(t, "f32")) cls = 'f';
      else if (strstr(t, "f64")) cls = 'd';
      else if (strstr(t, "s32")) cls = 'i';
      else if (strstr(t, "u32") || strstr(t, "b32") || strstr(t, "16") ||
               strstr(t, "8")) cls = 'u';
      if (cls == '?') { free(text); return -1; }
      types[n++] = cls;
    }
    free(text);
  }
  return n > 0 ? n : -1;
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
  typedef struct { char name[128]; int is_ptr; char ptype; } KParam;
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
        /* Classify: p=device pointer, u=uint, i=int, f=float, d=double.
           Note: a .u64 scalar (size_t n) is indistinguishable from a pointer
           in PTX; the shim's registry miss falls back to an 8-byte scalar. */
        char pt;
        int is_ptr = 0;
        if (strstr(type, "u64") || strstr(type, "b64") || strstr(type, "s64")) {
          is_ptr = 1;
          pt = 'p';
        } else if (!strcmp(type, ".f32")) pt = 'f';
        else if (!strcmp(type, ".f64")) pt = 'd';
        else if (!strcmp(type, ".s32")) pt = 'i';
        else if (!strcmp(type, ".u32") || !strcmp(type, ".b32") ||
                 strstr(type, "16") || strstr(type, "8")) pt = 'u';
        else {
          snprintf(err_buf, err_sz, "unsupported param type '%s'", type);
          free(text);
          return -1;
        }
        kparams[nkparams].is_ptr = is_ptr;
        kparams[nkparams].ptype = pt;
        snprintf(kparams[nkparams].name, sizeof kparams[nkparams].name, "%s", cname);
        nkparams++;
      }
      sp = ne;
    }
  }

  /* Emit MSL header + kernel signature, then the body into a separate
     buffer so declarations can be hoisted before assembly (goto must not
     cross initializations in C++). */
  char body_buf[12288];
  size_t bpos = 0;
  #define EMIT(...) do { if (bpos < sizeof body_buf - 1) bpos += (size_t)snprintf(body_buf + bpos, sizeof body_buf - 1 - bpos, __VA_ARGS__); } while(0)

  EMIT("#include <metal_stdlib>\n#include <metal_atomic>\nusing namespace metal;\n\n");
  /* PTX ISA shfl arithmetic, verbatim (docs.nvidia.com/cuda/parallel-thread-execution):
     lane[4:0]=laneid; bval=b[4:0]; cval=c[4:0]; mask=c[12:8];
     maxLane=(lane&mask)|(cval&~mask); minLane=(lane&mask);
     up:   j=lane-bval; p=(j>=maxLane)
     down: j=lane+bval; p=(j<=maxLane)
     bfly: j=lane^bval; p=(j<=maxLane)
     idx:  j=minLane|(bval&~mask); p=(j<=maxLane)
     if(!p) j=lane;  d = SourceA[j];  */
  EMIT("static float _shfl_up_p(float a, uint bval, uint cval, uint lane, thread bool *p) {\n"
       "  bval &= 31u; uint mask = (cval >> 8u) & 31u; uint clamp = cval & 31u;\n"
       "  uint maxLane = (lane & mask) | (clamp & ~mask);\n"
       "  int j = (int)lane - (int)bval; bool ok = (j >= (int)maxLane); if (!ok) j = (int)lane;\n"
       "  *p = ok; return simd_shuffle(a, (uint)j);\n"
       "}\n");
  EMIT("static float _shfl_up(float a, uint bval, uint cval, uint lane) {\n"
       "  bool p; return _shfl_up_p(a, bval, cval, lane, &p);\n"
       "}\n");
  EMIT("static float _shfl_down_p(float a, uint bval, uint cval, uint lane, thread bool *p) {\n"
       "  bval &= 31u; uint mask = (cval >> 8u) & 31u; uint clamp = cval & 31u;\n"
       "  uint maxLane = (lane & mask) | (clamp & ~mask);\n"
       "  int j = (int)lane + (int)bval; bool ok = (j <= (int)maxLane); if (!ok) j = (int)lane;\n"
       "  *p = ok; return simd_shuffle(a, (uint)j);\n"
       "}\n");
  EMIT("static float _shfl_down(float a, uint bval, uint cval, uint lane) {\n"
       "  bool p; return _shfl_down_p(a, bval, cval, lane, &p);\n"
       "}\n");
  EMIT("static float _shfl_bfly_p(float a, uint bval, uint cval, uint lane, thread bool *p) {\n"
       "  bval &= 31u; uint mask = (cval >> 8u) & 31u; uint clamp = cval & 31u;\n"
       "  uint maxLane = (lane & mask) | (clamp & ~mask);\n"
       "  int j = (int)(lane ^ bval); bool ok = (j <= (int)maxLane); if (!ok) j = (int)lane;\n"
       "  *p = ok; return simd_shuffle(a, (uint)j);\n"
       "}\n");
  EMIT("static float _shfl_bfly(float a, uint bval, uint cval, uint lane) {\n"
       "  bool p; return _shfl_bfly_p(a, bval, cval, lane, &p);\n"
       "}\n");
  EMIT("static float _shfl_idx_p(float a, uint bval, uint cval, uint lane, thread bool *p) {\n"
       "  bval &= 31u; uint mask = (cval >> 8u) & 31u; uint clamp = cval & 31u;\n"
       "  uint minLane = lane & mask;\n"
       "  uint maxLane = (lane & mask) | (clamp & ~mask);\n"
       "  int j = (int)(minLane | (bval & ~mask)); bool ok = (j <= (int)maxLane); if (!ok) j = (int)lane;\n"
       "  *p = ok; return simd_shuffle(a, (uint)j);\n"
       "}\n");
  EMIT("static float _shfl_idx(float a, uint bval, uint cval, uint lane) {\n"
       "  bool p; return _shfl_idx_p(a, bval, cval, lane, &p);\n"
       "}\n");
  EMIT("kernel void %s_msl(\n", kernel_name);
  for (int i = 0; i < nkparams; i++) {
    if (kparams[i].is_ptr) {
      /* Writable (no const): output params are stored to through these bindings */
      EMIT("  device float *%s [[buffer(%d)]]%s\n",
           kparams[i].name, i, i + 1 < nkparams ? "," : "");
    } else {
      const char *ts = kparams[i].ptype == 'f' ? "float"
                       : kparams[i].ptype == 'd' ? "double"
                       : kparams[i].ptype == 'i' ? "int" : "uint";
      EMIT("  constant %s &%s [[buffer(%d)]]%s\n",
           ts, kparams[i].name, i, i + 1 < nkparams ? "," : "");
    }
  }
  /* The true CUDA blockDim arrives as data (buffer 30), not the Metal
     builtin: the dispatcher pads the threadgroup to a multiple of 32
     (SIMD-group alignment), and padded lanes must see the CUDA-visible
     ntid, exactly like the idle tail of the last warp on real NVIDIA
     hardware. */
  EMIT(",\n"
       "  constant uint& _lsz_c [[buffer(30)]],\n"
       "  uint _gid [[threadgroup_position_in_grid]],\n"
       "  uint _lid [[thread_position_in_threadgroup]],\n"
       "  uint _ngid [[threadgroups_per_grid]]\n"
       ") {\n"
       "  uint _lsz = _lsz_c;\n");
  size_t header_len = bpos;

  char ptr_regs[64][32];
  int nptr_regs = 0;
  char sh_ptr_regs[64][32];
  int nsh_ptr = 0;
  char pred_decls[16][16];
  int npred_decls = 0;

  /* MSL has no goto/labels. Control flow is translated to a program-counter
     switch loop: each chunk of straight-line code between labels becomes a
     case; branches assign _pc; ret sets -1. Exact for arbitrary PTX flow. */

  /* Pass A: collect labels → chunk ids and count chunks. A chunk boundary
     is created by a label AND by a branch instruction (the code after a
     predicated branch is its own chunk). Chunk 0 = code before the first
     boundary. Also collects __shared__ declarations. */
  char label_names[32][64];
  int label_chunks[32];
  int nlabels = 0;
  int nchunks = 1;
  char sh_names[8][64];
  int sh_offsets[8];
  int sh_sizes[8];
  int nshared = 0;
  int sh_total = 0;
  char sdecls[2048];
  size_t spos = 0;
  {
    char *p = body;
    while (p < func_end) {
      char *nl2 = strchr(p, '\n');
      if (!nl2) nl2 = func_end;
      size_t ll2 = (size_t)(nl2 - p);
      if (ll2 < MAX_LINE) {
        char l2[MAX_LINE];
        memcpy(l2, p, ll2);
        l2[ll2] = 0;
        char *ls2 = l2;
        while (*ls2 && isspace((unsigned char)*ls2)) ls2++;
        int is_branch =
            (strncmp(ls2, "bra", 3) == 0 && !isalnum((unsigned char)ls2[3])) ||
            strstr(ls2, " bra ") != NULL;
        if (strncmp(ls2, ".shared", 7) == 0 && nshared < 8) {
          /* .shared .align 4 .b8 name[size]; → per-var base into a pooled
             threadgroup allocation (16-byte aligned offsets) */
          const char *br = strchr(ls2, '[');
          const char *brk = br ? strchr(br, ']') : NULL;
          if (br && brk) {
            const char *e = br;
            while (e > ls2 && isspace((unsigned char)e[-1])) e--;
            const char *b2 = e;
            while (b2 > ls2 &&
                   (isalnum((unsigned char)b2[-1]) || b2[-1] == '_' || b2[-1] == '$'))
              b2--;
            size_t nm_len = (size_t)(e - b2);
            if (nm_len > 0 && nm_len < 64) {
              char nm[64];
              memcpy(nm, b2, nm_len);
              nm[nm_len] = 0;
              char cnm[64];
              label_clean(nm, cnm, sizeof cnm);
              if (cnm[0]) {
                int elems = (atoi(br + 1) + 3) / 4;
                if (elems < 1) elems = 1;
                int off = (sh_total + 15) & ~15;
                snprintf(sh_names[nshared], sizeof sh_names[0], "%s", cnm);
                sh_offsets[nshared] = off;
                sh_sizes[nshared] = elems * 4;
                sh_total = off + elems * 4;
                nshared++;
              }
            }
          }
        } else if (*ls2 == '$') {
          if (nlabels < 32) {
            label_clean(ls2, label_names[nlabels], sizeof label_names[nlabels]);
            nchunks++;
            label_chunks[nlabels++] = nchunks - 1;
          }
        } else if (is_branch) {
          nchunks++;
        }
      }
      p = nl2 < func_end ? nl2 + 1 : func_end;
    }
  }
  int cur_chunk = 0;
  int chunk_closed = 0;

  /* pooled threadgroup allocation: one array, per-var base pointers */
  if (nshared > 0) {
    spos += (size_t)snprintf(sdecls + spos, sizeof sdecls - 1 - spos,
                             "  threadgroup float _shpool[%d];\n", (sh_total + 3) / 4);
    for (int i = 0; i < nshared; i++)
      spos += (size_t)snprintf(sdecls + spos, sizeof sdecls - 1 - spos,
                               "  threadgroup float *%s = (threadgroup float *)((threadgroup char *)_shpool + %d);\n",
                               sh_names[i], sh_offsets[i]);
  }

  EMIT("  int _pc = 0;\n");
  EMIT("  while (_pc >= 0) {\n");
  EMIT("    switch (_pc) {\n");
  EMIT("    case 0:\n");

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

    /* Skip empty, comments, declarations, closing brace.
       Key on line start: strstr(".param") would also swallow ld.param ops. */
    if (!*ls) continue;
    if (*ls == '/' || strstr(ls, "//")) continue;
    if (ls[0] == '.' || ls[0] == '}')
      continue;

    /* Labels $L__BB0_2: close the current chunk and open the next case */
    if (ls[0] == '$') {
      char ln[64];
      label_clean(ls, ln, sizeof ln);
      if (!ln[0]) continue;
      if (!chunk_closed) {
        EMIT("      _pc = %d;\n      break;\n",
             cur_chunk + 1 <= nchunks - 1 ? cur_chunk + 1 : -1);
        chunk_closed = 1;
      }
      cur_chunk++;
      EMIT("    case %d:\n", cur_chunk);
      chunk_closed = 0;
      continue;
    }

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

    /* ld.param.u64 %rd1, [_param_0]; → register aliases the param pointer.
       Typed scalars copy with their PTX type. */
    if (ntoks >= 2 && op_is(op, "ld.param")) {
      char pname[64];
      clean_param_name(toks[2], pname, sizeof pname);
      if (strstr(op, ".u64")) {
        EMIT("  device const void *%s = %s;\n", toks[1], pname);
        mark_ptr_reg(ptr_regs, &nptr_regs, toks[1]);
      } else if (strstr(op, ".f32")) {
        EMIT("  float %s = %s;\n", toks[1], pname);
      } else if (strstr(op, ".f64")) {
        EMIT("  double %s = %s;\n", toks[1], pname);
      } else if (strstr(op, ".s32")) {
        EMIT("  int %s = %s;\n", toks[1], pname);
      } else {
        EMIT("  uint %s = %s;\n", toks[1], pname);
      }
      continue;
    }

    /* mov.f32 %f, <imm|reg>; → float moves incl. 0f<8hex> bit patterns */
    if (ntoks >= 3 && !strcmp(op, "mov.f32")) {
      char v[96];
      ptx_val(toks[2], v, sizeof v);
      EMIT("  float %s = %s;\n", toks[1], v);
      continue;
    }

    /* mov.u32 %r3, %ctaid.x; → CUDA thread coordinates as MSL builtins.
       mov.u32 %r8, _sdata; → the shared base becomes a threadgroup pointer. */
    if (ntoks >= 3 && !strcmp(op, "mov.u32")) {
      char cand[64];
      label_clean(toks[2], cand, sizeof cand);
      int is_shared_base = 0;
      for (int i = 0; i < nshared; i++)
        if (strcmp(sh_names[i], cand) == 0) { is_shared_base = 1; break; }
      if (is_shared_base) {
        EMIT("  threadgroup float *%s = %s;\n", toks[1], cand);
        mark_ptr_reg(sh_ptr_regs, &nsh_ptr, toks[1]);
        continue;
      }
      /* check nctaid first: "nctaid" contains "ctaid" */
      if (strstr(toks[2], "nctaid"))
        EMIT("  uint %s = _ngid;\n", toks[1]);
      else if (strstr(toks[2], "ctaid"))
        EMIT("  uint %s = _gid;\n", toks[1]);
      else if (strstr(toks[2], "ntid"))
        EMIT("  uint %s = _lsz;\n", toks[1]);
      else if (strstr(toks[2], "tid"))
        EMIT("  uint %s = _lid;\n", toks[1]);
      else
        EMIT("  uint %s = %s;\n", toks[1], toks[2]);
      continue;
    }

    /* cvta.to.global — identity on unified memory, but the destination
       register must still be defined (it feeds add.s64 chains) */
    if (op_is(op, "cvta.to.global")) {
      if (ntoks >= 3) {
        EMIT("  device const void *%s = %s;\n", toks[1], toks[2]);
        mark_ptr_reg(ptr_regs, &nptr_regs, toks[1]);
      }
      continue;
    }

    /* cvta.to.shared — shared-space alias */
    if (op_is(op, "cvta.to.shared")) {
      if (ntoks >= 3) {
        if (!is_ptr_reg(sh_ptr_regs, nsh_ptr, toks[2])) {
          snprintf(err_buf, err_sz, "cvta.to.shared on non-shared register '%s'", toks[2]);
          free(text);
          return -1;
        }
        EMIT("  threadgroup float *%s = %s;\n", toks[1], toks[2]);
        mark_ptr_reg(sh_ptr_regs, &nsh_ptr, toks[1]);
      }
      continue;
    }

    /* mad.lo.s32 %r1, %r3, %r4, %r5; → r1 = r3 * r4 + r5 */
    if (op_is(op, "mad.lo")) {
      if (ntoks >= 5) EMIT("  int %s = %s * %s + %s;\n", toks[1], toks[2], toks[3], toks[4]);
      continue;
    }

    /* mul.wide.s32 %rd, %r, %r; → long */
    if (op_is(op, "mul.wide")) {
      if (ntoks >= 4) EMIT("  long %s = (long)%s * (long)%s;\n", toks[1], toks[2], toks[3]);
      continue;
    }

    /* mul.lo.s32 %r, %r, %r; → int multiply */
    if (op_is(op, "mul.lo")) {
      if (ntoks >= 4) EMIT("  int %s = %s * %s;\n", toks[1], toks[2], toks[3]);
      continue;
    }

    /* add.s32 / add.u32 — int adds and shared-pointer arithmetic */
    if (op_is(op, "add.s32") || op_is(op, "add.u32")) {
      if (ntoks >= 4) {
        if (is_ptr_reg(sh_ptr_regs, nsh_ptr, toks[2])) {
          EMIT("  threadgroup float *%s = (threadgroup float *)((threadgroup char *)%s + (long)%s);\n",
               toks[1], toks[2], toks[3]);
          mark_ptr_reg(sh_ptr_regs, &nsh_ptr, toks[1]);
        } else if (is_ptr_reg(sh_ptr_regs, nsh_ptr, toks[3])) {
          EMIT("  threadgroup float *%s = (threadgroup float *)((threadgroup char *)%s + (long)%s);\n",
               toks[1], toks[3], toks[2]);
          mark_ptr_reg(sh_ptr_regs, &nsh_ptr, toks[1]);
        } else {
          EMIT("  int %s = %s + %s;\n", toks[1], toks[2], toks[3]);
        }
      }
      continue;
    }

    /* shr.u32 / shr.s32 — the reduction loop halving */
    if (op_is(op, "shr.u32") || op_is(op, "shr.s32")) {
      if (ntoks >= 4)
        EMIT("  %s %s = %s >> %s;\n",
             op_is(op, "shr.s32") ? "int" : "uint", toks[1], toks[2], toks[3]);
      continue;
    }

    /* selp.<type> d, a, b, c; → d = c ? a : b. All PTX widths: float/double
       (0f immediates), s/u/b 16-bit → int, s/u/b 32-bit → int, s/u/b 64-bit → long. */
    if (op_is(op, "selp")) {
      if (ntoks >= 5) {
        const char *ts = op_is(op, "selp.f32") ? "float"
                         : op_is(op, "selp.f64") ? "double"
                         : (op_is(op, "selp.s64") || op_is(op, "selp.u64") ||
                            op_is(op, "selp.b64")) ? "long"
                         : "int";
        char va[96], vb[96];
        ptx_val(toks[2], va, sizeof va);
        ptx_val(toks[3], vb, sizeof vb);
        EMIT("  %s %s = (%s) ? (%s) : (%s);\n", ts, toks[1], toks[4], va, vb);
      }
      continue;
    }

    /* shfl[.sync].<mode>.b32 d[|p], a, b, c[, membermask]; → ISA-exact helpers
       above. membermask ignored: MSL simd ops are warp-synchronizing (CUDA
       leaves mismatched-mask behavior undefined). The d|p predicate
       destination emits the in-range flag for @p-guarded consumers. */
    if (op_is(op, "shfl")) {
      const char *fn = NULL;
      if (strstr(op, ".up")) fn = "_shfl_up";
      else if (strstr(op, ".down")) fn = "_shfl_down";
      else if (strstr(op, ".bfly") || strstr(op, ".xor") || strstr(op, ".fly"))
        fn = "_shfl_bfly";
      else if (strstr(op, ".idx")) fn = "_shfl_idx";
      if (!fn) {
        snprintf(err_buf, err_sz, "unsupported shuffle mode '%s'", op);
        free(text);
        return -1;
      }
      if (ntoks >= 5) {
        char dest[64];
        const char *pred = NULL;
        snprintf(dest, sizeof dest, "%s", toks[1]);
        char *bar = strchr(dest, '|');
        if (bar) { *bar = 0; pred = bar + 1; }
        if (pred && pred[0]) {
          int seen = 0;
          for (int i = 0; i < npred_decls; i++)
            if (strcmp(pred_decls[i], pred) == 0) { seen = 1; break; }
          if (!seen && npred_decls < 16) {
            snprintf(pred_decls[npred_decls++], sizeof pred_decls[0], "%s", pred);
            EMIT("  bool %s;\n", pred); /* vacuous decl — safe across cases */
          }
          EMIT("  float %s = %s_p(%s, %s, %s, (uint)(_lid & 31u), &%s);\n",
               dest, fn, toks[2], toks[3], toks[4], pred);
        } else {
          EMIT("  float %s = %s(%s, %s, %s, (uint)(_lid & 31u));\n",
               dest, fn, toks[2], toks[3], toks[4]);
        }
      }
      continue;
    }

    /* @%p1 add/sub/mul/fma (predicated float arithmetic — the ISA shuffle
       scan/reduction idiom). Fires only when toks[0] is a bare pN register. */
    if (ntoks >= 5) {
      int is_pred = toks[0][0] == 'p' && isdigit((unsigned char)toks[0][1]);
      for (const char *c = toks[0] + 1; is_pred && *c; c++)
        if (!isdigit((unsigned char)*c)) is_pred = 0;
      if (is_pred) {
        const char *iop = toks[1];
        const char *opc = op_is(iop, "add.f32") ? "+"
                        : op_is(iop, "sub.f32") ? "-"
                        : op_is(iop, "mul.f32") ? "*"
                        : NULL;
        if (opc) {
          /* bare assignment: the destination was defined earlier (PTX
             discipline); a declaration here would shadow the outer register */
          EMIT("  if (%s) {\n    %s = %s %s %s;\n  }\n",
               toks[0], toks[2], toks[3], opc, toks[4]);
          continue;
        }
        if (op_is(iop, "fma.rn.f32") && ntoks >= 6) {
          EMIT("  if (%s) {\n    %s = fma(%s, %s, %s);\n  }\n",
               toks[0], toks[2], toks[3], toks[4], toks[5]);
          continue;
        }
      }
    }

    /* bar.sync 0; → threadgroup barrier */
    if (!strcmp(op, "bar.sync")) {
      EMIT("      threadgroup_barrier(mem_flags::mem_threadgroup);\n");
      continue;
    }

    /* atom.global.add.u32 / red.global.add.u32 — device-memory atomics.
       atom: %rd,[%addr],%val; red: [%addr],%val; */
    if (op_is(op, "atom.global") || op_is(op, "red.global")) {
      if (!strstr(op, ".u32")) {
        snprintf(err_buf, err_sz, "unsupported atomic width '%s'", op);
        free(text);
        return -1;
      }
      int is_red = op_is(op, "red.global");
      const char *addr_tok = is_red ? toks[1] : toks[2];
      const char *val_tok = is_red ? toks[2] : toks[3];
      char base_reg[64] = "";
      char offset_str[64] = "0";
      const char *addr = addr_tok ? addr_tok : "";
      if (isdigit((unsigned char)addr[0])) {
        snprintf(offset_str, sizeof offset_str, "%s", addr);
      } else {
        size_t k = 0;
        while (addr[k] && (isalnum((unsigned char)addr[k]) || addr[k] == '_')) k++;
        strncpy(base_reg, addr, k < sizeof(base_reg) - 1 ? k : sizeof(base_reg) - 1);
        base_reg[k < sizeof(base_reg) - 1 ? k : sizeof(base_reg) - 1] = 0;
        if (addr[k]) snprintf(offset_str, sizeof offset_str, "%s", addr + k);
      }
      if (is_red)
        EMIT("      atomic_fetch_add_explicit((device atomic_uint *)((device char *)%s + %s), %s, memory_order_relaxed);\n",
             base_reg, offset_str, val_tok ? val_tok : "0");
      else if (ntoks >= 4)
        EMIT("  uint %s = atomic_fetch_add_explicit((device atomic_uint *)((device char *)%s + %s), %s, memory_order_relaxed);\n",
             toks[1], base_reg, offset_str, val_tok ? val_tok : "0");
      continue;
    }

    /* setp.ge.s32 %p1, %r1, %r2; → comparison */
    if (op_is(op, "setp.ge")) {
      if (ntoks >= 4) EMIT("  bool %s = (%s >= %s);\n", toks[1], toks[2], toks[3]);
      continue;
    }
    if (op_is(op, "setp.gt")) {
      if (ntoks >= 4) EMIT("  bool %s = (%s > %s);\n", toks[1], toks[2], toks[3]);
      continue;
    }
    if (op_is(op, "setp.lt")) {
      if (ntoks >= 4) EMIT("  bool %s = (%s < %s);\n", toks[1], toks[2], toks[3]);
      continue;
    }
    if (op_is(op, "setp.le")) {
      if (ntoks >= 4) EMIT("  bool %s = (%s <= %s);\n", toks[1], toks[2], toks[3]);
      continue;
    }
    if (op_is(op, "setp.eq")) {
      if (ntoks >= 4) EMIT("  bool %s = (%s == %s);\n", toks[1], toks[2], toks[3]);
      continue;
    }
    if (op_is(op, "setp.ne")) {
      if (ntoks >= 4) EMIT("  bool %s = (%s != %s);\n", toks[1], toks[2], toks[3]);
      continue;
    }

    /* @%p bra LABEL; → predicated _pc transition */
    if (ntoks >= 3 && !strcmp(toks[1], "bra")) {
      char ln[64];
      label_clean(toks[2], ln, sizeof ln);
      int tc = -1;
      for (int i = 0; i < nlabels; i++)
        if (strcmp(label_names[i], ln) == 0) { tc = label_chunks[i]; break; }
      if (!ln[0] || tc < 0) {
        snprintf(err_buf, err_sz, "branch to unknown label '%s'",
                 ln[0] ? ln : toks[2]);
        free(text);
        return -1;
      }
      EMIT("      _pc = %s ? %d : %d;\n      break;\n",
           toks[0], tc, cur_chunk + 1 <= nchunks - 1 ? cur_chunk + 1 : -1);
      /* the fallthrough code after the branch is a new chunk */
      cur_chunk++;
      EMIT("    case %d:\n", cur_chunk);
      continue;
    }

    /* bra.uni LABEL; → unconditional _pc transition */
    if ((!strcmp(op, "bra.uni") || !strcmp(op, "bra")) && ntoks >= 2) {
      char ln[64];
      label_clean(toks[1], ln, sizeof ln);
      int tc = -1;
      for (int i = 0; i < nlabels; i++)
        if (strcmp(label_names[i], ln) == 0) { tc = label_chunks[i]; break; }
      if (!ln[0] || tc < 0) {
        snprintf(err_buf, err_sz, "branch to unknown label '%s'",
                 ln[0] ? ln : toks[1]);
        free(text);
        return -1;
      }
      EMIT("      _pc = %d;\n      break;\n", tc);
      /* the code after an unconditional branch is unreachable; open the next
         chunk anyway so labels/ret line up with pass A's numbering */
      cur_chunk++;
      EMIT("    case %d:\n", cur_chunk);
      continue;
    }

    /* ld.shared.f32 [%rN]; → threadgroup-space load. The address register
       must be a known shared pointer (raw shared offsets cannot be resolved
       to a specific __shared__ array). */
    if (op_is(op, "ld.shared")) {
      if (!strstr(op, ".f32")) {
        snprintf(err_buf, err_sz, "unsupported load type '%s'", op);
        free(text);
        return -1;
      }
      char base_reg[64] = "";
      char offset_str[64] = "0";
      const char *addr = ntoks >= 3 ? toks[2] : "";
      size_t k = 0;
      while (addr[k] && (isalnum((unsigned char)addr[k]) || addr[k] == '_')) k++;
      strncpy(base_reg, addr, k < sizeof(base_reg) - 1 ? k : sizeof(base_reg) - 1);
      base_reg[k < sizeof(base_reg) - 1 ? k : sizeof(base_reg) - 1] = 0;
      if (addr[k]) snprintf(offset_str, sizeof offset_str, "%s", addr + k);
      if (!is_ptr_reg(sh_ptr_regs, nsh_ptr, base_reg)) {
        /* Single-__shared__ heuristic: a raw integer address is a byte
           offset into the only shared array. Multi-var kernels hard-fail. */
        if (nshared != 1) {
          snprintf(err_buf, err_sz, "ld.shared through non-shared register '%s'", base_reg);
          free(text);
          return -1;
        }
        const char *off = (base_reg[0] && strcmp(offset_str, "0") == 0) ? base_reg : offset_str;
        EMIT("  float %s = *(threadgroup float *)((threadgroup char *)(%s) + (long)(%s));\n",
             toks[1], sh_names[0], off);
        continue;
      }
      EMIT("  float %s = *(threadgroup float *)((threadgroup char *)%s + %s);\n",
           toks[1], base_reg, offset_str);
      continue;
    }

    /* st.shared.f32 [%rN], %fX; → threadgroup-space store */
    if (op_is(op, "st.shared")) {
      if (!strstr(op, ".f32")) {
        snprintf(err_buf, err_sz, "unsupported store type '%s'", op);
        free(text);
        return -1;
      }
      char base_reg[64] = "";
      char offset_str[64] = "0";
      const char *addr = ntoks >= 2 ? toks[1] : "";
      size_t k = 0;
      while (addr[k] && (isalnum((unsigned char)addr[k]) || addr[k] == '_')) k++;
      strncpy(base_reg, addr, k < sizeof(base_reg) - 1 ? k : sizeof(base_reg) - 1);
      base_reg[k < sizeof(base_reg) - 1 ? k : sizeof(base_reg) - 1] = 0;
      if (addr[k]) snprintf(offset_str, sizeof offset_str, "%s", addr + k);
      if (!is_ptr_reg(sh_ptr_regs, nsh_ptr, base_reg)) {
        if (nshared != 1) {
          snprintf(err_buf, err_sz, "st.shared through non-shared register '%s'", base_reg);
          free(text);
          return -1;
        }
        const char *off = (base_reg[0] && strcmp(offset_str, "0") == 0) ? base_reg : offset_str;
        EMIT("  *(threadgroup float *)((threadgroup char *)(%s) + (long)(%s)) = %s;\n",
             sh_names[0], off, ntoks >= 3 ? toks[2] : "0");
        continue;
      }
      EMIT("  *(threadgroup float *)((threadgroup char *)%s + %s) = %s;\n",
           base_reg, offset_str, ntoks >= 3 ? toks[2] : "0");
      continue;
    }

    /* ld.global.f32 %f1, [%rd8+4]; → device-space load. */
    if (op_is(op, "ld.global")) {
      if (!strstr(op, ".f32")) { /* only f32 loads in the measured subset */
        snprintf(err_buf, err_sz, "unsupported load type '%s'", op);
        free(text);
        return -1;
      }
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
         Emit the load in device address space. */
      EMIT("  float %s = *(device const float *)((device const char *)%s + %s);\n",
           toks[1], base_reg, offset_str);
      continue;
    }

    /* st.global.f32 [%rd10], %f3; → store to global memory */
    if (op_is(op, "st.global")) {
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
      EMIT("  *(device float *)((device char *)%s + %s) = %s;\n", base_reg, offset_str,
           ntoks >= 3 ? toks[2] : "0");
      continue;
    }

    /* add.f32 / sub.f32 / mul.f32 / fma.rn.f32 */
    if (op_is(op, "add.f32")) {
      if (ntoks >= 4) EMIT("  float %s = %s + %s;\n", toks[1], toks[2], toks[3]);
      continue;
    }
    if (op_is(op, "sub.f32")) {
      if (ntoks >= 4) EMIT("  float %s = %s - %s;\n", toks[1], toks[2], toks[3]);
      continue;
    }
    if (op_is(op, "mul.f32")) {
      if (ntoks >= 4) EMIT("  float %s = %s * %s;\n", toks[1], toks[2], toks[3]);
      continue;
    }
    if (op_is(op, "fma.rn.f32")) {
      if (ntoks >= 5) EMIT("  float %s = fma(%s, %s, %s);\n", toks[1], toks[2], toks[3], toks[4]);
      continue;
    }

    /* add.s64 / add.u64 — pointer-aware: shared-space sums stay in the
       threadgroup space, device-space sums stay in the device space */
    if (op_is(op, "add.s64") || op_is(op, "add.u64")) {
      if (ntoks >= 4) {
        if (is_ptr_reg(sh_ptr_regs, nsh_ptr, toks[2])) {
          EMIT("  threadgroup float *%s = (threadgroup float *)((threadgroup char *)%s + (long)%s);\n",
               toks[1], toks[2], toks[3]);
          mark_ptr_reg(sh_ptr_regs, &nsh_ptr, toks[1]);
        } else if (is_ptr_reg(sh_ptr_regs, nsh_ptr, toks[3])) {
          EMIT("  threadgroup float *%s = (threadgroup float *)((threadgroup char *)%s + (long)%s);\n",
               toks[1], toks[3], toks[2]);
          mark_ptr_reg(sh_ptr_regs, &nsh_ptr, toks[1]);
        } else if (is_ptr_reg(ptr_regs, nptr_regs, toks[2])) {
          EMIT("  device const char *%s = (device const char *)%s + (long)%s;\n",
               toks[1], toks[2], toks[3]);
          mark_ptr_reg(ptr_regs, &nptr_regs, toks[1]);
        } else if (is_ptr_reg(ptr_regs, nptr_regs, toks[3])) {
          EMIT("  device const char *%s = (device const char *)%s + (long)%s;\n",
               toks[1], toks[3], toks[2]);
          mark_ptr_reg(ptr_regs, &nptr_regs, toks[1]);
        } else {
          EMIT("  long %s = %s + %s;\n", toks[1], toks[2], toks[3]);
        }
      }
      continue;
    }

    /* ret */
    if (!strcmp(op, "ret")) {
      EMIT("      _pc = -1;\n      break;\n");
      chunk_closed = 1;
      continue;
    }

    /* Unknown instruction — hard fail. Skipping could silently change
       semantics; the contract is "never a wrong translation". */
    snprintf(err_buf, err_sz, "unsupported instruction '%s'", op);
    free(text);
    return -1;
  }

  /* close the final chunk, then the switch and the dispatch loop */
  if (!chunk_closed) {
    EMIT("      _pc = %d;\n      break;\n",
         cur_chunk + 1 <= nchunks - 1 ? cur_chunk + 1 : -1);
  }
  EMIT("    }\n"); /* switch */
  EMIT("  }\n");   /* while (_pc >= 0) */

  /* ---- assembly: hoist every declaration above the body so the dispatch
     cases share one scope (C++ forbids jumping over initialized decls),
     then append the processed body ---- */
  static const char *ktypes[] = {
      "device const void *", "device const char *", "threadgroup float *",
      "uint", "int", "long", "bool", "float", "double"};
  const int nktypes = (int)(sizeof ktypes / sizeof ktypes[0]);
  char hoist[6144];
  size_t hpos = 0;
  char seen[128][24];
  int nseen = 0;
  char fbody[12288];
  size_t fpos = 0;
  char *s = body_buf + header_len;
  while (s < body_buf + bpos) {
    char *nl = strchr(s, '\n');
    if (!nl) nl = body_buf + bpos;
    size_t ll = (size_t)(nl - s);
    if (ll == 0 || ll >= MAX_LINE) {
      s = nl < body_buf + bpos ? nl + 1 : body_buf + bpos;
      continue;
    }
    char line[MAX_LINE];
    memcpy(line, s, ll);
    line[ll] = 0;
    s = nl < body_buf + bpos ? nl + 1 : body_buf + bpos;

    /* try to parse "  <type> <name> = <rhs>;" (a declaration with init) */
    char *t = line;
    while (*t == ' ') t++;
    const char *ty = NULL;
    size_t tyl = 0;
    for (int i = 0; i < nktypes && !ty; i++) {
      size_t kl = strlen(ktypes[i]);
      if (strncmp(t, ktypes[i], kl) == 0 && (t[kl] == ' ' || ktypes[i][kl - 1] == '*')) {
        ty = ktypes[i];
        tyl = kl;
      }
    }
    char name[24] = "";
    const char *rhs = NULL;
    size_t nlen = 0;
    if (ty) {
      const char *np = t + tyl;
      while (*np == ' ') np++;
      const char *ns = np;
      while (*np && (isalnum((unsigned char)*np) || *np == '_')) np++;
      nlen = (size_t)(np - ns);
      if (nlen > 0 && nlen < sizeof name && np[0] == ' ' && np[1] == '=' && np[2] == ' ') {
        memcpy(name, ns, nlen);
        name[nlen] = 0;
        rhs = np + 3;
      }
    }
    if (rhs) {
      int seen_before = 0;
      for (int i = 0; i < nseen; i++)
        if (strcmp(seen[i], name) == 0) { seen_before = 1; break; }
      if (!seen_before && nseen < 128) {
        snprintf(seen[nseen++], sizeof seen[0], "%s", name);
        if (hpos + tyl + nlen + 16 < sizeof hoist)
          hpos += (size_t)snprintf(hoist + hpos, sizeof hoist - 1 - hpos,
                                   "  %.*s %s;\n", (int)tyl, ty, name);
      }
      if (fpos + nlen + strlen(rhs) + 8 < sizeof fbody)
        fpos += (size_t)snprintf(fbody + fpos, sizeof fbody - 1 - fpos,
                                 "  %s = %s\n", name, rhs);
    } else if (fpos + ll + 2 < sizeof fbody) {
      memcpy(fbody + fpos, line, ll);
      fpos += ll;
      fbody[fpos++] = '\n';
      fbody[fpos] = 0;
    }
  }

  size_t need = header_len + hpos + spos + fpos + 4;
  if (need > msl_buf_sz - 1 || bpos >= sizeof body_buf - 1 ||
      hpos >= sizeof hoist - 1 || fpos >= sizeof fbody - 1) {
    snprintf(err_buf, err_sz, "MSL output truncated");
    free(text);
    return -1;
  }
  memcpy(msl_buf, body_buf, header_len);
  memcpy(msl_buf + header_len, hoist, hpos);
  memcpy(msl_buf + header_len + hpos, sdecls, spos);
  memcpy(msl_buf + header_len + hpos + spos, fbody, fpos);
  msl_buf[header_len + hpos + spos + fpos] = '}';
  msl_buf[header_len + hpos + spos + fpos + 1] = '\n';
  msl_buf[header_len + hpos + spos + fpos + 2] = 0;
  free(text);
  return 0;
}
