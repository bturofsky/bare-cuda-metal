# cuda-metal v0 build
# shim: libcuda_shim.dylib (the interposition surface; v0 exports everything)
# test: the dlopen harness asserting the v0 contract
CC = clang
CFLAGS = -O2 -Wall -Wextra -dynamiclib
SHIM = shim/libcuda_shim.dylib

all: $(SHIM) test_run scan/scan gemm_ab v2_launch v2_fatbin v2_multi v2_loop v2_reduce v2_stream v2_realfat v2_section v2_simd

CUBLAS_SHIM = shim/libcublas_shim.dylib
$(CUBLAS_SHIM): shim/libcublas_shim.m
	$(CC) $(CFLAGS) -framework Metal -framework Foundation -framework MetalPerformanceShaders -o $@ shim/libcublas_shim.m

gemm_ab: test/gemm_ab.c $(CUBLAS_SHIM) $(SHIM)
	$(CC) -O2 -Wall -o $@ test/gemm_ab.c

$(SHIM): shim/libcuda_shim.m shim/module_launch.m translator/ptx2msl.c shim/exports.txt
	$(CC) $(CFLAGS) -framework Metal -framework Foundation -o $@ shim/libcuda_shim.m shim/module_launch.m translator/ptx2msl.c -lz

test/test_harness: test/test_harness.c
	$(CC) -O2 -Wall -o $@ test/test_harness.c

test_run: $(SHIM) test/test_harness
	./test/test_harness

v2_launch: $(SHIM) test/v2_launch.c
	$(CC) -O2 -Wall -o $@ test/v2_launch.c $(SHIM)

v2_fatbin: $(SHIM) test/v2_fatbin.c
	$(CC) -O2 -Wall -o $@ test/v2_fatbin.c $(SHIM)

v2_multi: $(SHIM) test/v2_multi.c
	$(CC) -O2 -Wall -o $@ test/v2_multi.c $(SHIM)

v2_loop: $(SHIM) test/v2_loop.c
	$(CC) -O2 -Wall -o $@ test/v2_loop.c $(SHIM)

v2_reduce: $(SHIM) test/v2_reduce.c
	$(CC) -O2 -Wall -o $@ test/v2_reduce.c $(SHIM)

v2_stream: $(SHIM) test/v2_stream.c
	$(CC) -O2 -Wall -o $@ test/v2_stream.c $(SHIM)

v2_realfat: $(SHIM) test/v2_realfat.c
	$(CC) -O2 -Wall -o $@ test/v2_realfat.c $(SHIM)

v2_simd: $(SHIM) test/v2_simd.c
	$(CC) -O2 -Wall -o $@ test/v2_simd.c $(SHIM)

test/nv_fatbin_section.h: docs/reference/vectorAdd.real.fatbin
	python3 -c "import sys; d=open('docs/reference/vectorAdd.real.fatbin','rb').read(); print('#include <stddef.h>'); print('__attribute__((used, section(\"__NV_CUDA,__nv_fatbin\"))) static const unsigned char g_nv_fatbin[] = {' + ','.join(str(b) for b in d) + '};')" > $@

v2_section: $(SHIM) test/v2_section.c test/nv_fatbin_section.h
	$(CC) -O2 -Wall -o $@ test/v2_section.c $(SHIM)

scan/scan: scan/scan.c scan/zstd_decomp.c
	$(CC) -O2 -Wall -DHAVE_ZSTD -I/opt/homebrew/include -o $@ scan/scan.c scan/zstd_decomp.c scan/lz4_block.c -lz -L/opt/homebrew/lib -lzstd -L/opt/homebrew/opt/lz4/lib -llz4

clean:
	rm -f $(SHIM) test/test_harness v2_launch v2_fatbin v2_multi v2_loop v2_reduce v2_stream v2_realfat v2_section v2_simd test/nv_fatbin_section.h

.PHONY: all test_run clean
