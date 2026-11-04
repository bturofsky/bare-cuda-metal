# cuda-metal v0 build
# shim: libcuda_shim.dylib (the interposition surface; v0 exports everything)
# test: the dlopen harness asserting the v0 contract
CC = clang
CFLAGS = -O2 -Wall -Wextra -dynamiclib
SHIM = shim/libcuda_shim.dylib

all: $(SHIM) test_run scan/scan

$(SHIM): shim/libcuda_shim.c shim/exports.txt
	$(CC) $(CFLAGS) -o $@ shim/libcuda_shim.c

test/test_harness: test/test_harness.c
	$(CC) -O2 -Wall -o $@ test/test_harness.c

test_run: $(SHIM) test/test_harness
	./test/test_harness

scan/scan: scan/scan.c scan/zstd_decomp.c
	$(CC) -O2 -Wall -DHAVE_ZSTD -I/opt/homebrew/include -o $@ scan/scan.c scan/zstd_decomp.c scan/lz4_block.c -lz -L/opt/homebrew/lib -lzstd -L/opt/homebrew/opt/lz4/lib -llz4

clean:
	rm -f $(SHIM) test/test_harness

.PHONY: all test_run clean
