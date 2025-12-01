CC=/usr/local/cuda-12.4/bin/nvcc
CUDA_PATH=/usr/local/cuda-12.4
CFLAGS=-Icommon -L$(CUDA_PATH)/lib64 -lcufft -Xlinker -rpath=$(CUDA_PATH)/lib64 -arch=native
HALF = half
FULL = full

ifeq ($(DTYPE), $(HALF))
$(eval CFLAGS += -DHALF_PRECISION)
else # default = FULL
$(eval CFLAGS += -DFULL_PRECISION)
endif

all: clean fft complex_poly real_poly

fft: fft.cu
	$(CC) -o $@ $< $(CFLAGS)

complex_poly: complex_poly.cu
	$(CC) -o $@ $< $(CFLAGS)

real_poly: real_poly.cu
	$(CC) -o $@ $< $(CFLAGS)

clean:
	rm -f fft complex_poly real_poly

