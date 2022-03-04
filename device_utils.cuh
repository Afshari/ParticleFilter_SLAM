#pragma once

#include "headers.h"

__inline__ void device_alloc_init_int(int* d_var, int* h_var, const size_t sz) {

	gpuErrchk(cudaMalloc((void**)&d_var, sz));
	gpuErrchk(cudaMemcpy(d_var, h_var, sz, cudaMemcpyHostToDevice));
}


