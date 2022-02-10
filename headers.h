#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define  WALL   2
#define  FREE   1

#define LOG_ODD_PRIOR     -13.862943611198908


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__device__ void swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}
