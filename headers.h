#ifndef _HEADERS_H_
#define _HEADERS_H_

#define _USE_MATH_DEFINES

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <vector>
#include <map>
#include <set>
#include <cmath>
#include <list>
#include <time.h>
#include <chrono>
#include <thread>
#include <memory>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <numeric>


using std::vector;
using std::set;
using std::tuple;
using std::make_tuple;
using std::pair;
using std::make_pair;
using std::shared_ptr;
using std::make_shared;

using ChronoTime = std::chrono::steady_clock::time_point;
using namespace thrust::placeholders;

#define NUM_PARTICLES   100

#define  WALL   2
#define  FREE   1

#define LOG_ODD_PRIOR     -13.862943611198908


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

struct thrust_exp {
    __device__
        double operator()(double x) {
        return exp(x);
    }
};

struct thrust_div_sum {

    float sum;
    thrust_div_sum(double sum) {
        this->sum = sum;
    }
    __device__
        double operator()(double x) {
        return x / this->sum;
    }
};

#endif