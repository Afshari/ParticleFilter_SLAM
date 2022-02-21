#pragma once

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

using namespace thrust::placeholders;

#define NUM_PARTICLES   100

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


template <typename T>
struct unorderLess {
    bool operator () (const std::pair<T, T>& lhs, const std::pair<T, T>& rhs) const {
        const auto lhs_order = lhs.first < lhs.second ? lhs : std::tie(lhs.second, lhs.first);
        const auto rhs_order = rhs.first < rhs.second ? rhs : std::tie(rhs.second, rhs.first);

        return lhs_order < rhs_order;
    }
};

struct pair_cmp {
    bool operator()(const pair<int, int>& lhs, const pair<int, int>& rhs) const {

        //if (lhs.first == rhs.first && lhs.second == rhs.second)
        //    return 0;
        //return 1;
        if (lhs.first == rhs.first) {
            if (lhs.second == rhs.second)   return 0;
            return lhs.second > rhs.second;
            // if (lhs.second > rhs.second)    return 1;
            // else                            return -1;
        }
        return lhs.first > rhs.second;
        //else if (lhs.first > rhs.first)          return 1;
        //else                                     return -1;
    }
};