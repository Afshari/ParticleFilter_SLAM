#ifndef _HEADERS_H_
#define _HEADERS_H_

#define _USE_MATH_DEFINES

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
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
#include <fstream>
#include <sstream>

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

using std::string;
using std::stringstream;

using ChronoTime = std::chrono::steady_clock::time_point;
using namespace thrust::placeholders;

#define NUM_PARTICLES   100
#define LOG_ODD_PRIOR     -13.862943611198908
#define SEP             0

#define  WALL   2
#define  FREE   1

static float h_transition_body_lidar[] = { 1.0, 0.0, 0.0, 0.0, 1.0, 0.015935, 0.0, 0.0, 1.0 };

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

static const char* ws = " \t\n\r\f\v";

// trim from end of string (right)
inline std::string& rtrim(std::string& s, const char* t = ws) {
    s.erase(s.find_last_not_of(t) + 1);
    return s;
}

// trim from beginning of string (left)
inline std::string& ltrim(std::string& s, const char* t = ws) {
    s.erase(0, s.find_first_not_of(t));
    return s;
}

// trim from both ends of string (right then left)
inline std::string& trim(std::string& s, const char* t = ws) {
    return ltrim(rtrim(s, t), t);
}


//struct thrust_exp {
//    __device__
//        double operator()(double x) {
//        return exp(x);
//    }
//};
//
//struct thrust_div_sum {
//
//    double sum;
//    thrust_div_sum(double sum) {
//        this->sum = sum;
//    }
//    __device__
//        double operator()(double x) {
//        return x / this->sum;
//    }
//};

#endif