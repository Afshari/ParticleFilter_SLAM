#ifndef _KERNELS_UTILS_H_
#define _KERNELS_UTILS_H_

#include "headers.h"

__global__ void kernel_arr_increase(int* arr, const int increase_value, const int start_index);
__global__ void kernel_arr_increase(float* arr, const float increase_value, const int start_index);
__global__ void kernel_arr_mult(float* arr, const float mult_value);
__global__ void kernel_arr_mult(float* arr, const float* mult_arr);
__global__ void kernel_arr_max(float* arr, float* result, const int LEN);
__global__ void kernel_arr_sum_exp(double* result, const float* arr, const int LEN);
__global__ void kernel_arr_normalize(float* arr, const double norm);
__global__ void kernel_update_unique_sum(int* unique_in_particle, const int NUM_ELEMS);
__global__ void kernel_update_unique_sum_col(int* unique_in_particle_col, const int GRID_WIDTH);

__global__ void kernel_index_init_const(int* indices, const int value);
__global__ void kernel_index_arr_const(float* indices, const float value, const int GRID_SIZE);
__global__ void kernel_index_expansion(int* extended_idx, const int* idx, const int NUM_ELEMS);

inline __device__ void kernel_matrix_mul_3x3(const float* A, const float* B, float* C, int start_i) {

    // A[i, j] --> A[i*3 + j]

    for (int i = 0; i < 3; i++) {

        for (int j = 0; j < 3; j++) {

            float currVal = 0;
            for (int k = 0; k < 3; k++) {
                currVal += A[start_i + (i * 3) + k] * B[k * 3 + j];
            }
            C[start_i + (i * 3) + j] = currVal;
        }
    }
}

inline __global__ void kernel_matrix_mul_3x3(const float* A, const float* B, float* C) {

    for (int i = 0; i < 3; i++) {

        for (int j = 0; j < 3; j++) {

            float currVal = 0;
            for (int k = 0; k < 3; k++) {
                currVal += A[(i * 3) + k] * B[k * 3 + j];
            }
            C[(i * 3) + j] = currVal;
        }
    }
}

inline __device__ void swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

#endif