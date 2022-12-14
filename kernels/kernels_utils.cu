
#include "kernels_utils.cuh"


__global__ void kernel_arr_increase(int* arr, const int increase_value, const int start_index) {

    int i = threadIdx.x;
    if (i >= start_index) {
        arr[i] += increase_value;
    }
}

__global__ void kernel_arr_increase(float* arr, const float increase_value, const int start_index) {

    int i = threadIdx.x;
    if (i >= start_index) {
        arr[i] += increase_value;
    }
}

__global__ void kernel_arr_mult(float* arr, const float mult_value) {

    int i = threadIdx.x;
    arr[i] = arr[i] * mult_value;
}

__global__ void kernel_arr_mult(float* arr, const float* mult_arr) {

    int i = threadIdx.x;
    arr[i] = arr[i] * mult_arr[i];
}

__global__ void kernel_arr_max(float* arr, float* result, const int LEN) { // ??????????????????

    float rs = arr[0];
    for (int i = 1; i < LEN; i++) {
        if (rs < arr[i])
            rs = arr[i];
    }
    result[0] = rs;
}

__global__ void kernel_arr_sum_exp(double* result, const float* arr, const int LEN) {

    double s = 0;
    for (int i = 0; i < LEN; i++) {
        s += exp(arr[i]);
    }
    result[0] = s;
}

__global__ void kernel_arr_sum_exp(float* result, const float* arr, const int LEN) {

    float s = 0;
    for (int i = 0; i < LEN; i++) {
        s += exp(arr[i]);
    }
    result[0] = s;
}

__global__ void kernel_arr_normalize(float* arr, const double norm) {

    int i = threadIdx.x;
    arr[i] = exp(arr[i]) / norm;
}

__global__ void kernel_update_unique_sum(int* unique_in_particle, const int NUM_ELEMS) {

    for (int j = 1; j < NUM_ELEMS; j++)
        unique_in_particle[j] = unique_in_particle[j] + unique_in_particle[j - 1];
}

__global__ void kernel_update_unique_sum_col(int* unique_in_particle_col, const int GRID_WIDTH) {

    int i = threadIdx.x;

    for (int j = (i * GRID_WIDTH) + 1; j < (i + 1) * GRID_WIDTH; j++)
        unique_in_particle_col[j] = unique_in_particle_col[j] + unique_in_particle_col[j - 1];
}

__global__ void kernel_index_expansion(int* extended_idx, const int* idx, const int NUM_ELEMS) {

    int i = blockIdx.x;
    int k = threadIdx.x;
    const int numThreads = blockDim.x;

    if (i < numThreads) {

        int first_idx = idx[i];
        int last_idx = (i < numThreads - 1) ? idx[i + 1] : NUM_ELEMS;
        int arr_len = last_idx - first_idx;

        int arr_end = last_idx;

        int start_idx = ((arr_len / blockDim.x) * k) + first_idx;
        int end_idx = ((arr_len / blockDim.x) * (k + 1)) + first_idx;
        end_idx = (k < blockDim.x - 1) ? end_idx : arr_end;

        for (int j = start_idx; j < end_idx; j++)
            extended_idx[j] = i;
    }
}

__global__ void kernel_index_init_const(int* indices, const int value) {

    int i = threadIdx.x;
    indices[i] = value;
}


__global__ void kernel_index_arr_const(uint8_t* arr, const uint8_t value, const int SIZE) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < SIZE)
        arr[i] = value;
}

__global__ void kernel_index_arr_const(int* arr, const int value, const int SIZE) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < SIZE)
        arr[i] = value;
}

__global__ void kernel_index_arr_const(float* arr, const float value, const int SIZE) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < SIZE)
        arr[i] = value;
}

__global__ void kernel_index_arr_const(double* arr, const double value, const int SIZE) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < SIZE)
        arr[i] = value;
}