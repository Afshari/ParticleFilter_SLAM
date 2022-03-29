
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

__global__ void kernel_arr_mult(float* arr, float* mult_arr) {

    int i = threadIdx.x;
    arr[i] = arr[i] * mult_arr[i];
}

__global__ void kernel_arr_max(float* arr, float* result, const int LEN) {

    float rs = arr[0];
    for (int i = 1; i < LEN; i++) {
        if (rs < arr[i])
            rs = arr[i];
    }
    result[0] = rs;
}

__global__ void kernel_arr_sum_exp(float* arr, double* result, const int LEN) {

    double s = 0;
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

__global__ void kernel_index_init_const(int* indices, const int value) {

    int i = threadIdx.x;
    indices[i] = value;
}

__global__ void kernel_index_arr_const(float* arr, const float value, const int GRID_SIZE) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < GRID_SIZE)
        arr[i] = value;
}
