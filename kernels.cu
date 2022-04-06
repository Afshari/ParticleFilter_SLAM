
#include "headers.h"
#include "kernels.cuh"
#include "kernels_utils.cuh"


__global__ void kernel_2d_copy_with_offset(int* des_map, float* des_log_odds, const int F_SEP,
    const int* src_map, const float* src_log_odds, const int row_offset, const int col_offset,
    const int PRE_GRID_HEIGHT, const int NEW_GRID_HEIGHT, const int NUM_ELEMS) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < NUM_ELEMS) {

        int y = i % PRE_GRID_HEIGHT;
        int x = i / PRE_GRID_HEIGHT;

        x += row_offset;
        y += col_offset;

        int new_idx = x * NEW_GRID_HEIGHT + y;
        des_map[new_idx] = src_map[i];
        des_log_odds[new_idx] = src_log_odds[i];
    }
}

__global__ void kernel_check_map_extend_less(int* should_extend, const int F_SEP,
    const float* particles_world_pos, const int value, const int result_idx, const int NUM_ELEMS) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < NUM_ELEMS) {
        if (particles_world_pos[i] < value) {
            atomicAdd(&should_extend[result_idx], 1);
        }
    }
}

__global__ void kernel_check_map_extend_greater(int* should_extend, const int F_SEP, 
    const float* particles_world_pos, const int value, const int result_idx, const int NUM_ELEMS) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < NUM_ELEMS) {
        if (particles_world_pos[i] > value) {
            atomicAdd(&should_extend[result_idx], 1);
        }
    }
}

__global__ void kernel_robot_advance(float* states_x, float* states_y, float* states_theta, float* rnd_v, float* rnd_w,
    float encoder_counts, float yaw, float dt, float nv, float nw) {

    int i = threadIdx.x;

    encoder_counts += nv * rnd_v[i];
    yaw += nw * rnd_w[i];

    float dtheta = yaw * dt;
    float theta = states_theta[i] + dtheta;

    states_x[i] = states_x[i] + encoder_counts * dt * (sin(dtheta / 2 / M_PI) / (dtheta / 2 / M_PI)) * cos(theta);
    states_y[i] = states_y[i] + encoder_counts * dt * (sin(dtheta / 2 / M_PI) / (dtheta / 2 / M_PI)) * sin(theta);
    states_theta[i] = theta;
}

