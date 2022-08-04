
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

__global__ void kernel_check_map_extend_less(int* c_should_extend, const int F_SEP,
    const float* particles_world_pos, const int value, const int result_idx, const int NUM_ELEMS) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < NUM_ELEMS) {
        if (particles_world_pos[i] < value) {
            atomicAdd(&c_should_extend[result_idx], 1);
        }
    }
}

__global__ void kernel_check_map_extend_greater(int* c_should_extend, const int F_SEP, 
    const float* particles_world_pos, const int value, const int result_idx, const int NUM_ELEMS) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < NUM_ELEMS) {
        if (particles_world_pos[i] > value) {
            atomicAdd(&c_should_extend[result_idx], 1);
        }
    }
}

__global__ void kernel_robot_advance(float* states_x, float* states_y, float* states_theta, const int F_SEP,
    const float* rnd_v, const float* rnd_w,
    float encoder_counts, float yaw, const float dt, float nv, float nw, const int NUM_ELEMS) {

    int i = threadIdx.x;

    if (i == NUM_ELEMS - 1) {
        nv = 0;
        nw = 0;
    }

    encoder_counts += nv * rnd_v[i];
    yaw += nw * rnd_w[i];

    float dtheta = yaw * dt;
    float theta = states_theta[i] + dtheta;

    //float x = states_x[i] + encoder_counts * dt * (sin(dtheta / 2 / M_PI) / (dtheta / 2 / M_PI)) * cos(theta);
    //if (isnan(x)) {
    //    printf("x: %f, states_x: %f, encoders_counts: %f, dt: %f sin: %f, cos: %f, div: %f, dtheta: %f\n", 
    //        x, states_x[i], encoder_counts, dt, sin(dtheta / 2 / M_PI), cos(theta), (dtheta / 2 / M_PI), dtheta);
    //}
    if (dtheta == 0)
        states_x[i] = states_x[i] + encoder_counts * dt * (1) * cos(theta);
    else
        states_x[i] = states_x[i] + encoder_counts * dt * (sin(dtheta / 2 / M_PI) / (dtheta / 2 / M_PI)) * cos(theta);
    
    if (dtheta == 0)
        states_y[i] = states_y[i] + encoder_counts * dt * (1) * sin(theta);
    else
        states_y[i] = states_y[i] + encoder_counts * dt * (sin(dtheta / 2 / M_PI) / (dtheta / 2 / M_PI)) * sin(theta);
    
    states_theta[i] = theta;
}

