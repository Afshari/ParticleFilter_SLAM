#ifndef _KERNELS_ROBOT_H_
#define _KERNELS_ROBOT_H_

#include "headers.h"

__global__ void kernel_correlation_max(float* correlation, const float* correlation_raw, const int PARTICLES_LEN);

__global__ void kernel_correlation(float* weights, const int F_SEP, const int* grid_map, const int* states_x, const int* states_y,
    const int* states_idx, const int GRID_WIDTH, const int GRID_HEIGHT, const int NUM_ELEMS);

__global__ void kernel_resampling(int* js, const float* weights, const float* rnd, const int NUM_ELEMS);

__global__ void kernel_create_2d_map(uint8_t* map_2d, int* unique_in_particle, int* unique_in_particle_col, const int F_SEP,
    const int* particles_x, const int* particles_y, const int* particles_idx, const int IDX_LEN,
    const int GRID_WIDTH, const int GRID_HEIGHT, const int NUM_ELEMS);

__global__ void kernel_update_2d_map_with_measure(uint8_t* map_2d, int* unique_in_particle, int* unique_in_particle_col, const int F_SEP,
    const int* measure_x, const int* measure_y, const int* measure_idx, const int IDX_LEN,
    const int GRID_WIDTH, const int GRID_HEIGHT, const int NUM_ELEMS);

__global__ void kernel_update_particles_states(float* transition_world_body, float* transition_world_lidar, const int F_SEP,
    const float* states_x, const float* states_y, const float* states_theta,
    const float* transition_body_lidar, const int NUM_ELEMS);

__global__ void kernel_update_particles_lidar(int* processed_measure_x, int* processed_measure_y, const int F_SEP,
    const float* transition_world_lidar, const float* lidar_coords,
    const float res, const int xmin, const int ymax, const int LIDAR_COORDS_LEN);

__global__ void kernel_update_unique_restructure(int* particles_x, int* particles_y, int* particles_idx, const int F_SEP,
    const uint8_t* map_2d, const int* unique_in_particle,
    const int* unique_in_particle_col, const int GRID_WIDTH, const int GRID_HEIGHT);

__global__ void kernel_rearrange_particles(int* particles_x, int* particles_y, const int F_SEP, 
    const int* particles_idx, const int* c_particles_x, const int* c_particles_y, const int* c_particles_idx, const int* js,
    const int GRID_WIDTH, const int GRID_HEIGHT, const int NUM_ELEMS, const int IDX_LEN, const int C_IDX_LEN);

__global__ void kernel_rearrange_states(float* states_x, float* states_y, float* states_theta, const int F_SEP,
    const float* c_states_x, const float* c_states_y, const float* c_states_theta, const int* js);

__global__ void kernel_rearrange_indecies(int* particles_idx, int* last_len, const int F_SEP, const int* c_particles_idx, const int* js, const int ARR_LEN);

#endif

