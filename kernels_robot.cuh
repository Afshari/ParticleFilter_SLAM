#ifndef _KERNELS_ROBOT_H_
#define _KERNELS_ROBOT_H_

#include "headers.h"

__global__ void kernel_correlation_max(const float* correlation_raw, float* correlation, const int PARTICLES_LEN);

__global__ void kernel_correlation(const int* grid_map, const int* states_x, const int* states_y,
    const int* states_idx, float* result, const int GRID_WIDTH, const int GRID_HEIGHT, const int NUM_ELEMS);

__global__ void kernel_resampling(const float* weights, int* js, const float* rnd, const int NUM_ELEMS);

__global__ void kernel_create_2d_map(const int* particles_x, const int* particles_y, const int* particles_idx, const int IDX_LEN, uint8_t* map_2d,
    int* unique_in_particle, int* unique_in_particle_col, const int GRID_WIDTH, const int GRID_HEIGHT, const int NUM_ELEMS);

__global__ void kernel_update_2d_map_with_measure(const int* measure_x, const int* measure_y, const int* measure_idx, const int IDX_LEN, uint8_t* map_2d,
    int* unique_in_particle, int* unique_in_particle_col, const int GRID_WIDTH, const int GRID_HEIGHT, const int NUM_ELEMS);

__global__ void kernel_update_particles_states(const float* states_x, const float* states_y, const float* states_theta,
    float* transition_world_body, const float* transition_body_lidar, float* transition_world_lidar, const int NUM_ELEMS);

__global__ void kernel_update_particles_lidar(float* transition_world_lidar, int* processed_measure_x, int* processed_measure_y, const float* lidar_coords, float res, int xmin, int ymax,
    const int LIDAR_COORDS_LEN);

__global__ void kernel_update_unique_restructure(uint8_t* map_2d, int* particles_x, int* particles_y, int* particles_idx,
    int* unique_in_each_particle, int* unique_in_each_particle_col, const int GRID_WIDTH, const int GRID_HEIGHT);

__global__ void kernel_rearrange_particles(int* particles_x, int* particles_y, const int* particles_idx,
    const int* c_particles_x, const int* c_particles_y, const int* c_particles_idx, const int* js,
    const int GRID_WIDTH, const int GRID_HEIGHT, const int NUM_ELEMS, const int IDX_LEN, const int C_IDX_LEN);

__global__ void kernel_rearrange_states(float* states_x, float* states_y, float* states_theta,
    float* c_states_x, float* c_states_y, float* c_states_theta, int* js);

__global__ void kernel_rearrange_indecies(int* particles_idx, int* c_particles_idx, int* js, int* last_len, const int ARR_LEN);

#endif

