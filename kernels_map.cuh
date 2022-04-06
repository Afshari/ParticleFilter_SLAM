#ifndef _KERNELS_MAP_H_
#define _KERNELS_MAP_H_

#include "headers.h"
#include "kernels_utils.cuh"

__global__ void kernel_bresenham_rearrange(int* particles_free_x, int* particles_free_y, const int F_SEP,
    int* particles_free_x_max, int* particles_free_y_max,
    int* particles_free_idx, const int MAX_DIST_IN_MAP, const int NUM_ELEMS);

__global__ void kernel_bresenham(int* particles_free_x, int* particles_free_y, int* particles_free_counter, const int F_SEP,
    const int* particles_occupied_x, const int* particles_occupied_y, const int* position_image_body,
    const int PARTICLES_LEN, const int MAX_DIST_IN_MAP);

__global__ void kernel_bresenham(int* particles_free_x, int* particles_free_y, const int F_SEP,
    const int* particles_occupied_x, const int* particles_occupied_y,
    const int* position_image_body, const int* particles_free_idx, const int PARTICLES_LEN);

__global__ void kernel_2d_map_counter(int* unique_counter, int* unique_counter_col, const int F_SEP,
    const uint8_t* map_2d, const int GRID_WIDHT, const int GRID_HEIGHT);

__global__ void kernel_update_log_odds(float* log_odds, const int F_SEP, 
    const int* f_x, const int* f_y, const float log_t,
    const int GRID_WIDTH, const int GRID_HEIGHT, const int NUM_ELEMS);

__global__ void kernel_position_to_image(int* position_image_body, const int F_SEP,
    const float transition_world_lidar_x, const float transition_world_lidar_y,
    const float res, const int xmin, const int ymax);

__global__ void kernel_position_to_image(int* position_image_body, const int F_SEP,
    const float* transition_world_lidar,
    const float res, const int xmin, const int ymax);

__global__ void kernel_update_map(int* grid_map, const int F_SEP,
    const float* log_odds, const float _LOG_ODD_PRIOR, const int _WALL, const int _FREE, const int NUM_ELEMS);

__global__ void kernel_update_particles_lidar(float* particles_world_x, float* particles_world_y, const int F_SEP,
    float* transition_world_lidar, const float* lidar_coords, const int LIDAR_COORDS_LEN);

__global__ void kernel_update_particles_lidar(int* processed_measure_x, int* processed_measure_y,
    float* particles_world_frame_x, float* particles_world_frame_y, const int F_SEP,
    float* transition_world_lidar, const float* lidar_coords, float res, int xmin, int ymax, const int LIDAR_COORDS_LEN);

__global__ void kernel_create_2d_map(uint8_t* map_2d, const int F_SEP,
    const int* particles_x, const int* particles_y, const int* particles_idx, const int IDX_LEN,
    const int GRID_WIDTH, const int GRID_HEIGHT, const int NUM_ELEMS);

__global__ void kernel_update_unique_restructure(int* particles_x, int* particles_y, const int F_SEP,
    const uint8_t* map_2d, const int* unique_in_particle_col,
    const int GRID_WIDTH, const int GRID_HEIGHT);


#endif