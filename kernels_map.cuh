#ifndef _KERNELS_MAP_H_
#define _KERNELS_MAP_H_

#include "headers.h"
#include "kernels_utils.cuh"

__global__ void kernel_bresenham(const int* particles_occupied_x, const int* particles_occupied_y, const int* position_image_body,
    int* particles_free_x, int* particles_free_y, int* particles_free_counter, const int PARTICLES_LEN, const int MAX_DIST_IN_MAP);

__global__ void kernel_bresenham(const int* particles_occupied_x, const int* particles_occupied_y,
    const int* position_image_body, int* particles_free_x, int* particles_free_y, const int* particles_free_idx, const int PARTICLES_LEN);

__global__ void kernel_bresenham_rearrange(int* particles_free_x, int* particles_free_y, int* particles_free_x_max, int* particles_free_y_max,
    int* particles_free_idx, const int MAX_DIST_IN_MAP, const int NUM_ELEMS);

__global__ void kernel_2d_map_counter(uint8_t* map_2d, int* unique_counter, int* unique_counter_col, const int GRID_WIDHT, const int GRID_HEIGHT);

__global__ void kernel_update_log_odds(float* log_odds, int* f_x, int* f_y, const float _log_t,
    const int GRID_WIDTH, const int GRID_HEIGHT, const int NUM_ELEMS);

__global__ void kernel_position_to_image(int* position_image_body, const float transition_world_lidar_x, const float transition_world_lidar_y,
    const float res, const int xmin, const int ymax);

__global__ void kernel_position_to_image(int* position_image_body, const float transition_world_lidar_x, const float transition_world_lidar_y,
    const float res, const float xmin, const float ymax);

__global__ void kernel_position_to_image(int* position_image_body, float* transition_world_lidar, float res, int xmin, int ymax);

__global__ void kernel_update_map(int* grid_map, const float* log_odds, const float _LOG_ODD_PRIOR, const int _WALL, const int _FREE, const int NUM_ELEMS);

__global__ void kernel_update_particles_lidar(float* particles_world_x, float* particles_world_y, float* transition_world_lidar,
    const float* lidar_coords, const int LIDAR_COORDS_LEN);

__global__ void kernel_update_particles_lidar(float* transition_world_frame, int* processed_measure_x, int* processed_measure_y,
    float* particles_wframe_x, float* particles_wframe_y, const float* lidar_coords, float res, int xmin, int ymax, const int LIDAR_COORDS_LEN);

__global__ void kernel_create_2d_map(const int* particles_x, const int* particles_y, const int* particles_idx, const int IDX_LEN, uint8_t* map_2d,
    const int GRID_WIDTH, const int GRID_HEIGHT, const int NUM_ELEMS);

__global__ void kernel_update_unique_restructure(uint8_t* map_2d, int* particles_x, int* particles_y, int* unique_in_particle_col,
    const int GRID_WIDTH, const int GRID_HEIGHT);


#endif