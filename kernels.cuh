#ifndef _KERNELS_H_
#define _KERNELS_H_

#include "headers.h"



__global__ void kernel_2d_copy_with_offset(int* des_map, float* des_log_odds, const int* src_map, const float* src_log_odds, const int row_offset, const int col_offset,
    const int PRE_GRID_HEIGHT, const int NEW_GRID_HEIGHT, const int NUM_ELEMS);

__global__ void kernel_robot_advance(float* states_x, float* states_y, float* states_theta, float* rnd_v, float* rnd_w,
    float encoder_counts, float yaw, float dt, float nv, float nw);

__global__ void kernel_check_map_extend_less(const float* lidar_coords, const float value, int* should_extend, const int result_idx, const int START_INDEX, const int NUM_ELEMS);
__global__ void kernel_check_map_extend_greater(const float* lidar_coords, const float value, int* should_extend, const int result_idx, const int START_INDEX, const int NUM_ELEMS);



#endif