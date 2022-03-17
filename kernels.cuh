#ifndef _KERNELS_H_
#define _KERNELS_H_

#include "headers.h"

__global__ void kernel_bresenham_rearrange(int* particles_free_x, int* particles_free_y, int* particles_free_x_max, int* particles_free_y_max,
    int* particles_free_idx, const int MAX_DIST_IN_MAP, const int NUM_ELEMS);

__global__ void kernel_bresenham(const int* particles_occupied_x, const int* particles_occupied_y, const int* position_image_body,
    int* particles_free_x, int* particles_free_y, int* particles_free_counter, const int PARTICLES_LEN, const int MAX_DIST_IN_MAP);

__global__ void kernel_bresenham(const int* particles_occupied_x, const int* particles_occupied_y,
    const int* position_image_body, int* particles_free_x, int* particles_free_y, const int* particles_free_idx, const int PARTICLES_LEN);

__global__ void kernel_index_init_const(int* indices, const int value);

__global__ void kernel_index_expansion(const int* idx, int* extended_idx, const int numElements);
__global__ void kernel_correlation_max(const float* correlation_raw, float* correlation, const int _NUM_PARTICLES);
__global__ void kernel_correlation(const int* grid_map, const int* states_x, const int* states_y,
    const int* states_idx, float* result, const int _GRID_WIDTH, const int _GRID_HEIGHT, int numElements);

__global__ void kernel_2d_copy_with_offset(int* dest, const int* source, const int row_offset, const int col_offset, 
    const int PRE_GRID_HEIGHT, const int NEW_GRID_HEIGHT, const int NUM_ELEMS);

__global__ void kernel_update_log_odds(float* log_odds, int* f_x, int* f_y, const float _log_t,
    const int _GRID_WIDTH, const int _GRID_HEIGHT, const int numElements);

__global__ void kernel_update_map(int* grid_map, const float* log_odds, const float _LOG_ODD_PRIOR, const int _WALL, const int _FREE, const int numElements);

__global__ void kernel_resampling(const float* weights, int* js, const float* rnd, const int numElements);

__global__ void kernel_update_particles_states(const float* states_x, const float* states_y, const float* states_theta,
    float* transition_body_frame, const float* transition_lidar_frame, float* transition_world_frame, const int numElements);

__global__ void kernel_update_particles_lidar(float* transition_world_frame, int* processed_measure_x, int* processed_measure_y,
    float* particles_wframe_x, float* particles_wframe_y, const float* _lidar_coords, float _res, int _xmin, int _ymax, const int _LIDAR_COORDS_LEN);

__global__ void kernel_update_particles_lidar(float* transition_world_frame, int* processed_measure_x, int* processed_measure_y, const float* _lidar_coords, float _res, int _xmin, int _ymax,
    const int _lidar_coords_LEN, const int numElements);

__global__ void kernel_matrix_mul_3x3(const float* A, const float* B, float* C);

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
inline __device__ void swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

__global__ void kernel_2d_map_counter(uint8_t* map_2d, int* unique_counter, int* unique_counter_col, const int _GRID_WIDHT, const int _GRID_HEIGHT);

__global__ void kernel_create_2d_map(const int* particles_x, const int* particles_y, const int* particles_idx, const int IDX_LEN, uint8_t* map_2d,
    const int _GRID_WIDTH, const int _GRID_HEIGHT, const int _NUM_ELEMS);

__global__ void kernel_create_2d_map(const int* particles_x, const int* particles_y, const int* particles_idx, const int IDX_LEN, uint8_t* map_2d,
    int* unique_in_particle, int* unique_in_particle_col, const int _GRID_WIDTH, const int _GRID_HEIGHT, const int _NUM_ELEMS);

__global__ void kernel_update_2d_map_with_measure(const int* measure_x, const int* measure_y, const int* measure_idx, const int IDX_LEN, uint8_t* map_2d,
    int* unique_in_particle, int* unique_in_particle_col, const int _GRID_WIDTH, const int _GRID_HEIGHT, const int _NUM_ELEMS);

__global__ void kernel_update_unique_restructure(uint8_t* map_2d, int* particles_x, int* particles_y, int* particles_idx, int* unique_in_particle_col,
    const int _GRID_WIDTH, const int _GRID_HEIGHT);

__global__ void kernel_update_unique_restructure(uint8_t* map_2d, int* particles_x, int* particles_y, int* particles_idx,
    int* unique_in_each_particle, int* unique_in_each_particle_col, const int _GRID_WIDTH, const int _GRID_HEIGHT);

__global__ void kernel_position_to_image(int* position_image_body, const float transition_world_lidar_x, const float transition_world_lidar_y,
    const float res, const int xmin, const int ymax);
__global__ void kernel_position_to_image(int* position_image_body, const float transition_world_lidar_x, const float transition_world_lidar_y,
    const float res, const float xmin, const float ymax);
__global__ void kernel_position_to_image(int* position_image_body, float* transition_world_lidar, float _res, int _xmin, int _ymax);

__global__ void kernel_rearrange_particles(int* particles_x, int* particles_y, const int* particles_idx,
    const int* c_particles_x, const int* c_particles_y, const int* c_particles_idx, const int* js,
    const int _GRID_WIDTH, const int _GRID_HEIGHT, const int _NUM_ELEMS, const int IDX_LEN, const int C_IDX_LEN);

__global__ void kernel_rearrange_states(float* states_x, float* states_y, float* states_theta,
    float* c_states_x, float* c_states_y, float* c_states_theta, int* js);

__global__ void kernel_robot_advance(float* states_x, float* states_y, float* states_theta, float* rnd_v, float* rnd_w,
    float encoder_counts, float yaw, float dt, float nv, float nw);

__global__ void kernel_check_map_extend_less(const float* lidar_coords, const float value, int* should_extend, const int result_idx, const int START_INDEX, const int NUM_ELEMS);
__global__ void kernel_check_map_extend_greater(const float* lidar_coords, const float value, int* should_extend, const int result_idx, const int START_INDEX, const int NUM_ELEMS);

__global__ void kernel_rearrange_indecies(int* particles_idx, int* c_particles_idx, int* js, int* last_len, const int ARR_LEN);
__global__ void kernel_arr_increase(int* arr, const int increase_value, const int start_index);
__global__ void kernel_arr_increase(float* arr, const float increase_value, const int start_index);
__global__ void kernel_arr_mult(float* arr, const float mult_value);
__global__ void kernel_arr_mult(float* arr, float* mult_arr);
__global__ void kernel_arr_max(float* arr, float* result, const int LEN);
__global__ void kernel_arr_sum_exp(float* arr, double* result, const int LEN);
__global__ void kernel_arr_normalize(float* arr, const double norm);
__global__ void kernel_update_unique_sum(int* unique_in_particle, const int _NUM_ELEMS);
__global__ void kernel_update_unique_sum_col(int* unique_in_particle_col, const int _GRID_WIDTH);


#endif