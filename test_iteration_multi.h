#ifndef _TEST_ITERATION_MULTI_H_
#define _TEST_ITERATION_MULTI_H_

//#define ADD_HEADER_DATA

#include "headers.h"
#include "host_asserts.h"
#include "host_utils.h"
#include "kernels.cuh"
#include "kernels_robot.cuh"
#include "kernels_map.cuh"
#include "kernels_utils.cuh"
#include "device_init_common.h"
#include "device_init_robot.h"
#include "device_init_map.h"
#include "device_exec_robot.h"
#include "device_exec_map.h"
#include "device_assert_robot.h"
#include "device_assert_map.h"


//#define VERBOSE_BORDER_LINE_COUNTER
//#define VERBOSE_TOTAL_INFO
//#define VERBOSE_BANNER
//#define VERBOSE_EXECUTION_TIME
#define GET_EXTRA_TRANSITION_WORLD_BODY


bool map_size_changed = false;


///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////


//void alloc_init_state_vars(float* h_states_x, float* h_states_y, float* h_states_theta) {
//
//    sz_states_pos = NUM_PARTICLES * sizeof(float);
//
//    gpuErrchk(cudaMalloc((void**)&d_states_x, sz_states_pos));
//    gpuErrchk(cudaMalloc((void**)&d_states_y, sz_states_pos));
//    gpuErrchk(cudaMalloc((void**)&d_states_theta, sz_states_pos));
//
//    gpuErrchk(cudaMemcpy(d_states_x, h_states_x, sz_states_pos, cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemcpy(d_states_y, h_states_y, sz_states_pos, cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemcpy(d_states_theta, h_states_theta, sz_states_pos, cudaMemcpyHostToDevice));
//
//    res_robot_state = (float*)malloc(3 * sizeof(float));
//}
//
////void alloc_init_lidar_coords_var(float* h_lidar_coords, const int LIDAR_COORDS_LEN) {
////
////    sz_lidar_coords = 2 * LIDAR_COORDS_LEN * sizeof(float);
////    gpuErrchk(cudaMalloc((void**)&d_lidar_coords, sz_lidar_coords));
////
////    gpuErrchk(cudaMemcpy(d_lidar_coords, h_lidar_coords, sz_lidar_coords, cudaMemcpyHostToDevice));
////}
//
//void alloc_init_grid_map(int* h_grid_map, const int GRID_WIDTH, const int GRID_HEIGHT) {
//
//    sz_grid_map = GRID_WIDTH * GRID_HEIGHT * sizeof(int);
//    gpuErrchk(cudaMalloc((void**)&d_grid_map, sz_grid_map));
//
//    gpuErrchk(cudaMemcpy(d_grid_map, h_grid_map, sz_grid_map, cudaMemcpyHostToDevice));
//}
//
//void alloc_init_particles_vars(int* h_particles_x, int* h_particles_y, int* h_particles_idx,
//    float* h_particles_weight, const int PARTICLES_ITEMS_LEN) {
//
//    sz_particles_pos = PARTICLES_ITEMS_LEN * sizeof(int);
//    sz_particles_idx = NUM_PARTICLES * sizeof(int);
//    sz_particles_weight = NUM_PARTICLES * sizeof(float);
//
//    gpuErrchk(cudaMalloc((void**)&d_particles_x, sz_particles_pos));
//    gpuErrchk(cudaMalloc((void**)&d_particles_y, sz_particles_pos));
//    gpuErrchk(cudaMalloc((void**)&d_particles_idx, sz_particles_idx));
//    gpuErrchk(cudaMalloc((void**)&d_particles_weight, sz_particles_weight));
//
//    gpuErrchk(cudaMemcpy(d_particles_x, h_particles_x, sz_particles_pos, cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemcpy(d_particles_y, h_particles_y, sz_particles_pos, cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemcpy(d_particles_idx, h_particles_idx, sz_particles_idx, cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemcpy(d_particles_weight, h_particles_weight, sz_particles_weight, cudaMemcpyHostToDevice));
//
//    res_particles_idx = (int*)malloc(sz_particles_idx);
//}
//
//void alloc_extended_idx(const int PARTICLES_ITEMS_LEN) {
//
//    sz_extended_idx = PARTICLES_ITEMS_LEN * sizeof(int);
//    gpuErrchk(cudaMalloc((void**)&d_extended_idx, sz_extended_idx));
//
//    res_extended_idx = (int*)malloc(sz_extended_idx);
//}
//
//void alloc_states_copy_vars() {
//
//    gpuErrchk(cudaMalloc((void**)&dc_states_x, sz_states_pos));
//    gpuErrchk(cudaMalloc((void**)&dc_states_y, sz_states_pos));
//    gpuErrchk(cudaMalloc((void**)&dc_states_theta, sz_states_pos));
//}
//
//void alloc_correlation_vars() {
//
//    sz_correlation_weights = NUM_PARTICLES * sizeof(float);
//    sz_correlation_weights_raw = 25 * sz_correlation_weights;
//
//    gpuErrchk(cudaMalloc((void**)&d_correlation_weights, sz_correlation_weights));
//    gpuErrchk(cudaMalloc((void**)&d_correlation_weights_raw, sz_correlation_weights_raw));
//    gpuErrchk(cudaMemset(d_correlation_weights_raw, 0, sz_correlation_weights_raw));
//
//    res_correlation_weights = (float*)malloc(sz_correlation_weights);
//    memset(res_correlation_weights, 0, sz_correlation_weights);
//
//    //res_extended_idx = (int*)malloc(sz_extended_idx);
//}
//
//void alloc_init_transition_vars(float* h_transition_body_lidar) {
//
//    sz_transition_multi_world_frame = 9 * NUM_PARTICLES * sizeof(float);
//    sz_transition_body_lidar = 9 * sizeof(float);
//
//    gpuErrchk(cudaMalloc((void**)&d_transition_multi_world_body, sz_transition_multi_world_frame));
//    gpuErrchk(cudaMalloc((void**)&d_transition_multi_world_lidar, sz_transition_multi_world_frame));
//    gpuErrchk(cudaMalloc((void**)&d_transition_body_lidar, sz_transition_body_lidar));
//
//    gpuErrchk(cudaMemcpy(d_transition_body_lidar, h_transition_body_lidar, sz_transition_body_lidar, cudaMemcpyHostToDevice));
//
//    gpuErrchk(cudaMemset(d_transition_multi_world_body, 0, sz_transition_multi_world_frame));
//    gpuErrchk(cudaMemset(d_transition_multi_world_lidar, 0, sz_transition_multi_world_frame));
//
//    res_transition_world_body = (float*)malloc(sz_transition_multi_world_frame);
//    //res_transition_world_lidar = (float*)malloc(sz_transition_world_frame);
//    res_robot_world_body = (float*)malloc(sz_transition_multi_world_frame);
//}
//
//void alloc_init_processed_measurement_vars(const int LIDAR_COORDS_LEN) {
//
//    sz_processed_measure_pos = NUM_PARTICLES * LIDAR_COORDS_LEN * sizeof(int);
//    sz_processed_measure_idx = NUM_PARTICLES * LIDAR_COORDS_LEN * sizeof(int);
//
//    gpuErrchk(cudaMalloc((void**)&d_processed_measure_x, sz_processed_measure_pos));
//    gpuErrchk(cudaMalloc((void**)&d_processed_measure_y, sz_processed_measure_pos));
//    gpuErrchk(cudaMalloc((void**)&d_processed_measure_idx, sz_processed_measure_idx));
//
//    gpuErrchk(cudaMemset(d_processed_measure_x, 0, sz_processed_measure_pos));
//    gpuErrchk(cudaMemset(d_processed_measure_y, 0, sz_processed_measure_pos));
//    gpuErrchk(cudaMemset(d_processed_measure_idx, 0, sz_processed_measure_idx));
//}
//
//void alloc_map_2d_var(const int GRID_WIDTH, const int GRID_HEIGHT) {
//
//    sz_map_2d = GRID_WIDTH * GRID_HEIGHT * NUM_PARTICLES * sizeof(uint8_t);
//
//    gpuErrchk(cudaMalloc((void**)&d_map_2d, sz_map_2d));
//    gpuErrchk(cudaMemset(d_map_2d, 0, sz_map_2d));
//}
//
//void alloc_map_2d_unique_counter_vars(const int UNIQUE_COUNTER_LEN, const int GRID_WIDTH) {
//
//    sz_unique_in_particle = UNIQUE_COUNTER_LEN * sizeof(int);
//    sz_unique_in_particle_col = UNIQUE_COUNTER_LEN * GRID_WIDTH * sizeof(int);
//
//    gpuErrchk(cudaMalloc((void**)&d_unique_in_particle, sz_unique_in_particle));
//    gpuErrchk(cudaMalloc((void**)&d_unique_in_particle_col, sz_unique_in_particle_col));
//
//    gpuErrchk(cudaMemset(d_unique_in_particle, 0, sz_unique_in_particle));
//    gpuErrchk(cudaMemset(d_unique_in_particle_col, 0, sz_unique_in_particle_col));
//
//    res_unique_in_particle = (int*)malloc(sz_unique_in_particle);
//}
//
//void alloc_correlation_weights_vars() {
//
//    sz_correlation_sum_exp = sizeof(double);
//    sz_correlation_weights_max = sizeof(float);
//
//    gpuErrchk(cudaMalloc((void**)&d_correlation_sum_exp, sz_correlation_sum_exp));
//    gpuErrchk(cudaMalloc((void**)&d_correlation_weights_max, sz_correlation_weights_max));
//
//    gpuErrchk(cudaMemset(d_correlation_sum_exp, 0, sz_correlation_sum_exp));
//    gpuErrchk(cudaMemset(d_correlation_weights_max, 0, sz_correlation_weights_max));
//
//    res_correlation_sum_exp = (double*)malloc(sz_correlation_sum_exp);
//    res_correlation_weights_max = (float*)malloc(sz_correlation_weights_max);
//}
//
//void alloc_resampling_vars(float* h_resampling_rnds, bool should_mem_allocate) {
//
//    if (should_mem_allocate == true) {
//
//        sz_resampling_js = NUM_PARTICLES * sizeof(int);
//        sz_resampling_rnd = NUM_PARTICLES * sizeof(float);
//
//        gpuErrchk(cudaMalloc((void**)&d_resampling_js, sz_resampling_js));
//        gpuErrchk(cudaMalloc((void**)&d_resampling_rnd, sz_resampling_rnd));
//    }
//
//    gpuErrchk(cudaMemcpy(d_resampling_rnd, h_resampling_rnds, sz_resampling_rnd, cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemset(d_resampling_js, 0, sz_resampling_js));
//}
//
//void exec_calc_transition() {
//
//    threadsPerBlock = NUM_PARTICLES;
//    blocksPerGrid = 1;
//    kernel_update_particles_states << <blocksPerGrid, threadsPerBlock >> > (d_transition_multi_world_body, d_transition_multi_world_lidar, SEP,
//        d_states_x, d_states_y, d_states_theta,
//        d_transition_body_lidar, NUM_PARTICLES);
//    cudaDeviceSynchronize();
//
//    gpuErrchk(cudaMemcpy(res_transition_world_body, d_transition_multi_world_body, sz_transition_multi_world_frame, cudaMemcpyDeviceToHost));
//}
//
//void exec_process_measurements(float res, const int xmin, const int ymax, const int LIDAR_COORDS_LEN) {
//
//    threadsPerBlock = NUM_PARTICLES;
//    blocksPerGrid = LIDAR_COORDS_LEN;
//    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (d_processed_measure_x, d_processed_measure_y, SEP,
//        d_transition_multi_world_lidar, d_lidar_coords, res, xmin, ymax, LIDAR_COORDS_LEN);
//    cudaDeviceSynchronize();
//
//    threadsPerBlock = NUM_PARTICLES;
//    blocksPerGrid = 1;
//    kernel_index_init_const << < blocksPerGrid, threadsPerBlock >> > (d_processed_measure_idx, LIDAR_COORDS_LEN);
//    cudaDeviceSynchronize();
//
//    thrust::exclusive_scan(thrust::device, d_processed_measure_idx, d_processed_measure_idx + NUM_PARTICLES, d_processed_measure_idx, 0);
//}
//
//void exec_create_2d_map(const int PARTICLES_ITEMS_LEN, const int GRID_WIDTH, const int GRID_HEIGHT) {
//
//    threadsPerBlock = 100;
//    blocksPerGrid = NUM_PARTICLES;
//    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (d_map_2d, d_unique_in_particle, d_unique_in_particle_col, SEP,
//        d_particles_x, d_particles_y, d_particles_idx, PARTICLES_ITEMS_LEN, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
//    cudaDeviceSynchronize();
//}
//
//void exec_update_map(const int MEASURE_LEN, const int GRID_WIDTH, const int GRID_HEIGHT) {
//
//    threadsPerBlock = NUM_PARTICLES;
//    blocksPerGrid = 1;
//
//    kernel_update_2d_map_with_measure << <blocksPerGrid, threadsPerBlock >> > (d_map_2d, d_unique_in_particle, d_unique_in_particle_col, SEP,
//        d_processed_measure_x, d_processed_measure_y, d_processed_measure_idx,
//        MEASURE_LEN, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
//    cudaDeviceSynchronize();
//}
//
//void exec_particle_unique_cum_sum(int& PARTICLES_ITEMS_LEN, int& C_PARTICLES_ITEMS_LEN, const int GRID_WIDTH) {
//
//    threadsPerBlock = UNIQUE_COUNTER_LEN;
//    blocksPerGrid = 1;
//    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (d_unique_in_particle_col, GRID_WIDTH);
//    thrust::exclusive_scan(thrust::device, d_unique_in_particle, d_unique_in_particle + UNIQUE_COUNTER_LEN, d_unique_in_particle, 0);
//    cudaDeviceSynchronize();
//
//    gpuErrchk(cudaMemcpy(res_unique_in_particle, d_unique_in_particle, sz_unique_in_particle, cudaMemcpyDeviceToHost));
//
//    PARTICLES_ITEMS_LEN = res_unique_in_particle[UNIQUE_COUNTER_LEN - 1];
//    C_PARTICLES_ITEMS_LEN = 0;
//}
//
//void reinit_map_vars(const int PARTICLES_ITEMS_LEN) {
//
//    //gpuErrchk(cudaFree(d_particles_x));
//    //gpuErrchk(cudaFree(d_particles_y));
//    //gpuErrchk(cudaFree(d_extended_idx));
//
//    sz_particles_pos = PARTICLES_ITEMS_LEN * sizeof(int);
//    sz_extended_idx = PARTICLES_ITEMS_LEN * sizeof(int);
//
//    gpuErrchk(cudaMalloc((void**)&d_particles_x, sz_particles_pos));
//    gpuErrchk(cudaMalloc((void**)&d_particles_y, sz_particles_pos));
//    gpuErrchk(cudaMalloc((void**)&d_extended_idx, sz_extended_idx));
//}
//
//void exec_map_restructure(const int GRID_WIDTH, const int GRID_HEIGHT) {
//
//    threadsPerBlock = GRID_WIDTH;
//    blocksPerGrid = NUM_PARTICLES;
//
//    cudaMemset(d_particles_idx, 0, sz_particles_idx);
//    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (d_particles_x, d_particles_y, d_particles_idx, SEP,
//        d_map_2d, d_unique_in_particle, d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT);
//    cudaDeviceSynchronize();
//
//    thrust::exclusive_scan(thrust::device, d_particles_idx, d_particles_idx + NUM_PARTICLES, d_particles_idx, 0);
//}
//
//void exec_index_expansion(const int PARTICLES_ITEMS_LEN) {
//
//    threadsPerBlock = 100;
//    blocksPerGrid = NUM_PARTICLES;
//    kernel_index_expansion << <blocksPerGrid, threadsPerBlock >> > (d_extended_idx, d_particles_idx, PARTICLES_ITEMS_LEN);
//    cudaDeviceSynchronize();
//
//    res_extended_idx = (int*)malloc(sz_extended_idx);
//    gpuErrchk(cudaMemcpy(res_extended_idx, d_extended_idx, sz_extended_idx, cudaMemcpyDeviceToHost));
//    gpuErrchk(cudaMemcpy(res_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToHost));
//}
//
//void exec_correlation(const int PARTICLES_ITEMS_LEN, const int GRID_WIDTH, const int GRID_HEIGHT) {
//
//    threadsPerBlock = 256;
//    blocksPerGrid = (PARTICLES_ITEMS_LEN + threadsPerBlock - 1) / threadsPerBlock;
//
//    kernel_correlation << <blocksPerGrid, threadsPerBlock >> > (d_correlation_weights_raw, SEP,
//        d_grid_map, d_particles_x, d_particles_y,
//        d_extended_idx, GRID_WIDTH, GRID_HEIGHT, PARTICLES_ITEMS_LEN);
//    cudaDeviceSynchronize();
//
//    threadsPerBlock = NUM_PARTICLES;
//    blocksPerGrid = 1;
//    kernel_correlation_max << <blocksPerGrid, threadsPerBlock >> > (d_correlation_weights, d_correlation_weights_raw, NUM_PARTICLES);
//    cudaDeviceSynchronize();
//
//    gpuErrchk(cudaMemcpy(res_extended_idx, d_extended_idx, sz_extended_idx, cudaMemcpyDeviceToHost));
//}
//
//void exec_update_weights() {
//
//    threadsPerBlock = 1;
//    blocksPerGrid = 1;
//
//    kernel_arr_max << < blocksPerGrid, threadsPerBlock >> > (d_correlation_weights, d_correlation_weights_max, NUM_PARTICLES);
//    cudaDeviceSynchronize();
//
//    gpuErrchk(cudaMemcpy(res_correlation_weights_max, d_correlation_weights_max, sz_correlation_weights_max, cudaMemcpyDeviceToHost));
//
//    float norm_value = -res_correlation_weights_max[0] + 50;
//
//    threadsPerBlock = NUM_PARTICLES;
//    blocksPerGrid = 1;
//    kernel_arr_increase << < blocksPerGrid, threadsPerBlock >> > (d_correlation_weights, norm_value, 0);
//    cudaDeviceSynchronize();
//
//    threadsPerBlock = 1;
//    blocksPerGrid = 1;
//    kernel_arr_sum_exp << < blocksPerGrid, threadsPerBlock >> > (d_correlation_sum_exp, d_correlation_weights, NUM_PARTICLES);
//    cudaDeviceSynchronize();
//
//    gpuErrchk(cudaMemcpy(res_correlation_sum_exp, d_correlation_sum_exp, sz_correlation_sum_exp, cudaMemcpyDeviceToHost));
//
//    threadsPerBlock = NUM_PARTICLES;
//    blocksPerGrid = 1;
//    kernel_arr_normalize << < blocksPerGrid, threadsPerBlock >> > (d_correlation_weights, res_correlation_sum_exp[0]);
//    cudaDeviceSynchronize();
//
//    threadsPerBlock = NUM_PARTICLES;
//    blocksPerGrid = 1;
//    kernel_arr_mult << < blocksPerGrid, threadsPerBlock >> > (d_particles_weight, d_correlation_weights);
//    cudaDeviceSynchronize();
//}
//
//void exec_resampling() {
//
//    threadsPerBlock = NUM_PARTICLES;
//    blocksPerGrid = 1;
//
//    kernel_resampling << <blocksPerGrid, threadsPerBlock >> > (d_resampling_js, d_correlation_weights, d_resampling_rnd, NUM_PARTICLES);
//    cudaDeviceSynchronize();
//}
//
//void reinit_particles_vars(const int PARTICLES_ITEMS_LEN) {
//
//    sz_last_len = sizeof(int);
//    d_last_len = NULL;
//    res_last_len = (int*)malloc(sizeof(int));
//
//    gpuErrchk(cudaMalloc((void**)&d_last_len, sizeof(int)));
//    gpuErrchk(cudaMalloc((void**)&dc_particles_x, sz_particles_pos));
//    gpuErrchk(cudaMalloc((void**)&dc_particles_y, sz_particles_pos));
//    gpuErrchk(cudaMalloc((void**)&dc_particles_idx, sz_particles_idx));
//
//    gpuErrchk(cudaMemcpy(dc_particles_x, d_particles_x, sz_particles_pos, cudaMemcpyDeviceToDevice));
//    gpuErrchk(cudaMemcpy(dc_particles_y, d_particles_y, sz_particles_pos, cudaMemcpyDeviceToDevice));
//    gpuErrchk(cudaMemcpy(dc_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToDevice));
//
//    gpuErrchk(cudaMemcpy(dc_states_x, d_states_x, sz_states_pos, cudaMemcpyDeviceToDevice));
//    gpuErrchk(cudaMemcpy(dc_states_y, d_states_y, sz_states_pos, cudaMemcpyDeviceToDevice));
//    gpuErrchk(cudaMemcpy(dc_states_theta, d_states_theta, sz_states_pos, cudaMemcpyDeviceToDevice));
//
//    threadsPerBlock = NUM_PARTICLES;
//    blocksPerGrid = 1;
//    kernel_rearrange_indecies << <blocksPerGrid, threadsPerBlock >> > (d_particles_idx, d_last_len, SEP,
//        dc_particles_idx, d_resampling_js, PARTICLES_ITEMS_LEN);
//    cudaDeviceSynchronize();
//
//    gpuErrchk(cudaMemcpy(res_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToHost));
//    gpuErrchk(cudaMemcpy(res_last_len, d_last_len, sz_last_len, cudaMemcpyDeviceToHost));
//}
//
//void exec_rearrangement(int& PARTICLES_ITEMS_LEN, int& C_PARTICLES_ITEMS_LEN, 
//    const int GRID_WIDTH, const int GRID_HEIGHT) {
//
//    thrust::exclusive_scan(thrust::device, d_particles_idx, d_particles_idx + NUM_PARTICLES, d_particles_idx, 0);
//
//    gpuErrchk(cudaMemcpy(res_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToHost));
//    C_PARTICLES_ITEMS_LEN = PARTICLES_ITEMS_LEN;
//    PARTICLES_ITEMS_LEN = res_particles_idx[NUM_PARTICLES - 1] + res_last_len[0];
//
//    sz_particles_pos = PARTICLES_ITEMS_LEN * sizeof(int);
//
//    threadsPerBlock = 100;
//    blocksPerGrid = NUM_PARTICLES;
//    kernel_rearrange_particles << <blocksPerGrid, threadsPerBlock >> > (d_particles_x, d_particles_y, SEP,
//        d_particles_idx, dc_particles_x, dc_particles_y, dc_particles_idx, d_resampling_js,
//        GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES, PARTICLES_ITEMS_LEN, C_PARTICLES_ITEMS_LEN);
//
//    kernel_rearrange_states << <blocksPerGrid, threadsPerBlock >> > (d_states_x, d_states_y, d_states_theta, SEP,
//        dc_states_x, dc_states_y, dc_states_theta, d_resampling_js);
//    cudaDeviceSynchronize();
//}
//
//void exec_update_states() {
//
//    thrust::device_vector<float> d_vec_states_x(d_states_x, d_states_x + NUM_PARTICLES);
//    thrust::device_vector<float> d_vec_states_y(d_states_y, d_states_y + NUM_PARTICLES);
//    thrust::device_vector<float> d_vec_states_theta(d_states_theta, d_states_theta + NUM_PARTICLES);
//
//    thrust::host_vector<float> h_vec_states_x(d_vec_states_x.begin(), d_vec_states_x.end());
//    thrust::host_vector<float> h_vec_states_y(d_vec_states_y.begin(), d_vec_states_y.end());
//    thrust::host_vector<float> h_vec_states_theta(d_vec_states_theta.begin(), d_vec_states_theta.end());
//
//    std_vec_states_x.clear();
//    std_vec_states_y.clear();
//    std_vec_states_theta.clear();
//    std_vec_states_x.resize(h_vec_states_x.size());
//    std_vec_states_y.resize(h_vec_states_y.size());
//    std_vec_states_theta.resize(h_vec_states_theta.size());
//
//    std::copy(h_vec_states_x.begin(), h_vec_states_x.end(), std_vec_states_x.begin());
//    std::copy(h_vec_states_y.begin(), h_vec_states_y.end(), std_vec_states_y.begin());
//    std::copy(h_vec_states_theta.begin(), h_vec_states_theta.end(), std_vec_states_theta.begin());
//
//    std::map<std::tuple<float, float, float>, int> states;
//
//    for (int i = 0; i < NUM_PARTICLES; i++) {
//        if (states.find(std::make_tuple(std_vec_states_x[i], std_vec_states_y[i], std_vec_states_theta[i])) == states.end())
//            states.insert({ std::make_tuple(std_vec_states_x[i], std_vec_states_y[i], std_vec_states_theta[i]), 1 });
//        else
//            states[std::make_tuple(std_vec_states_x[i], std_vec_states_y[i], std_vec_states_theta[i])] += 1;
//    }
//
//    std::map<std::tuple<float, float, float>, int>::iterator best
//        = std::max_element(states.begin(), states.end(), [](const std::pair<std::tuple<float, float, float>, int>& a,
//            const std::pair<std::tuple<float, float, float>, int>& b)->bool { return a.second < b.second; });
//
//    auto key = best->first;
//
//    float theta = std::get<2>(key);
//
//    res_robot_world_body[0] = cos(theta);	res_robot_world_body[1] = -sin(theta);	res_robot_world_body[2] = std::get<0>(key);
//    res_robot_world_body[3] = sin(theta);   res_robot_world_body[4] = cos(theta);	res_robot_world_body[5] = std::get<1>(key);
//    res_robot_world_body[6] = 0;			res_robot_world_body[7] = 0;			res_robot_world_body[8] = 1;
//
//#ifndef GET_EXTRA_TRANSITION_WORLD_BODY
//    gpuErrchk(cudaMemcpy(d_transition_single_world_body, res_robot_world_body, sz_transition_single_frame, cudaMemcpyHostToDevice));
//#endif
//
//    res_robot_state[0] = std::get<0>(key); res_robot_state[1] = std::get<1>(key); res_robot_state[2] = std::get<2>(key);
//}
//
//void assert_robot_results(float* new_weights, float* particles_weight_post, float* h_robot_transition_world_body,
//    float* h_robot_state) {
//
//    //gpuErrchk(cudaMemcpy(res_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToHost));
//    //ASSERT_resampling_particles_index(h_particles_idx_after_resampling, res_particles_idx, NUM_PARTICLES, false, negative_after_counter);
//
//    res_correlation_weights = (float*)malloc(sz_correlation_weights);
//    res_particles_weight = (float*)malloc(sz_particles_weight);
//
//    gpuErrchk(cudaMemcpy(res_correlation_weights, d_correlation_weights, sz_correlation_weights, cudaMemcpyDeviceToHost));
//    gpuErrchk(cudaMemcpy(res_particles_weight, d_particles_weight, sz_particles_weight, cudaMemcpyDeviceToHost));
//
//    ASSERT_update_particle_weights(res_correlation_weights, new_weights, NUM_PARTICLES, "weights", false, true, true, true);
//    ASSERT_update_particle_weights(res_particles_weight, particles_weight_post, NUM_PARTICLES, "particles weight", false, true, false, true);
//
//    printf("\n");
//    printf("~~$ Transition World to Body (Result): ");
//    for (int i = 0; i < 9; i++) {
//        printf("%f ", res_transition_world_body[i]);
//    }
//    printf("\n");
//    printf("~~$ Transition World to Body (Host)  : ");
//    for (int i = 0; i < 9; i++) {
//        printf("%f ", h_robot_transition_world_body[i]);
//    }
//
//    printf("\n\n");
//    printf("~~$ Robot State (Result): %f, %f, %f\n", res_robot_state[0], res_robot_state[1], res_robot_state[2]);
//    printf("~~$ Robot State (Host)  : %f, %f, %f\n", h_robot_state[0], h_robot_state[1], h_robot_state[2]);
//}


///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////


//void alloc_init_transition_vars(float* h_transition_body_lidar, float* h_transition_world_body) {
//
//    sz_transition_single_frame = 9 * sizeof(float);
//    sz_transition_body_lidar = 9 * sizeof(float);
//
//    gpuErrchk(cudaMalloc((void**)&d_transition_single_world_body, sz_transition_single_frame));
//    gpuErrchk(cudaMalloc((void**)&d_transition_single_world_lidar, sz_transition_single_frame));
//    gpuErrchk(cudaMalloc((void**)&d_transition_body_lidar, sz_transition_body_lidar));
//
//    gpuErrchk(cudaMemcpy(d_transition_body_lidar, h_transition_body_lidar, sz_transition_body_lidar, cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemcpy(d_transition_single_world_body, h_transition_world_body, sz_transition_single_frame, cudaMemcpyHostToDevice));
//}
//
//void alloc_init_lidar_coords_var(float* h_lidar_coords, const int LIDAR_COORDS_LEN) {
//
//    sz_lidar_coords = 2 * LIDAR_COORDS_LEN * sizeof(float);
//    gpuErrchk(cudaMalloc((void**)&d_lidar_coords, sz_lidar_coords));
//
//    gpuErrchk(cudaMemcpy(d_lidar_coords, h_lidar_coords, sz_lidar_coords, cudaMemcpyHostToDevice));
//}
//
//void alloc_particles_world_vars(const int LIDAR_COORDS_LEN) {
//
//    sz_particles_world_pos = LIDAR_COORDS_LEN * sizeof(float);
//
//    gpuErrchk(cudaMalloc((void**)&d_particles_world_x, sz_particles_world_pos));
//    gpuErrchk(cudaMalloc((void**)&d_particles_world_y, sz_particles_world_pos));
//}
//
//void alloc_particles_free_vars(int PARTICLES_OCCUPIED_LEN, int PARTICLE_UNIQUE_COUNTER, int MAX_DIST_IN_MAP) {
//
//    sz_particles_free_pos = 0;
//    sz_particles_free_pos_max = PARTICLES_OCCUPIED_LEN * MAX_DIST_IN_MAP * sizeof(int);
//    sz_particles_free_counter = PARTICLE_UNIQUE_COUNTER * sizeof(int);
//    sz_particles_free_idx = PARTICLES_OCCUPIED_LEN * sizeof(int);
//
//    gpuErrchk(cudaMalloc((void**)&d_particles_free_x_max, sz_particles_free_pos_max));
//    gpuErrchk(cudaMalloc((void**)&d_particles_free_y_max, sz_particles_free_pos_max));
//    gpuErrchk(cudaMalloc((void**)&d_particles_free_counter, sz_particles_free_counter));
//    gpuErrchk(cudaMalloc((void**)&d_particles_free_idx, sz_particles_free_idx));
//
//    gpuErrchk(cudaMemset(d_particles_free_x_max, 0, sz_particles_free_pos_max));
//    gpuErrchk(cudaMemset(d_particles_free_y_max, 0, sz_particles_free_pos_max));
//    gpuErrchk(cudaMemset(d_particles_free_counter, 0, sz_particles_free_counter));
//
//    res_particles_free_counter = (int*)malloc(sz_particles_free_counter);
//}
//
//void alloc_particles_occupied_vars(int LIDAR_COORDS_LEN) {
//
//    sz_particles_occupied_pos = LIDAR_COORDS_LEN * sizeof(int);
//
//    gpuErrchk(cudaMalloc((void**)&d_particles_occupied_x, sz_particles_occupied_pos));
//    gpuErrchk(cudaMalloc((void**)&d_particles_occupied_y, sz_particles_occupied_pos));
//}
//
//void alloc_bresenham_vars() {
//
//    sz_position_image_body = 2 * sizeof(int);
//
//    gpuErrchk(cudaMalloc((void**)&d_position_image_body, sz_position_image_body));
//}
//
//void alloc_init_map_vars(int* h_grid_map, const int GRID_WIDTH, const int GRID_HEIGHT) {
//
//    sz_grid_map = (GRID_WIDTH * GRID_HEIGHT) * sizeof(int);
//    gpuErrchk(cudaMalloc((void**)&d_grid_map, sz_grid_map));
//
//    gpuErrchk(cudaMemcpy(d_grid_map, h_grid_map, sz_grid_map, cudaMemcpyHostToDevice));
//
//    res_grid_map = (int*)malloc(sz_grid_map);
//}
//
//void alloc_log_odds_vars(int GRID_WIDTH, int GRID_HEIGHT) {
//
//    sz_log_odds = (GRID_WIDTH * GRID_HEIGHT) * sizeof(float);
//    gpuErrchk(cudaMalloc((void**)&d_log_odds, sz_log_odds));
//
//    res_log_odds = (float*)malloc(sz_log_odds);
//}
//
//void alloc_init_log_odds_free_vars(bool should_mem_allocate, int GRID_WIDTH, int GRID_HEIGHT) {
//
//    if (should_mem_allocate == true) {
//
//        sz_free_map_idx = 2 * sizeof(int);
//        sz_free_map_2d = GRID_WIDTH * GRID_HEIGHT * sizeof(uint8_t);
//        sz_free_unique_counter = 1 * sizeof(int);
//        sz_free_unique_counter_col = (GRID_WIDTH + 1) * sizeof(int);
//
//        gpuErrchk(cudaMalloc((void**)&d_free_map_2d, sz_free_map_2d));
//        gpuErrchk(cudaMalloc((void**)&d_free_unique_counter, sz_free_unique_counter));
//        gpuErrchk(cudaMalloc((void**)&d_free_unique_counter_col, sz_free_unique_counter_col));
//        gpuErrchk(cudaMalloc((void**)&d_free_map_idx, sz_free_map_idx));
//
//        res_free_unique_counter = (int*)malloc(sz_free_unique_counter);
//    }
//
//    gpuErrchk(cudaMemset(d_free_map_2d, 0, sz_free_map_2d));
//    gpuErrchk(cudaMemset(d_free_unique_counter, 0, sz_free_unique_counter));
//    gpuErrchk(cudaMemset(d_free_unique_counter_col, 0, sz_free_unique_counter_col));
//    gpuErrchk(cudaMemset(d_free_map_idx, 0, sz_free_map_idx));
//}
//
//void alloc_init_log_odds_occupied_vars(bool should_mem_allocate, int GRID_WIDTH, int GRID_HEIGHT) {
//
//    if (should_mem_allocate == true) {
//
//        sz_occupied_map_idx = 2 * sizeof(int);
//        sz_occupied_map_2d = GRID_WIDTH * GRID_HEIGHT * sizeof(uint8_t);
//        sz_occupied_unique_counter = 1 * sizeof(int);
//        sz_occupied_unique_counter_col = (GRID_WIDTH + 1) * sizeof(int);
//
//        gpuErrchk(cudaMalloc((void**)&d_occupied_map_2d, sz_occupied_map_2d));
//        gpuErrchk(cudaMalloc((void**)&d_occupied_unique_counter, sz_occupied_unique_counter));
//        gpuErrchk(cudaMalloc((void**)&d_occupied_unique_counter_col, sz_occupied_unique_counter_col));
//        gpuErrchk(cudaMalloc((void**)&d_occupied_map_idx, sz_occupied_map_idx));
//
//        res_occupied_unique_counter = (int*)malloc(sz_occupied_unique_counter);
//    }
//
//    gpuErrchk(cudaMemset(d_occupied_map_2d, 0, sz_occupied_map_2d));
//    gpuErrchk(cudaMemset(d_occupied_unique_counter, 0, sz_occupied_unique_counter));
//    gpuErrchk(cudaMemset(d_occupied_unique_counter_col, 0, sz_occupied_unique_counter_col));
//    gpuErrchk(cudaMemset(d_occupied_map_idx, 0, sz_occupied_map_idx));
//}
//
//void init_log_odds_vars(float* h_log_odds, int PARTICLES_OCCUPIED_LEN) {
//
//    h_occupied_map_idx[1] = PARTICLES_OCCUPIED_LEN;
//    h_free_map_idx[1] = 0;
//
//    memset(res_log_odds, 0, sz_log_odds);
//
//    gpuErrchk(cudaMemcpy(d_occupied_map_idx, h_occupied_map_idx, sz_occupied_map_idx, cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemcpy(d_free_map_idx, h_free_map_idx, sz_free_map_idx, cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemcpy(d_log_odds, h_log_odds, sz_log_odds, cudaMemcpyHostToDevice));
//}
//
//void exec_world_to_image_transform_step_1(int xmin, int ymax, float res, int LIDAR_COORDS_LEN) {
//
//    kernel_matrix_mul_3x3 << < 1, 1 >> > (d_transition_single_world_body, d_transition_body_lidar, d_transition_single_world_lidar);
//    cudaDeviceSynchronize();
//
//    threadsPerBlock = 1;
//    blocksPerGrid = LIDAR_COORDS_LEN;
//    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (d_particles_world_x, d_particles_world_y, SEP,
//        d_transition_single_world_lidar, d_lidar_coords, LIDAR_COORDS_LEN);
//    cudaDeviceSynchronize();
//
//    kernel_position_to_image << < 1, 1 >> > (d_position_image_body, SEP, d_transition_single_world_lidar, res, xmin, ymax);
//    cudaDeviceSynchronize();
//}
//
//void exec_map_extend(int& xmin, int& xmax, int& ymin, int& ymax, float res, int LIDAR_COORDS_LEN, int& GRID_WIDTH, int& GRID_HEIGHT) {
//
//    int xmin_pre = xmin;
//    int ymax_pre = ymax;
//
//    sz_should_extend = 4 * sizeof(int);
//
//    gpuErrchk(cudaMalloc((void**)&d_should_extend, sz_should_extend));
//    res_should_extend = (int*)malloc(sz_should_extend);
//    memset(res_should_extend, 0, sz_should_extend);
//    gpuErrchk(cudaMemset(d_should_extend, 0, sz_should_extend));
//
//    threadsPerBlock = 256;
//    blocksPerGrid = (LIDAR_COORDS_LEN + threadsPerBlock - 1) / threadsPerBlock;
//    kernel_check_map_extend_less << <blocksPerGrid, threadsPerBlock >> > (d_should_extend, SEP, d_particles_world_x, xmin, 0, LIDAR_COORDS_LEN);
//    kernel_check_map_extend_less << <blocksPerGrid, threadsPerBlock >> > (d_should_extend, SEP, d_particles_world_y, ymin, 1, LIDAR_COORDS_LEN);
//
//    kernel_check_map_extend_greater << <blocksPerGrid, threadsPerBlock >> > (d_should_extend, SEP, d_particles_world_x, xmax, 2, LIDAR_COORDS_LEN);
//    kernel_check_map_extend_greater << <blocksPerGrid, threadsPerBlock >> > (d_should_extend, SEP, d_particles_world_y, ymax, 3, LIDAR_COORDS_LEN);
//    cudaDeviceSynchronize();
//    gpuErrchk(cudaMemcpy(res_should_extend, d_should_extend, sz_should_extend, cudaMemcpyDeviceToHost));
//
//    bool EXTEND = false;
//    if (res_should_extend[0] != 0) {
//        EXTEND = true;
//        xmin = xmin * 2;
//    }
//    else if (res_should_extend[2] != 0) {
//        EXTEND = true;
//        xmax = xmax * 2;
//    }
//    else if (res_should_extend[1] != 0) {
//        EXTEND = true;
//        ymin = ymin * 2;
//    }
//    else if (res_should_extend[3] != 0) {
//        EXTEND = true;
//        ymax = ymax * 2;
//    }
//
//    //printf("EXTEND = %d\n", EXTEND);
//
//    if (EXTEND == true) {
//
//        map_size_changed = true;
//
//        sz_coord = 2 * sizeof(int);
//        gpuErrchk(cudaMalloc((void**)&d_coord, sz_coord));
//        res_coord = (int*)malloc(sz_coord);
//
//        kernel_position_to_image << <1, 1 >> > (d_coord, SEP, xmin_pre, ymax_pre, res, xmin, ymax);
//        cudaDeviceSynchronize();
//
//        gpuErrchk(cudaMemcpy(res_coord, d_coord, sz_coord, cudaMemcpyDeviceToHost));
//
//        int* dc_grid_map = NULL;
//        gpuErrchk(cudaMalloc((void**)&dc_grid_map, sz_grid_map));
//        gpuErrchk(cudaMemcpy(dc_grid_map, d_grid_map, sz_grid_map, cudaMemcpyDeviceToDevice));
//
//        float* dc_log_odds = NULL;
//        gpuErrchk(cudaMalloc((void**)&dc_log_odds, sz_log_odds));
//        gpuErrchk(cudaMemcpy(dc_log_odds, d_log_odds, sz_log_odds, cudaMemcpyDeviceToDevice));
//
//        const int PRE_GRID_WIDTH = GRID_WIDTH;
//        const int PRE_GRID_HEIGHT = GRID_HEIGHT;
//        GRID_WIDTH = ceil((ymax - ymin) / res + 1);
//        GRID_HEIGHT = ceil((xmax - xmin) / res + 1);
//        //printf("GRID_WIDTH=%d, GRID_HEIGHT=%d, PRE_GRID_WIDTH=%d, PRE_GRID_HEIGHT=%d\n", GRID_WIDTH, GRID_HEIGHT, PRE_GRID_WIDTH, PRE_GRID_HEIGHT);
//        //assert(GRID_WIDTH == AF_GRID_WIDTH);
//        //assert(GRID_HEIGHT == AF_GRID_HEIGHT);
//
//        const int PRE_GRID_SIZE = PRE_GRID_WIDTH * PRE_GRID_HEIGHT;
//        const int NEW_GRID_SIZE = GRID_WIDTH * GRID_HEIGHT;
//
//        //gpuErrchk(cudaFree(d_grid_map));
//
//        sz_grid_map = GRID_WIDTH * GRID_HEIGHT * sizeof(int);
//        gpuErrchk(cudaMalloc((void**)&d_grid_map, sz_grid_map));
//        gpuErrchk(cudaMemset(d_grid_map, 0, sz_grid_map));
//
//        sz_log_odds = GRID_WIDTH * GRID_HEIGHT * sizeof(float);
//        gpuErrchk(cudaMalloc((void**)&d_log_odds, sz_log_odds));
//
//        threadsPerBlock = 256;
//        blocksPerGrid = (NEW_GRID_SIZE + threadsPerBlock - 1) / threadsPerBlock;
//        kernel_index_arr_const << <blocksPerGrid, threadsPerBlock >> > (d_log_odds, LOG_ODD_PRIOR, NEW_GRID_SIZE);
//        cudaDeviceSynchronize();
//
//        res_grid_map = (int*)malloc(sz_grid_map);
//        res_log_odds = (float*)malloc(sz_log_odds);
//
//        threadsPerBlock = 256;
//        blocksPerGrid = (PRE_GRID_SIZE + threadsPerBlock - 1) / threadsPerBlock;
//        kernel_2d_copy_with_offset << <blocksPerGrid, threadsPerBlock >> > (d_grid_map, d_log_odds, SEP,
//            dc_grid_map, dc_log_odds, res_coord[0], res_coord[1],
//            PRE_GRID_HEIGHT, GRID_HEIGHT, PRE_GRID_SIZE);
//        cudaDeviceSynchronize();
//
//        sz_free_unique_counter_col = (GRID_WIDTH + 1) * sizeof(int);
//        gpuErrchk(cudaMalloc((void**)&d_free_unique_counter_col, sz_free_unique_counter_col));
//        res_free_unique_counter_col = (int*)malloc(sz_free_unique_counter_col);
//
//        sz_free_map_2d = GRID_WIDTH * GRID_HEIGHT * sizeof(uint8_t);
//        gpuErrchk(cudaMalloc((void**)&d_free_map_2d, sz_free_map_2d));
//
//
//        sz_occupied_unique_counter_col = (GRID_WIDTH + 1) * sizeof(int);
//        gpuErrchk(cudaMalloc((void**)&d_occupied_unique_counter_col, sz_occupied_unique_counter_col));
//        res_occupied_unique_counter_col = (int*)malloc(sz_occupied_unique_counter_col);
//
//        sz_occupied_map_2d = GRID_WIDTH * GRID_HEIGHT * sizeof(uint8_t);
//        gpuErrchk(cudaMalloc((void**)&d_occupied_map_2d, sz_occupied_map_2d));
//
//        gpuErrchk(cudaMemcpy(res_grid_map, d_grid_map, sz_grid_map, cudaMemcpyDeviceToHost));
//        gpuErrchk(cudaMemcpy(res_log_odds, d_log_odds, sz_log_odds, cudaMemcpyDeviceToHost));
//    }
//}
//
//void exec_world_to_image_transform_step_2(int xmin, int ymax, float res, int LIDAR_COORDS_LEN) {
//
//    threadsPerBlock = 1;
//    blocksPerGrid = LIDAR_COORDS_LEN;
//    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (d_particles_occupied_x, d_particles_occupied_y, SEP,
//        d_transition_single_world_lidar, d_lidar_coords, res, xmin, ymax, LIDAR_COORDS_LEN);
//    cudaDeviceSynchronize();
//
//    kernel_position_to_image << < 1, 1 >> > (d_position_image_body, SEP, d_transition_single_world_lidar, res, xmin, ymax);
//    cudaDeviceSynchronize();
//}
//
//void exec_bresenham(int PARTICLES_OCCUPIED_LEN, int& PARTICLES_FREE_LEN, int PARTICLE_UNIQUE_COUNTER, int MAX_DIST_IN_MAP) {
//
//    threadsPerBlock = 256;
//    blocksPerGrid = (PARTICLES_OCCUPIED_LEN + threadsPerBlock - 1) / threadsPerBlock;
//    kernel_bresenham << <blocksPerGrid, threadsPerBlock >> > (d_particles_free_x_max, d_particles_free_y_max, d_particles_free_counter, SEP,
//        d_particles_occupied_x, d_particles_occupied_y, d_position_image_body,
//        PARTICLES_OCCUPIED_LEN, MAX_DIST_IN_MAP);
//    cudaDeviceSynchronize();
//    thrust::exclusive_scan(thrust::device, d_particles_free_counter, d_particles_free_counter + PARTICLE_UNIQUE_COUNTER, d_particles_free_counter, 0);
//
//    gpuErrchk(cudaMemcpy(res_particles_free_counter, d_particles_free_counter, sz_particles_free_counter, cudaMemcpyDeviceToHost));
//    gpuErrchk(cudaMemcpy(d_particles_free_idx, d_particles_free_counter, sz_particles_free_idx, cudaMemcpyDeviceToDevice));
//
//    PARTICLES_FREE_LEN = res_particles_free_counter[PARTICLE_UNIQUE_COUNTER - 1];
//    sz_particles_free_pos = PARTICLES_FREE_LEN * sizeof(int);
//    gpuErrchk(cudaMalloc((void**)&d_particles_free_x, sz_particles_free_pos));
//    gpuErrchk(cudaMalloc((void**)&d_particles_free_y, sz_particles_free_pos));
//
//    kernel_bresenham_rearrange << <blocksPerGrid, threadsPerBlock >> > (d_particles_free_x, d_particles_free_y, SEP,
//        d_particles_free_x_max, d_particles_free_y_max,
//        d_particles_free_counter, MAX_DIST_IN_MAP, PARTICLES_OCCUPIED_LEN);
//    cudaDeviceSynchronize();
//}
//
//void reinit_map_idx_vars(int PARTICLES_OCCUPIED_LEN, int PARTICLES_FREE_LEN) {
//
//    h_occupied_map_idx[1] = PARTICLES_OCCUPIED_LEN;
//    h_free_map_idx[1] = PARTICLES_FREE_LEN;
//
//    gpuErrchk(cudaMemcpy(d_occupied_map_idx, h_occupied_map_idx, sz_occupied_map_idx, cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemcpy(d_free_map_idx, h_free_map_idx, sz_free_map_idx, cudaMemcpyHostToDevice));
//}
//
//void exec_create_map(int PARTICLES_OCCUPIED_LEN, int PARTICLES_FREE_LEN, int GRID_WIDTH, int GRID_HEIGHT) {
//
//    threadsPerBlock = 256;
//    blocksPerGrid = 1;
//    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (d_occupied_map_2d, SEP,
//        d_particles_occupied_x, d_particles_occupied_y, d_occupied_map_idx,
//        PARTICLES_OCCUPIED_LEN, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
//    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (d_free_map_2d, SEP,
//        d_particles_free_x, d_particles_free_y, d_free_map_idx,
//        PARTICLES_FREE_LEN, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
//    cudaDeviceSynchronize();
//
//    threadsPerBlock = GRID_WIDTH;
//    blocksPerGrid = 1;
//    kernel_2d_map_counter << <blocksPerGrid, threadsPerBlock >> > (d_occupied_unique_counter, d_occupied_unique_counter_col, SEP,
//        d_occupied_map_2d, GRID_WIDTH, GRID_HEIGHT);
//    kernel_2d_map_counter << <blocksPerGrid, threadsPerBlock >> > (d_free_unique_counter, d_free_unique_counter_col, SEP,
//        d_free_map_2d, GRID_WIDTH, GRID_HEIGHT);
//    cudaDeviceSynchronize();
//
//    threadsPerBlock = 1;
//    blocksPerGrid = 1;
//    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (d_occupied_unique_counter_col, GRID_WIDTH);
//    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (d_free_unique_counter_col, GRID_WIDTH);
//    cudaDeviceSynchronize();
//
//    gpuErrchk(cudaMemcpy(res_occupied_unique_counter, d_occupied_unique_counter, sz_occupied_unique_counter, cudaMemcpyDeviceToHost));
//    gpuErrchk(cudaMemcpy(res_free_unique_counter, d_free_unique_counter, sz_free_unique_counter, cudaMemcpyDeviceToHost));
//}
//
//void reinit_map_vars(int& PARTICLES_OCCUPIED_UNIQUE_LEN, int& PARTICLES_FREE_UNIQUE_LEN, int GRID_WIDTH, int GRID_HEIGHT) {
//
//    PARTICLES_OCCUPIED_UNIQUE_LEN = res_occupied_unique_counter[0];
//    PARTICLES_FREE_UNIQUE_LEN = res_free_unique_counter[0];
//
//    //gpuErrchk(cudaFree(d_particles_occupied_x));
//    //gpuErrchk(cudaFree(d_particles_occupied_y));
//    //gpuErrchk(cudaFree(d_particles_free_x));
//    //gpuErrchk(cudaFree(d_particles_free_y));
//
//    sz_particles_occupied_pos = PARTICLES_OCCUPIED_UNIQUE_LEN * sizeof(int);
//    sz_particles_free_pos = PARTICLES_FREE_UNIQUE_LEN * sizeof(int);
//
//    gpuErrchk(cudaMalloc((void**)&d_particles_occupied_x, sz_particles_occupied_pos));
//    gpuErrchk(cudaMalloc((void**)&d_particles_occupied_y, sz_particles_occupied_pos));
//    gpuErrchk(cudaMalloc((void**)&d_particles_free_x, sz_particles_free_pos));
//    gpuErrchk(cudaMalloc((void**)&d_particles_free_y, sz_particles_free_pos));
//
//    threadsPerBlock = GRID_WIDTH;
//    blocksPerGrid = 1;
//    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (d_particles_occupied_x, d_particles_occupied_y, SEP,
//        d_occupied_map_2d, d_occupied_unique_counter_col, GRID_WIDTH, GRID_HEIGHT);
//    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (d_particles_free_x, d_particles_free_y, SEP,
//        d_free_map_2d, d_free_unique_counter_col, GRID_WIDTH, GRID_HEIGHT);
//    cudaDeviceSynchronize();
//}
//
//void exec_log_odds(float log_t, int PARTICLES_OCCUPIED_UNIQUE_LEN, int PARTICLES_FREE_UNIQUE_LEN,
//    int GRID_WIDTH, int GRID_HEIGHT) {
//
//    threadsPerBlock = 256;
//    blocksPerGrid = (PARTICLES_OCCUPIED_UNIQUE_LEN + threadsPerBlock - 1) / threadsPerBlock;
//    kernel_update_log_odds << <blocksPerGrid, threadsPerBlock >> > (d_log_odds, SEP,
//        d_particles_occupied_x, d_particles_occupied_y,
//        2 * log_t, GRID_WIDTH, GRID_HEIGHT, PARTICLES_OCCUPIED_UNIQUE_LEN);
//    cudaDeviceSynchronize();
//
//    threadsPerBlock = 256;
//    blocksPerGrid = (PARTICLES_FREE_UNIQUE_LEN + threadsPerBlock - 1) / threadsPerBlock;
//    kernel_update_log_odds << <blocksPerGrid, threadsPerBlock >> > (d_log_odds, SEP,
//        d_particles_free_x, d_particles_free_y, (-1) * log_t, GRID_WIDTH, GRID_HEIGHT, PARTICLES_FREE_UNIQUE_LEN);
//    cudaDeviceSynchronize();
//
//    threadsPerBlock = 256;
//    blocksPerGrid = ((GRID_WIDTH * GRID_HEIGHT) + threadsPerBlock - 1) / threadsPerBlock;
//    kernel_update_map << <blocksPerGrid, threadsPerBlock >> > (d_grid_map, SEP,
//        d_log_odds, LOG_ODD_PRIOR, WALL, FREE, GRID_WIDTH * GRID_HEIGHT);
//    cudaDeviceSynchronize();
//}
//
//void assert_map_results(float* h_log_odds, int* h_grid_map,
//    float* h_post_log_odds, int* h_post_grid_map, int PARTICLES_FREE_LEN,
//    int PARTICLES_OCCUPIED_UNIQUE_LEN, int PARTICLES_FREE_UNIQUE_LEN, int GRID_WIDTH, int GRID_HEIGHT) {
//
//    printf("\n");
//    printf("--> Occupied Unique: \t\t%d\n", PARTICLES_OCCUPIED_UNIQUE_LEN);
//    printf("--> Free Unique: \t\t%d\n", PARTICLES_FREE_UNIQUE_LEN);
//    printf("~~$ PARTICLES_FREE_LEN: \t%d\n", PARTICLES_FREE_LEN);
//    printf("~~$ sz_log_odds: \t\t%d\n", sz_log_odds / sizeof(float));
//
//    gpuErrchk(cudaMemcpy(res_log_odds, d_log_odds, sz_log_odds, cudaMemcpyDeviceToHost));
//    gpuErrchk(cudaMemcpy(res_grid_map, d_grid_map, sz_grid_map, cudaMemcpyDeviceToHost));
//
//    ASSERT_log_odds(res_log_odds, h_log_odds, h_post_log_odds, (GRID_WIDTH * GRID_HEIGHT), true);
//    ASSERT_log_odds_maps(res_grid_map, h_grid_map, h_post_grid_map, (GRID_WIDTH * GRID_HEIGHT), true);
//
//    printf("\n~~$ Verification All Passed\n");
//}


///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

//void alloc_init_movement_vars(float* h_rnds_encoder_counts, float* h_rnds_yaws, bool should_mem_allocate) {
//
//    if (should_mem_allocate == true) {
//
//        gpuErrchk(cudaMalloc((void**)&d_rnds_encoder_counts, sz_states_pos));
//        gpuErrchk(cudaMalloc((void**)&d_rnds_yaws, sz_states_pos));
//    }
//
//    gpuErrchk(cudaMemcpy(d_rnds_encoder_counts, h_rnds_encoder_counts, sz_states_pos, cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemcpy(d_rnds_yaws, h_rnds_yaws, sz_states_pos, cudaMemcpyHostToDevice));
//}
//
//void exec_robot_move(float encoder_counts, float yaw, float dt, float nv, float nw) {
//
//    int threadsPerBlock = NUM_PARTICLES;
//    int blocksPerGrid = 1;
//    kernel_robot_advance << <blocksPerGrid, threadsPerBlock >> > (d_states_x, d_states_y, d_states_theta, SEP,
//        d_rnds_encoder_counts, d_rnds_yaws,
//        encoder_counts, yaw, dt, nv, nw, NUM_PARTICLES);
//    cudaDeviceSynchronize();
//}
//
//void assert_robot_move_results(float* post_states_x, float* post_states_y, float* post_states_theta) {
//
//    res_states_x = (float*)malloc(sz_states_pos);
//    res_states_y = (float*)malloc(sz_states_pos);
//    res_states_theta = (float*)malloc(sz_states_pos);
//
//    gpuErrchk(cudaMemcpy(res_states_x, d_states_x, sz_states_pos, cudaMemcpyDeviceToHost));
//    gpuErrchk(cudaMemcpy(res_states_y, d_states_y, sz_states_pos, cudaMemcpyDeviceToHost));
//    gpuErrchk(cudaMemcpy(res_states_theta, d_states_theta, sz_states_pos, cudaMemcpyDeviceToHost));
//
//    for (int i = 0; i < NUM_PARTICLES; i++) {
//
//        if (abs(res_states_x[i] - post_states_x[i]) > 1e-4)
//            printf("i=%d, x=%f, %f\n", i, res_states_x[i], post_states_x[i]);
//        if (abs(res_states_y[i] - post_states_y[i]) > 1e-4)
//            printf("i=%d, y=%f, %f\n", i, res_states_y[i], post_states_y[i]);
//        if (abs(res_states_theta[i] - post_states_theta[i]) > 1e-4)
//            printf("i=%d, theta=%f, %f\n", i, res_states_theta[i], post_states_theta[i]);
//    }
//    printf("~~$ Robot Move Check Finished\n");
//}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

void test_setup() {

    const int data_len = 6;
    int data[data_len] = { 1, 0, 2, 2, 1, 3 };
    int* d_data = NULL;
    gpuErrchk(cudaMalloc((void**)&d_data, data_len * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_data, data, data_len * sizeof(int), cudaMemcpyHostToDevice));

    thrust::exclusive_scan(thrust::host, data, data + 6, data, 0);
    thrust::exclusive_scan(thrust::device, d_data, d_data + 6, d_data, 0);
}


void test_alloc_init_robot(DeviceState& d_state, DeviceState& d_clone_state, DeviceMeasurements& d_measurements,
    DeviceMap& d_map, DeviceRobotParticles& d_robot_particles, DeviceCorrelation& d_correlation, DeviceParticlesTransition& d_particles_transition,
    DeviceParticlesPosition& d_particles_position, DeviceParticlesRotation& d_particles_rotation, DeviceTransition& d_transition,
    DeviceProcessedMeasure& d_processed_measure, Device2DUniqueFinder& d_2d_unique, DeviceResampling& d_resampling,
    HostState& h_state, HostRobotState& h_robot_state, HostMeasurements& h_measurements, HostMap& h_map, HostRobotParticles& h_robot_particles,
    HostCorrelation& h_correlation, HostParticlesTransition& h_particles_transition, HostParticlesPosition& h_particles_position,
    HostParticlesRotation& h_particles_rotation, HostProcessedMeasure& h_processed_measure, Host2DUniqueFinder& h_2d_unique, HostResampling& h_resampling,
    HostState& pre_state, HostMeasurements& pre_measurements, HostMap& pre_map, HostRobotParticles& pre_robot_particles, HostResampling& pre_resampling) {

#ifdef VERBOSE_BANNER
    printf("/****************** ALLOCATIONS & INITIALIZATIONS  ******************/\n");
#endif

#ifdef VERBOSE_TOTAL_INFO
    printf("~~$ GRID_WIDTH: \t\t%d\n", GRID_WIDTH);
    printf("~~$ GRID_HEIGHT: \t\t%d\n", GRID_HEIGHT);

    printf("~~$ PARTICLES_OCCUPIED_LEN: \t%d\n", PARTICLES_OCCUPIED_LEN);
    printf("~~$ PARTICLE_UNIQUE_COUNTER: \t%d\n", PARTICLE_UNIQUE_COUNTER);
    printf("~~$ MAX_DIST_IN_MAP: \t\t%d\n", MAX_DIST_IN_MAP);
    printf("~~$ LIDAR_COORDS_LEN: \t\t%d\n", LIDAR_COORDS_LEN);
    printf("~~$ PARTICLES_ITEMS_LEN: \t%d\n", PARTICLES_ITEMS_LEN);
#endif

    auto start_robot_particles_alloc = std::chrono::high_resolution_clock::now();
    alloc_init_state_vars(d_state, d_clone_state, h_state, h_robot_state, pre_state);
    alloc_init_measurement_vars(d_measurements, h_measurements, pre_measurements);
    alloc_init_map_vars(d_map, h_map, pre_map);
    alloc_init_robot_particles_vars(d_robot_particles, h_robot_particles, pre_robot_particles);
    alloc_correlation_vars(d_correlation, h_correlation);
    alloc_particles_transition_vars(d_particles_transition, d_particles_position, d_particles_rotation,
        h_particles_transition, h_particles_position, h_particles_rotation);
    alloc_init_body_lidar(d_transition);
    alloc_init_processed_measurement_vars(d_processed_measure, h_processed_measure, h_measurements);
    alloc_map_2d_var(d_2d_unique, h_2d_unique, h_map, true);
    alloc_resampling_vars(d_resampling, h_resampling, pre_resampling);
    auto stop_robot_particles_alloc = std::chrono::high_resolution_clock::now();


#ifdef VERBOSE_EXECUTION_TIME
    auto duration_robot_particles_alloc = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_particles_alloc - start_robot_particles_alloc);
    auto duration_mapping_alloc = std::chrono::duration_cast<std::chrono::microseconds>(stop_mapping_alloc - start_mapping_alloc);

    std::cout << "Time taken by function (Particles Allocation): " << duration_robot_particles_alloc.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Mapping Allocation): " << duration_mapping_alloc.count() << " microseconds" << std::endl;
    std::cout << std::endl;
#endif
}

void test_alloc_init_map(DevicePosition& d_position, DeviceTransition& d_transition, DeviceParticles& d_particles,
    Device2DUniqueFinder& d_unique_occupied, Device2DUniqueFinder& d_unique_free,
    HostMap& h_map, HostPosition& h_position, HostTransition& h_transition, HostParticles& h_particles,
    HostMeasurements& h_measurements, Host2DUniqueFinder& h_unique_occupied, Host2DUniqueFinder& h_unique_free,
    HostTransition& pre_transition, HostMap& pre_map, HostParticles& pre_particles,
    host_vector<int>& hvec_occupied_map_idx, host_vector<int>& hvec_free_map_idx) {

    hvec_occupied_map_idx.resize(2, 0);
    hvec_free_map_idx.resize(2, 0);

    auto start_mapping_alloc = std::chrono::high_resolution_clock::now();
    alloc_init_transition_vars(d_position, d_transition, h_position, h_transition, pre_transition);
    int MAX_DIST_IN_MAP = sqrt(pow(pre_map.GRID_WIDTH, 2) + pow(pre_map.GRID_HEIGHT, 2));
    alloc_init_particles_vars(d_particles, h_particles, h_measurements, pre_particles, MAX_DIST_IN_MAP);
    hvec_occupied_map_idx[1] = h_particles.PARTICLES_OCCUPIED_LEN;
    hvec_free_map_idx[1] = 0;
    alloc_init_unique_map_vars(d_unique_occupied, h_unique_occupied, h_map, hvec_occupied_map_idx);
    alloc_init_unique_map_vars(d_unique_free, h_unique_free, h_map, hvec_free_map_idx);
    auto stop_mapping_alloc = std::chrono::high_resolution_clock::now();

#ifdef VERBOSE_EXECUTION_TIME
    auto duration_mapping_alloc = std::chrono::duration_cast<std::chrono::microseconds>(stop_mapping_alloc - start_mapping_alloc);
    std::cout << "Time taken by function (Mapping Allocation): " << duration_mapping_alloc.count() << " microseconds" << std::endl;
    std::cout << std::endl;
#endif
}


void test_robot(DeviceState& d_state, DeviceState& d_clone_state, DeviceMap& d_map, DeviceTransition& d_transition, DeviceMeasurements& d_measurements, 
    DeviceProcessedMeasure& d_processed_measure, DeviceCorrelation& d_correlation, DeviceResampling& d_resampling,
    DeviceParticlesTransition& d_particles_transition, Device2DUniqueFinder& d_2d_unique, 
    DeviceRobotParticles& d_robot_particles, DeviceRobotParticles& d_clone_robot_particles,
    HostMap& h_map, HostMeasurements& h_measurements, HostParticlesTransition& h_particles_transition, HostResampling& h_resampling,
    HostRobotParticles& h_robot_particles, HostRobotParticles& h_clone_robot_particles, HostProcessedMeasure& h_processed_measure,
    Host2DUniqueFinder& h_2d_unique, HostCorrelation& h_correlation, HostState& h_state, HostRobotState& h_robot_state,
    HostRobotParticles& pre_robot_particles, HostRobotParticles& post_resampling_robot_particles, HostState& post_robot_move_state,
    HostResampling& pre_resampling, HostProcessedMeasure& post_processed_measure, HostRobotParticles& post_unique_robot_particles,
    HostState& post_state, host_vector<float>& pre_weights, host_vector<float>& post_loop_weights,
    GeneralInfo& general_info, bool should_assert) {

#ifdef VERBOSE_BANNER
    printf("/****************************** ROBOT *******************************/\n");
#endif

    const int MEASURE_LEN = NUM_PARTICLES * h_measurements.LIDAR_COORDS_LEN;
    int* h_last_len = (int*)malloc(sizeof(int));

    int negative_before_counter = 0;
    int count_bigger_than_height = 0;
    int negative_after_counter = 0;

    if (should_assert == true) {
        negative_before_counter = getNegativeCounter(pre_robot_particles.x.data(), pre_robot_particles.y.data(), pre_robot_particles.LEN);
        count_bigger_than_height = getGreaterThanCounter(pre_robot_particles.y.data(), h_map.GRID_HEIGHT, pre_robot_particles.LEN);
        negative_after_counter = getNegativeCounter(post_resampling_robot_particles.x.data(), post_resampling_robot_particles.y.data(), post_resampling_robot_particles.LEN);
    }


#ifdef VERBOSE_BORDER_LINE_COUNTER
    printf("~~$ negative_before_counter: \t%d\n", negative_before_counter);
    printf("~~$ count_bigger_than_height: \t%d\n", count_bigger_than_height);
#endif

    auto start_robot_particles_kernel = std::chrono::high_resolution_clock::now();
    exec_robot_move(d_state, h_state);
    if (should_assert == true) assert_robot_move_results(d_state, h_state, post_robot_move_state);

    exec_calc_transition(d_particles_transition, d_state, d_transition, h_particles_transition);
    exec_process_measurements(d_processed_measure, d_particles_transition, d_measurements, h_map, h_measurements, general_info);
    if (should_assert == true)
        assert_processed_measures(d_particles_transition, d_processed_measure, h_particles_transition,
            h_measurements, h_processed_measure, post_processed_measure);

    exec_create_2d_map(d_2d_unique, d_robot_particles, h_map, h_robot_particles);
    if (should_assert == true) assert_create_2d_map(d_2d_unique, h_2d_unique, h_map, h_robot_particles, negative_before_counter);

    exec_update_map(d_2d_unique, d_processed_measure, h_map, MEASURE_LEN);
    exec_particle_unique_cum_sum(d_2d_unique, h_map, h_2d_unique, h_robot_particles);
    if (should_assert == true) assert_particles_unique(h_robot_particles, post_unique_robot_particles, negative_after_counter);

    reinit_map_vars(d_robot_particles, h_robot_particles);
    exec_map_restructure(d_robot_particles, d_2d_unique, h_map);
    if (should_assert == true) assert_particles_unique(d_robot_particles, h_robot_particles, post_unique_robot_particles, negative_after_counter);

    exec_index_expansion(d_robot_particles, h_robot_particles);
    exec_correlation(d_map, d_robot_particles, d_correlation, h_map, h_robot_particles);
    if (should_assert == true) assert_correlation(d_correlation, d_robot_particles, h_correlation, h_robot_particles, pre_weights);

    exec_update_weights(d_robot_particles, d_correlation, h_robot_particles, h_correlation);
    if (should_assert == true) assert_update_weights(d_correlation, d_robot_particles, h_correlation, h_robot_particles, post_loop_weights);

    exec_resampling(d_correlation, d_resampling);
    reinit_particles_vars(d_state, d_robot_particles, d_resampling, d_clone_robot_particles, d_clone_state, h_robot_particles, h_state, h_last_len);
    if (should_assert == true) assert_resampling(d_resampling, h_resampling, pre_resampling, post_robot_move_state, post_state);

    exec_rearrangement(d_robot_particles, d_state, d_resampling, d_clone_robot_particles, d_clone_state, h_map,
        h_robot_particles, h_clone_robot_particles, h_last_len);
    exec_update_states(d_state, h_state, h_robot_state);
    auto stop_robot_particles_kernel = std::chrono::high_resolution_clock::now();

#ifdef VERBOSE_EXECUTION_TIME
    auto duration_robot_particles_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_particles_kernel - start_robot_particles_kernel);

    std::cout << std::endl;
    std::cout << "Time taken by function (Kernel): " << duration_robot_particles_kernel.count() << " microseconds" << std::endl;
    std::cout << std::endl;
#endif
}

void test_map(DevicePosition& d_position, DeviceTransition& d_transition, DeviceParticles& d_particles,
    DeviceMeasurements& d_measurements, DeviceMap& d_map, 
    HostMeasurements& h_measurements, HostMap& h_map, HostPosition& h_position, HostTransition& h_transition,
    Device2DUniqueFinder& d_unique_occupied, Device2DUniqueFinder& d_unique_free,
    Host2DUniqueFinder& h_unique_occupied, Host2DUniqueFinder& h_unique_free,
    HostMap& pre_map, HostMap& post_bg_map, HostMap& post_map, HostParticles& post_particles, HostPosition& post_position, HostTransition& post_transition,
    HostParticles& h_particles, GeneralInfo& general_info, bool should_assert) {

#ifdef VERBOSE_BANNER
    printf("/****************************** MAP MAIN ****************************/\n");
#endif

    auto start_mapping_kernel = std::chrono::high_resolution_clock::now();

    exec_world_to_image_transform_step_1(d_position, d_transition, d_particles, d_measurements, h_measurements);

    bool EXTEND = false;
    exec_map_extend(d_map, d_measurements, d_particles, d_unique_occupied, d_unique_free,
        h_map, h_measurements, h_unique_occupied, h_unique_free, general_info, EXTEND);
    if (should_assert == true) assert_map_extend(h_map, pre_map, post_bg_map, post_map, EXTEND);

    exec_world_to_image_transform_step_2(d_measurements, d_particles, d_position, d_transition,
        h_map, h_measurements, general_info);
    if (should_assert == true)
        assert_world_to_image_transform(d_particles, d_position, d_transition,
            h_measurements, h_particles, h_position, h_transition, post_particles, post_position, post_transition);

    int MAX_DIST_IN_MAP = sqrt(pow(h_map.GRID_WIDTH, 2) + pow(h_map.GRID_HEIGHT, 2));

    exec_bresenham(d_particles, d_position, d_transition, h_particles, MAX_DIST_IN_MAP);
    if (should_assert == true) assert_bresenham(d_particles, h_particles, h_measurements, d_measurements, post_particles);

    reinit_map_idx_vars(d_unique_free, h_particles, h_unique_free);
    exec_create_map(d_particles, d_unique_occupied, d_unique_free, h_map, h_particles);
    
    reinit_map_vars(d_particles, d_unique_occupied, d_unique_free, h_particles, h_unique_occupied, h_unique_free);
    exec_map_restructure(d_particles, d_unique_occupied, d_unique_free, h_map);
    if (should_assert == true) assert_map_restructure(d_particles, h_particles, post_particles);

    exec_log_odds(d_map, d_particles, h_map, h_particles, general_info);
    if (should_assert == true) assert_log_odds(d_map, h_map, pre_map, post_map);
    
    auto stop_mapping_kernel = std::chrono::high_resolution_clock::now();

#ifdef VERBOSE_EXECUTION_TIME
    auto duration_mapping_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_mapping_kernel - start_mapping_kernel);

    std::cout << std::endl;
    std::cout << "Time taken by function (Mapping Kernel): " << duration_mapping_kernel.count() << " microseconds" << std::endl;
    std::cout << std::endl;
#endif
}



void resetMiddleVariables(DeviceCorrelation& d_correlation, DeviceProcessedMeasure& d_processed_measure, DeviceResampling& d_resampling,
    Device2DUniqueFinder& d_2d_unique, DeviceRobotParticles& d_robot_particles, 
    HostMap& h_map, HostMeasurements& h_measurements, HostProcessedMeasure& h_processed_measure) {

    int num_items = NUM_PARTICLES * h_measurements.LIDAR_COORDS_LEN;
    d_processed_measure.x.clear();
    d_processed_measure.y.clear();
    d_processed_measure.idx.clear();
    d_processed_measure.x.resize(num_items, 0);
    d_processed_measure.y.resize(num_items, 0);
    d_processed_measure.idx.resize(num_items, 0);

    h_processed_measure.x.clear();
    h_processed_measure.y.clear();
    h_processed_measure.idx.clear();
    h_processed_measure.x.resize(num_items, 0);
    h_processed_measure.y.resize(num_items, 0);
    h_processed_measure.idx.resize(num_items, 0);

    thrust::fill(d_2d_unique.map.begin(), d_2d_unique.map.end(), 0);
    thrust::fill(d_2d_unique.in_map.begin(), d_2d_unique.in_map.end(), 0);
    thrust::fill(d_2d_unique.in_col.begin(), d_2d_unique.in_col.end(), 0);

    thrust::fill(d_correlation.raw.begin(), d_correlation.raw.end(), 0);
    thrust::fill(d_correlation.sum_exp.begin(), d_correlation.sum_exp.end(), 0);
    thrust::fill(d_correlation.max.begin(), d_correlation.max.end(), 0);

    thrust::fill(d_resampling.js.begin(), d_resampling.js.end(), 0);
    thrust::fill(d_robot_particles.weight.begin(), d_robot_particles.weight.end(), 0);
}


void test_iterations() {

    vector<vector<float>> vec_arr_rnds_encoder_counts;
    vector<vector<float>> vec_arr_lidar_coords;
    vector<vector<float>> vec_arr_rnds;
    vector<vector<float>> vec_arr_transition;
    vector<vector<float>> vec_arr_rnds_yaws;

    vector<float> vec_encoder_counts;
    vector<float> vec_yaws;
    vector<float> vec_dt;

    printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
    printf("Reading Data Files\n");
    printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");

    const int INPUT_VEC_SIZE = 800;
    read_small_steps_vec("encoder_counts", vec_encoder_counts, INPUT_VEC_SIZE);
    read_small_steps_vec_arr("rnds_encoder_counts", vec_arr_rnds_encoder_counts, INPUT_VEC_SIZE);
    read_small_steps_vec_arr("lidar_coords", vec_arr_lidar_coords, INPUT_VEC_SIZE);
    read_small_steps_vec_arr("rnds", vec_arr_rnds, INPUT_VEC_SIZE);
    read_small_steps_vec_arr("transition", vec_arr_transition, INPUT_VEC_SIZE);
    read_small_steps_vec("yaws", vec_yaws, INPUT_VEC_SIZE);
    read_small_steps_vec_arr("rnds_yaws", vec_arr_rnds_yaws, INPUT_VEC_SIZE);
    read_small_steps_vec("dt", vec_dt, INPUT_VEC_SIZE);

    bool should_assert = false;

    HostState pre_state;
    HostState post_robot_move_state;
    HostState post_state;
    HostRobotParticles pre_robot_particles;
    HostRobotParticles post_unique_robot_particles;
    HostRobotParticles pre_resampling_robot_particles;
    HostRobotParticles post_resampling_robot_particles;
    HostProcessedMeasure post_processed_measure;
    HostParticlesTransition post_particles_transition;
    HostResampling pre_resampling;
    HostRobotState post_robot_state;
    HostMap pre_map;
    HostMap post_bg_map;
    HostMap post_map;
    HostMeasurements pre_measurements;
    HostPosition post_position;
    HostTransition pre_transition;
    HostTransition post_transition;
    HostParticles pre_particles;
    HostParticles post_particles;
    GeneralInfo general_info;

    DeviceState d_state;
    DeviceState d_clone_state;
    DeviceMeasurements d_measurements;
    DeviceMap d_map;
    DeviceRobotParticles d_robot_particles;
    DeviceRobotParticles d_clone_robot_particles;
    DeviceCorrelation d_correlation;
    DeviceParticlesTransition d_particles_transition;
    DeviceParticlesPosition d_particles_position;
    DeviceParticlesRotation d_particles_rotation;
    DeviceProcessedMeasure d_processed_measure;
    Device2DUniqueFinder d_2d_unique;
    DeviceResampling d_resampling;
    DeviceTransition d_transition;
    DevicePosition d_position;
    DeviceParticles d_particles;
    Device2DUniqueFinder d_unique_occupied;
    Device2DUniqueFinder d_unique_free;


    HostState h_state;
    HostRobotState h_robot_state;
    HostMeasurements h_measurements;
    HostMap h_map;
    HostRobotParticles h_robot_particles;
    HostRobotParticles h_clone_robot_particles;
    HostCorrelation h_correlation;
    HostParticlesTransition h_particles_transition;
    HostParticlesPosition h_particles_position;
    HostParticlesRotation h_particles_rotation;
    HostProcessedMeasure h_processed_measure;
    Host2DUniqueFinder h_2d_unique;
    HostResampling h_resampling;
    HostPosition h_position;
    HostTransition h_transition;
    HostParticles h_particles;
    Host2DUniqueFinder h_unique_occupied;
    Host2DUniqueFinder h_unique_free;

    host_vector<float> pre_weights;
    host_vector<float> post_loop_weights;
    host_vector<float> post_weights;

    host_vector<int> hvec_occupied_map_idx; 
    host_vector<int> hvec_free_map_idx;

    const int LOOP_LEN = 501;
    const int ST_FILE_NUMBER = 2800;
    const int CHECK_STEP = 500;
    const int DIFF_FROM_START = ST_FILE_NUMBER - 100;

    for (int file_number = ST_FILE_NUMBER; file_number < ST_FILE_NUMBER + LOOP_LEN; file_number += 1) {

        Host2DUniqueFinder h_unique_occupied;
        Host2DUniqueFinder h_unique_free;
        HostProcessedMeasure h_processed_measure;

        Device2DUniqueFinder d_unique_occupied;
        Device2DUniqueFinder d_unique_free;
        DeviceProcessedMeasure d_processed_measure;

        //auto start_run_step = std::chrono::high_resolution_clock::now();
        should_assert = (file_number % CHECK_STEP == 0);

        if (file_number == ST_FILE_NUMBER) {

            printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
            printf("Iteration: %d\n", file_number);
            printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");

            auto start_read_file = std::chrono::high_resolution_clock::now();
            read_iteration(file_number, pre_state, post_robot_move_state, post_state,
                pre_robot_particles, post_unique_robot_particles,
                pre_resampling_robot_particles, post_resampling_robot_particles,
                post_processed_measure, post_particles_transition,
                pre_resampling, post_robot_state,
                pre_map, post_bg_map, post_map, pre_measurements,
                post_position, pre_transition, post_transition,
                pre_particles, post_particles, general_info,
                pre_weights, post_loop_weights, post_weights);
            auto stop_read_file = std::chrono::high_resolution_clock::now();

            test_alloc_init_robot(d_state, d_clone_state, d_measurements,
                d_map, d_robot_particles, d_correlation, d_particles_transition,
                d_particles_position, d_particles_rotation, d_transition,
                d_processed_measure, d_2d_unique, d_resampling,
                h_state, h_robot_state, h_measurements, h_map, h_robot_particles,
                h_correlation, h_particles_transition, h_particles_position,
                h_particles_rotation, h_processed_measure, h_2d_unique, h_resampling,
                pre_state, pre_measurements, pre_map, pre_robot_particles, pre_resampling);

            test_alloc_init_map(d_position, d_transition, d_particles,
                d_unique_occupied, d_unique_free,
                h_map, h_position, h_transition, h_particles,
                h_measurements, h_unique_occupied, h_unique_free,
                pre_transition, pre_map, pre_particles,
                hvec_occupied_map_idx, hvec_free_map_idx);

            auto duration_read_file = std::chrono::duration_cast<std::chrono::microseconds>(stop_read_file - start_read_file);
            std::cout << std::endl;
            std::cout << "Time taken by function (Read Data Files): " << duration_read_file.count() << " microseconds" << std::endl;
            std::cout << std::endl;
        }
        else {

            //should_assert = (file_number % CHECK_STEP == 0);
            if (should_assert == true) {
                printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
                printf("Iteration: %d\n", file_number);
                printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
            }

            if (should_assert == true) {
                read_iteration(file_number, pre_state, post_robot_move_state, post_state,
                    pre_robot_particles, post_unique_robot_particles,
                    pre_resampling_robot_particles, post_resampling_robot_particles,
                    post_processed_measure, post_particles_transition,
                    pre_resampling, post_robot_state,
                    pre_map, post_bg_map, post_map, pre_measurements,
                    post_position, pre_transition, post_transition,
                    pre_particles, post_particles, general_info,
                    pre_weights, post_loop_weights, post_weights);
            }

            int curr_idx = file_number - ST_FILE_NUMBER + DIFF_FROM_START;

            auto start_alloc_init_step = std::chrono::high_resolution_clock::now();
            h_measurements.LIDAR_COORDS_LEN = vec_arr_lidar_coords[curr_idx].size() / 2;
            int MEASURE_LEN = NUM_PARTICLES * h_measurements.LIDAR_COORDS_LEN;
            h_particles.PARTICLES_OCCUPIED_LEN = h_measurements.LIDAR_COORDS_LEN;
            int PARTICLE_UNIQUE_COUNTER = h_particles.PARTICLES_OCCUPIED_LEN + 1;

            d_measurements.lidar_coords.resize(2 * h_measurements.LIDAR_COORDS_LEN);
            d_measurements.lidar_coords.assign(vec_arr_lidar_coords[curr_idx].begin(), vec_arr_lidar_coords[curr_idx].end());

            resetMiddleVariables(d_correlation, d_processed_measure, d_resampling, d_2d_unique, d_robot_particles, 
                h_map, h_measurements, h_processed_measure);


#if defined(GET_EXTRA_TRANSITION_WORLD_BODY)
            d_transition.single_world_body.assign(vec_arr_transition[curr_idx].begin(), vec_arr_transition[curr_idx].end());
#endif
            d_state.rnds_encoder_counts.assign(vec_arr_rnds_encoder_counts[curr_idx].begin(), vec_arr_rnds_encoder_counts[curr_idx].end());
            d_state.rnds_yaws.assign(vec_arr_rnds_yaws[curr_idx].begin(), vec_arr_rnds_yaws[curr_idx].end());
            h_state.rnds_encoder_counts.assign(vec_arr_rnds_encoder_counts[curr_idx].begin(), vec_arr_rnds_encoder_counts[curr_idx].end());
            h_state.rnds_yaws.assign(vec_arr_rnds_yaws[curr_idx].begin(), vec_arr_rnds_yaws[curr_idx].end());

            h_state.encoder_counts = vec_encoder_counts[curr_idx];
            h_state.yaw = vec_yaws[curr_idx];
            h_state.dt = vec_dt[curr_idx];
            h_state.nv = pre_state.nv;
            h_state.nw = pre_state.nw;
            
            d_resampling.js.resize(NUM_PARTICLES, 0);
            d_resampling.rnds.resize(NUM_PARTICLES, 0);
            d_resampling.rnds.assign(vec_arr_rnds[curr_idx].begin(), vec_arr_rnds[curr_idx].end());

            int MAX_DIST_IN_MAP = sqrt(pow(pre_map.GRID_WIDTH, 2) + pow(pre_map.GRID_HEIGHT, 2));
            alloc_init_particles_vars(d_particles, h_particles, h_measurements, h_particles, MAX_DIST_IN_MAP);
            hvec_occupied_map_idx[1] = h_particles.PARTICLES_OCCUPIED_LEN;
            hvec_free_map_idx[1] = 0;
            alloc_init_unique_map_vars(d_unique_occupied, h_unique_occupied, h_map, hvec_occupied_map_idx);
            alloc_init_unique_map_vars(d_unique_free, h_unique_free, h_map, hvec_free_map_idx);

            if (should_assert == true || (file_number - 1) % CHECK_STEP == 0) {
                if (should_assert == false) {
                    printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
                    printf("Iteration: %d\n", file_number);
                    printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
                }

                auto stop_alloc_init_step = std::chrono::high_resolution_clock::now();
                auto duration_alloc_init_step = std::chrono::duration_cast<std::chrono::microseconds>(stop_alloc_init_step - start_alloc_init_step);
                std::cout << std::endl;
                std::cout << "Time taken by function (Alloc & Init Step): " << duration_alloc_init_step.count() << " microseconds" << std::endl;
                std::cout << std::endl;
            }
            
            map_size_changed = false;
        }

        auto start_run_step = std::chrono::high_resolution_clock::now();
        test_robot(d_state, d_clone_state, d_map, d_transition, d_measurements, d_processed_measure, d_correlation, d_resampling,
            d_particles_transition, d_2d_unique, d_robot_particles, d_clone_robot_particles,
            h_map, h_measurements, h_particles_transition, h_resampling, h_robot_particles, h_clone_robot_particles, h_processed_measure,
            h_2d_unique, h_correlation, h_state, h_robot_state,
            pre_robot_particles, post_resampling_robot_particles, post_robot_move_state, pre_resampling, post_processed_measure, post_unique_robot_particles,
            post_state, pre_weights, post_loop_weights, general_info, should_assert);
        test_map(d_position, d_transition, d_particles, d_measurements, d_map,
            h_measurements, h_map, h_position, h_transition, d_unique_occupied, d_unique_free, h_unique_occupied, h_unique_free,
            pre_map, post_bg_map, post_map, post_particles, post_position, post_transition, h_particles, general_info, should_assert);

        if (should_assert == true || (file_number - 1) % CHECK_STEP == 0) {

            auto stop_run_step = std::chrono::high_resolution_clock::now();
            auto duration_run_step = std::chrono::duration_cast<std::chrono::microseconds>(stop_run_step - start_run_step);
            std::cout << std::endl;
            std::cout << "Time taken by function (Run Step): " << duration_run_step.count() << " microseconds" << std::endl;
            std::cout << std::endl;
        }
    }
}

#endif
