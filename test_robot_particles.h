#ifndef _TEST_ROBOT_PARTICLES_H_
#define _TEST_ROBOT_PARTICLES_H_

#include "headers.h"
#include "host_asserts.h"
#include "host_utils.h"
#include "kernels.cuh"
#include "kernels_robot.cuh"
#include "kernels_utils.cuh"

#include "data/update_func/1500.h"

void alloc_state_vars();
void alloc_particles_vars();
void alloc_particles_copy_vars();
void alloc_correlation_vars();
void alloc_transition_vars();
void alloc_map_vars();
void alloc_weights_vars();
void alloc_resampling_vars();

void init_state_vars();
void init_particles_vars();
void init_transition_vars();

void reinit_map_vars();
void reinit_particles_vars();

void exec_transition();
void exec_create_2d_map();
void exec_update_map();
void exec_cum_sum();
void exec_map_restructure();
void exec_index_expansion();
void exec_correlation();
void exec_update_weights();
void exec_resampling();
void exec_rearrangement();
void exec_update_states();

void assertResults();

int GRID_WIDTH = ST_GRID_WIDTH;
int GRID_HEIGHT = ST_GRID_HEIGHT;
int LIDAR_COORDS_LEN = ST_LIDAR_COORDS_LEN;
float res = ST_res;
int xmin = ST_xmin;
int ymax = ST_ymax;
int PARTICLES_ITEMS_LEN = ST_PARTICLES_ITEMS_LEN;
int C_PARTICLES_ITEMS_LEN = 0;


int UNIQUE_COUNTER_LEN = NUM_PARTICLES + 1;
int MEASURE_LEN = NUM_PARTICLES * LIDAR_COORDS_LEN;

int negative_before_counter = 0;
int count_bigger_than_height = 0;
int negative_after_counter = 0;


int threadsPerBlock = 1;
int blocksPerGrid = 1;

/********************************************************************/
/************************** STATES VARIABLES ************************/
/********************************************************************/
size_t sz_states_pos = 0;
size_t sz_lidar_coords = 0;

float* d_states_x = NULL;
float* d_states_y = NULL;
float* d_states_theta = NULL;
float* d_lidar_coords = NULL;

float* res_robot_state = NULL;

/********************************************************************/
/************************* PARTICLES VARIABLES **********************/
/********************************************************************/
size_t sz_particles_pos = 0;
size_t sz_particles_idx = 0;
size_t sz_particles_weight = 0;
size_t sz_extended_idx = 0;
size_t sz_grid_map = 0;

int* d_grid_map = NULL;
int* d_particles_x = NULL;
int* d_particles_y = NULL;
int* d_particles_idx = NULL;
float* d_particles_weight = NULL;
int* d_extended_idx = NULL;

int* res_particles_idx = NULL;
int* res_extended_idx = NULL;

/********************************************************************/
/******************** PARTICLES COPY VARIABLES **********************/
/********************************************************************/
int* dc_particles_x = NULL;
int* dc_particles_y = NULL;
int* dc_particles_idx = NULL;

float* dc_states_x = NULL;
float* dc_states_y = NULL;
float* dc_states_theta = NULL;

size_t sz_last_len = 0;
int* d_last_len = NULL;
int* res_last_len = NULL;


/********************************************************************/
/********************** CORRELATION VARIABLES ***********************/
/********************************************************************/
size_t sz_weights = 0;
size_t sz_correlation_raw = 0;

float* h_weights = NULL;
int* h_extended_idx = NULL;
float* d_weights = NULL;
float* d_weights_raw = NULL;

/********************************************************************/
/*********************** TRANSITION VARIABLES ***********************/
/********************************************************************/
size_t sz_transition_world_body = 0;
size_t sz_transition_body_lidar = 0;
size_t sz_transition_world_lidar = 0;
size_t sz_processed_measure_pos = 0;
size_t sz_processed_measure_idx = 0;

float* d_transition_world_body = NULL;
float* d_transition_body_lidar = NULL;
float* d_transition_world_lidar = NULL;
int* d_processed_measure_x = NULL;
int* d_processed_measure_y = NULL;
int* d_processed_measure_idx = NULL;

float* res_transition_world_body = NULL;
float* res_transition_world_lidar = NULL;
float* res_robot_world_body = NULL;

/********************************************************************/
/**************************** MAP VARIABLES *************************/
/********************************************************************/
size_t sz_map_2d = 0;
size_t sz_unique_in_particle = 0;
size_t sz_unique_in_particle_col = 0;

uint8_t* d_map_2d = NULL;
int* d_unique_in_particle = NULL;
int* d_unique_in_particle_col = NULL;

int* h_unique_in_particle = NULL;

/********************************************************************/
/************************ WEIGHTS VARIABLES *************************/
/********************************************************************/
size_t sz_weights_max = 0;
size_t sz_sum_exp = 0;

float* d_weights_max = NULL;
double* d_sum_exp = NULL;
float* res_weights_max = NULL;
double* res_sum_exp = NULL;

/********************************************************************/
/*********************** RESAMPLING VARIABLES ***********************/
/********************************************************************/
size_t sz_js = 0;
size_t sz_rnd = 0;

int* d_js = NULL;
float* d_rnd = NULL;

/********************************************************************/
/********************* UPDATE STATES VARIABLES **********************/
/********************************************************************/
std::vector<float> std_vec_states_x;
std::vector<float> std_vec_states_y;
std::vector<float> std_vec_states_theta;


void test_robot_particles_main() {

	thrust::device_vector<float> d_temp(h_states_x, h_states_x + NUM_PARTICLES);

	negative_before_counter = getNegativeCounter(h_particles_x, h_particles_y, PARTICLES_ITEMS_LEN);
	count_bigger_than_height = getGreaterThanCounter(h_particles_y, GRID_HEIGHT, PARTICLES_ITEMS_LEN);
	negative_after_counter = getNegativeCounter(h_particles_x_after_resampling, h_particles_y_after_resampling, AF_PARTICLES_ITEMS_LEN_RESAMPLING);;

	//printf("GRID_WIDTH: %d, GRID_HEIGHT: %d\n", GRID_WIDTH, GRID_HEIGHT);
	//printf("negative_before_counter: %d\n", negative_before_counter);
	//printf("negative_after_counter: %d\n", negative_after_counter);
	//printf("count_bigger_than_height: %d\n", count_bigger_than_height);

	auto start_robot_particles_alloc = std::chrono::high_resolution_clock::now();
	alloc_state_vars();
	alloc_particles_vars();
	alloc_particles_copy_vars();
	alloc_correlation_vars();
	alloc_transition_vars();
	alloc_map_vars();
	alloc_weights_vars();
	alloc_resampling_vars();
	auto stop_robot_particles_alloc = std::chrono::high_resolution_clock::now();

	auto start_robot_particles_init = std::chrono::high_resolution_clock::now();
	init_state_vars();
	init_particles_vars();
	init_transition_vars();
	auto stop_robot_particles_init = std::chrono::high_resolution_clock::now();

	auto start_robot_particles_kernel = std::chrono::high_resolution_clock::now();
	exec_transition();
	exec_create_2d_map();
	exec_update_map();
	exec_cum_sum();
	reinit_map_vars();
	exec_map_restructure();
	exec_index_expansion();
	exec_correlation();
	exec_update_weights();
	exec_resampling();
	reinit_particles_vars();
	exec_rearrangement();
	exec_update_states();
	auto stop_robot_particles_kernel = std::chrono::high_resolution_clock::now();

	assertResults();

	auto duration_robot_particles_alloc = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_particles_alloc - start_robot_particles_alloc);
	auto duration_robot_particles_init = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_particles_init - start_robot_particles_init);
	auto duration_robot_particles_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_particles_kernel - start_robot_particles_kernel);
	auto duration_robot_particles_total = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_particles_kernel - start_robot_particles_alloc);

	std::cout << std::endl;
	std::cout << "Time taken by function (Allocation): " << duration_robot_particles_alloc.count() << " microseconds" << std::endl;
	std::cout << "Time taken by function (Initialization): " << duration_robot_particles_init.count() << " microseconds" << std::endl;
	std::cout << "Time taken by function (Kernel): " << duration_robot_particles_kernel.count() << " microseconds" << std::endl;
	std::cout << "Time taken by function (Total): " << duration_robot_particles_total.count() << " microseconds" << std::endl;
}

void alloc_state_vars() {

	sz_states_pos = NUM_PARTICLES * sizeof(float);
	sz_lidar_coords = 2 * LIDAR_COORDS_LEN * sizeof(float);

	gpuErrchk(cudaMalloc((void**)&d_states_x, sz_states_pos));
	gpuErrchk(cudaMalloc((void**)&d_states_y, sz_states_pos));
	gpuErrchk(cudaMalloc((void**)&d_states_theta, sz_states_pos));
	gpuErrchk(cudaMalloc((void**)&d_lidar_coords, sz_lidar_coords));

	res_robot_state = (float*)malloc(3 * sizeof(float));
}

void init_state_vars() {

	gpuErrchk(cudaMemcpy(d_states_x, h_states_x, sz_states_pos, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_states_y, h_states_y, sz_states_pos, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_states_theta, h_states_theta, sz_states_pos, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_lidar_coords, h_lidar_coords, sz_lidar_coords, cudaMemcpyHostToDevice));
}

void alloc_particles_vars() {

	sz_particles_pos = PARTICLES_ITEMS_LEN * sizeof(int);
	sz_particles_idx = NUM_PARTICLES * sizeof(int);
	sz_particles_weight = NUM_PARTICLES * sizeof(float);
	sz_extended_idx = PARTICLES_ITEMS_LEN * sizeof(int);
	sz_grid_map = GRID_WIDTH * GRID_HEIGHT * sizeof(int);

	res_particles_idx = (int*)malloc(sz_particles_idx);
	res_extended_idx = (int*)malloc(sz_extended_idx);

	gpuErrchk(cudaMalloc((void**)&d_grid_map, sz_grid_map));
	gpuErrchk(cudaMalloc((void**)&d_particles_x, sz_particles_pos));
	gpuErrchk(cudaMalloc((void**)&d_particles_y, sz_particles_pos));
	gpuErrchk(cudaMalloc((void**)&d_particles_idx, sz_particles_idx));
	gpuErrchk(cudaMalloc((void**)&d_particles_weight, sz_particles_weight));
	gpuErrchk(cudaMalloc((void**)&d_extended_idx, sz_extended_idx));
}

void init_particles_vars() {

	gpuErrchk(cudaMemcpy(d_grid_map, h_grid_map, sz_grid_map, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_particles_x, h_particles_x, sz_particles_pos, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_particles_y, h_particles_y, sz_particles_pos, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_particles_idx, h_particles_idx, sz_particles_idx, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_particles_weight, particles_weight_pre, sz_particles_weight, cudaMemcpyHostToDevice));
}

void alloc_particles_copy_vars() {

	gpuErrchk(cudaMalloc((void**)&dc_states_x, sz_states_pos));
	gpuErrchk(cudaMalloc((void**)&dc_states_y, sz_states_pos));
	gpuErrchk(cudaMalloc((void**)&dc_states_theta, sz_states_pos));
}

void alloc_correlation_vars() {

	sz_weights = NUM_PARTICLES * sizeof(float);
	sz_correlation_raw = 25 * sz_weights;

	h_weights = (float*)malloc(sz_weights);
	h_extended_idx = (int*)malloc(sz_extended_idx);
	d_weights = NULL;
	d_weights_raw = NULL;

	memset(h_weights, 0, sz_weights);

	gpuErrchk(cudaMalloc((void**)&d_weights, sz_weights));
	gpuErrchk(cudaMalloc((void**)&d_weights_raw, sz_correlation_raw));
	gpuErrchk(cudaMemset(d_weights_raw, 0, sz_correlation_raw));
}

void alloc_transition_vars() {

	sz_transition_world_body = 9 * NUM_PARTICLES * sizeof(float);
	sz_transition_body_lidar = 9 * sizeof(float);
	sz_transition_world_lidar = 9 * NUM_PARTICLES * sizeof(float);
	sz_processed_measure_pos = NUM_PARTICLES * LIDAR_COORDS_LEN * sizeof(int);
	sz_processed_measure_idx = NUM_PARTICLES * LIDAR_COORDS_LEN * sizeof(int);

	d_transition_world_body = NULL;
	d_transition_body_lidar = NULL;
	d_transition_world_lidar = NULL;
	d_processed_measure_x = NULL;
	d_processed_measure_y = NULL;
	d_processed_measure_idx = NULL;

	res_transition_world_body = (float*)malloc(sz_transition_world_body);
	res_transition_world_lidar = (float*)malloc(sz_transition_world_lidar);
	res_robot_world_body = (float*)malloc(sz_transition_world_body);

	gpuErrchk(cudaMalloc((void**)&d_transition_world_body, sz_transition_world_body));
	gpuErrchk(cudaMalloc((void**)&d_transition_body_lidar, sz_transition_body_lidar));
	gpuErrchk(cudaMalloc((void**)&d_transition_world_lidar, sz_transition_world_lidar));
	gpuErrchk(cudaMalloc((void**)&d_processed_measure_x, sz_processed_measure_pos));
	gpuErrchk(cudaMalloc((void**)&d_processed_measure_y, sz_processed_measure_pos));
	gpuErrchk(cudaMalloc((void**)&d_processed_measure_idx, sz_processed_measure_idx));
}

void init_transition_vars() {

	gpuErrchk(cudaMemcpy(d_transition_body_lidar, h_transition_body_lidar, sz_transition_body_lidar, cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemset(d_transition_world_body, 0, sz_transition_world_body));
	gpuErrchk(cudaMemset(d_transition_world_lidar, 0, sz_transition_world_lidar));
	gpuErrchk(cudaMemset(d_processed_measure_x, 0, sz_processed_measure_pos));
	gpuErrchk(cudaMemset(d_processed_measure_y, 0, sz_processed_measure_pos));
	gpuErrchk(cudaMemset(d_processed_measure_idx, 0, sz_processed_measure_idx));
}

void alloc_map_vars() {

	sz_map_2d = GRID_WIDTH * GRID_HEIGHT * NUM_PARTICLES * sizeof(uint8_t);
	sz_unique_in_particle = UNIQUE_COUNTER_LEN * sizeof(int);
	sz_unique_in_particle_col = UNIQUE_COUNTER_LEN * GRID_WIDTH * sizeof(int);

	h_unique_in_particle = (int*)malloc(sz_unique_in_particle);

	gpuErrchk(cudaMalloc((void**)&d_map_2d, sz_map_2d));
	gpuErrchk(cudaMalloc((void**)&d_unique_in_particle, sz_unique_in_particle));
	gpuErrchk(cudaMalloc((void**)&d_unique_in_particle_col, sz_unique_in_particle_col));

	gpuErrchk(cudaMemset(d_map_2d, 0, sz_map_2d));
	gpuErrchk(cudaMemset(d_unique_in_particle, 0, sz_unique_in_particle));
	gpuErrchk(cudaMemset(d_unique_in_particle_col, 0, sz_unique_in_particle_col));
}

void alloc_weights_vars() {

	sz_weights_max = sizeof(float);
	sz_sum_exp = sizeof(double);

	res_weights_max = (float*)malloc(sz_weights_max);
	res_sum_exp = (double*)malloc(sz_sum_exp);

	gpuErrchk(cudaMalloc((void**)&d_weights_max, sz_weights_max));
	gpuErrchk(cudaMalloc((void**)&d_sum_exp, sz_sum_exp));

	gpuErrchk(cudaMemset(d_weights_max, 0, sz_weights_max));
	gpuErrchk(cudaMemset(d_sum_exp, 0, sz_sum_exp));
}

void alloc_resampling_vars() {

	sz_js = NUM_PARTICLES * sizeof(int);
	sz_rnd = NUM_PARTICLES * sizeof(float);

	d_js = NULL;
	d_rnd = NULL;

	gpuErrchk(cudaMalloc((void**)&d_js, sz_js));
	gpuErrchk(cudaMalloc((void**)&d_rnd, sz_rnd));

	gpuErrchk(cudaMemcpy(d_rnd, h_rnds, sz_rnd, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemset(d_js, 0, sz_js));
}

void exec_transition() {

	threadsPerBlock = NUM_PARTICLES;
	blocksPerGrid = 1;
	kernel_update_particles_states << <blocksPerGrid, threadsPerBlock >> > (d_states_x, d_states_y, d_states_theta, d_transition_world_body, d_transition_body_lidar, d_transition_world_lidar, NUM_PARTICLES);
	cudaDeviceSynchronize();

	threadsPerBlock = NUM_PARTICLES;
	blocksPerGrid = LIDAR_COORDS_LEN;
	kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (d_transition_world_lidar, d_processed_measure_x, d_processed_measure_y, d_lidar_coords, res, xmin, ymax, LIDAR_COORDS_LEN, NUM_PARTICLES);
	cudaDeviceSynchronize();

	threadsPerBlock = NUM_PARTICLES;
	blocksPerGrid = 1;
	kernel_index_init_const << < blocksPerGrid, threadsPerBlock >> > (d_processed_measure_idx, LIDAR_COORDS_LEN);
	cudaDeviceSynchronize();

	thrust::exclusive_scan(thrust::device, d_processed_measure_idx, d_processed_measure_idx + NUM_PARTICLES, d_processed_measure_idx, 0);

	gpuErrchk(cudaMemcpy(res_transition_world_body, d_transition_world_body, sz_transition_world_body, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(res_transition_world_lidar, d_transition_world_lidar, sz_transition_world_lidar, cudaMemcpyDeviceToHost));
}

void exec_create_2d_map() {

	threadsPerBlock = 100;
	blocksPerGrid = NUM_PARTICLES;
	kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (d_particles_x, d_particles_y, d_particles_idx, PARTICLES_ITEMS_LEN, d_map_2d, d_unique_in_particle,
		d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
	cudaDeviceSynchronize();
}

void exec_update_map() {

	threadsPerBlock = NUM_PARTICLES;
	blocksPerGrid = 1;

	kernel_update_2d_map_with_measure << <blocksPerGrid, threadsPerBlock >> > (d_processed_measure_x, d_processed_measure_y, d_processed_measure_idx,
		MEASURE_LEN, d_map_2d, d_unique_in_particle,
		d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
	cudaDeviceSynchronize();
}

void exec_cum_sum() {

	threadsPerBlock = UNIQUE_COUNTER_LEN;
	blocksPerGrid = 1;
	kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (d_unique_in_particle_col, GRID_WIDTH);
	thrust::exclusive_scan(thrust::device, d_unique_in_particle, d_unique_in_particle + UNIQUE_COUNTER_LEN, d_unique_in_particle, 0);
	cudaDeviceSynchronize();

	gpuErrchk(cudaMemcpy(h_unique_in_particle, d_unique_in_particle, sz_unique_in_particle, cudaMemcpyDeviceToHost));

	PARTICLES_ITEMS_LEN = h_unique_in_particle[UNIQUE_COUNTER_LEN - 1];
	C_PARTICLES_ITEMS_LEN = 0;
}

void reinit_map_vars() {

	gpuErrchk(cudaFree(d_particles_x));
	gpuErrchk(cudaFree(d_particles_y));
	gpuErrchk(cudaFree(d_extended_idx));

	sz_particles_pos = PARTICLES_ITEMS_LEN * sizeof(int);
	sz_extended_idx = PARTICLES_ITEMS_LEN * sizeof(int);

	gpuErrchk(cudaMalloc((void**)&d_particles_x, sz_particles_pos));
	gpuErrchk(cudaMalloc((void**)&d_particles_y, sz_particles_pos));
	gpuErrchk(cudaMalloc((void**)&d_extended_idx, sz_extended_idx));
}

void exec_map_restructure() {

	threadsPerBlock = GRID_WIDTH;
	blocksPerGrid = NUM_PARTICLES;

	cudaMemset(d_particles_idx, 0, sz_particles_idx);
	kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (d_map_2d, d_particles_x, d_particles_y, d_particles_idx,
		d_unique_in_particle, d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT);
	cudaDeviceSynchronize();

	thrust::exclusive_scan(thrust::device, d_particles_idx, d_particles_idx + NUM_PARTICLES, d_particles_idx, 0);
}

void exec_index_expansion() {

	threadsPerBlock = 100;
	blocksPerGrid = NUM_PARTICLES;
	kernel_index_expansion << <blocksPerGrid, threadsPerBlock >> > (d_particles_idx, d_extended_idx, PARTICLES_ITEMS_LEN);
	cudaDeviceSynchronize();

	res_extended_idx = (int*)malloc(sz_extended_idx);
	gpuErrchk(cudaMemcpy(res_extended_idx, d_extended_idx, sz_extended_idx, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(res_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToHost));
}

void exec_correlation() {

	threadsPerBlock = 256;
	blocksPerGrid = (PARTICLES_ITEMS_LEN + threadsPerBlock - 1) / threadsPerBlock;

	kernel_correlation << <blocksPerGrid, threadsPerBlock >> > (d_grid_map, d_particles_x, d_particles_y, 
		d_extended_idx, d_weights_raw, GRID_WIDTH, GRID_HEIGHT, PARTICLES_ITEMS_LEN);
	cudaDeviceSynchronize();

	threadsPerBlock = NUM_PARTICLES;
	blocksPerGrid = 1;
	kernel_correlation_max << <blocksPerGrid, threadsPerBlock >> > (d_weights_raw, d_weights, NUM_PARTICLES);
	cudaDeviceSynchronize();

	gpuErrchk(cudaMemcpy(res_extended_idx, d_extended_idx, sz_extended_idx, cudaMemcpyDeviceToHost));
}

void exec_update_weights() {

	threadsPerBlock = 1;
	blocksPerGrid = 1;

	kernel_arr_max << < blocksPerGrid, threadsPerBlock >> > (d_weights, d_weights_max, NUM_PARTICLES);
	cudaDeviceSynchronize();

	gpuErrchk(cudaMemcpy(res_weights_max, d_weights_max, sz_weights_max, cudaMemcpyDeviceToHost));

	float norm_value = -res_weights_max[0] + 50;

	threadsPerBlock = NUM_PARTICLES;
	blocksPerGrid = 1;
	kernel_arr_increase << < blocksPerGrid, threadsPerBlock >> > (d_weights, norm_value, 0);
	cudaDeviceSynchronize();

	threadsPerBlock = 1;
	blocksPerGrid = 1;
	kernel_arr_sum_exp << < blocksPerGrid, threadsPerBlock >> > (d_weights, d_sum_exp, NUM_PARTICLES);
	cudaDeviceSynchronize();

	gpuErrchk(cudaMemcpy(res_sum_exp, d_sum_exp, sz_sum_exp, cudaMemcpyDeviceToHost));

	threadsPerBlock = NUM_PARTICLES;
	blocksPerGrid = 1;
	kernel_arr_normalize << < blocksPerGrid, threadsPerBlock >> > (d_weights, res_sum_exp[0]);
	cudaDeviceSynchronize();

	threadsPerBlock = NUM_PARTICLES;
	blocksPerGrid = 1;
	kernel_arr_mult << < blocksPerGrid, threadsPerBlock >> > (d_particles_weight, d_weights);
	cudaDeviceSynchronize();
}

void exec_resampling() {

	threadsPerBlock = NUM_PARTICLES;
	blocksPerGrid = 1;
	kernel_resampling << <blocksPerGrid, threadsPerBlock >> > (d_weights, d_js, d_rnd, NUM_PARTICLES);
	cudaDeviceSynchronize();
}

void reinit_particles_vars() {

	sz_last_len = sizeof(int);
	d_last_len = NULL;
	res_last_len = (int*)malloc(sizeof(int));

	gpuErrchk(cudaMalloc((void**)&d_last_len, sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dc_particles_x, sz_particles_pos));
	gpuErrchk(cudaMalloc((void**)&dc_particles_y, sz_particles_pos));
	gpuErrchk(cudaMalloc((void**)&dc_particles_idx, sz_particles_idx));

	gpuErrchk(cudaMemcpy(dc_particles_x, d_particles_x, sz_particles_pos, cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaMemcpy(dc_particles_y, d_particles_y, sz_particles_pos, cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaMemcpy(dc_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToDevice));

	gpuErrchk(cudaMemcpy(dc_states_x, d_states_x, sz_states_pos, cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaMemcpy(dc_states_y, d_states_y, sz_states_pos, cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaMemcpy(dc_states_theta, d_states_theta, sz_states_pos, cudaMemcpyDeviceToDevice));

	threadsPerBlock = NUM_PARTICLES;
	blocksPerGrid = 1;
	kernel_rearrange_indecies << <blocksPerGrid, threadsPerBlock >> > (d_particles_idx, dc_particles_idx, d_js, d_last_len, PARTICLES_ITEMS_LEN);
	cudaDeviceSynchronize();

	gpuErrchk(cudaMemcpy(res_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(res_last_len, d_last_len, sz_last_len, cudaMemcpyDeviceToHost));
}

void exec_rearrangement() {

	thrust::exclusive_scan(thrust::device, d_particles_idx, d_particles_idx + NUM_PARTICLES, d_particles_idx, 0);

	gpuErrchk(cudaMemcpy(res_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToHost));
	C_PARTICLES_ITEMS_LEN = PARTICLES_ITEMS_LEN;
	PARTICLES_ITEMS_LEN = res_particles_idx[NUM_PARTICLES - 1] + res_last_len[0];

	sz_particles_pos = PARTICLES_ITEMS_LEN * sizeof(int);

	threadsPerBlock = 100;
	blocksPerGrid = NUM_PARTICLES;
	kernel_rearrange_particles << <blocksPerGrid, threadsPerBlock >> > (d_particles_x, d_particles_y, d_particles_idx,
		dc_particles_x, dc_particles_y, dc_particles_idx, d_js,
		GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES, PARTICLES_ITEMS_LEN, C_PARTICLES_ITEMS_LEN);

	kernel_rearrange_states << <blocksPerGrid, threadsPerBlock >> > (d_states_x, d_states_y, d_states_theta,
		dc_states_x, dc_states_y, dc_states_theta, d_js);
	cudaDeviceSynchronize();
}

void exec_update_states() {

	thrust::device_vector<float> d_vec_states_x(d_states_x, d_states_x + NUM_PARTICLES);
	thrust::device_vector<float> d_vec_states_y(d_states_y, d_states_y + NUM_PARTICLES);
	thrust::device_vector<float> d_vec_states_theta(d_states_theta, d_states_theta + NUM_PARTICLES);

	thrust::host_vector<float> h_vec_states_x(d_vec_states_x.begin(), d_vec_states_x.end());
	thrust::host_vector<float> h_vec_states_y(d_vec_states_y.begin(), d_vec_states_y.end());
	thrust::host_vector<float> h_vec_states_theta(d_vec_states_theta.begin(), d_vec_states_theta.end());

	std_vec_states_x.clear();
	std_vec_states_y.clear();
	std_vec_states_theta.clear();
	std_vec_states_x.resize(h_vec_states_x.size());
	std_vec_states_y.resize(h_vec_states_y.size());
	std_vec_states_theta.resize(h_vec_states_theta.size());

	std::copy(h_vec_states_x.begin(), h_vec_states_x.end(), std_vec_states_x.begin());
	std::copy(h_vec_states_y.begin(), h_vec_states_y.end(), std_vec_states_y.begin());
	std::copy(h_vec_states_theta.begin(), h_vec_states_theta.end(), std_vec_states_theta.begin());

	std::map<std::tuple<float, float, float>, int> states;

	for (int i = 0; i < NUM_PARTICLES; i++) {
		if (states.find(std::make_tuple(std_vec_states_x[i], std_vec_states_y[i], std_vec_states_theta[i])) == states.end())
			states.insert({ std::make_tuple(std_vec_states_x[i], std_vec_states_y[i], std_vec_states_theta[i]), 1 });
		else
			states[std::make_tuple(std_vec_states_x[i], std_vec_states_y[i], std_vec_states_theta[i])] += 1;
	}

	std::map<std::tuple<float, float, float>, int>::iterator best
		= std::max_element(states.begin(), states.end(), [](const std::pair<std::tuple<float, float, float>, int>& a,
			const std::pair<std::tuple<float, float, float>, int>& b)->bool { return a.second < b.second; });

	auto key = best->first;

	float theta = std::get<2>(key);

	res_robot_world_body[0] = cos(theta);	res_robot_world_body[1] = -sin(theta);	res_robot_world_body[2] = std::get<0>(key);
	res_robot_world_body[3] = sin(theta);   res_robot_world_body[4] = cos(theta);	res_robot_world_body[5] = std::get<1>(key);
	res_robot_world_body[6] = 0;			res_robot_world_body[7] = 0;			res_robot_world_body[8] = 1;

	res_robot_state[0] = std::get<0>(key); res_robot_state[1] = std::get<1>(key); res_robot_state[2] = std::get<2>(key);
}

void assertResults() {

	printf("\n");
	printf("~~$ Transition World to Body (Result): ");
	for (int i = 0; i < 9; i++) {
		printf("%f ", res_transition_world_body[i]);
	}
	printf("\n");
	printf("~~$ Transition World to Body (Host)  : ");
	for (int i = 0; i < 9; i++) {
		printf("%f ", h_robot_transition_world_body[i]);
	}

	printf("\n\n");
	printf("~~$ Robot State (Result): %f, %f, %f\n", res_robot_state[0], res_robot_state[1], res_robot_state[2]);
	printf("~~$ Robot State (Host)  : %f, %f, %f\n", h_robot_state[0], h_robot_state[1], h_robot_state[2]);
}

#endif
