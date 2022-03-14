#ifndef _TEST_MAP_H_
#define _TEST_MAP_H_

#include "headers.h"
#include "host_asserts.h"
#include "host_utils.h"
#include "kernels.cuh"

#include "data/map/100.h"

inline void alloc_image_transform_vars(int LIDAR_COORDS_LEN);
inline void init_image_transform_vars();
inline void alloc_bresenham_vars();
inline void alloc_map_vars();
inline void init_map_vars();
inline void alloc_log_odds_vars();
inline void init_log_odds_vars();
inline void reinit_map_idx_vars();
inline void reinit_map_vars();

inline void exec_world_to_image_transform(float res, int xmin, int ymax, const int LIDAR_COORDS_LEN);
inline void exec_bresenham();
inline void exec_create_map();
inline void exec_log_odds(float log_t, int GRID_WIDTH, int GRID_HEIGHT);

void assertResults();

/********************************************************************/
/************************* GENERAL VARIABLES ************************/
/********************************************************************/
int PARTICLES_OCCUPIED_LEN = ST_PARTICLES_OCCUPIED_LEN;
int PARTICLES_FREE_LEN = 0;
int PARTICLES_OCCUPIED_UNIQUE_LEN = 0;
int PARTICLES_FREE_UNIQUE_LEN = 0;

int GRID_WIDTH = ST_GRID_WIDTH;
int GRID_HEIGHT = ST_GRID_HEIGHT;

float log_t = ST_log_t;

int MAX_DIST_IN_MAP = sqrt(pow(GRID_WIDTH, 2) + pow(GRID_HEIGHT, 2));
int PARTICLE_UNIQUE_COUNTER = PARTICLES_OCCUPIED_LEN + 1;

/********************************************************************/
/*********************** EXEC TIME VARIABLES ************************/
/********************************************************************/


int threadsPerBlock = 1;
int blocksPerGrid   = 1;

/********************************************************************/
/********************* IMAGE TRANSFORM VARIABLES ********************/
/********************************************************************/
size_t sz_transition_frames;
size_t sz_lidar_coords;
size_t sz_particles_occupied_pos;
size_t sz_particles_wframe_pos;
size_t sz_position_image;

float* d_lidar_coords = NULL;
float* d_transition_body_lidar = NULL;
float* d_transition_world_body = NULL;
float* d_transition_world_lidar = NULL;
int* d_particles_occupied_x = NULL;
int* d_particles_occupied_y = NULL;
float* d_particles_wframe_x = NULL;
float* d_particles_wframe_y = NULL;

/********************************************************************/
/************************ BRESENHAM VARIABLES ***********************/
/********************************************************************/
size_t sz_particles_free_pos = 0;
size_t sz_particles_free_pos_max = 0;
size_t sz_particles_free_counter = 0;
size_t sz_particles_free_idx = 0;
size_t sz_position_image_body = 0;

int* d_particles_occupied_idx = NULL;
int* d_particles_free_x = NULL;
int* d_particles_free_y = NULL;
int* d_particles_free_x_max = NULL;
int* d_particles_free_y_max = NULL;
int* d_particles_free_counter = NULL;
int* d_particles_free_idx = NULL;
int* d_position_image_body = NULL;

int* res_particles_free_counter = NULL;

/********************************************************************/
/**************************** MAP VARIABLES *************************/
/********************************************************************/
size_t sz_map = 0;
int* d_grid_map = NULL;

/********************************************************************/
/************************* LOG-ODDS VARIABLES ***********************/
/********************************************************************/
size_t  sz_map_2d = 0;
size_t  sz_unique_counter = 0;
size_t  sz_unique_counter_col = 0;
size_t  sz_log_odds = 0;
size_t  sz_map_idx = 0;

uint8_t* d_map_occupied_2d = NULL;
uint8_t* d_map_free_2d = NULL;
int* d_unique_occupied_counter = NULL;
int* d_unique_free_counter = NULL;
int* d_unique_occupied_counter_col = NULL;
int* d_unique_free_counter_col = NULL;
float* d_log_odds = NULL;
int* d_map_occupied_idx = NULL;
int* d_map_free_idx = NULL;

int h_map_occupied_idx[] = { 0, 0 };
int h_map_free_idx[] = { 0, 0 };

int* res_unique_occupied_counter = NULL;
int* res_unique_free_counter = NULL;
int* res_grid_map = NULL;
float* res_log_odds = NULL;



void test_map_main() {

	const int data_len = 6;
	int data[data_len] = { 1, 0, 2, 2, 1, 3 };
	int* d_data = NULL;
	gpuErrchk(cudaMalloc((void**)&d_data, data_len * sizeof(int)));
	gpuErrchk(cudaMemcpy(d_data, data, data_len * sizeof(int), cudaMemcpyHostToDevice));

	thrust::exclusive_scan(thrust::host, data, data + 6, data, 0);
	thrust::exclusive_scan(thrust::device, d_data, d_data + 6, d_data, 0);

	auto start_mapping_alloc = std::chrono::high_resolution_clock::now();
	alloc_image_transform_vars(ST_LIDAR_COORDS_LEN);
	alloc_bresenham_vars();
	alloc_map_vars();
	alloc_log_odds_vars();
	auto stop_mapping_alloc = std::chrono::high_resolution_clock::now();

	auto start_mapping_init = std::chrono::high_resolution_clock::now();
	init_image_transform_vars();
	init_map_vars();
	init_log_odds_vars();
	auto stop_mapping_init = std::chrono::high_resolution_clock::now();

	auto start_mapping_kernel = std::chrono::high_resolution_clock::now();
	exec_world_to_image_transform(ST_res, ST_xmin, ST_ymax, ST_LIDAR_COORDS_LEN);
	exec_bresenham();

	reinit_map_idx_vars();

	exec_create_map();
	reinit_map_vars();

	exec_log_odds(ST_log_t, ST_GRID_WIDTH, ST_GRID_HEIGHT);
	auto stop_mapping_kernel = std::chrono::high_resolution_clock::now();

	auto duration_mapping_alloc = std::chrono::duration_cast<std::chrono::microseconds>(stop_mapping_alloc - start_mapping_alloc);
	auto duration_mapping_init = std::chrono::duration_cast<std::chrono::microseconds>(stop_mapping_init - start_mapping_init);
	auto duration_mapping_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_mapping_kernel - start_mapping_kernel);

	std::cout << std::endl << "Time taken by function (Mapping Allocation): " << duration_mapping_alloc.count() << " microseconds" << std::endl;
	std::cout << std::endl << "Time taken by function (Mapping Initialization): " << duration_mapping_init.count() << " microseconds" << std::endl;
	std::cout << std::endl << "Time taken by function (Mapping Kernel): " << duration_mapping_kernel.count() << " microseconds" << std::endl;


	assertResults();
}

inline void alloc_image_transform_vars(int LIDAR_COORDS_LEN) {

	sz_transition_frames = 9 * sizeof(float);
	sz_lidar_coords = 2 * LIDAR_COORDS_LEN * sizeof(float);
	sz_particles_occupied_pos = LIDAR_COORDS_LEN * sizeof(int);
	sz_particles_wframe_pos = LIDAR_COORDS_LEN * sizeof(float);
	sz_position_image = 2 * sizeof(int);

	gpuErrchk(cudaMalloc((void**)&d_lidar_coords, sz_lidar_coords));
	gpuErrchk(cudaMalloc((void**)&d_transition_body_lidar, sz_transition_frames));
	gpuErrchk(cudaMalloc((void**)&d_transition_world_body, sz_transition_frames));
	gpuErrchk(cudaMalloc((void**)&d_transition_world_lidar, sz_transition_frames));
	gpuErrchk(cudaMalloc((void**)&d_particles_occupied_x, sz_particles_occupied_pos));
	gpuErrchk(cudaMalloc((void**)&d_particles_occupied_y, sz_particles_occupied_pos));
	gpuErrchk(cudaMalloc((void**)&d_particles_wframe_x, sz_particles_wframe_pos));
	gpuErrchk(cudaMalloc((void**)&d_particles_wframe_y, sz_particles_wframe_pos));
}

void init_image_transform_vars() {

	gpuErrchk(cudaMemcpy(d_lidar_coords, lidar_coords, sz_lidar_coords, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_transition_body_lidar, h_transition_body_lidar, sz_transition_frames, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_transition_world_body, h_transition_world_body, sz_transition_frames, cudaMemcpyHostToDevice));
}

void alloc_bresenham_vars() {

	sz_particles_free_pos = 0;
	sz_particles_free_pos_max = PARTICLES_OCCUPIED_LEN * MAX_DIST_IN_MAP * sizeof(int);
	sz_particles_free_counter = PARTICLE_UNIQUE_COUNTER * sizeof(int);
	sz_particles_free_idx = PARTICLES_OCCUPIED_LEN * sizeof(int);
	sz_position_image_body = 2 * sizeof(int);

	gpuErrchk(cudaMalloc((void**)&d_particles_free_x_max, sz_particles_free_pos_max));
	gpuErrchk(cudaMalloc((void**)&d_particles_free_y_max, sz_particles_free_pos_max));
	gpuErrchk(cudaMalloc((void**)&d_particles_free_counter, sz_particles_free_counter));
	gpuErrchk(cudaMalloc((void**)&d_particles_free_idx, sz_particles_free_idx));
	gpuErrchk(cudaMalloc((void**)&d_position_image_body, sz_position_image_body));

	res_particles_free_counter = (int*)malloc(sz_particles_free_counter);

	gpuErrchk(cudaMemset(d_particles_free_x, 0, sz_particles_free_pos));
	gpuErrchk(cudaMemset(d_particles_free_y, 0, sz_particles_free_pos));
	gpuErrchk(cudaMemset(d_particles_free_x_max, 0, sz_particles_free_pos_max));
	gpuErrchk(cudaMemset(d_particles_free_y_max, 0, sz_particles_free_pos_max));
	gpuErrchk(cudaMemset(d_particles_free_counter, 0, sz_particles_free_counter));
}

void alloc_map_vars() {

	sz_map = (GRID_WIDTH * GRID_HEIGHT) * sizeof(int);
	gpuErrchk(cudaMalloc((void**)&d_grid_map, sz_map));
}

void init_map_vars() {

	gpuErrchk(cudaMemcpy(d_grid_map, pre_grid_map, sz_map, cudaMemcpyHostToDevice));
}

void alloc_log_odds_vars() {

	sz_map_2d = GRID_WIDTH * GRID_HEIGHT * sizeof(uint8_t);
	sz_unique_counter = 1 * sizeof(int);
	sz_unique_counter_col = (GRID_WIDTH + 1) * sizeof(int);
	sz_log_odds = (GRID_WIDTH * GRID_HEIGHT) * sizeof(float);
	sz_map_idx = 2 * sizeof(int);

	res_unique_occupied_counter = (int*)malloc(sz_unique_counter);
	res_unique_free_counter = (int*)malloc(sz_unique_counter);
	res_grid_map = (int*)malloc(sz_map);
	res_log_odds = (float*)malloc(sz_log_odds);

	gpuErrchk(cudaMalloc((void**)&d_map_occupied_2d, sz_map_2d));
	gpuErrchk(cudaMalloc((void**)&d_map_free_2d, sz_map_2d));
	gpuErrchk(cudaMalloc((void**)&d_unique_occupied_counter, sz_unique_counter));
	gpuErrchk(cudaMalloc((void**)&d_unique_occupied_counter_col, sz_unique_counter_col));
	gpuErrchk(cudaMalloc((void**)&d_unique_free_counter, sz_unique_counter));
	gpuErrchk(cudaMalloc((void**)&d_unique_free_counter_col, sz_unique_counter_col));
	gpuErrchk(cudaMalloc((void**)&d_log_odds, sz_log_odds));

	gpuErrchk(cudaMalloc((void**)&d_map_occupied_idx, sz_map_idx));
	gpuErrchk(cudaMalloc((void**)&d_map_free_idx, sz_map_idx));
}

void init_log_odds_vars() {

	memset(res_log_odds, 0, sz_log_odds);

	gpuErrchk(cudaMemcpy(d_map_occupied_idx, h_map_occupied_idx, sz_map_idx, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_map_free_idx, h_map_free_idx, sz_map_idx, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_log_odds, pre_log_odds, sz_log_odds, cudaMemcpyHostToDevice));
}

void exec_world_to_image_transform(float res, int xmin, int ymax, const int LIDAR_COORDS_LEN) {

	kernel_matrix_mul_3x3 << < 1, 1 >> > (d_transition_world_body, d_transition_body_lidar, d_transition_world_lidar);
	cudaDeviceSynchronize();

	threadsPerBlock = 1;
	blocksPerGrid = LIDAR_COORDS_LEN;
	kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (d_transition_world_lidar, d_particles_occupied_x, d_particles_occupied_y,
		d_particles_wframe_x, d_particles_wframe_y, d_lidar_coords, res, xmin, ymax, LIDAR_COORDS_LEN);
	cudaDeviceSynchronize();

	kernel_position_to_image << < 1, 1 >> > (d_position_image_body, d_transition_world_lidar, res, xmin, ymax);
	cudaDeviceSynchronize();
}

void exec_bresenham() {

	threadsPerBlock = 256;
	blocksPerGrid = (PARTICLES_OCCUPIED_LEN + threadsPerBlock - 1) / threadsPerBlock;
	kernel_bresenham << <blocksPerGrid, threadsPerBlock >> > (d_particles_occupied_x, d_particles_occupied_y, d_position_image_body,
		d_particles_free_x_max, d_particles_free_y_max, d_particles_free_counter, PARTICLES_OCCUPIED_LEN, MAX_DIST_IN_MAP);
	cudaDeviceSynchronize();
	thrust::exclusive_scan(thrust::device, d_particles_free_counter, d_particles_free_counter + PARTICLE_UNIQUE_COUNTER, d_particles_free_counter, 0); // in-place scan

	gpuErrchk(cudaMemcpy(res_particles_free_counter, d_particles_free_counter, sz_particles_free_counter, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(d_particles_free_idx, d_particles_free_counter, sz_particles_free_idx, cudaMemcpyDeviceToDevice));

	PARTICLES_FREE_LEN = res_particles_free_counter[PARTICLE_UNIQUE_COUNTER - 1];
	sz_particles_free_pos = PARTICLES_FREE_LEN * sizeof(int);
	gpuErrchk(cudaMalloc((void**)&d_particles_free_x, sz_particles_free_pos));
	gpuErrchk(cudaMalloc((void**)&d_particles_free_y, sz_particles_free_pos));

	kernel_bresenham_rearrange << <blocksPerGrid, threadsPerBlock >> > (d_particles_free_x, d_particles_free_y, d_particles_free_x_max, d_particles_free_y_max,
		d_particles_free_counter, MAX_DIST_IN_MAP, PARTICLES_OCCUPIED_LEN);
	cudaDeviceSynchronize();
}

void reinit_map_idx_vars() {

	h_map_occupied_idx[1] = PARTICLES_OCCUPIED_LEN;
	h_map_free_idx[1] = PARTICLES_FREE_LEN;

	gpuErrchk(cudaMemcpy(d_map_occupied_idx, h_map_occupied_idx, sz_map_idx, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_map_free_idx, h_map_free_idx, sz_map_idx, cudaMemcpyHostToDevice));
}

void exec_create_map() {

	threadsPerBlock = 256;
	blocksPerGrid = 1;
	kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (d_particles_occupied_x, d_particles_occupied_y, d_map_occupied_idx,
		PARTICLES_OCCUPIED_LEN, d_map_occupied_2d, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
	kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (d_particles_free_x, d_particles_free_y, d_map_free_idx,
		PARTICLES_FREE_LEN, d_map_free_2d, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
	cudaDeviceSynchronize();

	threadsPerBlock = GRID_WIDTH;
	blocksPerGrid = 1;
	kernel_2d_map_counter << <blocksPerGrid, threadsPerBlock >> > (d_map_occupied_2d, d_unique_occupied_counter, d_unique_occupied_counter_col, GRID_WIDTH, GRID_HEIGHT);
	kernel_2d_map_counter << <blocksPerGrid, threadsPerBlock >> > (d_map_free_2d, d_unique_free_counter, d_unique_free_counter_col, GRID_WIDTH, GRID_HEIGHT);
	cudaDeviceSynchronize();

	threadsPerBlock = 1;
	blocksPerGrid = 1;
	kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (d_unique_occupied_counter_col, GRID_WIDTH);
	kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (d_unique_free_counter_col, GRID_WIDTH);
	cudaDeviceSynchronize();

	gpuErrchk(cudaMemcpy(res_unique_occupied_counter, d_unique_occupied_counter, sz_unique_counter, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(res_unique_free_counter, d_unique_free_counter, sz_unique_counter, cudaMemcpyDeviceToHost));
}

void reinit_map_vars() {

	PARTICLES_OCCUPIED_UNIQUE_LEN = res_unique_occupied_counter[0];
	PARTICLES_FREE_UNIQUE_LEN = res_unique_free_counter[0];

	gpuErrchk(cudaFree(d_particles_occupied_x));
	gpuErrchk(cudaFree(d_particles_occupied_y));
	gpuErrchk(cudaFree(d_particles_free_x));
	gpuErrchk(cudaFree(d_particles_free_y));

	sz_particles_occupied_pos = PARTICLES_OCCUPIED_UNIQUE_LEN * sizeof(int);
	sz_particles_free_pos = PARTICLES_FREE_UNIQUE_LEN * sizeof(int);

	gpuErrchk(cudaMalloc((void**)&d_particles_occupied_x, sz_particles_occupied_pos));
	gpuErrchk(cudaMalloc((void**)&d_particles_occupied_y, sz_particles_occupied_pos));
	gpuErrchk(cudaMalloc((void**)&d_particles_free_x, sz_particles_free_pos));
	gpuErrchk(cudaMalloc((void**)&d_particles_free_y, sz_particles_free_pos));

	threadsPerBlock = GRID_WIDTH;
	blocksPerGrid = 1;
	kernel_update_unique_restructure2 << <blocksPerGrid, threadsPerBlock >> > (d_map_occupied_2d, d_particles_occupied_x, d_particles_occupied_y, d_particles_occupied_idx,
		d_unique_occupied_counter_col, GRID_WIDTH, GRID_HEIGHT);
	kernel_update_unique_restructure2 << <blocksPerGrid, threadsPerBlock >> > (d_map_free_2d, d_particles_free_x, d_particles_free_y, d_particles_free_idx,
		d_unique_free_counter_col, GRID_WIDTH, GRID_HEIGHT);
	cudaDeviceSynchronize();
}

void exec_log_odds(float log_t, int GRID_WIDTH, int GRID_HEIGHT) {

	threadsPerBlock = 256;
	blocksPerGrid = (PARTICLES_OCCUPIED_UNIQUE_LEN + threadsPerBlock - 1) / threadsPerBlock;
	kernel_update_log_odds << <blocksPerGrid, threadsPerBlock >> > (d_log_odds, d_particles_occupied_x, d_particles_occupied_y, 
		2 * log_t, GRID_WIDTH, GRID_HEIGHT, PARTICLES_OCCUPIED_UNIQUE_LEN);
	cudaDeviceSynchronize();

	threadsPerBlock = 256;
	blocksPerGrid = (PARTICLES_FREE_UNIQUE_LEN + threadsPerBlock - 1) / threadsPerBlock;
	kernel_update_log_odds << <blocksPerGrid, threadsPerBlock >> > (d_log_odds, d_particles_free_x, d_particles_free_y, (-1) * log_t, GRID_WIDTH, GRID_HEIGHT, PARTICLES_FREE_UNIQUE_LEN);
	cudaDeviceSynchronize();

	threadsPerBlock = 256;
	blocksPerGrid = ((GRID_WIDTH * GRID_HEIGHT) + threadsPerBlock - 1) / threadsPerBlock;
	kernel_update_map << <blocksPerGrid, threadsPerBlock >> > (d_grid_map, d_log_odds, LOG_ODD_PRIOR, WALL, FREE, GRID_WIDTH * GRID_HEIGHT);
	cudaDeviceSynchronize();
}


void assertResults() {

	printf("\n--> Occupied Unique: %d, %d\n", PARTICLES_OCCUPIED_UNIQUE_LEN, ST_PARTICLES_OCCUPIED_UNIQUE_LEN);
	assert(PARTICLES_OCCUPIED_UNIQUE_LEN == ST_PARTICLES_OCCUPIED_UNIQUE_LEN);
	printf("\n--> Free Unique: %d, %d\n", PARTICLES_FREE_UNIQUE_LEN, ST_PARTICLES_FREE_UNIQUE_LEN);
	assert(PARTICLES_FREE_UNIQUE_LEN == ST_PARTICLES_FREE_UNIQUE_LEN);

	printf("PARTICLES_FREE_LEN=%d\n", PARTICLES_FREE_LEN);
	ASSERT_particles_free_index(res_particles_free_counter, h_particles_free_idx, PARTICLES_OCCUPIED_LEN, false);
	ASSERT_particles_free_new_len(PARTICLES_FREE_LEN, ST_PARTICLES_FREE_LEN);

	gpuErrchk(cudaMemcpy(res_log_odds, d_log_odds, sz_log_odds, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(res_grid_map, d_grid_map, sz_map, cudaMemcpyDeviceToHost));

	ASSERT_log_odds(res_log_odds, pre_log_odds, post_log_odds, (GRID_WIDTH * GRID_HEIGHT));
	ASSERT_log_odds_maps(res_grid_map, pre_grid_map, post_grid_map, (GRID_WIDTH * GRID_HEIGHT));
	printf("\n");

}

#endif
