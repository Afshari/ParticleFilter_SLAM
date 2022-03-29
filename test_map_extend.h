#ifndef _TEST_MAP_EXTEND_H_
#define _TEST_MAP_EXTEND_H_

#include "headers.h"
#include "host_asserts.h"
#include "host_utils.h"
#include "kernels.cuh"
#include "kernels_map.cuh"

//#include "data/map_ignore_extend/1000.h"
//#include "data/map_extend/721.h"
//#include "data/map_extend/2789.h"


//#include "data/map/3000.h"
//#include "data/map/721.h"
#include "data/map/2789.h"

void alloc_();

size_t sz_lidar_coords = 0;
size_t sz_should_extend = 0;
size_t sz_coord = 0;
size_t sz_grid_map = 0;
size_t sz_log_odds = 0;

float* d_lidar_coords = NULL;
int* d_should_extend = NULL;
int* d_coord = NULL;
int* d_grid_map = NULL;
float* d_log_odds = NULL;

int* res_should_extend = NULL;
int* res_coord = NULL;
int* res_grid_map = NULL;
float* res_log_odds = NULL;

int threadsPerBlock = 0;
int blocksPerGrid = 0;


void test_map_extend_main() {

	std::cout << "Starting 'Test Map Extend'" << std::endl;

	int LIDAR_COORDS_LEN = ST_LIDAR_COORDS_LEN;
	int GRID_WIDTH = ST_GRID_WIDTH;
	int GRID_HEIGHT = ST_GRID_HEIGHT;
	
	int xmin = ST_xmin;
	int xmax = ST_xmax;
	int ymin = ST_ymin;
	int ymax = ST_ymax;
	
	float res = ST_res;

	int xmin_pre = xmin;
	int xmax_pre = xmax;
	int ymin_pre = ymin;
	int ymax_pre = ymax;

	printf("xmin=%d, xmax=%d, ymin=%d, ymax=%d\n", xmin, xmax, ymin, ymax);
	printf("xmin=%d, xmax=%d, ymin=%d, ymax=%d\n", ST_xmin, ST_xmax, ST_ymin, ST_ymax);

	sz_lidar_coords = 2 * LIDAR_COORDS_LEN * sizeof(float);
	sz_should_extend = 4 * sizeof(int);
	sz_grid_map = GRID_WIDTH * GRID_HEIGHT * sizeof(int);
	sz_log_odds = GRID_WIDTH * GRID_HEIGHT * sizeof(float);

	gpuErrchk(cudaMalloc((void**)&d_lidar_coords, sz_lidar_coords));
	gpuErrchk(cudaMalloc((void**)&d_should_extend, sz_should_extend));
	gpuErrchk(cudaMalloc((void**)&d_grid_map, sz_grid_map));
	gpuErrchk(cudaMalloc((void**)&d_log_odds, sz_log_odds));

	res_should_extend = (int*)malloc(sz_should_extend);

	gpuErrchk(cudaMemcpy(d_lidar_coords, h_lidar_coords, sz_lidar_coords, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_grid_map, h_grid_map, sz_grid_map, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_log_odds, h_log_odds, sz_log_odds, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemset(d_should_extend, 0, sz_should_extend));


	size_t sz_transition_frames = 9 * sizeof(float);
	size_t sz_particles_world_pos = LIDAR_COORDS_LEN * sizeof(float);
	size_t sz_position_image = 2 * sizeof(int);

	float* res_transition_world_lidar = (float*)malloc(sz_transition_frames);
	float* res_particles_world_x = (float*)malloc(sz_particles_world_pos);
	float* res_particles_world_y = (float*)malloc(sz_particles_world_pos);
	int* res_position_image_body = (int*)malloc(sz_position_image);

	float* d_transition_body_lidar = NULL;
	float* d_transition_world_body = NULL;
	float* d_transition_world_lidar = NULL;
	float* d_particles_world_x = NULL;
	float* d_particles_world_y = NULL;
	int* d_position_image_body = NULL;

	gpuErrchk(cudaMalloc((void**)&d_transition_body_lidar, sz_transition_frames));
	gpuErrchk(cudaMalloc((void**)&d_transition_world_body, sz_transition_frames));
	gpuErrchk(cudaMalloc((void**)&d_transition_world_lidar, sz_transition_frames));
	gpuErrchk(cudaMalloc((void**)&d_particles_world_x, sz_particles_world_pos));
	gpuErrchk(cudaMalloc((void**)&d_particles_world_y, sz_particles_world_pos));
	gpuErrchk(cudaMalloc((void**)&d_position_image_body, sz_position_image));

	gpuErrchk(cudaMemcpy(d_transition_body_lidar, h_transition_body_lidar, sz_transition_frames, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_transition_world_body, h_transition_world_body, sz_transition_frames, cudaMemcpyHostToDevice));


	auto start_check_extend = std::chrono::high_resolution_clock::now();
	kernel_matrix_mul_3x3 << < 1, 1 >> > (d_transition_world_body, d_transition_body_lidar, d_transition_world_lidar);
	cudaDeviceSynchronize();

	threadsPerBlock = 1;
	blocksPerGrid = LIDAR_COORDS_LEN;
	kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (d_particles_world_x, d_particles_world_y, d_transition_world_lidar,
		d_lidar_coords, LIDAR_COORDS_LEN);
	cudaDeviceSynchronize();

	threadsPerBlock = 256;
	blocksPerGrid = (LIDAR_COORDS_LEN + threadsPerBlock - 1) / threadsPerBlock;
	kernel_check_map_extend_less << <blocksPerGrid, threadsPerBlock >> > (d_particles_world_x, xmin, d_should_extend, 0, 0, LIDAR_COORDS_LEN);
	kernel_check_map_extend_less << <blocksPerGrid, threadsPerBlock >> > (d_particles_world_y, ymin, d_should_extend, 1, 0, LIDAR_COORDS_LEN);

	kernel_check_map_extend_greater << <blocksPerGrid, threadsPerBlock >> > (d_particles_world_x, xmax, d_should_extend, 2, 0, LIDAR_COORDS_LEN);
	kernel_check_map_extend_greater << <blocksPerGrid, threadsPerBlock >> > (d_particles_world_y, ymax, d_should_extend, 3, 0, LIDAR_COORDS_LEN);
	cudaDeviceSynchronize();
	gpuErrchk(cudaMemcpy(res_should_extend, d_should_extend, sz_should_extend, cudaMemcpyDeviceToHost));

	bool EXTEND = false;
	if (res_should_extend[0] != 0) {
		EXTEND = true;
		xmin = xmin * 2;
	}
	else if (res_should_extend[2] != 0) {
		EXTEND = true;
		xmax = xmax * 2;
	}
	else if (res_should_extend[1] != 0) {
		EXTEND = true;
		ymin = ymin * 2;
	}
	else if (res_should_extend[3] != 0) {
		EXTEND = true;
		ymax = ymax * 2;
	}
	auto stop_check_extend = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < 4; i++)
		std::cout << "Should Extend: " << res_should_extend[i] << std::endl;

	printf("xmin=%d, xmax=%d, ymin=%d, ymax=%d\n", xmin, xmax, ymin, ymax);
	printf("xmin=%d, xmax=%d, ymin=%d, ymax=%d\n", AF_xmin, AF_xmax, AF_ymin, AF_ymax);
	assert(EXTEND == ST_EXTEND);

	if (EXTEND == true) {

		auto start_extend = std::chrono::high_resolution_clock::now();

		sz_coord = 2 * sizeof(int);
		gpuErrchk(cudaMalloc((void**)&d_coord, sz_coord));
		res_coord = (int*)malloc(sz_coord);

		kernel_position_to_image << <1, 1 >> > (d_coord, xmin_pre, ymax_pre, res, xmin, ymax);
		cudaDeviceSynchronize();
		
		gpuErrchk(cudaMemcpy(res_coord, d_coord, sz_coord, cudaMemcpyDeviceToHost));

		int* dc_grid_map = NULL;
		gpuErrchk(cudaMalloc((void**)&dc_grid_map, sz_grid_map));
		gpuErrchk(cudaMemcpy(dc_grid_map, d_grid_map, sz_grid_map, cudaMemcpyDeviceToDevice));

		float* dc_log_odds = NULL;
		gpuErrchk(cudaMalloc((void**)&dc_log_odds, sz_log_odds));
		gpuErrchk(cudaMemcpy(dc_log_odds, d_log_odds, sz_log_odds, cudaMemcpyDeviceToDevice));

		const int PRE_GRID_WIDTH = GRID_WIDTH;
		const int PRE_GRID_HEIGHT = GRID_HEIGHT;
		GRID_WIDTH  = ceil((ymax - ymin) / res + 1);
		GRID_HEIGHT = ceil((xmax - xmin) / res + 1);
		//printf("GRID_WIDTH=%d, AF_GRID_WIDTH=%d, GRID_HEIGHT=%d, AF_GRID_HEIGHT=%d\n", GRID_WIDTH, GRID_HEIGHT, AF_GRID_WIDTH, AF_GRID_HEIGHT);
		assert(GRID_WIDTH == AF_GRID_WIDTH);
		assert(GRID_HEIGHT == AF_GRID_HEIGHT);

		//gpuErrchk(cudaFree(d_grid_map));

		sz_grid_map = GRID_WIDTH * GRID_HEIGHT * sizeof(int);
		gpuErrchk(cudaMalloc((void**)&d_grid_map, sz_grid_map));
		gpuErrchk(cudaMemset(d_grid_map, 0, sz_grid_map));

		sz_log_odds = GRID_WIDTH * GRID_HEIGHT * sizeof(float);
		gpuErrchk(cudaMalloc((void**)&d_log_odds, sz_log_odds));
		gpuErrchk(cudaMemset(d_log_odds, LOG_ODD_PRIOR, sz_log_odds));

		res_grid_map = (int*)malloc(sz_grid_map);
		res_log_odds = (float*)malloc(sz_log_odds);

		const int GRID_SIZE = PRE_GRID_WIDTH * PRE_GRID_HEIGHT;
		threadsPerBlock = 256;
		blocksPerGrid = (GRID_SIZE + threadsPerBlock - 1) / threadsPerBlock;
		kernel_2d_copy_with_offset << <blocksPerGrid, threadsPerBlock >> > (d_grid_map, d_log_odds, dc_grid_map, dc_log_odds, res_coord[0], res_coord[1],
				PRE_GRID_HEIGHT, GRID_HEIGHT, GRID_SIZE);
		cudaDeviceSynchronize();

		auto stop_extend = std::chrono::high_resolution_clock::now();
		auto duration_extend = std::chrono::duration_cast<std::chrono::microseconds>(stop_extend - start_extend);

		std::cout << std::endl;
		std::cout << "Time taken by function (Extend): " << duration_extend.count() << " microseconds" << std::endl;

		gpuErrchk(cudaMemcpy(res_grid_map, d_grid_map, sz_grid_map, cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(res_log_odds, d_log_odds, sz_log_odds, cudaMemcpyDeviceToHost));

		int error_map = 0;
		int error_log = 0;
		for (int i = 0; i < (GRID_WIDTH * GRID_HEIGHT); i++) {
			if (res_grid_map[i] != bg_grid_map[i]) {
				error_map += 1;
				//printf("Grid Map: %d <> %d\n", res_grid_map[i], bg_grid_map[i]);
			}
			if ( (res_log_odds[i] - bg_log_odds[i]) > 1e-4 ) {
				error_log += 1;
				//printf("Log Odds: (%d) %f <> %f\n", i, res_log_odds[i], bg_log_odds[i]);
			}
			if (error_log > 200)
				break;
		}
		printf("Map Erros: %d\n", error_map);
		printf("Log Erros: %d\n", error_log);
	}

	auto duration_check_extend = std::chrono::duration_cast<std::chrono::microseconds>(stop_check_extend - start_check_extend);

	std::cout << std::endl;
	std::cout << "Time taken by function (Check Extend): " << duration_check_extend.count() << " microseconds" << std::endl;
	std::cout << std::endl;
}

#endif
