#ifndef _TEST_MAP_EXTEND_H_
#define _TEST_MAP_EXTEND_H_

#include "headers.h"
#include "host_asserts.h"
#include "host_utils.h"
#include "kernels.cuh"

#include "data/map_ignore_extend/1000.h"
//#include "data//map_extend/721.h"
//#include "data/map_extend/2789.h"


#ifndef ST_res
#define ST_GRID_WIDTH 0
#define ST_GRID_HEIGHT 0
#define ST_xmin 0
#define ST_xmax 0
#define ST_ymin 0
#define ST_ymax 0
#define ST_res 0
int* pre_grid_map = NULL;
int* post_grid_map = NULL;
#endif

void alloc_();

size_t sz_lidar_coords = 0;
size_t sz_should_extend = 0;
size_t sz_coord = 0;
size_t sz_grid_map = 0;


float* d_lidar_coords = NULL;
int* d_should_extend = NULL;
int* d_coord = NULL;
int* d_grid_map = NULL;

int* res_should_extend = NULL;
int* res_coord = NULL;
int* res_grid_map = NULL;

int threadsPerBlock = 0;
int blocksPerGrid = 0;


void test_map_extend_main() {

	std::cout << "Starting 'Test Map Extend'" << std::endl;

	int LIDAR_COORDS_LEN = ST_LIDAR_COORDS_LEN;
	int GRID_WIDTH = ST_GRID_WIDTH;
	int GRID_HEIGHT = ST_GRID_HEIGHT;
	int xmin = ST_xmin_pre;
	int xmin_pre = xmin;
	int xmax = ST_xmax_pre;
	int xmax_pre = xmax;
	int ymin = ST_ymin_pre;
	int ymin_pre = ymin;
	int ymax = ST_ymax_pre;
	int ymax_pre = ymax;
	float res = ST_res;

	printf("xmin=%d, xmax=%d, ymin=%d, ymax=%d\n", xmin, xmax, ymin, ymax);
	printf("xmin=%d, xmax=%d, ymin=%d, ymax=%d\n", ST_xmin, ST_xmax, ST_ymin, ST_ymax);

	sz_lidar_coords = 2 * LIDAR_COORDS_LEN * sizeof(float);
	sz_should_extend = 4 * sizeof(int);
	sz_grid_map = GRID_WIDTH * GRID_HEIGHT * sizeof(int);

	gpuErrchk(cudaMalloc((void**)&d_lidar_coords, sz_lidar_coords));
	gpuErrchk(cudaMalloc((void**)&d_should_extend, sz_lidar_coords));
	gpuErrchk(cudaMalloc((void**)&d_grid_map, sz_grid_map));

	res_should_extend = (int*)malloc(sz_should_extend);

	gpuErrchk(cudaMemcpy(d_lidar_coords, h_lidar_coords, sz_lidar_coords, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_grid_map, pre_grid_map, sz_grid_map, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemset(d_should_extend, 0, sz_should_extend));

	auto start_kernel = std::chrono::high_resolution_clock::now();
	threadsPerBlock = 256;
	blocksPerGrid = (LIDAR_COORDS_LEN + threadsPerBlock - 1) / threadsPerBlock;
	kernel_check_map_extend_less << <blocksPerGrid, threadsPerBlock >> > (d_lidar_coords, xmin, d_should_extend, 0, 0, LIDAR_COORDS_LEN);
	kernel_check_map_extend_less << <blocksPerGrid, threadsPerBlock >> > (d_lidar_coords, ymin, d_should_extend, 1, LIDAR_COORDS_LEN, LIDAR_COORDS_LEN);

	kernel_check_map_extend_greater << <blocksPerGrid, threadsPerBlock >> > (d_lidar_coords, xmax, d_should_extend, 2, 0, LIDAR_COORDS_LEN);
	kernel_check_map_extend_greater << <blocksPerGrid, threadsPerBlock >> > (d_lidar_coords, ymax, d_should_extend, 3, LIDAR_COORDS_LEN, LIDAR_COORDS_LEN);
	cudaDeviceSynchronize();
	auto stop_kernel = std::chrono::high_resolution_clock::now();

	auto start_copy = std::chrono::high_resolution_clock::now();
	gpuErrchk(cudaMemcpy(res_should_extend, d_should_extend, sz_should_extend, cudaMemcpyDeviceToHost));
	auto stop_copy = std::chrono::high_resolution_clock::now();

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

	printf("xmin=%d, xmax=%d, ymin=%d, ymax=%d\n", xmin, xmax, ymin, ymax);

	if (EXTEND == true) {
		sz_coord = 2 * sizeof(int);
		gpuErrchk(cudaMalloc((void**)&d_coord, sz_coord));
		res_coord = (int*)malloc(sz_coord);

		kernel_position_to_image << <1, 1 >> > (d_coord, xmin_pre, ymax_pre, res, xmin, ymax);
		cudaDeviceSynchronize();
		
		gpuErrchk(cudaMemcpy(res_coord, d_coord, sz_coord, cudaMemcpyDeviceToHost));

		int* dc_grid_map = NULL;
		gpuErrchk(cudaMalloc((void**)&dc_grid_map, sz_grid_map));
		gpuErrchk(cudaMemcpy(dc_grid_map, d_grid_map, sz_grid_map, cudaMemcpyDeviceToDevice));

		const int PRE_GRID_WIDTH = GRID_WIDTH;
		const int PRE_GRID_HEIGHT = GRID_HEIGHT;
		GRID_WIDTH  = ceil((ymax - ymin) / res + 1);
		GRID_HEIGHT = ceil((xmax - xmin) / res + 1);
		printf("GRID_WIDTH=%d, AF_GRID_WIDTH=%d, GRID_HEIGHT=%d, AF_GRID_HEIGHT=%d\n", GRID_WIDTH, GRID_HEIGHT, AF_GRID_WIDTH, AF_GRID_HEIGHT);
		assert(GRID_WIDTH == AF_GRID_WIDTH);
		assert(GRID_HEIGHT == AF_GRID_HEIGHT);

		gpuErrchk(cudaFree(d_grid_map));

		sz_grid_map = GRID_WIDTH * GRID_HEIGHT * sizeof(int);
		gpuErrchk(cudaMalloc((void**)&d_grid_map, sz_grid_map));
		gpuErrchk(cudaMemset(d_grid_map, 0, sz_grid_map));
		res_grid_map = (int*)malloc(sz_grid_map);

		const int GRID_SIZE = PRE_GRID_WIDTH * PRE_GRID_HEIGHT;
		threadsPerBlock = 256;
		blocksPerGrid = (GRID_SIZE + threadsPerBlock - 1) / threadsPerBlock;
		kernel_2d_copy_with_offset << <blocksPerGrid, threadsPerBlock >> > (d_grid_map, dc_grid_map, res_coord[0], res_coord[1],
			PRE_GRID_HEIGHT, GRID_HEIGHT, GRID_SIZE);
		cudaDeviceSynchronize();

		gpuErrchk(cudaMemcpy(res_grid_map, d_grid_map, sz_grid_map, cudaMemcpyDeviceToHost));

		int error_count = 0;
		for (int i = 0; i < (GRID_WIDTH * GRID_HEIGHT); i++) {
			if (res_grid_map[i] != post_grid_map[i]) {
				error_count += 1;
				printf("%d <> %d\n", res_grid_map[i], post_grid_map[i]);
			}
		}
		printf("Erros: %d\n", error_count);
	}

	for (int i = 0; i < 4; i++)
		std::cout << "Should Extend: " << res_should_extend[i] << std::endl;

	auto duration_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_kernel - start_kernel);
	auto duration_copy = std::chrono::duration_cast<std::chrono::microseconds>(stop_copy - start_copy);

	std::cout << std::endl;
	std::cout << "Time taken by function (Kernel): " << duration_kernel.count() << " microseconds" << std::endl;
	std::cout << "Time taken by function (Copy): " << duration_copy.count() << " microseconds" << std::endl;
}

#endif
