#ifndef _TEST_ROBOT_ADVANCE_
#define _TEST_ROBOT_ADVANCE_

#include "headers.h"
#include "host_asserts.h"
#include "host_utils.h"
#include "kernels.cuh"

#include "data/robot_advance/4900.h"

void alloc_states();
void init_states();
void exec_robot_advance();

void assertResults();

/********************************************************************/
/************************* STATES VARIABLES *************************/
/********************************************************************/
size_t  sz_states_pos = 0;

float* d_states_x;
float* d_states_y;
float* d_states_theta;

float* res_states_x;
float* res_states_y;
float* res_states_theta;

float* d_rnds_encoder_counts;
float* d_rnds_yaws;


void test_robot_advance_main() {

	std::cout << "Start Robot Advance" << std::endl;

	alloc_states();
	init_states();

	auto start_robot_advance_kernel = std::chrono::high_resolution_clock::now();
	exec_robot_advance();
	auto stop_robot_advance_kernel = std::chrono::high_resolution_clock::now();

	assertResults();

	auto duration_robot_advance_total = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_advance_kernel - start_robot_advance_kernel);
	std::cout << std::endl;
	std::cout << "Time taken by function (Robot Advance Kernel): " << duration_robot_advance_total.count() << " microseconds" << std::endl;
}

void alloc_states() {

	sz_states_pos = NUM_PARTICLES * sizeof(float);

	gpuErrchk(cudaMalloc((void**)&d_states_x, sz_states_pos));
	gpuErrchk(cudaMalloc((void**)&d_states_y, sz_states_pos));
	gpuErrchk(cudaMalloc((void**)&d_states_theta, sz_states_pos));

	gpuErrchk(cudaMalloc((void**)&d_rnds_encoder_counts, sz_states_pos));
	gpuErrchk(cudaMalloc((void**)&d_rnds_yaws, sz_states_pos));

	res_states_x = (float*)malloc(sz_states_pos);
	res_states_y = (float*)malloc(sz_states_pos);
	res_states_theta = (float*)malloc(sz_states_pos);
}

void init_states() {

	gpuErrchk(cudaMemcpy(d_states_x, pre_states_x, sz_states_pos, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_states_y, pre_states_y, sz_states_pos, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_states_theta, pre_states_theta, sz_states_pos, cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpy(d_rnds_encoder_counts, h_rnds_encoder_counts, sz_states_pos, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_rnds_yaws, h_rnds_yaws, sz_states_pos, cudaMemcpyHostToDevice));
}

void exec_robot_advance() {

	int threadsPerBlock = NUM_PARTICLES;
	int blocksPerGrid = 1;
	kernel_robot_advance << <blocksPerGrid, threadsPerBlock >> > (d_states_x, d_states_y, d_states_theta,
		d_rnds_encoder_counts, d_rnds_yaws,
		ST_encoder_counts, ST_yaw, ST_dt, ST_nv, ST_nw);
	cudaDeviceSynchronize();
}

void assertResults() {

	gpuErrchk(cudaMemcpy(res_states_x, d_states_x, sz_states_pos, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(res_states_y, d_states_y, sz_states_pos, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(res_states_theta, d_states_theta, sz_states_pos, cudaMemcpyDeviceToHost));

	for (int i = 0; i < NUM_PARTICLES; i++) {

		if(abs(res_states_x[i] - post_states_x[i]) > 1e-4)
			printf("i=%d, x=%f, %f\n", i, res_states_x[i], post_states_x[i]);
		if(abs(res_states_y[i] - post_states_y[i]) > 1e-4)
			printf("i=%d, y=%f, %f\n", i, res_states_y[i], post_states_y[i]);
		if(abs(res_states_theta[i] - post_states_theta[i]) > 1e-4)
			printf("i=%d, theta=%f, %f\n", i, res_states_theta[i], post_states_theta[i]);
	}
}

#endif
