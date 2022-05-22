#ifndef _DEVICE_EXEC_ROBOT_H_
#define _DEVICE_EXEC_ROBOT_H_

#include "headers.h"
#include "structures.h"
#include "kernels.cuh"


void exec_robot_advance(DeviceState& d_state, HostState& res_state) {

	int threadsPerBlock = NUM_PARTICLES;
	int blocksPerGrid = 1;
	kernel_robot_advance << <blocksPerGrid, threadsPerBlock >> > (
		THRUST_RAW_CAST(d_state.x), THRUST_RAW_CAST(d_state.y), THRUST_RAW_CAST(d_state.theta), SEP,
		THRUST_RAW_CAST(d_state.rnds_encoder_counts), THRUST_RAW_CAST(d_state.rnds_yaws),
		res_state.encoder_counts, res_state.yaw, res_state.dt, res_state.nv, res_state.nw, NUM_PARTICLES);
	cudaDeviceSynchronize();
}


#endif
