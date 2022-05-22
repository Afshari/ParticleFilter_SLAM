#ifndef _DEVICE_ASSERT_ROBOT_H_
#define _DEVICE_ASSERT_ROBOT_H_

#include "headers.h"
#include "host_asserts.h"
#include "structures.h"


void assert_robot_advance_results(DeviceState& d_state, HostState& res_state, HostState& post_state) {

	res_state.x.assign(d_state.x.begin(), d_state.x.end());
	res_state.y.assign(d_state.y.begin(), d_state.y.end());
	res_state.theta.assign(d_state.theta.begin(), d_state.theta.end());

	for (int i = 0; i < NUM_PARTICLES; i++) {

		if (abs(res_state.x[i] - post_state.x[i]) > 1e-4)
			printf("i=%d, x=%f, %f\n", i, res_state.x[i], post_state.x[i]);
		if (abs(res_state.y[i] - post_state.y[i]) > 1e-4)
			printf("i=%d, y=%f, %f\n", i, res_state.y[i], post_state.y[i]);
		if (abs(res_state.theta[i] - post_state.theta[i]) > 1e-4)
			printf("i=%d, theta=%f, %f\n", i, res_state.theta[i], post_state.theta[i]);
	}
}


#endif
