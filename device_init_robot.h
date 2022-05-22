#ifndef _DEVICE_INIT_ROBOT_H_
#define _DEVICE_INIT_ROBOT_H_

#include "headers.h"
#include "host_utils.h"


void alloc_init_state_vars(DeviceState& d_state, HostState& res_state, HostState& h_state) {

	d_state.x.resize(NUM_PARTICLES, 0);
	d_state.y.resize(NUM_PARTICLES, 0);
	d_state.theta.resize(NUM_PARTICLES, 0);
	d_state.rnds_encoder_counts.resize(NUM_PARTICLES, 0);
	d_state.rnds_yaws.resize(NUM_PARTICLES, 0);

	d_state.x.assign(h_state.x.begin(), h_state.x.end());
	d_state.y.assign(h_state.y.begin(), h_state.y.end());
	d_state.theta.assign(h_state.theta.begin(), h_state.theta.end());
	d_state.rnds_encoder_counts.assign(h_state.rnds_encoder_counts.begin(), h_state.rnds_encoder_counts.end());
	d_state.rnds_yaws.assign(h_state.rnds_yaws.begin(), h_state.rnds_yaws.end());

	res_state.x.resize(NUM_PARTICLES, 0);
	res_state.y.resize(NUM_PARTICLES, 0);
	res_state.theta.resize(NUM_PARTICLES, 0);
	res_state.rnds_encoder_counts.resize(NUM_PARTICLES, 0);
	res_state.rnds_yaws.resize(NUM_PARTICLES, 0);

	res_state.x.assign(h_state.x.begin(), h_state.x.end());
	res_state.y.assign(h_state.y.begin(), h_state.y.end());
	res_state.theta.assign(h_state.theta.begin(), h_state.theta.end());
	res_state.rnds_encoder_counts.assign(h_state.rnds_encoder_counts.begin(), h_state.rnds_encoder_counts.end());
	res_state.rnds_yaws.assign(h_state.rnds_yaws.begin(), h_state.rnds_yaws.end());

	res_state.encoder_counts = h_state.encoder_counts;
	res_state.yaw = h_state.yaw;
	res_state.dt = h_state.dt;
	res_state.nv = h_state.nv;
	res_state.nw = h_state.nw;
}



#endif
