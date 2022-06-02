#ifndef _DEVICE_INIT_ROBOT_H_
#define _DEVICE_INIT_ROBOT_H_

#include "headers.h"
#include "host_utils.h"


void alloc_init_state_vars(DeviceState& d_state, DeviceState& d_clone_state, HostState& h_state, 
    HostRobotState& h_robot_state, HostState& pre_state) {

	d_state.c_x.resize(NUM_PARTICLES, 0);
	d_state.c_y.resize(NUM_PARTICLES, 0);
	d_state.c_theta.resize(NUM_PARTICLES, 0);
	d_state.c_rnds_encoder_counts.resize(NUM_PARTICLES, 0);
	d_state.c_rnds_yaws.resize(NUM_PARTICLES, 0);

	d_state.c_x.assign(pre_state.c_x.begin(), pre_state.c_x.end());
	d_state.c_y.assign(pre_state.c_y.begin(), pre_state.c_y.end());
	d_state.c_theta.assign(pre_state.c_theta.begin(), pre_state.c_theta.end());
	d_state.c_rnds_encoder_counts.assign(pre_state.c_rnds_encoder_counts.begin(), pre_state.c_rnds_encoder_counts.end());
	d_state.c_rnds_yaws.assign(pre_state.c_rnds_yaws.begin(), pre_state.c_rnds_yaws.end());

    d_clone_state.c_x.resize(NUM_PARTICLES, 0);
    d_clone_state.c_y.resize(NUM_PARTICLES, 0);
    d_clone_state.c_theta.resize(NUM_PARTICLES, 0);

	h_state.c_x.resize(NUM_PARTICLES, 0);
	h_state.c_y.resize(NUM_PARTICLES, 0);
	h_state.c_theta.resize(NUM_PARTICLES, 0);
	h_state.c_rnds_encoder_counts.resize(NUM_PARTICLES, 0);
	h_state.c_rnds_yaws.resize(NUM_PARTICLES, 0);

	h_state.c_x.assign(pre_state.c_x.begin(), pre_state.c_x.end());
	h_state.c_y.assign(pre_state.c_y.begin(), pre_state.c_y.end());
	h_state.c_theta.assign(pre_state.c_theta.begin(), pre_state.c_theta.end());
	h_state.c_rnds_encoder_counts.assign(pre_state.c_rnds_encoder_counts.begin(), pre_state.c_rnds_encoder_counts.end());
	h_state.c_rnds_yaws.assign(pre_state.c_rnds_yaws.begin(), pre_state.c_rnds_yaws.end());

	h_state.encoder_counts = pre_state.encoder_counts;
	h_state.yaw = pre_state.yaw;
	h_state.dt = pre_state.dt;
	h_state.nv = pre_state.nv;
	h_state.nw = pre_state.nw;

    h_robot_state.state.resize(3, 0);
}


void alloc_init_robot_particles_vars(DeviceRobotParticles& d_robot_particles, HostRobotParticles& h_robot_particles,
    HostRobotParticles& pre_robot_particles) {

    h_robot_particles.LEN = pre_robot_particles.LEN;
    h_robot_particles.f_x.resize(h_robot_particles.LEN, 0);
    h_robot_particles.f_y.resize(h_robot_particles.LEN, 0);
    h_robot_particles.c_idx.resize(NUM_PARTICLES, 0);
    h_robot_particles.f_extended_idx.resize(h_robot_particles.LEN, 0);
    h_robot_particles.c_weight.resize(NUM_PARTICLES, 0);
    h_robot_particles.f_extended_idx.resize(h_robot_particles.LEN, 0);

    d_robot_particles.f_x.resize(h_robot_particles.LEN, 0);
    d_robot_particles.f_y.resize(h_robot_particles.LEN, 0);
    d_robot_particles.c_idx.resize(NUM_PARTICLES, 0);
    d_robot_particles.c_weight.resize(NUM_PARTICLES, 0);
    d_robot_particles.f_extended_idx.resize(h_robot_particles.LEN, 0);

    d_robot_particles.f_x.assign(pre_robot_particles.f_x.begin(), pre_robot_particles.f_x.end());
    d_robot_particles.f_y.assign(pre_robot_particles.f_y.begin(), pre_robot_particles.f_y.end());
    d_robot_particles.c_idx.assign(pre_robot_particles.c_idx.begin(), pre_robot_particles.c_idx.end());
    d_robot_particles.c_weight.assign(pre_robot_particles.c_weight.begin(), pre_robot_particles.c_weight.end());
}

void alloc_correlation_vars(DeviceCorrelation& d_correlation, HostCorrelation& h_correlation) {

    d_correlation.c_weight.resize(NUM_PARTICLES, 0);
    d_correlation.c_raw.resize(25 * NUM_PARTICLES, 0);
    d_correlation.c_sum_exp.resize(1, 0);
    d_correlation.c_max.resize(1, 0);
                  
    h_correlation.c_weight.resize(NUM_PARTICLES, 0);
    h_correlation.c_raw.resize(25 * NUM_PARTICLES, 0);
    h_correlation.c_sum_exp.resize(1, 0);
    h_correlation.c_max.resize(1, 0);
}

void alloc_particles_transition_vars(DeviceParticlesTransition& d_particles_transition, 
    DeviceParticlesPosition& d_particles_position, DeviceParticlesRotation& d_particles_rotation,
    HostParticlesTransition& h_particles_transition, HostParticlesPosition& h_particles_position,
    HostParticlesRotation& h_particles_rotation) {

    d_particles_transition.c_world_body.resize(9 * NUM_PARTICLES, 0);
    d_particles_transition.c_world_lidar.resize(9 * NUM_PARTICLES, 0);

    d_particles_position.c_world_body.resize(2 * NUM_PARTICLES, 0);
    d_particles_rotation.c_world_body.resize(4 * NUM_PARTICLES, 0);

    h_particles_transition.c_world_body.resize(9 * NUM_PARTICLES, 0);
    h_particles_transition.c_world_lidar.resize(9 * NUM_PARTICLES, 0);

    h_particles_position.c_world_body.resize(2 * NUM_PARTICLES, 0);
    //h_particles_rotation.world_body.resize(4 * NUM_PARTICLES, 0);
}

void alloc_init_processed_measurement_vars(DeviceProcessedMeasure& d_processed_measure, HostProcessedMeasure& h_processed_measure,
    HostMeasurements& h_measurements) {

    d_processed_measure.v_x.resize(NUM_PARTICLES * h_measurements.LEN, 0);
    d_processed_measure.v_y.resize(NUM_PARTICLES * h_measurements.LEN, 0);
    d_processed_measure.v_idx.resize(NUM_PARTICLES * h_measurements.LEN, 0);

    h_processed_measure.v_x.resize(NUM_PARTICLES * h_measurements.LEN);
    h_processed_measure.v_y.resize(NUM_PARTICLES * h_measurements.LEN);
}

void alloc_map_2d_var(Device2DUniqueFinder& d_2d_unique, Host2DUniqueFinder& h_2d_unique, HostMap& h_map, bool alloc) {

    const int UNIQUE_COUNTER_LEN = NUM_PARTICLES + 1;

    if (alloc == true) {
        d_2d_unique.s_map.clear();
        d_2d_unique.s_map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT * NUM_PARTICLES, 0);
        d_2d_unique.c_in_map.clear();
        d_2d_unique.c_in_map.resize(UNIQUE_COUNTER_LEN, 0);
        d_2d_unique.s_in_col.clear();
        d_2d_unique.s_in_col.resize(UNIQUE_COUNTER_LEN * h_map.GRID_WIDTH, 0);

        h_2d_unique.s_map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT * NUM_PARTICLES, 0);
        h_2d_unique.c_in_map.resize(UNIQUE_COUNTER_LEN, 0);
        h_2d_unique.s_in_col.resize(UNIQUE_COUNTER_LEN * h_map.GRID_WIDTH, 0);
    }
    else {
        thrust::fill(d_2d_unique.s_map.begin(), d_2d_unique.s_map.end(), 0);
        thrust::fill(d_2d_unique.c_in_map.begin(), d_2d_unique.c_in_map.end(), 0);
        thrust::fill(d_2d_unique.s_in_col.begin(), d_2d_unique.s_in_col.end(), 0);

        thrust::fill(h_2d_unique.s_map.begin(), h_2d_unique.s_map.end(), 0);
        thrust::fill(h_2d_unique.c_in_map.begin(), h_2d_unique.c_in_map.end(), 0);
        thrust::fill(h_2d_unique.s_in_col.begin(), h_2d_unique.s_in_col.end(), 0);
    }
}

void alloc_resampling_vars(DeviceResampling& d_resampling, HostResampling& h_resampling, HostResampling& pre_resampling) {

    h_resampling.c_js.resize(NUM_PARTICLES, 0);

    d_resampling.c_js.resize(NUM_PARTICLES, 0);
    d_resampling.c_rnds.resize(NUM_PARTICLES, 0);

    d_resampling.c_rnds.assign(pre_resampling.c_rnds.begin(), pre_resampling.c_rnds.end());
}


#endif
