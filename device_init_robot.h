#ifndef _DEVICE_INIT_ROBOT_H_
#define _DEVICE_INIT_ROBOT_H_

#include "headers.h"
#include "host_utils.h"


void alloc_init_state_vars(DeviceState& d_state, DeviceState& d_clone_state, HostState& h_state, 
    HostRobotState& h_robot_state, HostState& pre_state) {

	d_state.x.resize(NUM_PARTICLES, 0);
	d_state.y.resize(NUM_PARTICLES, 0);
	d_state.theta.resize(NUM_PARTICLES, 0);
	d_state.rnds_encoder_counts.resize(NUM_PARTICLES, 0);
	d_state.rnds_yaws.resize(NUM_PARTICLES, 0);

	d_state.x.assign(pre_state.x.begin(), pre_state.x.end());
	d_state.y.assign(pre_state.y.begin(), pre_state.y.end());
	d_state.theta.assign(pre_state.theta.begin(), pre_state.theta.end());
	d_state.rnds_encoder_counts.assign(pre_state.rnds_encoder_counts.begin(), pre_state.rnds_encoder_counts.end());
	d_state.rnds_yaws.assign(pre_state.rnds_yaws.begin(), pre_state.rnds_yaws.end());

    d_clone_state.x.resize(NUM_PARTICLES, 0);
    d_clone_state.y.resize(NUM_PARTICLES, 0);
    d_clone_state.theta.resize(NUM_PARTICLES, 0);

	h_state.x.resize(NUM_PARTICLES, 0);
	h_state.y.resize(NUM_PARTICLES, 0);
	h_state.theta.resize(NUM_PARTICLES, 0);
	h_state.rnds_encoder_counts.resize(NUM_PARTICLES, 0);
	h_state.rnds_yaws.resize(NUM_PARTICLES, 0);

	h_state.x.assign(pre_state.x.begin(), pre_state.x.end());
	h_state.y.assign(pre_state.y.begin(), pre_state.y.end());
	h_state.theta.assign(pre_state.theta.begin(), pre_state.theta.end());
	h_state.rnds_encoder_counts.assign(pre_state.rnds_encoder_counts.begin(), pre_state.rnds_encoder_counts.end());
	h_state.rnds_yaws.assign(pre_state.rnds_yaws.begin(), pre_state.rnds_yaws.end());

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
    h_robot_particles.x.resize(h_robot_particles.LEN, 0);
    h_robot_particles.y.resize(h_robot_particles.LEN, 0);
    h_robot_particles.idx.resize(NUM_PARTICLES, 0);
    h_robot_particles.extended_idx.resize(h_robot_particles.LEN, 0);
    h_robot_particles.weight.resize(NUM_PARTICLES, 0);
    h_robot_particles.extended_idx.resize(h_robot_particles.LEN, 0);

    d_robot_particles.x.resize(h_robot_particles.LEN, 0);
    d_robot_particles.y.resize(h_robot_particles.LEN, 0);
    d_robot_particles.idx.resize(NUM_PARTICLES, 0);
    d_robot_particles.weight.resize(NUM_PARTICLES, 0);
    d_robot_particles.extended_idx.resize(h_robot_particles.LEN, 0);

    d_robot_particles.x.assign(pre_robot_particles.x.begin(), pre_robot_particles.x.end());
    d_robot_particles.y.assign(pre_robot_particles.y.begin(), pre_robot_particles.y.end());
    d_robot_particles.idx.assign(pre_robot_particles.idx.begin(), pre_robot_particles.idx.end());
    d_robot_particles.weight.assign(pre_robot_particles.weight.begin(), pre_robot_particles.weight.end());
}

void alloc_correlation_vars(DeviceCorrelation& d_correlation, HostCorrelation& h_correlation) {

    d_correlation.weight.resize(NUM_PARTICLES, 0);
    d_correlation.raw.resize(25 * NUM_PARTICLES, 0);
    d_correlation.sum_exp.resize(1, 0);
    d_correlation.max.resize(1, 0);

    h_correlation.weight.resize(NUM_PARTICLES, 0);
    h_correlation.raw.resize(25 * NUM_PARTICLES, 0);
    h_correlation.sum_exp.resize(1, 0);
    h_correlation.max.resize(1, 0);
}

void alloc_particles_transition_vars(DeviceParticlesTransition& d_particles_transition, 
    DeviceParticlesPosition& d_particles_position, DeviceParticlesRotation& d_particles_rotation,
    HostParticlesTransition& h_particles_transition, HostParticlesPosition& h_particles_position,
    HostParticlesRotation& h_particles_rotation) {

    d_particles_transition.world_body.resize(9 * NUM_PARTICLES, 0);
    d_particles_transition.world_lidar.resize(9 * NUM_PARTICLES, 0);

    d_particles_position.world_body.resize(2 * NUM_PARTICLES, 0);
    d_particles_rotation.world_body.resize(4 * NUM_PARTICLES, 0);

    h_particles_transition.world_body.resize(9 * NUM_PARTICLES, 0);
    h_particles_transition.world_lidar.resize(9 * NUM_PARTICLES, 0);

    h_particles_position.world_body.resize(2 * NUM_PARTICLES, 0);
    //h_particles_rotation.world_body.resize(4 * NUM_PARTICLES, 0);
}

void alloc_init_processed_measurement_vars(DeviceProcessedMeasure& d_processed_measure, HostProcessedMeasure& h_processed_measure,
    HostMeasurements& h_measurements) {

    d_processed_measure.x.resize(NUM_PARTICLES * h_measurements.LIDAR_COORDS_LEN, 0);
    d_processed_measure.y.resize(NUM_PARTICLES * h_measurements.LIDAR_COORDS_LEN, 0);
    d_processed_measure.idx.resize(NUM_PARTICLES * h_measurements.LIDAR_COORDS_LEN, 0);

    h_processed_measure.x.resize(NUM_PARTICLES * h_measurements.LIDAR_COORDS_LEN);
    h_processed_measure.y.resize(NUM_PARTICLES * h_measurements.LIDAR_COORDS_LEN);
}

void alloc_map_2d_var(Device2DUniqueFinder& d_2d_unique, Host2DUniqueFinder& h_2d_unique, HostMap& h_map,
    bool alloc) {

    const int UNIQUE_COUNTER_LEN = NUM_PARTICLES + 1;

    if (alloc == true) {
        d_2d_unique.map.clear();
        d_2d_unique.map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT * NUM_PARTICLES, 0);
        d_2d_unique.in_map.clear();
        d_2d_unique.in_map.resize(UNIQUE_COUNTER_LEN, 0);
        d_2d_unique.in_col.clear();
        d_2d_unique.in_col.resize(UNIQUE_COUNTER_LEN * h_map.GRID_WIDTH, 0);

        h_2d_unique.map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT * NUM_PARTICLES, 0);
        h_2d_unique.in_map.resize(UNIQUE_COUNTER_LEN, 0);
        h_2d_unique.in_col.resize(UNIQUE_COUNTER_LEN * h_map.GRID_WIDTH, 0);
    }
    else {
        thrust::fill(d_2d_unique.map.begin(), d_2d_unique.map.end(), 0);
        thrust::fill(d_2d_unique.in_map.begin(), d_2d_unique.in_map.end(), 0);
        thrust::fill(d_2d_unique.in_col.begin(), d_2d_unique.in_col.end(), 0);

        thrust::fill(h_2d_unique.map.begin(), h_2d_unique.map.end(), 0);
        thrust::fill(h_2d_unique.in_map.begin(), h_2d_unique.in_map.end(), 0);
        thrust::fill(h_2d_unique.in_col.begin(), h_2d_unique.in_col.end(), 0);
    }
}

void alloc_resampling_vars(DeviceResampling& d_resampling, HostResampling& h_resampling, HostResampling& pre_resampling) {

    h_resampling.js.resize(NUM_PARTICLES, 0);

    d_resampling.js.resize(NUM_PARTICLES, 0);
    d_resampling.rnds.resize(NUM_PARTICLES, 0);

    d_resampling.rnds.assign(pre_resampling.rnds.begin(), pre_resampling.rnds.end());
}


#endif
