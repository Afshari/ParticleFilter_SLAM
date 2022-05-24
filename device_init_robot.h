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

void alloc_init_state_vars(DeviceState& d_state, HostState& res_state, HostRobotState& res_robot_state, HostState& h_state) {

    d_state.x.resize(NUM_PARTICLES, 0);
    d_state.y.resize(NUM_PARTICLES, 0);
    d_state.theta.resize(NUM_PARTICLES, 0);

    res_state.x.resize(NUM_PARTICLES, 0);
    res_state.y.resize(NUM_PARTICLES, 0);
    res_state.theta.resize(NUM_PARTICLES, 0);

    d_state.x.assign(h_state.x.begin(), h_state.x.end());
    d_state.y.assign(h_state.y.begin(), h_state.y.end());
    d_state.theta.assign(h_state.theta.begin(), h_state.theta.end());

    res_robot_state.state.resize(3, 0);
}

void alloc_init_lidar_coords_var(DeviceMeasurements& d_measurements, HostMeasurements& res_measurements,
    HostMeasurements& h_measurements) {

    d_measurements.lidar_coords.resize(2 * h_measurements.LIDAR_COORDS_LEN, 0);
    d_measurements.lidar_coords.assign(h_measurements.lidar_coords.begin(), h_measurements.lidar_coords.end());

    res_measurements.LIDAR_COORDS_LEN = h_measurements.LIDAR_COORDS_LEN;
}

void alloc_init_grid_map(DeviceMapData& d_map, HostMapData& res_map, HostMapData& h_map) {

    res_map.GRID_WIDTH = h_map.GRID_WIDTH;
    res_map.GRID_HEIGHT = h_map.GRID_HEIGHT;
    res_map.xmin = h_map.xmin;
    res_map.xmax = h_map.xmax;
    res_map.ymin = h_map.ymin;
    res_map.ymax = h_map.ymax;

    d_map.grid_map.resize(res_map.GRID_WIDTH * res_map.GRID_HEIGHT, 0);
    d_map.grid_map.assign(h_map.grid_map.begin(), h_map.grid_map.end());
}

void alloc_init_particles_vars(DeviceRobotParticles& d_robot_particles, HostRobotParticles& res_robot_particles,
    HostRobotParticles& h_robot_particles) {

    res_robot_particles.LEN = h_robot_particles.LEN;
    res_robot_particles.x.resize(res_robot_particles.LEN, 0);
    res_robot_particles.y.resize(res_robot_particles.LEN, 0);
    res_robot_particles.idx.resize(NUM_PARTICLES, 0);
    res_robot_particles.extended_idx.resize(res_robot_particles.LEN, 0);
    res_robot_particles.weight.resize(NUM_PARTICLES, 0);


    d_robot_particles.x.resize(res_robot_particles.LEN, 0);
    d_robot_particles.y.resize(res_robot_particles.LEN, 0);
    d_robot_particles.idx.resize(NUM_PARTICLES, 0);
    d_robot_particles.weight.resize(NUM_PARTICLES, 0);

    d_robot_particles.x.assign(h_robot_particles.x.begin(), h_robot_particles.x.end());
    d_robot_particles.y.assign(h_robot_particles.y.begin(), h_robot_particles.y.end());
    d_robot_particles.idx.assign(h_robot_particles.idx.begin(), h_robot_particles.idx.end());
    d_robot_particles.weight.assign(h_robot_particles.weight.begin(), h_robot_particles.weight.end());
}

void alloc_extended_idx(DeviceRobotParticles& d_robot_particles, HostRobotParticles& res_robot_particles) {

    d_robot_particles.extended_idx.resize(res_robot_particles.LEN, 0);
    res_robot_particles.extended_idx.resize(res_robot_particles.LEN, 0);
}

void alloc_states_copy_vars(DeviceState& d_clone_state) {

    d_clone_state.x.resize(NUM_PARTICLES, 0);
    d_clone_state.y.resize(NUM_PARTICLES, 0);
    d_clone_state.theta.resize(NUM_PARTICLES, 0);
}

void alloc_correlation_vars(DeviceCorrelation& d_correlation, HostCorrelation& res_correlation) {

    d_correlation.weight.resize(NUM_PARTICLES, 0);
    d_correlation.raw.resize(25 * NUM_PARTICLES, 0);

    res_correlation.weight.resize(NUM_PARTICLES, 0);
    res_correlation.raw.resize(25 * NUM_PARTICLES, 0);
}

void alloc_init_transition_vars(DevicePositionTransition& d_transition, DeviceParticlesTransition& d_particles_transition,
    HostParticlesTransition& res_particles_transition) {

    d_particles_transition.transition_multi_world_body.resize(9 * NUM_PARTICLES, 0);
    d_particles_transition.transition_multi_world_lidar.resize(9 * NUM_PARTICLES, 0);

    res_particles_transition.transition_multi_world_body.resize(9 * NUM_PARTICLES, 0);
    res_particles_transition.transition_multi_world_lidar.resize(9 * NUM_PARTICLES, 0);

    d_transition.transition_body_lidar.resize(9, 0);
    d_transition.transition_body_lidar.assign(hvec_transition_body_lidar.begin(), hvec_transition_body_lidar.end());
}

void alloc_init_processed_measurement_vars(DeviceProcessedMeasure& d_processed_measure, HostProcessedMeasure& res_processed_measure,
    HostMeasurements& res_measurements) {

    d_processed_measure.x.resize(NUM_PARTICLES * res_measurements.LIDAR_COORDS_LEN, 0);
    d_processed_measure.y.resize(NUM_PARTICLES * res_measurements.LIDAR_COORDS_LEN, 0);
    d_processed_measure.idx.resize(NUM_PARTICLES * res_measurements.LIDAR_COORDS_LEN, 0);

    res_processed_measure.x.resize(NUM_PARTICLES * res_measurements.LIDAR_COORDS_LEN);
    res_processed_measure.y.resize(NUM_PARTICLES * res_measurements.LIDAR_COORDS_LEN);
}

void alloc_map_2d_var(Device2DUniqueFinder& d_2d_unique, Host2DUniqueFinder& res_2d_unique, HostMapData& res_map) {

    d_2d_unique.map.resize(res_map.GRID_WIDTH * res_map.GRID_HEIGHT * NUM_PARTICLES, 0);
    res_2d_unique.map.resize(res_map.GRID_WIDTH * res_map.GRID_HEIGHT * NUM_PARTICLES, 0);
}

void alloc_map_2d_unique_counter_vars(Device2DUniqueFinder& d_2d_unique, Host2DUniqueFinder& res_2d_unique, HostMapData& res_map) {

    const int UNIQUE_COUNTER_LEN = NUM_PARTICLES + 1;

    d_2d_unique.in_map.resize(UNIQUE_COUNTER_LEN, 0);
    d_2d_unique.in_col.resize(UNIQUE_COUNTER_LEN * res_map.GRID_WIDTH, 0);

    res_2d_unique.in_map.resize(UNIQUE_COUNTER_LEN, 0);
}

void alloc_correlation_weights_vars(DeviceCorrelation& d_correlation, HostCorrelation& res_correlation) {

    d_correlation.vec_sum_exp.resize(1, 0);
    d_correlation.vec_max.resize(1, 0);

    res_correlation.vec_sum_exp.resize(1, 0);
    res_correlation.vec_max.resize(1, 0);
}

void alloc_resampling_vars(DeviceResampling& d_resampling, HostResampling& res_resampling, HostResampling& h_resampling) {

    res_resampling.js.resize(NUM_PARTICLES, 0);

    d_resampling.js.resize(NUM_PARTICLES, 0);
    d_resampling.rnds.resize(NUM_PARTICLES, 0);

    d_resampling.rnds.assign(h_resampling.rnds.begin(), h_resampling.rnds.end());
}


#endif
