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

void assert_processed_measures(DeviceParticlesTransition& d_particles_transition, DeviceProcessedMeasure& d_processed_measure,
    HostParticlesTransition& res_particles_transition, HostMeasurements& res_measurements, HostProcessedMeasure& res_processed_measure,
    HostProcessedMeasure& h_processed_measure) {

    res_particles_transition.transition_multi_world_body.assign(
        d_particles_transition.transition_multi_world_body.begin(), d_particles_transition.transition_multi_world_body.end());
    res_particles_transition.transition_multi_world_lidar.assign(
        d_particles_transition.transition_multi_world_lidar.begin(), d_particles_transition.transition_multi_world_lidar.end());

    res_processed_measure.x.assign(d_processed_measure.x.begin(), d_processed_measure.x.end());
    res_processed_measure.y.assign(d_processed_measure.y.begin(), d_processed_measure.y.end());
    res_processed_measure.idx.assign(d_processed_measure.idx.begin(), d_processed_measure.idx.end());

    //ASSERT_transition_frames(res_transition_world_body, res_transition_world_lidar,
    //    h_transition_world_body, h_transition_world_lidar, NUM_PARTICLES, false);
    // ASSERT_processed_measurements(res_processed_measure_x, res_processed_measure_y, processed_measure, NUM_PARTICLES, LIDAR_COORDS_LEN);

    ASSERT_processed_measurements(res_processed_measure.x.data(), res_processed_measure.y.data(),
        res_processed_measure.idx.data(), h_processed_measure.x.data(), h_processed_measure.y.data(),
        (NUM_PARTICLES * res_measurements.LIDAR_COORDS_LEN), res_measurements.LIDAR_COORDS_LEN, false, true, true);
}

void assert_create_2d_map(Device2DUniqueFinder& d_2d_unique, Host2DUniqueFinder& res_2d_unique, HostMapData& res_map, HostRobotParticles& res_robot_particles,
    const int negative_before_counter) {

    res_2d_unique.map.assign(d_2d_unique.map.begin(), d_2d_unique.map.end());
    ASSERT_create_2d_map_elements(res_2d_unique.map.data(), negative_before_counter, res_map.GRID_WIDTH, res_map.GRID_HEIGHT, NUM_PARTICLES, res_robot_particles.LEN, true, true);
}

void assert_particles_unique(HostRobotParticles& res_robot_particles, HostRobotParticles& h_robot_particles_unique,
    const int negative_after_counter) {

    ASSERT_new_len_calculation(res_robot_particles.LEN, h_robot_particles_unique.LEN, negative_after_counter);
}

void assert_particles_unique(DeviceRobotParticles& d_robot_particles, HostRobotParticles& res_robot_particles,
    HostRobotParticles& h_robot_particles_unique, const int negative_after_counter) {

    res_robot_particles.x.assign(d_robot_particles.x.begin(), d_robot_particles.x.end());
    res_robot_particles.y.assign(d_robot_particles.y.begin(), d_robot_particles.y.end());
    res_robot_particles.idx.assign(d_robot_particles.idx.begin(), d_robot_particles.idx.end());

    ASSERT_particles_pos_unique(res_robot_particles.x.data(), res_robot_particles.y.data(),
        h_robot_particles_unique.x.data(), h_robot_particles_unique.y.data(), res_robot_particles.LEN, false, true, true);
    ASSERT_particles_idx_unique(res_robot_particles.idx.data(), h_robot_particles_unique.idx.data(), negative_after_counter, NUM_PARTICLES, false, true);
}

void assert_correlation(DeviceCorrelation& d_correlation, DeviceRobotParticles& d_robot_particles,
    HostCorrelation& res_correlation, HostRobotParticles& res_robot_particles, host_vector<float>& weights_pre) {

    res_correlation.weight.assign(d_correlation.weight.begin(), d_correlation.weight.end());
    res_robot_particles.extended_idx.assign(d_robot_particles.extended_idx.begin(), d_robot_particles.extended_idx.end());

    ASSERT_correlation_Equality(res_correlation.weight.data(), weights_pre.data(), NUM_PARTICLES, false, true);
}

void assert_update_weights(DeviceCorrelation& d_correlation, DeviceRobotParticles& d_robot_particles,
    HostCorrelation& res_correlation, HostRobotParticles& res_robot_particles, host_vector<float>& weights_new) {

    res_correlation.weight.assign(d_correlation.weight.begin(), d_correlation.weight.end());
    res_robot_particles.weight.assign(d_robot_particles.weight.begin(), d_robot_particles.weight.end());

    ASSERT_update_particle_weights(res_correlation.weight.data(), weights_new.data(), NUM_PARTICLES, "weights", false, false, true);
    //ASSERT_update_particle_weights(res_robot_particles.weight.data(), h_robot_particles_unique.weight.data(), NUM_PARTICLES, "particles weight", true, false, true);
}

void assert_resampling(DeviceResampling& d_resampling, HostResampling& res_resampling, HostResampling& h_resampling,
    HostState& h_state, HostState& h_state_updated) {

    res_resampling.js.assign(d_resampling.js.begin(), d_resampling.js.end());

    ASSERT_resampling_indices(res_resampling.js.data(), h_resampling.js.data(), NUM_PARTICLES, false, false, true);
    ASSERT_resampling_states(h_state.x.data(), h_state.y.data(), h_state.theta.data(),
        h_state_updated.x.data(), h_state_updated.y.data(), h_state_updated.theta.data(), res_resampling.js.data(), NUM_PARTICLES, false, true, true);
}


#endif
