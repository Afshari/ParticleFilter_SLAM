
#include "gtest/gtest.h"
#include "../device_init_robot.h"

TEST(DeviceInitRobot, AllocInitStateVars) {

	DeviceState d_state; 
	DeviceState d_clone_state; 
	HostState h_state;
	HostRobotState h_robot_state; 
	HostState pre_state;

	pre_state.c_x.resize(NUM_PARTICLES, 0);
	pre_state.c_y.resize(NUM_PARTICLES, 0);
	pre_state.c_theta.resize(NUM_PARTICLES, 0);
	pre_state.c_rnds_encoder_counts.resize(NUM_PARTICLES, 0);
	pre_state.c_rnds_yaws.resize(NUM_PARTICLES, 0);

	alloc_init_state_vars(d_state, d_clone_state, h_state, h_robot_state, pre_state);

	EXPECT_EQ(d_state.c_x.size(), NUM_PARTICLES);
	EXPECT_EQ(d_state.c_y.size(), NUM_PARTICLES);
	EXPECT_EQ(d_state.c_theta.size(), NUM_PARTICLES);
	EXPECT_EQ(d_state.c_rnds_encoder_counts.size(), NUM_PARTICLES);
	EXPECT_EQ(d_state.c_rnds_yaws.size(), NUM_PARTICLES);

	EXPECT_EQ(d_clone_state.c_x.size(), NUM_PARTICLES);
	EXPECT_EQ(d_clone_state.c_y.size(), NUM_PARTICLES);
	EXPECT_EQ(d_clone_state.c_theta.size(), NUM_PARTICLES);

	EXPECT_EQ(h_state.c_x.size(), NUM_PARTICLES);
	EXPECT_EQ(h_state.c_y.size(), NUM_PARTICLES);
	EXPECT_EQ(h_state.c_theta.size(), NUM_PARTICLES);
	EXPECT_EQ(h_state.c_rnds_encoder_counts.size(), NUM_PARTICLES);
	EXPECT_EQ(h_state.c_rnds_yaws.size(), NUM_PARTICLES);

	EXPECT_EQ(h_state.encoder_counts, pre_state.encoder_counts);
	EXPECT_EQ(h_state.yaw, pre_state.yaw);
	EXPECT_EQ(h_state.dt, pre_state.dt);
	EXPECT_EQ(h_state.nv, pre_state.nv);
	EXPECT_EQ(h_state.nw, pre_state.nw);

	EXPECT_EQ(h_robot_state.state.size(), 3);
}

TEST(DeviceInitRobot, AllocInitRobotParticlesVars) {

	DeviceRobotParticles d_robot_particles;
	DeviceRobotParticles d_clone_robot_particles;
	HostRobotParticles h_robot_particles;
	HostRobotParticles pre_robot_particles;

	pre_robot_particles.LEN = 100;
	pre_robot_particles.c_idx.resize(NUM_PARTICLES, 0);
	pre_robot_particles.c_weight.resize(NUM_PARTICLES, 0);

	alloc_init_robot_particles_vars(d_robot_particles, d_clone_robot_particles, h_robot_particles, pre_robot_particles);

	EXPECT_EQ(h_robot_particles.LEN, pre_robot_particles.LEN);
	
	EXPECT_EQ(h_robot_particles.f_x.size(), h_robot_particles.MAX_LEN);
	EXPECT_EQ(h_robot_particles.f_y.size(), h_robot_particles.MAX_LEN);
	EXPECT_EQ(h_robot_particles.c_idx.size(), NUM_PARTICLES);
	EXPECT_EQ(h_robot_particles.c_weight.size(), NUM_PARTICLES);
	EXPECT_EQ(h_robot_particles.f_extended_idx.size(), h_robot_particles.MAX_LEN);

	EXPECT_EQ(d_robot_particles.f_x.size(), h_robot_particles.MAX_LEN);
	EXPECT_EQ(d_robot_particles.f_y.size(), h_robot_particles.MAX_LEN);
	EXPECT_EQ(d_robot_particles.c_idx.size(), NUM_PARTICLES);
	EXPECT_EQ(d_robot_particles.c_weight.size(), NUM_PARTICLES);
	EXPECT_EQ(d_robot_particles.f_extended_idx.size(), h_robot_particles.MAX_LEN);

	EXPECT_EQ(d_clone_robot_particles.f_x.size(), h_robot_particles.MAX_LEN);
	EXPECT_EQ(d_clone_robot_particles.f_y.size(), h_robot_particles.MAX_LEN);
	EXPECT_EQ(d_clone_robot_particles.c_idx.size(), NUM_PARTICLES);
	EXPECT_EQ(d_clone_robot_particles.c_weight.size(), NUM_PARTICLES);
	EXPECT_EQ(d_clone_robot_particles.f_extended_idx.size(), h_robot_particles.MAX_LEN);
}

TEST(DeviceInitRobot, AllocCorrelationVars) {

	DeviceCorrelation d_correlation; 
	HostCorrelation h_correlation;

	alloc_correlation_vars(d_correlation, h_correlation);

	EXPECT_EQ(d_correlation.c_weight.size(), NUM_PARTICLES);
	EXPECT_EQ(d_correlation.c_raw.size(), 25 * NUM_PARTICLES);
	EXPECT_EQ(d_correlation.c_sum_exp.size(), 1);
	EXPECT_EQ(d_correlation.c_max.size(), 1);

	EXPECT_EQ(h_correlation.c_weight.size(), NUM_PARTICLES);
	EXPECT_EQ(h_correlation.c_raw.size(), 25 * NUM_PARTICLES);
	EXPECT_EQ(h_correlation.c_sum_exp.size(), 1);
	EXPECT_EQ(h_correlation.c_max.size(), 1);
}


TEST(DeviceInitRobot, AllocParticlesTransitionVars) {

	DeviceParticlesTransition d_particles_transition;
	DeviceParticlesPosition d_particles_position; 
	DeviceParticlesRotation d_particles_rotation;
	HostParticlesTransition h_particles_transition; 
	HostParticlesPosition h_particles_position;
	HostParticlesRotation h_particles_rotation;

	alloc_particles_transition_vars(d_particles_transition, d_particles_position, d_particles_rotation,
		h_particles_transition, h_particles_position, h_particles_rotation);

	EXPECT_EQ(d_particles_transition.c_world_body.size(), 9 * NUM_PARTICLES);
	EXPECT_EQ(d_particles_transition.c_world_lidar.size(), 9 * NUM_PARTICLES);

	EXPECT_EQ(d_particles_position.c_world_body.size(), 2 * NUM_PARTICLES);
	EXPECT_EQ(d_particles_rotation.c_world_body.size(), 4 * NUM_PARTICLES);

	EXPECT_EQ(h_particles_transition.c_world_body.size(), 9 * NUM_PARTICLES);
	EXPECT_EQ(h_particles_transition.c_world_lidar.size(), 9 * NUM_PARTICLES);

	EXPECT_EQ(h_particles_position.c_world_body.size(), 2 * NUM_PARTICLES);
}

TEST(DeviceInitRobot, AllocInitProcessedMeasurementVars) {

	DeviceProcessedMeasure d_processed_measure; 
	HostProcessedMeasure h_processed_measure;
	HostMeasurements h_measurements;

	alloc_init_processed_measurement_vars(d_processed_measure, h_processed_measure, h_measurements);

	EXPECT_EQ(d_processed_measure.v_x.size(), NUM_PARTICLES * h_measurements.MAX_LEN);
	EXPECT_EQ(d_processed_measure.v_y.size(), NUM_PARTICLES * h_measurements.MAX_LEN);
	EXPECT_EQ(d_processed_measure.c_idx.size(), NUM_PARTICLES);

	EXPECT_EQ(h_processed_measure.v_x.size(), NUM_PARTICLES * h_measurements.MAX_LEN);
	EXPECT_EQ(h_processed_measure.v_y.size(), NUM_PARTICLES * h_measurements.MAX_LEN);
	EXPECT_EQ(h_processed_measure.c_idx.size(), NUM_PARTICLES);
}

TEST(DeviceInitRobot, AllocMap2DVar) {

	Device2DUniqueFinder d_2d_unique; 
	Host2DUniqueFinder h_2d_unique; 
	HostMap h_map;

	alloc_map_2d_var(d_2d_unique, h_2d_unique, h_map);

	EXPECT_EQ(d_2d_unique.s_map.size(), h_map.GRID_WIDTH * h_map.GRID_HEIGHT * NUM_PARTICLES);
	EXPECT_EQ(d_2d_unique.c_in_map.size(), UNIQUE_COUNTER_LEN);
	EXPECT_EQ(d_2d_unique.s_in_col.size(), UNIQUE_COUNTER_LEN * h_map.GRID_WIDTH);

	EXPECT_EQ(h_2d_unique.s_map.size(), h_map.GRID_WIDTH * h_map.GRID_HEIGHT * NUM_PARTICLES);
	EXPECT_EQ(h_2d_unique.c_in_map.size(), UNIQUE_COUNTER_LEN);
	EXPECT_EQ(h_2d_unique.s_in_col.size(), UNIQUE_COUNTER_LEN * h_map.GRID_WIDTH);
}

TEST(DeviceInitRobot, AllocResamplingVars1) {

	DeviceResampling d_resampling; 
	HostResampling h_resampling; 
	HostResampling pre_resampling;

	pre_resampling.c_rnds.resize(NUM_PARTICLES, 0);

	alloc_resampling_vars(d_resampling, h_resampling, pre_resampling);

	EXPECT_EQ(h_resampling.c_js.size(), NUM_PARTICLES);

	EXPECT_EQ(d_resampling.c_js.size(), NUM_PARTICLES);
	EXPECT_EQ(d_resampling.c_rnds.size(), NUM_PARTICLES);
}

TEST(DeviceInitRobot, AllocResamplingVars2) {

	DeviceResampling d_resampling; 
	HostResampling h_resampling;

	alloc_resampling_vars(d_resampling, h_resampling);

	EXPECT_EQ(h_resampling.c_js.size(), NUM_PARTICLES);

	EXPECT_EQ(d_resampling.c_js.size(), NUM_PARTICLES);
	EXPECT_EQ(d_resampling.c_rnds.size(), NUM_PARTICLES);
}
