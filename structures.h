#pragma once

#include "headers.h"

struct GeneralInfo {

	float res = 0.0f;
	float log_t = 0.0f;
};

struct HostMap {

	int GRID_WIDTH				=  0;
	int GRID_HEIGHT				=  0;

	int xmin					=  0;
	int xmax					=  0;
	int ymin					=  0;
	int ymax					=  0;

	bool b_should_extend		=  false;

	host_vector<int> c_should_extend;
	host_vector<int> s_grid_map;
	host_vector<float> s_log_odds;
};

struct DeviceMap {

	device_vector<int> c_should_extend;
	device_vector<int> s_grid_map;
	device_vector<float> s_log_odds;
};

struct HostPosition {

	host_vector<float>	world_body;
	host_vector<int>	c_image_body;
};

struct DevicePosition {

	device_vector<float> world_body;
	device_vector<int>   c_image_body;
};

//struct HostRotation {
//	//host_vector<float> world_body;
//};
//
//struct DeviceRotation {
//	//device_vector<float> world_body;
//};

struct HostTransition {

	host_vector<float>	c_world_body;
	host_vector<float>	c_world_lidar;
};

struct DeviceTransition {

	device_vector<float> c_world_body;
	device_vector<float> c_world_lidar;

	device_vector<float> c_body_lidar;
};

struct HostParticles {

	int OCCUPIED_LEN = 0;
	int OCCUPIED_UNIQUE_LEN = 0;
	int FREE_LEN = 0;
	int FREE_UNIQUE_LEN = 0;

	//int MAX_OCCUPIED_LEN = 2000;
	int MAX_OCCUPIED_UNIQUE_LEN = 1000;

	int MAX_FREE_LEN = 40000;
	int MAX_FREE_UNIQUE_LEN = 7000;

	host_vector<int> v_occupied_x;
	host_vector<int> v_occupied_y;
	host_vector<int> f_occupied_unique_x;
	host_vector<int> f_occupied_unique_y;

	host_vector<int> f_free_x;
	host_vector<int> f_free_y;
	host_vector<int> f_free_unique_x;
	host_vector<int> f_free_unique_y;
	host_vector<int> v_free_idx;

	host_vector<int> v_free_counter;

	host_vector<float> v_world_x;
	host_vector<float> v_world_y;
};

struct DeviceParticles {

	device_vector<int> v_occupied_x;
	device_vector<int> v_occupied_y;
	device_vector<int> f_occupied_unique_x;
	device_vector<int> f_occupied_unique_y;

	device_vector<int> f_free_x;
	device_vector<int> f_free_y;
	device_vector<int> f_free_unique_x;
	device_vector<int> f_free_unique_y;
	device_vector<int> v_free_idx;

	device_vector<int> sv_free_x_max;
	device_vector<int> sv_free_y_max;

	device_vector<int> v_free_counter;

	device_vector<float> v_world_x;
	device_vector<float> v_world_y;
};

struct HostParticlesTransition {

	host_vector<float> c_world_body;
	host_vector<float> c_world_lidar;
};

struct DeviceParticlesTransition {

	device_vector<float> c_world_body;
	device_vector<float> c_world_lidar;
};

struct HostParticlesPosition {
	host_vector<float> c_world_body;
};

struct DeviceParticlesPosition {
	device_vector<float> c_world_body;
};

struct HostParticlesRotation {
	//host_vector<float> world_body;
};

struct DeviceParticlesRotation {
	device_vector<float> c_world_body;
};

struct HostRobotParticles {

	int LEN = 0;
	int MAX_LEN = 1700000;

	host_vector<int> f_x;
	host_vector<int> f_y;
	host_vector<int> c_idx;
	host_vector<float> c_weight;
	host_vector<int> f_extended_idx;
};

struct DeviceRobotParticles {

	device_vector<int> f_x;
	device_vector<int> f_y;
	device_vector<int> c_idx;
	device_vector<float> c_weight;
	device_vector<int> f_extended_idx;
};

struct HostCorrelation {

	host_vector<float> c_weight;
	host_vector<float> c_raw;
	host_vector<float> c_max;
	host_vector<float> c_sum_exp;
};

struct DeviceCorrelation {

	device_vector<float> c_weight;
	device_vector<float> c_raw;
	device_vector<float> c_max;
	device_vector<float> c_sum_exp;
};

struct HostProcessedMeasure {

	host_vector<int> v_x;
	host_vector<int> v_y;
	host_vector<int> c_idx;
};

struct DeviceProcessedMeasure {

	device_vector<int> v_x;
	device_vector<int> v_y;
	device_vector<int> c_idx;
};

struct HostMeasurements {

	int LEN		=  0;
	int MAX_LEN = 2000;

	host_vector<float> v_lidar_coords;
	host_vector<int> c_coord;

	host_vector<int> v_processed_measure_x;
	host_vector<int> v_processed_measure_y;

	int getParticlesOccupiedLen() { return LEN; }
};

struct DeviceMeasurements {

	device_vector<float> v_lidar_coords;
	device_vector<int> c_coord;

	device_vector<int> v_processed_measure_x;
	device_vector<int> v_processed_measure_y;
};

struct Host2DUniqueFinder {

	host_vector<uint8_t> s_map;
	host_vector<int> c_in_map;
	host_vector<int> s_in_col;
	host_vector<int> c_idx;
};

struct Device2DUniqueFinder {

	device_vector<uint8_t> s_map;
	device_vector<int> c_in_map;
	device_vector<int> s_in_col;
	device_vector<int> c_idx;
};

struct HostState {

	float encoder_counts = 0.0f;
	float yaw = 0.0f;
	float dt = 0.0f;
	float nv = 0.0f;
	float nw = 0.0f;

	host_vector<float> c_x;
	host_vector<float> c_y;
	host_vector<float> c_theta;
	host_vector<float> c_rnds_encoder_counts;
	host_vector<float> c_rnds_yaws;
};

struct DeviceState {

	device_vector<float> c_x;
	device_vector<float> c_y;
	device_vector<float> c_theta;
	device_vector<float> c_rnds_encoder_counts;
	device_vector<float> c_rnds_yaws;
};

struct HostRobotState {

	host_vector<float> state;
	host_vector<float> transition_world_body;
};

struct DeviceRobotState {

	device_vector<float> state;
	device_vector<float> transition_world_body;
};

struct HostResampling {

	host_vector<float> c_rnds;
	host_vector<int> c_js;
};

struct DeviceResampling {

	device_vector<float> c_rnds;
	device_vector<int> c_js;
};


struct DeviceRobot {

	DeviceState state;
	DeviceState clone_state;
	DeviceTransition transition;
	DeviceCorrelation correlation;
	DeviceResampling resampling;
	DeviceParticlesTransition particles_transition;
	Device2DUniqueFinder _2d_unique;
	DeviceRobotParticles robot_particles;
	DeviceRobotParticles clone_robot_particles;
};

struct HostRobot {

	HostState state;
	HostRobotState robot_state;
	HostResampling resampling;
	HostCorrelation correlation;
	HostRobotParticles robot_particles;
	HostRobotParticles clone_robot_particles;
	Host2DUniqueFinder _2d_unique;
	HostParticlesTransition particles_transition;
};

struct DeviceMapping {

	DeviceMap map;
	DevicePosition position;
	DeviceParticles particles;
	Device2DUniqueFinder unique_occupied;
	Device2DUniqueFinder unique_free;
};

struct HostMapping {

	HostMap map;
	HostPosition position;
	HostTransition transition;
	HostParticles particles;
	Host2DUniqueFinder unique_occupied;
	Host2DUniqueFinder unique_free;
};

struct DeviceLidar {

	DeviceMeasurements measurements;
	DeviceProcessedMeasure processed_measure;
};

struct HostLidar {

	HostMeasurements measurements;
	HostProcessedMeasure processed_measure;
};

template <class T>
void vector_reset(host_vector<T>& vec, int sz, T val) {
	vec.clear();
	vec.resize(sz, val);
}

template <class T>
void vector_reset(device_vector<T>& vec, int sz, T val) {
	vec.clear();
	vec.resize(sz, val);
}



