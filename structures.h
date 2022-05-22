#pragma once

#include "headers.h"

struct GeneralInfo {

	float res = 0.0f;
	float log_t = 0.0f;
};

struct HostMapData {

	int GRID_WIDTH				=  0;
	int GRID_HEIGHT				=  0;

	int xmin					=  0;
	int xmax					=  0;
	int ymin					=  0;
	int ymax					=  0;

	bool b_should_extend		=  false;

	host_vector<int> should_extend;
	host_vector<int> grid_map;
	host_vector<float> log_odds;
};

struct DeviceMapData {

	device_vector<int> should_extend;
	device_vector<int> grid_map;
	device_vector<float> log_odds;

	DeviceMapData() {
		should_extend.clear();
		should_extend.resize(4);
	}
};

struct HostPositionTransition {

	host_vector<float>	position_world_body;
	host_vector<int>	position_image_body;

	host_vector<float>	transition_world_body;
	host_vector<float>	transition_world_lidar;

	host_vector<float> transition_single_world_body;
	
	host_vector<float> rotation_world_body;
};

struct DevicePositionTransition {

	device_vector<float> position_world_body;
	device_vector<int>   position_image_body;

	device_vector<float> transition_world_body;
	device_vector<float> transition_world_lidar;

	device_vector<float> transition_single_world_body;

	device_vector<float> rotation_world_body;

	device_vector<float> transition_body_lidar;
	device_vector<float> transition_single_world_lidar;
};

struct HostParticlesData {

	int PARTICLES_OCCUPIED_LEN = 0;
	int PARTICLES_OCCUPIED_UNIQUE_LEN = 0;
	int PARTICLES_FREE_LEN = 0;
	int PARTICLES_FREE_UNIQUE_LEN = 0;

	host_vector<int> particles_occupied_x;
	host_vector<int> particles_occupied_y;
	host_vector<int> particles_occupied_unique_x;
	host_vector<int> particles_occupied_unique_y;

	host_vector<int> particles_free_x;
	host_vector<int> particles_free_y;
	host_vector<int> particles_free_idx;
	host_vector<int> particles_free_unique_x;
	host_vector<int> particles_free_unique_y;
	host_vector<int> particles_free_counter;

	host_vector<float> particles_world_x;
	host_vector<float> particles_world_y;
};

struct DeviceParticlesData {

	device_vector<int> particles_occupied_x;
	device_vector<int> particles_occupied_y;
	device_vector<int> particles_occupied_unique_x;
	device_vector<int> particles_occupied_unique_y;

	device_vector<int> particles_free_x;
	device_vector<int> particles_free_y;
	device_vector<int> particles_free_idx;
	device_vector<int> particles_free_unique_x;
	device_vector<int> particles_free_unique_y;
	device_vector<int> particles_free_x_max;
	device_vector<int> particles_free_y_max;
	device_vector<int> particles_free_counter;

	device_vector<float> particles_world_x;
	device_vector<float> particles_world_y;
};

struct HostParticlesTransition {

	host_vector<float> world;
	host_vector<float> world_homo;

	host_vector<float> transition_multi_world_body;
	host_vector<float> transition_multi_world_lidar;
};

struct DeviceParticlesTransition {

	device_vector<float> world;
	device_vector<float> world_homo;

	device_vector<float> transition_multi_world_body;
	device_vector<float> transition_multi_world_lidar;
};

struct HostRobotParticles {

	int LEN = 0;

	host_vector<int> x;
	host_vector<int> y;
	host_vector<int> idx;
	host_vector<float> weight;
	host_vector<int> extended_idx;
};

struct DeviceRobotParticles {

	device_vector<int> x;
	device_vector<int> y;
	device_vector<int> idx;
	device_vector<float> weight;
	device_vector<int> extended_idx;
};

struct HostCorrelation {

	float max;
	float sum;

	host_vector<float> weight;
	host_vector<float> raw;
	host_vector<float> vec_max;
	host_vector<float> vec_sum_exp;
};

struct DeviceCorrelation {

	device_vector<float> weight;
	device_vector<float> raw;
	device_vector<float> vec_max;
	device_vector<float> vec_sum_exp;
};

struct HostProcessedMeasure {

	host_vector<int> x;
	host_vector<int> y;
	host_vector<int> idx;
};

struct DeviceProcessedMeasure {

	device_vector<int> x;
	device_vector<int> y;
	device_vector<int> idx;
};

struct HostMeasurements {

	int LIDAR_COORDS_LEN		=  0;

	host_vector<float> lidar_coords;
	host_vector<int> coord;

	host_vector<int> processed_single_measure_x;
	host_vector<int> processed_single_measure_y;
};

struct DeviceMeasurements {

	device_vector<float> lidar_coords;
	device_vector<int> coord;

	device_vector<int> processed_single_measure_x;
	device_vector<int> processed_single_measure_y;


	void resize(size_t sz, const float val) {

		lidar_coords.clear();
		lidar_coords.resize(sz, val);
	}
};


struct HostUniqueManager {

	host_vector<int> occupied_unique_counter;
	host_vector<int> occupied_unique_counter_col;
	host_vector<int> free_unique_counter;
	host_vector<int> free_unique_counter_col;

	host_vector<int> occupied_map_idx;
	host_vector<int> free_map_idx;
};

struct DeviceUniqueManager {

	device_vector<uint8_t> occupied_map_2d;
	device_vector<uint8_t> free_map_2d;

	device_vector<int> occupied_unique_counter;
	device_vector<int> occupied_unique_counter_col;
	device_vector<int> free_unique_counter;
	device_vector<int> free_unique_counter_col;

	device_vector<int> occupied_map_idx;
	device_vector<int> free_map_idx;
};

struct Host2DUniqueFinder {

	host_vector<uint8_t> map;
	host_vector<int> in_map;
	host_vector<int> in_col;
};

struct Device2DUniqueFinder {

	device_vector<uint8_t> map;
	device_vector<int> in_map;
	device_vector<int> in_col;
};

struct HostState {

	float encoder_counts = 0.0f;
	float yaw = 0.0f;
	float dt = 0.0f;
	float nv = 0.0f;
	float nw = 0.0f;

	host_vector<float> x;
	host_vector<float> y;
	host_vector<float> theta;
	host_vector<float> rnds_encoder_counts;
	host_vector<float> rnds_yaws;
};

struct DeviceState {

	device_vector<float> x;
	device_vector<float> y;
	device_vector<float> theta;
	device_vector<float> rnds_encoder_counts;
	device_vector<float> rnds_yaws;
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

	host_vector<float> rnds;
	host_vector<int> js;
};

struct DeviceResampling {

	device_vector<float> rnds;
	device_vector<int> js;
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



