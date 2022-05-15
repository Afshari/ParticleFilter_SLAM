#pragma once

#include "headers.h"

struct GeneralInfo {

	float res = 0.0f;
	float log_t = 0.0f;
};

struct MapData {

	int GRID_WIDTH				=  0;
	int GRID_HEIGHT				=  0;

	int xmin					=  0;
	int xmax					=  0;
	int ymin					=  0;
	int ymax					=  0;

	bool should_extend			=  false;

	vector<int> grid_map;
	vector<float> log_odds;
};

struct PositionTransition {

	vector<float> position_world_body;
	vector<int> position_image_body;
	vector<float> transition_world_body;
	vector<float> transition_world_lidar;
};

struct ParticlesData {

	int PARTICLES_OCCUPIED_LEN = 0;
	int PARTICLES_OCCUPIED_UNIQUE_LEN = 0;
	int PARTICLES_FREE_LEN = 0;
	int PARTICLES_FREE_UNIQUE_LEN = 0;

	vector<int> particles_occupied_x;
	vector<int> particles_occupied_y;
	vector<int> particles_occupied_unique_x;
	vector<int> particles_occupied_unique_y;

	vector<int> particles_free_x;
	vector<int> particles_free_y;
	vector<int> particles_free_idx;
	vector<int> particles_free_unique_x;
	vector<int> particles_free_unique_y;

	vector<float> particles_world_x;
	vector<float> particles_world_y;
};

struct Measurements {

	int LIDAR_COORDS_LEN		=  0;

	vector<float> lidar_coords;
	vector<float> coord;
};

