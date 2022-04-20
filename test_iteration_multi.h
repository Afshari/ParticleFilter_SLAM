#ifndef _TEST_ITERATION_MULTI_H_
#define _TEST_ITERATION_MULTI_H_

#include "headers.h"
#include "host_asserts.h"
#include "host_utils.h"
#include "kernels.cuh"
#include "kernels_robot.cuh"
#include "kernels_map.cuh"
#include "kernels_utils.cuh"

//#define ADD_HEADER_DATA

//#define VERBOSE_BORDER_LINE_COUNTER
//#define VERBOSE_TOTAL_INFO
//#define VERBOSE_BANNER
//#define VERBOSE_EXECUTION_TIME

#ifdef ADD_HEADER_DATA
#include "data/robot_advance/300.h"
#include "data/robot_iteration/300.h"
#include "data/map_iteration/300.h"
#endif

#define ST_nv   0.5
#define ST_nw   0.5

const int UNIQUE_COUNTER_LEN = NUM_PARTICLES + 1;

int threadsPerBlock = 1;
int blocksPerGrid = 1;

/********************* IMAGE TRANSFORM VARIABLES ********************/
size_t sz_transition_multi_world_frame = 0;
size_t sz_transition_body_lidar = 0;

float* d_transition_multi_world_body = NULL;
float* d_transition_multi_world_lidar = NULL;
float* d_transition_body_lidar = NULL;

/*------------------------ RESULT VARIABLES -----------------------*/
float* res_transition_world_body = NULL;
float* res_transition_world_lidar = NULL;

/************************* STATES VARIABLES *************************/
size_t sz_states_pos = 0;

float* d_states_x = NULL;
float* d_states_y = NULL;
float* d_states_theta = NULL;

/*------------------------ RESULT VARIABLES -----------------------*/
float* res_states_x = NULL;
float* res_states_y = NULL;
float* res_states_theta = NULL;

/********************** STATES COPY VARIABLES **********************/
float* dc_states_x = NULL;
float* dc_states_y = NULL;
float* dc_states_theta = NULL;


/************************ PARTICLES VARIABLES ***********************/
size_t sz_particles_pos = 0;
size_t sz_particles_idx = 0;

int* d_particles_x = NULL;
int* d_particles_y = NULL;
int* d_particles_idx = NULL;

/*------------------------ RESULT VARIABLES -----------------------*/
int* res_particles_x = NULL;
int* res_particles_y = NULL;
int* res_particles_idx = NULL;

/******************** PARTICLES COPY VARIABLES **********************/
int* dc_particles_x = NULL;
int* dc_particles_y = NULL;
int* dc_particles_idx = NULL;

/*********************** MEASUREMENT VARIABLES **********************/
size_t sz_lidar_coords = 0;
float* d_lidar_coords = NULL;

/**************** PROCESSED MEASUREMENTS VARIABLES ******************/
size_t sz_processed_measure_pos = 0;
size_t sz_processed_measure_idx = 0;

int* d_processed_measure_x = NULL;
int* d_processed_measure_y = NULL;
int* d_processed_measure_idx = NULL;

/*------------------------ RESULT VARIABLES -----------------------*/
int* res_processed_measure_x = NULL;
int* res_processed_measure_y = NULL;
int* res_processed_measure_idx = NULL;


/************************ WEIGHTS VARIABLES *************************/
size_t sz_correlation_weights = 0;
size_t sz_correlation_weights_raw = 0;
size_t sz_correlation_weights_max = 0;
size_t sz_correlation_sum_exp = 0;

float* d_correlation_weights = NULL;
float* d_correlation_weights_raw = NULL;
float* d_correlation_weights_max = NULL;
double* d_correlation_sum_exp = NULL;

/*------------------------ RESULT VARIABLES -----------------------*/
float* res_correlation_weights = NULL;
float* res_correlation_weights_max = NULL;
double* res_correlation_sum_exp = NULL;

/*********************** RESAMPLING VARIABLES ***********************/
size_t sz_resampling_js = 0;
size_t sz_resampling_rnd = 0;

int* d_resampling_js = NULL;
float* d_resampling_rnd = NULL;

int* res_resampling_js = NULL;

/**************************** MAP VARIABLES *************************/
size_t sz_grid_map = 0;
size_t sz_extended_idx = 0;

int* d_grid_map = NULL;
int* d_extended_idx = NULL;

/*------------------------ RESULT VARIABLES -----------------------*/
int* res_extended_idx = NULL;

/************************** 2D MAP VARIABLES ************************/
size_t sz_map_2d = 0;
size_t sz_unique_in_particle = 0;
size_t sz_unique_in_particle_col = 0;

uint8_t* d_map_2d = NULL;
int* d_unique_in_particle = NULL;
int* d_unique_in_particle_col = NULL;

/*------------------------ RESULT VARIABLES -----------------------*/
uint8_t* res_map_2d = NULL;
int* res_unique_in_particle = NULL;
int* res_unique_in_particle_col = NULL;


/********************* RESIZE PARTICLES VARIABLES *******************/
size_t sz_last_len = 0;
int* d_last_len = NULL;
int* res_last_len = NULL;

size_t sz_particles_weight = 0;
float* d_particles_weight = NULL;
float* res_particles_weight = NULL;

float* res_robot_state = NULL;
float* res_robot_world_body = NULL;

/********************* UPDATE STATES VARIABLES **********************/
std::vector<float> std_vec_states_x;
std::vector<float> std_vec_states_y;
std::vector<float> std_vec_states_theta;


/********************* IMAGE TRANSFORM VARIABLES ********************/
size_t sz_transition_single_frame = 0;

float* d_transition_single_world_body = NULL;
float* d_transition_single_world_lidar = NULL;

/**************** PROCESSED MEASUREMENTS VARIABLES ******************/
size_t sz_processed_single_measure_pos = 0;

int* d_processed_single_measure_x = NULL;
int* d_processed_single_measure_y = NULL;

/*------------------------ RESULT VARIABLES -----------------------*/
int* res_processed_single_measure_x = NULL;
int* res_processed_single_measure_y = NULL;


/******************* OCCUPIED PARTICLES VARIABLES *******************/
size_t sz_particles_occupied_pos = 0;

int* d_particles_occupied_x = NULL;
int* d_particles_occupied_y = NULL;

size_t sz_occupied_map_idx = 0;
int* d_occupied_map_idx = NULL;

size_t sz_occupied_map_2d = 0;
size_t sz_occupied_unique_counter = 0;
size_t sz_occupied_unique_counter_col = 0;

uint8_t* d_occupied_map_2d = NULL;
int* d_occupied_unique_counter = NULL;
int* d_occupied_unique_counter_col = NULL;


/*------------------------ RESULT VARIABLES -----------------------*/
int* res_occupied_unique_counter = NULL;
int* res_occupied_unique_counter_col = NULL;

int* res_particles_occupied_x = NULL;
int* res_particles_occupied_y = NULL;


/********************** FREE PARTICLES VARIABLES ********************/
size_t sz_particles_free_pos = 0;
size_t sz_particles_free_pos_max = 0;
size_t sz_particles_free_counter = 0;
size_t sz_particles_free_idx = 0;

int* d_particles_free_x = NULL;
int* d_particles_free_y = NULL;
int* d_particles_free_idx = NULL;

int* d_particles_free_x_max = NULL;
int* d_particles_free_y_max = NULL;
int* d_particles_free_counter = NULL;

size_t sz_free_map_idx = 0;
int* d_free_map_idx = NULL;

size_t sz_free_map_2d = 0;
size_t sz_free_unique_counter = 0;
size_t sz_free_unique_counter_col = 0;

uint8_t* d_free_map_2d = NULL;
int* d_free_unique_counter = NULL;
int* d_free_unique_counter_col = NULL;

/*------------------------ RESULT VARIABLES -----------------------*/
int* res_particles_free_x = NULL;
int* res_particles_free_y = NULL;
int* res_particles_free_counter = NULL;

int* res_free_unique_counter = NULL;
int* res_free_unique_counter_col = NULL;


/**************************** MAP VARIABLES *************************/
//int* d_grid_map = NULL;
int* res_grid_map = NULL;


/************************* LOG-ODDS VARIABLES ***********************/
size_t sz_log_odds = 0;
float* d_log_odds = NULL;

/*------------------------ RESULT VARIABLES -----------------------*/
float* res_log_odds = NULL;


/********************************************************************/
/********************************************************************/
size_t sz_position_image_body = 0;
size_t sz_particles_world_pos = 0;

float* d_particles_world_x = NULL;
float* d_particles_world_y = NULL;
int* d_position_image_body = NULL;

float* res_particles_world_x = NULL;
float* res_particles_world_y = NULL;
int* res_position_image_body = NULL;

int h_occupied_map_idx[] = { 0, 0 };
int h_free_map_idx[] = { 0, 0 };


size_t sz_should_extend = 0;
size_t sz_coord = 0;

int* d_should_extend = NULL;
int* d_coord = NULL;

int* res_should_extend = NULL;
int* res_coord = NULL;


float* d_rnds_encoder_counts;
float* d_rnds_yaws;


///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

int NEW_GRID_WIDTH = 0;
int NEW_GRID_HEIGHT = 0;
int NEW_LIDAR_COORDS_LEN = 0;

float encoder_counts = 0;
float yaw = 0;
float dt = 0;

vector<int> vec_grid_map;
vector<float> vec_log_odds;
vector<float> vec_lidar_coords;

vector<float> vec_robot_transition_world_body;
vector<float> vec_robot_state;
vector<float> vec_particles_weight_post;
vector<float> vec_rnds;

vector<float> vec_rnds_encoder_counts;
vector<float> vec_rnds_yaws;
vector<float> vec_states_x;
vector<float> vec_states_y;
vector<float> vec_states_theta;


int EXTRA_GRID_WIDTH = 0;
int EXTRA_GRID_HEIGHT = 0;
int EXTRA_PARTICLES_ITEMS_LEN = 0;

vector<int> extra_grid_map;
vector<int> extra_particles_x;
vector<int> extra_particles_y;
vector<int> extra_particles_idx;
vector<float> extra_states_x;
vector<float> extra_states_y;
vector<float> extra_states_theta;
vector<float> extra_new_weights;
vector<float> extra_particles_weight_pre;

int extra_xmin = 0;
int extra_xmax = 0;
int extra_ymin = 0;
int extra_ymax = 0;
float extra_res = 0;
float extra_log_t = 0;

vector<float> extra_log_odds;
vector<float> extra_transition_single_world_body;

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

void read_robot_move_data(int file_number, bool check_rnds_encoder_counts = true, bool check_rnds_yaws = true,
	bool check_states = true) {

    string file_name = std::to_string(file_number);

	const int SCALAR_VALUES = 1;
	const int RNDS_ENCODER_COUNTS_VALUES = 2;
	const int RNDS_YAWS_VALUES = 3;
	const int STATES_X_VALUES = 4;
	const int STATES_Y_VALUES = 5;
	const int STATES_THETA_VALUES = 6;
	const int SEPARATE_VALUES = 10;

	int curr_state = SCALAR_VALUES;
	string str_rnds_encoder_counts = "";
	string str_rnds_yaws = "";
	string str_states_x = "";
	string str_states_y = "";
	string str_states_theta = "";
	string segment;

	std::ifstream data("data/steps/robot_advance_" + file_name + ".txt");
	string line;

	while (getline(data, line)) {

		line = trim(line);

		if (curr_state == SCALAR_VALUES) {

			if (line == "encoder_counts") {
				getline(data, line);
				encoder_counts = std::stof(line);
			}
			else if (line == "yaw") {
				getline(data, line);
				yaw = std::stof(line);
			}
			else if (line == "dt") {
				getline(data, line);
				dt = std::stof(line);
			}
		}

		if (line == "rnds_encoder_counts") {
			curr_state = RNDS_ENCODER_COUNTS_VALUES;
			continue;
		}
		else if (line == "rnds_yaws") {
			curr_state = RNDS_YAWS_VALUES;
			continue;
		}
		else if (line == "states_x") {
			curr_state = STATES_X_VALUES;
			continue;
		}
		else if (line == "states_y") {
			curr_state = STATES_Y_VALUES;
			continue;
		}
		else if (line == "states_theta") {
			curr_state = STATES_THETA_VALUES;
			continue;
		}

		if (curr_state == RNDS_ENCODER_COUNTS_VALUES) {
			if (line == "SEPARATE") {
				curr_state = SEPARATE_VALUES;
			}
			else {
				str_rnds_encoder_counts += line;
			}
		}
		else if (curr_state == RNDS_YAWS_VALUES) {
			if (line == "SEPARATE") {
				curr_state = SEPARATE_VALUES;
			}
			else {
				str_rnds_yaws += line;
			}
		}
		else if (curr_state == STATES_X_VALUES) {
			if (line == "SEPARATE") {
				curr_state = SEPARATE_VALUES;
			}
			else {
				str_states_x += line;
			}
		}
		else if (curr_state == STATES_Y_VALUES) {
			if (line == "SEPARATE") {
				curr_state = SEPARATE_VALUES;
			}
			else {
				str_states_y += line;
			}
		}
		else if (curr_state == STATES_THETA_VALUES) {
			if (line == "SEPARATE") {
				curr_state = SEPARATE_VALUES;
			}
			else {
				str_states_theta += line;
			}
		}
	}

	stringstream stream_rnds_encoder_counts(str_rnds_encoder_counts);
	vec_rnds_encoder_counts.resize(NUM_PARTICLES);
	for (int i = 0; std::getline(stream_rnds_encoder_counts, segment, ','); i++) {
		vec_rnds_encoder_counts[i] = std::stof(segment);
	}
	stringstream stream_rnds_yaw(str_rnds_yaws);
	vec_rnds_yaws.resize(NUM_PARTICLES);
	for (int i = 0; std::getline(stream_rnds_yaw, segment, ','); i++) {
		vec_rnds_yaws[i] = std::stof(segment);
	}
	stringstream stream_states_x(str_states_x);
	vec_states_x.resize(NUM_PARTICLES);
	for (int i = 0; std::getline(stream_states_x, segment, ','); i++) {
		vec_states_x[i] = std::stof(segment);
	}
	stringstream stream_states_y(str_states_y);
	vec_states_y.resize(NUM_PARTICLES);
	for (int i = 0; std::getline(stream_states_y, segment, ','); i++) {
		vec_states_y[i] = std::stof(segment);
	}
	stringstream stream_states_theta(str_states_theta);
	vec_states_theta.resize(NUM_PARTICLES);
	for (int i = 0; std::getline(stream_states_theta, segment, ','); i++) {
		vec_states_theta[i] = std::stof(segment);
	}

#ifdef ADD_HEADER_DATA
	int num_equals = 0;

	if (check_rnds_encoder_counts == true) {
		num_equals = 0;
		for (int i = 0; i < vec_rnds_encoder_counts.size(); i++) {
			if (vec_rnds_encoder_counts[i] != h_rnds_encoder_counts[i])
				printf("%f <> %f\n", vec_rnds_encoder_counts[i], h_rnds_encoder_counts[i]);
			else
				num_equals += 1;
		}
		printf("Rnds Encoder Counts Num Equals=%d, Num Errors=%d\n\n", num_equals, int(vec_rnds_encoder_counts.size() - num_equals));
	}
	if (check_rnds_yaws == true) {
		num_equals = 0;
		for (int i = 0; i < vec_rnds_yaws.size(); i++) {
			if (vec_rnds_yaws[i] != h_rnds_yaws[i])
				printf("%f <> %f\n", vec_rnds_yaws[i], h_rnds_yaws[i]);
			else
				num_equals += 1;
		}
		printf("Rnds Yaws Num Equals=%d, Num Errors=%d\n\n", num_equals, int(vec_rnds_yaws.size() - num_equals));
	}
	if (check_states == true) {
		num_equals = 0;
		for (int i = 0; i < vec_states_x.size(); i++) {
			if (vec_states_x[i] != h_states_x[i])
				printf("%f <> %f\n", vec_states_x[i], h_states_x[i]);
			else
				num_equals += 1;
		}
		printf("States X Num Equals=%d, Num Errors=%d\n\n", num_equals, int(vec_states_x.size() - num_equals));

		num_equals = 0;
		for (int i = 0; i < vec_states_y.size(); i++) {
			if (vec_states_y[i] != h_states_y[i])
				printf("%f <> %f\n", vec_states_y[i], h_states_y[i]);
			else
				num_equals += 1;
		}
		printf("States Y Num Equals=%d, Num Errors=%d\n\n", num_equals, int(vec_states_y.size() - num_equals));

		num_equals = 0;
		for (int i = 0; i < vec_states_theta.size(); i++) {
			if (vec_states_theta[i] != h_states_theta[i])
				printf("%f <> %f\n", vec_states_theta[i], h_states_theta[i]);
			else
				num_equals += 1;
		}
		printf("States Theta Num Equals=%d, Num Errors=%d\n\n", num_equals, int(vec_states_theta.size() - num_equals));
	}
#endif
}

void read_robot_data(int file_number, bool check_robot_transition = true, bool check_state = true,
	bool check_particles_weight = true, bool check_rnds = true) {

    string file_name = std::to_string(file_number);

	const int ROBOT_TRANSITION_WORLD_BODY_VALUES = 2;
	const int ROBOT_STATE_VALUES = 3;
	const int PARTICLES_WEIGHT_POST_VALUES = 4;
	const int RNDS_VALUES = 5;
	const int SEPARATE_VALUES = 10;

	int curr_state = SEPARATE_VALUES;
	string str_robot_transition_world_body = "";
	string str_robot_state = "";
	string str_particles_weight_post = "";
	string str_rnds = "";
	string segment;

	std::ifstream data("data/steps/robot_" + file_name + ".txt");
	string line;

	while (getline(data, line)) {

		line = trim(line);

		if (line == "robot_transition_world_body") {
			curr_state = ROBOT_TRANSITION_WORLD_BODY_VALUES;
			continue;
		}
		else if (line == "robot_state") {
			curr_state = ROBOT_STATE_VALUES;
			continue;
		}
		else if (line == "particles_weight_post") {
			curr_state = PARTICLES_WEIGHT_POST_VALUES;
			continue;
		}
		else if (line == "rnds") {
			curr_state = RNDS_VALUES;
			continue;
		}

		if (curr_state == ROBOT_TRANSITION_WORLD_BODY_VALUES) {
			if (line == "SEPARATE") {
				curr_state = SEPARATE_VALUES;
			}
			else {
				str_robot_transition_world_body += line;
			}
		}
		else if (curr_state == ROBOT_STATE_VALUES) {
			if (line == "SEPARATE") {
				curr_state = SEPARATE_VALUES;
			}
			else {
				str_robot_state += line;
			}
		}
		else if (curr_state == PARTICLES_WEIGHT_POST_VALUES) {
			if (line == "SEPARATE") {
				curr_state = SEPARATE_VALUES;
			}
			else {
				str_particles_weight_post += line;
			}
		}
		else if (curr_state == RNDS_VALUES) {
			if (line == "SEPARATE") {
				curr_state = SEPARATE_VALUES;
			}
			else {
				str_rnds += line;
			}
		}
	}

	stringstream stream_robot_transition_world_body(str_robot_transition_world_body);
	vec_robot_transition_world_body.resize(9);
	for (int i = 0; std::getline(stream_robot_transition_world_body, segment, ','); i++) {
		vec_robot_transition_world_body[i] = std::stof(segment);
	}
	stringstream stream_robot_state(str_robot_state);
	vec_robot_state.resize(3);
	for (int i = 0; std::getline(stream_robot_state, segment, ','); i++) {
		vec_robot_state[i] = std::stof(segment);
	}
	stringstream stream_particles_weight_post(str_particles_weight_post);
	vec_particles_weight_post.resize(NUM_PARTICLES);
	for (int i = 0; std::getline(stream_particles_weight_post, segment, ','); i++) {
		vec_particles_weight_post[i] = std::stof(segment);
	}
	stringstream stream_rnds(str_rnds);
	vec_rnds.resize(NUM_PARTICLES);
	for (int i = 0; std::getline(stream_rnds, segment, ','); i++) {
		vec_rnds[i] = std::stof(segment);
	}

#ifdef ADD_HEADER_DATA
	int num_equals = 0;

	if (check_robot_transition == true) {
		num_equals = 0;
		for (int i = 0; i < vec_robot_transition_world_body.size(); i++) {
			if (vec_robot_transition_world_body[i] != h_robot_transition_world_body[i])
				printf("%f <> %f\n", vec_robot_transition_world_body[i], h_robot_transition_world_body[i]);
			else
				num_equals += 1;
		}
		printf("Robot Transition World Body Num Equals=%d, Num Errors=%d\n\n", num_equals, int(vec_robot_transition_world_body.size() - num_equals));
	}

	if (check_state == true) {
		num_equals = 0;
		for (int i = 0; i < vec_robot_state.size(); i++) {
			if (vec_robot_state[i] != h_robot_state[i])
				printf("%f <> %f\n", vec_robot_state[i], h_robot_state[i]);
			else
				num_equals += 1;
		}
		printf("Robot State Num Equals=%d, Num Errors=%d\n\n", num_equals, int(vec_robot_state.size() - num_equals));
	}

	if (check_particles_weight == true) {
		num_equals = 0;
		for (int i = 0; i < vec_particles_weight_post.size(); i++) {
			if (vec_particles_weight_post[i] != h_particles_weight_post[i])
				printf("%f <> %f\n", vec_particles_weight_post[i], h_particles_weight_post[i]);
			else
				num_equals += 1;
		}
		printf("Particles Weights Num Equals=%d, Num Errors=%d\n\n", num_equals, int(vec_particles_weight_post.size() - num_equals));
	}

	if (check_rnds == true) {
		num_equals = 0;
		for (int i = 0; i < vec_rnds.size(); i++) {
			if (vec_rnds[i] != h_rnds[i])
				printf("%f <> %f\n", vec_rnds[i], h_rnds[i]);
			else
				num_equals += 1;
		}
		printf("Rnds Num Equals=%d, Num Errors=%d\n\n", num_equals, int(vec_rnds.size() - num_equals));
	}
#endif
}

void read_robot_extra(int file_number, bool check_grid_map = true, bool check_particles = true,
    bool check_states = true, bool check_weights = true) {

    string file_name = std::to_string(file_number);

    const int SCALAR_VALUES = 1;
    const int GRID_MAP_VALUES = 2;
    const int PARTICLES_X_VALUES = 3;
    const int PARTICLES_Y_VALUES = 4;
    const int PARTICLES_IDX_VALUES = 5;
    const int STATES_X_VALUES = 6;
    const int STATES_Y_VALUES = 7;
    const int STATES_THETA_VALUES = 8;
    const int NEW_WEIGHTS_VALUES = 9;
    const int PARTICLES_WEIGHT_VALUES = 10;
    const int SEPARATE_VALUES = 11;

    int curr_state = SCALAR_VALUES;
    string str_grid_map = "";
    string str_particles_x = "";
    string str_particles_y = "";
    string str_particles_idx = "";
    string str_states_x = "";
    string str_states_y = "";
    string str_states_theta = "";
    string str_new_weights = "";
    string str_particles_weight = "";
    string segment;

    std::ifstream data("data/extra/robot_" + file_name + ".txt");
    string line;

    while (getline(data, line)) {

        line = trim(line);
        if (line == "") continue;

        if (curr_state == SCALAR_VALUES) {
            
            if (line == "GRID_WIDTH") {
                getline(data, line);
                EXTRA_GRID_WIDTH = std::stoi(line);
            }
            else if (line == "GRID_HEIGHT") {
                getline(data, line);
                EXTRA_GRID_HEIGHT = std::stoi(line);
            }
            else if (line == "PARTICLES_ITEMS_LEN") {
                getline(data, line);
                EXTRA_PARTICLES_ITEMS_LEN = std::stoi(line);
            }
        }

        if (line == "grid_map") {
            curr_state = GRID_MAP_VALUES;
            continue;
        }
        else if (line == "particles_x") {
            curr_state = PARTICLES_X_VALUES;
            continue;
        }
        else if (line == "particles_y") {
            curr_state = PARTICLES_Y_VALUES;
            continue;
        }
        else if (line == "particles_idx") {
            curr_state = PARTICLES_IDX_VALUES;
            continue;
        }
        else if (line == "states_x") {
            curr_state = STATES_X_VALUES;
            continue;
        }
        else if (line == "states_y") {
            curr_state = STATES_Y_VALUES;
            continue;
        }
        else if (line == "states_theta") {
            curr_state = STATES_THETA_VALUES;
            continue;
        }
        else if (line == "new_weights") {
            curr_state = NEW_WEIGHTS_VALUES;
            continue;
        }
        else if (line == "particles_weight_pre") {
            curr_state = PARTICLES_WEIGHT_VALUES;
            continue;
        }

        if (curr_state == GRID_MAP_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_grid_map += line;
            }
        }
        else if (curr_state == PARTICLES_X_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_particles_x += line;
            }
        }
        else if (curr_state == PARTICLES_Y_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_particles_y += line;
            }
        }
        else if (curr_state == PARTICLES_IDX_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_particles_idx += line;
            }
        }
        else if (curr_state == STATES_X_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_states_x += line;
            }
        }
        else if (curr_state == STATES_Y_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_states_y += line;
            }
        }
        else if (curr_state == STATES_THETA_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_states_theta += line;
            }
        }
        else if (curr_state == NEW_WEIGHTS_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_new_weights += line;
            }
        }
        else if (curr_state == PARTICLES_WEIGHT_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_particles_weight += line;
            }
        }

    }

    int GRID_SIZE = EXTRA_GRID_WIDTH * EXTRA_GRID_HEIGHT;

    stringstream stream_grid_map(str_grid_map);
    extra_grid_map.resize(GRID_SIZE);
    for (int i = 0; std::getline(stream_grid_map, segment, ','); i++) {
        extra_grid_map[i] = std::stoi(segment);
    }
    stringstream stream_particles_x(str_particles_x);
    extra_particles_x.resize(EXTRA_PARTICLES_ITEMS_LEN);
    for (int i = 0; std::getline(stream_particles_x, segment, ','); i++) {
        extra_particles_x[i] = std::stoi(segment);
    }
    stringstream stream_particles_y(str_particles_y);
    extra_particles_y.resize(EXTRA_PARTICLES_ITEMS_LEN);
    for (int i = 0; std::getline(stream_particles_y, segment, ','); i++) {
        extra_particles_y[i] = std::stoi(segment);
    }
    stringstream stream_particles_idx(str_particles_idx);
    extra_particles_idx.resize(NUM_PARTICLES);
    for (int i = 0; std::getline(stream_particles_idx, segment, ','); i++) {
        extra_particles_idx[i] = std::stoi(segment);
    }
    stringstream stream_states_x(str_states_x);
    extra_states_x.resize(NUM_PARTICLES);
    for (int i = 0; std::getline(stream_states_x, segment, ','); i++) {
        extra_states_x[i] = std::stof(segment);
    }
    stringstream stream_states_y(str_states_y);
    extra_states_y.resize(NUM_PARTICLES);
    for (int i = 0; std::getline(stream_states_y, segment, ','); i++) {
        extra_states_y[i] = std::stof(segment);
    }
    stringstream stream_states_theta(str_states_theta);
    extra_states_theta.resize(NUM_PARTICLES);
    for (int i = 0; std::getline(stream_states_theta, segment, ','); i++) {
        extra_states_theta[i] = std::stof(segment);
    }
    stringstream stream_new_weights(str_new_weights);
    extra_new_weights.resize(NUM_PARTICLES);
    for (int i = 0; std::getline(stream_new_weights, segment, ','); i++) {
        extra_new_weights[i] = std::stof(segment);
    }
    stringstream stream_particles_weight_pre(str_particles_weight);
    extra_particles_weight_pre.resize(NUM_PARTICLES);
    for (int i = 0; std::getline(stream_particles_weight_pre, segment, ','); i++) {
        extra_particles_weight_pre[i] = std::stof(segment);
    }

#ifdef ADD_HEADER_DATA
    int num_equals = 0;

    if (check_grid_map == true) {
        num_equals = 0;
        for (int i = 0; i < extra_grid_map.size(); i++) {
            if (extra_grid_map[i] != h_grid_map[i])
                printf("%d <> %d\n", extra_grid_map[i], h_grid_map[i]);
            else
                num_equals += 1;
        }
        printf("Grid Map Num Equals=%d, Num Errors=%d\n\n", num_equals, int(extra_grid_map.size() - num_equals));
    }

    if (check_particles == true) {
        num_equals = 0;
        for (int i = 0; i < extra_particles_x.size(); i++) {
            if (extra_particles_x[i] != h_particles_x[i])
                printf("%d <> %d\n", extra_particles_x[i], h_particles_x[i]);
            else
                num_equals += 1;
        }
        printf("Particles X Num Equals=%d, Num Errors=%d\n\n", num_equals, int(extra_particles_x.size() - num_equals));

        num_equals = 0;
        for (int i = 0; i < extra_particles_y.size(); i++) {
            if (extra_particles_y[i] != h_particles_y[i])
                printf("%d <> %d\n", extra_particles_x[i], h_particles_y[i]);
            else
                num_equals += 1;
        }
        printf("Particles Y Num Equals=%d, Num Errors=%d\n\n", num_equals, int(extra_particles_y.size() - num_equals));

        num_equals = 0;
        for (int i = 0; i < extra_particles_idx.size(); i++) {
            if (extra_particles_idx[i] != h_particles_idx[i])
                printf("%d <> %d\n", extra_particles_idx[i], h_particles_idx[i]);
            else
                num_equals += 1;
        }
        printf("Particles Idx Num Equals=%d, Num Errors=%d\n\n", num_equals, int(extra_particles_idx.size() - num_equals));
    }

    if (check_states == true) {
        num_equals = 0;
        for (int i = 0; i < extra_states_x.size(); i++) {
            if (extra_states_x[i] != post_states_x[i])
                printf("%f <> %f\n", extra_states_x[i], post_states_x[i]);
            else
                num_equals += 1;
        }
        printf("States X Num Equals=%d, Num Errors=%d\n\n", num_equals, int(extra_states_x.size() - num_equals));

        num_equals = 0;
        for (int i = 0; i < extra_states_y.size(); i++) {
            if (extra_states_y[i] != post_states_y[i])
                printf("%f <> %f\n", extra_states_y[i], post_states_y[i]);
            else
                num_equals += 1;
        }
        printf("States Y Num Equals=%d, Num Errors=%d\n\n", num_equals, int(extra_states_y.size() - num_equals));

        num_equals = 0;
        for (int i = 0; i < extra_states_theta.size(); i++) {
            if (extra_states_theta[i] != post_states_theta[i])
                printf("%f <> %f\n", extra_states_theta[i], post_states_theta[i]);
            else
                num_equals += 1;
        }
        printf("States Theta Num Equals=%d, Num Errors=%d\n\n", num_equals, int(extra_states_theta.size() - num_equals));
    }

    if (check_weights == true) {
        num_equals = 0;
        for (int i = 0; i < extra_new_weights.size(); i++) {
            if (extra_new_weights[i] != h_new_weights[i])
                printf("%f <> %f\n", extra_new_weights[i], h_new_weights[i]);
            else
                num_equals += 1;
        }
        printf("New Weights Num Equals=%d, Num Errors=%d\n\n", num_equals, int(extra_new_weights.size() - num_equals));

        num_equals = 0;
        for (int i = 0; i < extra_particles_weight_pre.size(); i++) {
            if (extra_particles_weight_pre[i] != h_particles_weight_pre[i])
                printf("%f <> %f\n", extra_particles_weight_pre[i], h_particles_weight_pre[i]);
            else
                num_equals += 1;
        }
        printf("Particles Weight Num Equals=%d, Num Errors=%d\n\n", num_equals, int(extra_particles_weight_pre.size() - num_equals));
    }
#endif
}


void read_map_data(int file_number, bool check_grid_map = true, bool check_log_odds = true, bool check_lidar_coords = true) {

    const int SCALAR_VALUES = 1;
    const int GRID_MAP_VALUES = 2;
    const int LOG_ODDS_VALUES = 3;
    const int LIDAR_COORDS_VALUES = 4;
    const int SEPARATE_VALUES = 10;

    int curr_state = SCALAR_VALUES;
    string str_grid_map = "";
    string str_log_odds = "";
    string str_lidar_coords = "";
    string segment;

    string file_name = std::to_string(file_number);

    std::ifstream data("data/steps/map_" + file_name + ".txt");
    string line;

    while (getline(data, line)) {

        line = trim(line);

        if (curr_state == SCALAR_VALUES) {

            if (line == "GRID_WIDTH") {
                getline(data, line);
                NEW_GRID_WIDTH = std::stoi(line);
            }
            else if (line == "GRID_HEIGHT") {
                getline(data, line);
                NEW_GRID_HEIGHT = std::stoi(line);
            }
            else if (line == "LIDAR_COORDS_LEN") {
                getline(data, line);
                NEW_LIDAR_COORDS_LEN = std::stoi(line);
            }
        }

        if (line == "grid_map") {
            curr_state = GRID_MAP_VALUES;
            continue;
        }
        else if (line == "log_odds") {
            curr_state = LOG_ODDS_VALUES;
            continue;
        }
        else if (line == "lidar_coords") {
            curr_state = LIDAR_COORDS_VALUES;
            continue;
        }

        if (curr_state == GRID_MAP_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_grid_map += line;
            }
        }
        else if (curr_state == LOG_ODDS_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_log_odds += line;
            }
        }
        else if (curr_state == LIDAR_COORDS_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_lidar_coords += line;
            }
        }
    }

    int GRID_SIZE = NEW_GRID_WIDTH * NEW_GRID_HEIGHT;

    stringstream stream_grid_map(str_grid_map);
    vec_grid_map.resize(GRID_SIZE);
    for (int i = 0; std::getline(stream_grid_map, segment, ','); i++) {
        vec_grid_map[i] = std::stoi(segment);
    }
    stringstream stream_log_odds(str_log_odds);
    vec_log_odds.resize(GRID_SIZE);
    for (int i = 0; std::getline(stream_log_odds, segment, ','); i++) {
        vec_log_odds[i] = std::stof(segment);
    }
    stringstream stream_lidar_coords(str_lidar_coords);
    vec_lidar_coords.resize(2 * NEW_LIDAR_COORDS_LEN);
    for (int i = 0; std::getline(stream_lidar_coords, segment, ','); i++) {
        vec_lidar_coords[i] = std::stof(segment);
    }

    int num_equals = 0;

#ifdef ADD_HEADER_DATA
    if (check_grid_map == true) {
        for (int i = 0; i < vec_grid_map.size(); i++) {
            if (vec_grid_map[i] != h_post_grid_map[i])
                printf("%d <> %d\n", vec_grid_map[i], h_post_grid_map[i]);
            else
                num_equals += 1;
        }
        printf("Grid Map Num Equals=%d\n\n", num_equals);
    }

    if (check_log_odds == true) {
        num_equals = 0;
        for (int i = 0; i < vec_log_odds.size(); i++) {
            if (vec_log_odds[i] != h_post_log_odds[i])
                printf("%f <> %f\n", vec_log_odds[i], h_post_log_odds[i]);
            else
                num_equals += 1;
        }
        printf("Log Odds Num Equals=%d, Num Errors=%d\n\n", num_equals, int(vec_log_odds.size() - num_equals));
    }
    if (check_lidar_coords == true) {
        num_equals = 0;
        for (int i = 0; i < 2 * NEW_LIDAR_COORDS_LEN; i++) {
            if (vec_lidar_coords[i] != h_lidar_coords[i])
                printf("%f <> %f\n", vec_lidar_coords[i], h_lidar_coords[i]);
            else
                num_equals += 1;
        }
        printf("LIDAR Coords Num Equals=%d, Num Errors=%d\n\n", num_equals, int(vec_lidar_coords.size() - num_equals));
    }
#endif
}

void read_map_extra(int file_number, bool check_grid_map = true, bool check_log_odds = true, bool check_transition = true) {

    const int SCALAR_VALUES = 1;
    const int GRID_MAP_VALUES = 2;
    const int LOG_ODDS_VALUES = 3;
    const int TRANSITION_VALUES = 4;
    const int SEPARATE_VALUES = 10;

    int curr_state = SCALAR_VALUES;
    string str_grid_map = "";
    string str_log_odds = "";
    string str_transition = "";
    string segment;

    string file_name = std::to_string(file_number);

    std::ifstream data("data/extra/map_" + file_name + ".txt");
    string line;

    while (getline(data, line)) {

        line = trim(line);

        if (curr_state == SCALAR_VALUES) {

            if (line == "GRID_WIDTH") {
                getline(data, line);
                EXTRA_GRID_WIDTH = std::stoi(line);
            }
            else if (line == "GRID_HEIGHT") {
                getline(data, line);
                EXTRA_GRID_HEIGHT = std::stoi(line);
            }
            else if (line == "xmin") {
                getline(data, line);
                extra_xmin = std::stoi(line);
            }
            else if (line == "xmax") {
                getline(data, line);
                extra_xmax = std::stoi(line);
            }
            else if (line == "ymin") {
                getline(data, line);
                extra_ymin = std::stoi(line);
            }
            else if (line == "ymax") {
                getline(data, line);
                extra_ymax = std::stoi(line);
            }
            else if (line == "res") {
                getline(data, line);
                extra_res = std::stof(line);
            }
            else if (line == "log_t") {
                getline(data, line);
                extra_log_t = std::stof(line);
            }
        }

        if (line == "grid_map") {
            curr_state = GRID_MAP_VALUES;
            continue;
        }
        else if (line == "log_odds") {
            curr_state = LOG_ODDS_VALUES;
            continue;
        }
        else if (line == "transition_single_world_body") {
            curr_state = TRANSITION_VALUES;
            continue;
        }

        if (curr_state == GRID_MAP_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_grid_map += line;
            }
        }
        else if (curr_state == LOG_ODDS_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_log_odds += line;
            }
        }
        else if (curr_state == TRANSITION_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_transition += line;
            }
        }
    }

    int GRID_SIZE = NEW_GRID_WIDTH * NEW_GRID_HEIGHT;

    stringstream stream_grid_map(str_grid_map);
    extra_grid_map.resize(GRID_SIZE);
    for (int i = 0; std::getline(stream_grid_map, segment, ','); i++) {
        extra_grid_map[i] = std::stoi(segment);
    }
    stringstream stream_log_odds(str_log_odds);
    extra_log_odds.resize(GRID_SIZE);
    for (int i = 0; std::getline(stream_log_odds, segment, ','); i++) {
        extra_log_odds[i] = std::stof(segment);
    }
    stringstream stream_transition(str_transition);
    extra_transition_single_world_body.resize(9);
    for (int i = 0; std::getline(stream_transition, segment, ','); i++) {
        extra_transition_single_world_body[i] = std::stof(segment);
    }

    int num_equals = 0;

#ifdef ADD_HEADER_DATA
    if (check_grid_map == true) {
        for (int i = 0; i < extra_grid_map.size(); i++) {
            if (extra_grid_map[i] != h_grid_map[i])
                printf("%d <> %d\n", extra_grid_map[i], h_grid_map[i]);
            else
                num_equals += 1;
        }
        printf("Extra Grid Map Num Equals=%d\n\n", num_equals);
    }

    if (check_log_odds == true) {
        num_equals = 0;
        for (int i = 0; i < extra_log_odds.size(); i++) {
            if (extra_log_odds[i] != h_log_odds[i])
                printf("%f <> %f\n", extra_log_odds[i], h_log_odds[i]);
            else
                num_equals += 1;
        }
        printf("Extra Log Odds Num Equals=%d, Num Errors=%d\n\n", num_equals, int(vec_log_odds.size() - num_equals));
    }

    if (check_transition == true) {
        num_equals = 0;
        for (int i = 0; i < extra_transition_single_world_body.size(); i++) {
            if (extra_transition_single_world_body[i] != h_transition_single_world_body[i])
                printf("%f <> %f\n", extra_transition_single_world_body[i], h_transition_single_world_body[i]);
            else
                num_equals += 1;
        }
        printf("Transition World Body Num Equals=%d, Num Errors=%d\n\n", num_equals, int(extra_transition_single_world_body.size() - num_equals));
    }
#endif


}


///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

void alloc_init_state_vars(float* h_states_x, float* h_states_y, float* h_states_theta) {

    sz_states_pos = NUM_PARTICLES * sizeof(float);

    gpuErrchk(cudaMalloc((void**)&d_states_x, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&d_states_y, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&d_states_theta, sz_states_pos));

    gpuErrchk(cudaMemcpy(d_states_x, h_states_x, sz_states_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_states_y, h_states_y, sz_states_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_states_theta, h_states_theta, sz_states_pos, cudaMemcpyHostToDevice));

    res_robot_state = (float*)malloc(3 * sizeof(float));
}

//void alloc_init_lidar_coords_var(float* h_lidar_coords, const int LIDAR_COORDS_LEN) {
//
//    sz_lidar_coords = 2 * LIDAR_COORDS_LEN * sizeof(float);
//    gpuErrchk(cudaMalloc((void**)&d_lidar_coords, sz_lidar_coords));
//
//    gpuErrchk(cudaMemcpy(d_lidar_coords, h_lidar_coords, sz_lidar_coords, cudaMemcpyHostToDevice));
//}

void alloc_init_grid_map(int* h_grid_map, const int GRID_WIDTH, const int GRID_HEIGHT) {

    sz_grid_map = GRID_WIDTH * GRID_HEIGHT * sizeof(int);
    gpuErrchk(cudaMalloc((void**)&d_grid_map, sz_grid_map));

    gpuErrchk(cudaMemcpy(d_grid_map, h_grid_map, sz_grid_map, cudaMemcpyHostToDevice));
}

void alloc_init_particles_vars(int* h_particles_x, int* h_particles_y, int* h_particles_idx,
    float* h_particles_weight, const int PARTICLES_ITEMS_LEN) {

    sz_particles_pos = PARTICLES_ITEMS_LEN * sizeof(int);
    sz_particles_idx = NUM_PARTICLES * sizeof(int);
    sz_particles_weight = NUM_PARTICLES * sizeof(float);

    gpuErrchk(cudaMalloc((void**)&d_particles_x, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_y, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_idx, sz_particles_idx));
    gpuErrchk(cudaMalloc((void**)&d_particles_weight, sz_particles_weight));

    gpuErrchk(cudaMemcpy(d_particles_x, h_particles_x, sz_particles_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particles_y, h_particles_y, sz_particles_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particles_idx, h_particles_idx, sz_particles_idx, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particles_weight, h_particles_weight, sz_particles_weight, cudaMemcpyHostToDevice));

    res_particles_idx = (int*)malloc(sz_particles_idx);
}

void alloc_extended_idx(const int PARTICLES_ITEMS_LEN) {

    sz_extended_idx = PARTICLES_ITEMS_LEN * sizeof(int);
    gpuErrchk(cudaMalloc((void**)&d_extended_idx, sz_extended_idx));

    res_extended_idx = (int*)malloc(sz_extended_idx);
}

void alloc_states_copy_vars() {

    gpuErrchk(cudaMalloc((void**)&dc_states_x, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&dc_states_y, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&dc_states_theta, sz_states_pos));
}

void alloc_correlation_vars() {

    sz_correlation_weights = NUM_PARTICLES * sizeof(float);
    sz_correlation_weights_raw = 25 * sz_correlation_weights;

    gpuErrchk(cudaMalloc((void**)&d_correlation_weights, sz_correlation_weights));
    gpuErrchk(cudaMalloc((void**)&d_correlation_weights_raw, sz_correlation_weights_raw));
    gpuErrchk(cudaMemset(d_correlation_weights_raw, 0, sz_correlation_weights_raw));

    res_correlation_weights = (float*)malloc(sz_correlation_weights);
    memset(res_correlation_weights, 0, sz_correlation_weights);

    //res_extended_idx = (int*)malloc(sz_extended_idx);
}

void alloc_init_transition_vars(float* h_transition_body_lidar) {

    sz_transition_multi_world_frame = 9 * NUM_PARTICLES * sizeof(float);
    sz_transition_body_lidar = 9 * sizeof(float);

    gpuErrchk(cudaMalloc((void**)&d_transition_multi_world_body, sz_transition_multi_world_frame));
    gpuErrchk(cudaMalloc((void**)&d_transition_multi_world_lidar, sz_transition_multi_world_frame));
    gpuErrchk(cudaMalloc((void**)&d_transition_body_lidar, sz_transition_body_lidar));

    gpuErrchk(cudaMemcpy(d_transition_body_lidar, h_transition_body_lidar, sz_transition_body_lidar, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemset(d_transition_multi_world_body, 0, sz_transition_multi_world_frame));
    gpuErrchk(cudaMemset(d_transition_multi_world_lidar, 0, sz_transition_multi_world_frame));

    res_transition_world_body = (float*)malloc(sz_transition_multi_world_frame);
    //res_transition_world_lidar = (float*)malloc(sz_transition_world_frame);
    res_robot_world_body = (float*)malloc(sz_transition_multi_world_frame);
}

void alloc_init_processed_measurement_vars(const int LIDAR_COORDS_LEN) {

    sz_processed_measure_pos = NUM_PARTICLES * LIDAR_COORDS_LEN * sizeof(int);
    sz_processed_measure_idx = NUM_PARTICLES * LIDAR_COORDS_LEN * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_processed_measure_x, sz_processed_measure_pos));
    gpuErrchk(cudaMalloc((void**)&d_processed_measure_y, sz_processed_measure_pos));
    gpuErrchk(cudaMalloc((void**)&d_processed_measure_idx, sz_processed_measure_idx));

    gpuErrchk(cudaMemset(d_processed_measure_x, 0, sz_processed_measure_pos));
    gpuErrchk(cudaMemset(d_processed_measure_y, 0, sz_processed_measure_pos));
    gpuErrchk(cudaMemset(d_processed_measure_idx, 0, sz_processed_measure_idx));
}

void alloc_map_2d_var(const int GRID_WIDTH, const int GRID_HEIGHT) {

    sz_map_2d = GRID_WIDTH * GRID_HEIGHT * NUM_PARTICLES * sizeof(uint8_t);

    gpuErrchk(cudaMalloc((void**)&d_map_2d, sz_map_2d));
    gpuErrchk(cudaMemset(d_map_2d, 0, sz_map_2d));
}

void alloc_map_2d_unique_counter_vars(const int UNIQUE_COUNTER_LEN, const int GRID_WIDTH) {

    sz_unique_in_particle = UNIQUE_COUNTER_LEN * sizeof(int);
    sz_unique_in_particle_col = UNIQUE_COUNTER_LEN * GRID_WIDTH * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_unique_in_particle, sz_unique_in_particle));
    gpuErrchk(cudaMalloc((void**)&d_unique_in_particle_col, sz_unique_in_particle_col));

    gpuErrchk(cudaMemset(d_unique_in_particle, 0, sz_unique_in_particle));
    gpuErrchk(cudaMemset(d_unique_in_particle_col, 0, sz_unique_in_particle_col));

    res_unique_in_particle = (int*)malloc(sz_unique_in_particle);
}

void alloc_correlation_weights_vars() {

    sz_correlation_sum_exp = sizeof(double);
    sz_correlation_weights_max = sizeof(float);

    gpuErrchk(cudaMalloc((void**)&d_correlation_sum_exp, sz_correlation_sum_exp));
    gpuErrchk(cudaMalloc((void**)&d_correlation_weights_max, sz_correlation_weights_max));

    gpuErrchk(cudaMemset(d_correlation_sum_exp, 0, sz_correlation_sum_exp));
    gpuErrchk(cudaMemset(d_correlation_weights_max, 0, sz_correlation_weights_max));

    res_correlation_sum_exp = (double*)malloc(sz_correlation_sum_exp);
    res_correlation_weights_max = (float*)malloc(sz_correlation_weights_max);
}

void alloc_resampling_vars(float* h_resampling_rnds) {

    sz_resampling_js = NUM_PARTICLES * sizeof(int);
    sz_resampling_rnd = NUM_PARTICLES * sizeof(float);

    gpuErrchk(cudaMalloc((void**)&d_resampling_js, sz_resampling_js));
    gpuErrchk(cudaMalloc((void**)&d_resampling_rnd, sz_resampling_rnd));

    gpuErrchk(cudaMemcpy(d_resampling_rnd, h_resampling_rnds, sz_resampling_rnd, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(d_resampling_js, 0, sz_resampling_js));
}

void exec_calc_transition() {

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_update_particles_states << <blocksPerGrid, threadsPerBlock >> > (d_transition_multi_world_body, d_transition_multi_world_lidar, SEP,
        d_states_x, d_states_y, d_states_theta,
        d_transition_body_lidar, NUM_PARTICLES);
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(res_transition_world_body, d_transition_multi_world_body, sz_transition_multi_world_frame, cudaMemcpyDeviceToHost));
}

void exec_process_measurements(float res, const int xmin, const int ymax, const int LIDAR_COORDS_LEN) {

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = LIDAR_COORDS_LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (d_processed_measure_x, d_processed_measure_y, SEP,
        d_transition_multi_world_lidar, d_lidar_coords, res, xmin, ymax, LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_index_init_const << < blocksPerGrid, threadsPerBlock >> > (d_processed_measure_idx, LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, d_processed_measure_idx, d_processed_measure_idx + NUM_PARTICLES, d_processed_measure_idx, 0);
}

void exec_create_2d_map(const int PARTICLES_ITEMS_LEN, const int GRID_WIDTH, const int GRID_HEIGHT) {

    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (d_map_2d, d_unique_in_particle, d_unique_in_particle_col, SEP,
        d_particles_x, d_particles_y, d_particles_idx, PARTICLES_ITEMS_LEN, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();
}

void exec_update_map(const int MEASURE_LEN, const int GRID_WIDTH, const int GRID_HEIGHT) {

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;

    kernel_update_2d_map_with_measure << <blocksPerGrid, threadsPerBlock >> > (d_map_2d, d_unique_in_particle, d_unique_in_particle_col, SEP,
        d_processed_measure_x, d_processed_measure_y, d_processed_measure_idx,
        MEASURE_LEN, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();
}

void exec_particle_unique_cum_sum(int& PARTICLES_ITEMS_LEN, int& C_PARTICLES_ITEMS_LEN, const int GRID_WIDTH) {

    threadsPerBlock = UNIQUE_COUNTER_LEN;
    blocksPerGrid = 1;
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (d_unique_in_particle_col, GRID_WIDTH);
    thrust::exclusive_scan(thrust::device, d_unique_in_particle, d_unique_in_particle + UNIQUE_COUNTER_LEN, d_unique_in_particle, 0);
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(res_unique_in_particle, d_unique_in_particle, sz_unique_in_particle, cudaMemcpyDeviceToHost));

    PARTICLES_ITEMS_LEN = res_unique_in_particle[UNIQUE_COUNTER_LEN - 1];
    C_PARTICLES_ITEMS_LEN = 0;
}

void reinit_map_vars(const int PARTICLES_ITEMS_LEN) {

    //gpuErrchk(cudaFree(d_particles_x));
    //gpuErrchk(cudaFree(d_particles_y));
    //gpuErrchk(cudaFree(d_extended_idx));

    sz_particles_pos = PARTICLES_ITEMS_LEN * sizeof(int);
    sz_extended_idx = PARTICLES_ITEMS_LEN * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_particles_x, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_y, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_extended_idx, sz_extended_idx));
}

void exec_map_restructure(const int GRID_WIDTH, const int GRID_HEIGHT) {

    threadsPerBlock = GRID_WIDTH;
    blocksPerGrid = NUM_PARTICLES;

    cudaMemset(d_particles_idx, 0, sz_particles_idx);
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (d_particles_x, d_particles_y, d_particles_idx, SEP,
        d_map_2d, d_unique_in_particle, d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, d_particles_idx, d_particles_idx + NUM_PARTICLES, d_particles_idx, 0);
}

void exec_index_expansion(const int PARTICLES_ITEMS_LEN) {

    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;
    kernel_index_expansion << <blocksPerGrid, threadsPerBlock >> > (d_extended_idx, d_particles_idx, PARTICLES_ITEMS_LEN);
    cudaDeviceSynchronize();

    res_extended_idx = (int*)malloc(sz_extended_idx);
    gpuErrchk(cudaMemcpy(res_extended_idx, d_extended_idx, sz_extended_idx, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToHost));
}

void exec_correlation(const int PARTICLES_ITEMS_LEN, const int GRID_WIDTH, const int GRID_HEIGHT) {

    threadsPerBlock = 256;
    blocksPerGrid = (PARTICLES_ITEMS_LEN + threadsPerBlock - 1) / threadsPerBlock;

    kernel_correlation << <blocksPerGrid, threadsPerBlock >> > (d_correlation_weights_raw, SEP,
        d_grid_map, d_particles_x, d_particles_y,
        d_extended_idx, GRID_WIDTH, GRID_HEIGHT, PARTICLES_ITEMS_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_correlation_max << <blocksPerGrid, threadsPerBlock >> > (d_correlation_weights, d_correlation_weights_raw, NUM_PARTICLES);
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(res_extended_idx, d_extended_idx, sz_extended_idx, cudaMemcpyDeviceToHost));
}

void exec_update_weights() {

    threadsPerBlock = 1;
    blocksPerGrid = 1;

    kernel_arr_max << < blocksPerGrid, threadsPerBlock >> > (d_correlation_weights, d_correlation_weights_max, NUM_PARTICLES);
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(res_correlation_weights_max, d_correlation_weights_max, sz_correlation_weights_max, cudaMemcpyDeviceToHost));

    float norm_value = -res_correlation_weights_max[0] + 50;

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_arr_increase << < blocksPerGrid, threadsPerBlock >> > (d_correlation_weights, norm_value, 0);
    cudaDeviceSynchronize();

    threadsPerBlock = 1;
    blocksPerGrid = 1;
    kernel_arr_sum_exp << < blocksPerGrid, threadsPerBlock >> > (d_correlation_sum_exp, d_correlation_weights, NUM_PARTICLES);
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(res_correlation_sum_exp, d_correlation_sum_exp, sz_correlation_sum_exp, cudaMemcpyDeviceToHost));

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_arr_normalize << < blocksPerGrid, threadsPerBlock >> > (d_correlation_weights, res_correlation_sum_exp[0]);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_arr_mult << < blocksPerGrid, threadsPerBlock >> > (d_particles_weight, d_correlation_weights);
    cudaDeviceSynchronize();
}

void exec_resampling() {

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;

    kernel_resampling << <blocksPerGrid, threadsPerBlock >> > (d_resampling_js, d_correlation_weights, d_resampling_rnd, NUM_PARTICLES);
    cudaDeviceSynchronize();
}

void reinit_particles_vars(const int PARTICLES_ITEMS_LEN) {

    sz_last_len = sizeof(int);
    d_last_len = NULL;
    res_last_len = (int*)malloc(sizeof(int));

    gpuErrchk(cudaMalloc((void**)&d_last_len, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dc_particles_x, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&dc_particles_y, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&dc_particles_idx, sz_particles_idx));

    gpuErrchk(cudaMemcpy(dc_particles_x, d_particles_x, sz_particles_pos, cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(dc_particles_y, d_particles_y, sz_particles_pos, cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(dc_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToDevice));

    gpuErrchk(cudaMemcpy(dc_states_x, d_states_x, sz_states_pos, cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(dc_states_y, d_states_y, sz_states_pos, cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(dc_states_theta, d_states_theta, sz_states_pos, cudaMemcpyDeviceToDevice));

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_rearrange_indecies << <blocksPerGrid, threadsPerBlock >> > (d_particles_idx, d_last_len, SEP,
        dc_particles_idx, d_resampling_js, PARTICLES_ITEMS_LEN);
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(res_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_last_len, d_last_len, sz_last_len, cudaMemcpyDeviceToHost));
}

void exec_rearrangement(int& PARTICLES_ITEMS_LEN, int& C_PARTICLES_ITEMS_LEN, 
    const int GRID_WIDTH, const int GRID_HEIGHT) {

    thrust::exclusive_scan(thrust::device, d_particles_idx, d_particles_idx + NUM_PARTICLES, d_particles_idx, 0);

    gpuErrchk(cudaMemcpy(res_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToHost));
    C_PARTICLES_ITEMS_LEN = PARTICLES_ITEMS_LEN;
    PARTICLES_ITEMS_LEN = res_particles_idx[NUM_PARTICLES - 1] + res_last_len[0];

    sz_particles_pos = PARTICLES_ITEMS_LEN * sizeof(int);

    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;
    kernel_rearrange_particles << <blocksPerGrid, threadsPerBlock >> > (d_particles_x, d_particles_y, SEP,
        d_particles_idx, dc_particles_x, dc_particles_y, dc_particles_idx, d_resampling_js,
        GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES, PARTICLES_ITEMS_LEN, C_PARTICLES_ITEMS_LEN);

    kernel_rearrange_states << <blocksPerGrid, threadsPerBlock >> > (d_states_x, d_states_y, d_states_theta, SEP,
        dc_states_x, dc_states_y, dc_states_theta, d_resampling_js);
    cudaDeviceSynchronize();
}

void exec_update_states() {

    thrust::device_vector<float> d_vec_states_x(d_states_x, d_states_x + NUM_PARTICLES);
    thrust::device_vector<float> d_vec_states_y(d_states_y, d_states_y + NUM_PARTICLES);
    thrust::device_vector<float> d_vec_states_theta(d_states_theta, d_states_theta + NUM_PARTICLES);

    thrust::host_vector<float> h_vec_states_x(d_vec_states_x.begin(), d_vec_states_x.end());
    thrust::host_vector<float> h_vec_states_y(d_vec_states_y.begin(), d_vec_states_y.end());
    thrust::host_vector<float> h_vec_states_theta(d_vec_states_theta.begin(), d_vec_states_theta.end());

    std_vec_states_x.clear();
    std_vec_states_y.clear();
    std_vec_states_theta.clear();
    std_vec_states_x.resize(h_vec_states_x.size());
    std_vec_states_y.resize(h_vec_states_y.size());
    std_vec_states_theta.resize(h_vec_states_theta.size());

    std::copy(h_vec_states_x.begin(), h_vec_states_x.end(), std_vec_states_x.begin());
    std::copy(h_vec_states_y.begin(), h_vec_states_y.end(), std_vec_states_y.begin());
    std::copy(h_vec_states_theta.begin(), h_vec_states_theta.end(), std_vec_states_theta.begin());

    std::map<std::tuple<float, float, float>, int> states;

    for (int i = 0; i < NUM_PARTICLES; i++) {
        if (states.find(std::make_tuple(std_vec_states_x[i], std_vec_states_y[i], std_vec_states_theta[i])) == states.end())
            states.insert({ std::make_tuple(std_vec_states_x[i], std_vec_states_y[i], std_vec_states_theta[i]), 1 });
        else
            states[std::make_tuple(std_vec_states_x[i], std_vec_states_y[i], std_vec_states_theta[i])] += 1;
    }

    std::map<std::tuple<float, float, float>, int>::iterator best
        = std::max_element(states.begin(), states.end(), [](const std::pair<std::tuple<float, float, float>, int>& a,
            const std::pair<std::tuple<float, float, float>, int>& b)->bool { return a.second < b.second; });

    auto key = best->first;

    float theta = std::get<2>(key);

    res_robot_world_body[0] = cos(theta);	res_robot_world_body[1] = -sin(theta);	res_robot_world_body[2] = std::get<0>(key);
    res_robot_world_body[3] = sin(theta);   res_robot_world_body[4] = cos(theta);	res_robot_world_body[5] = std::get<1>(key);
    res_robot_world_body[6] = 0;			res_robot_world_body[7] = 0;			res_robot_world_body[8] = 1;

    res_robot_state[0] = std::get<0>(key); res_robot_state[1] = std::get<1>(key); res_robot_state[2] = std::get<2>(key);
}

void assert_robot_results(float* new_weights, float* particles_weight_post, float* h_robot_transition_world_body,
    float* h_robot_state) {

    //gpuErrchk(cudaMemcpy(res_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToHost));
    //ASSERT_resampling_particles_index(h_particles_idx_after_resampling, res_particles_idx, NUM_PARTICLES, false, negative_after_counter);

    res_correlation_weights = (float*)malloc(sz_correlation_weights);
    res_particles_weight = (float*)malloc(sz_particles_weight);

    gpuErrchk(cudaMemcpy(res_correlation_weights, d_correlation_weights, sz_correlation_weights, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_weight, d_particles_weight, sz_particles_weight, cudaMemcpyDeviceToHost));

    ASSERT_update_particle_weights(res_correlation_weights, new_weights, NUM_PARTICLES, "weights", false, true, true, true);
    ASSERT_update_particle_weights(res_particles_weight, particles_weight_post, NUM_PARTICLES, "particles weight", false, true, false, true);

    printf("\n");
    printf("~~$ Transition World to Body (Result): ");
    for (int i = 0; i < 9; i++) {
        printf("%f ", res_transition_world_body[i]);
    }
    printf("\n");
    printf("~~$ Transition World to Body (Host)  : ");
    for (int i = 0; i < 9; i++) {
        printf("%f ", h_robot_transition_world_body[i]);
    }

    printf("\n\n");
    printf("~~$ Robot State (Result): %f, %f, %f\n", res_robot_state[0], res_robot_state[1], res_robot_state[2]);
    printf("~~$ Robot State (Host)  : %f, %f, %f\n", h_robot_state[0], h_robot_state[1], h_robot_state[2]);
}


///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

void alloc_init_transition_vars(float* h_transition_body_lidar, float* h_transition_world_body) {

    sz_transition_single_frame = 9 * sizeof(float);
    sz_transition_body_lidar = 9 * sizeof(float);

    gpuErrchk(cudaMalloc((void**)&d_transition_single_world_body, sz_transition_single_frame));
    gpuErrchk(cudaMalloc((void**)&d_transition_single_world_lidar, sz_transition_single_frame));
    gpuErrchk(cudaMalloc((void**)&d_transition_body_lidar, sz_transition_body_lidar));

    gpuErrchk(cudaMemcpy(d_transition_body_lidar, h_transition_body_lidar, sz_transition_body_lidar, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_transition_single_world_body, h_transition_world_body, sz_transition_single_frame, cudaMemcpyHostToDevice));
}

void alloc_init_lidar_coords_var(float* h_lidar_coords, const int LIDAR_COORDS_LEN) {

    sz_lidar_coords = 2 * LIDAR_COORDS_LEN * sizeof(float);
    gpuErrchk(cudaMalloc((void**)&d_lidar_coords, sz_lidar_coords));

    gpuErrchk(cudaMemcpy(d_lidar_coords, h_lidar_coords, sz_lidar_coords, cudaMemcpyHostToDevice));
}

void alloc_particles_world_vars(const int LIDAR_COORDS_LEN) {

    sz_particles_world_pos = LIDAR_COORDS_LEN * sizeof(float);

    gpuErrchk(cudaMalloc((void**)&d_particles_world_x, sz_particles_world_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_world_y, sz_particles_world_pos));
}

void alloc_particles_free_vars(int PARTICLES_OCCUPIED_LEN, int PARTICLE_UNIQUE_COUNTER, int MAX_DIST_IN_MAP) {

    sz_particles_free_pos = 0;
    sz_particles_free_pos_max = PARTICLES_OCCUPIED_LEN * MAX_DIST_IN_MAP * sizeof(int);
    sz_particles_free_counter = PARTICLE_UNIQUE_COUNTER * sizeof(int);
    sz_particles_free_idx = PARTICLES_OCCUPIED_LEN * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_particles_free_x_max, sz_particles_free_pos_max));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_y_max, sz_particles_free_pos_max));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_counter, sz_particles_free_counter));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_idx, sz_particles_free_idx));

    gpuErrchk(cudaMemset(d_particles_free_x_max, 0, sz_particles_free_pos_max));
    gpuErrchk(cudaMemset(d_particles_free_y_max, 0, sz_particles_free_pos_max));
    gpuErrchk(cudaMemset(d_particles_free_counter, 0, sz_particles_free_counter));

    res_particles_free_counter = (int*)malloc(sz_particles_free_counter);
}

void alloc_particles_occupied_vars(int LIDAR_COORDS_LEN) {

    sz_particles_occupied_pos = LIDAR_COORDS_LEN * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_particles_occupied_x, sz_particles_occupied_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_occupied_y, sz_particles_occupied_pos));
}

void alloc_bresenham_vars() {

    sz_position_image_body = 2 * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_position_image_body, sz_position_image_body));
}

void alloc_init_map_vars(int* h_grid_map, const int GRID_WIDTH, const int GRID_HEIGHT) {

    sz_grid_map = (GRID_WIDTH * GRID_HEIGHT) * sizeof(int);
    gpuErrchk(cudaMalloc((void**)&d_grid_map, sz_grid_map));

    gpuErrchk(cudaMemcpy(d_grid_map, h_grid_map, sz_grid_map, cudaMemcpyHostToDevice));

    res_grid_map = (int*)malloc(sz_grid_map);
}

void alloc_log_odds_vars(int GRID_WIDTH, int GRID_HEIGHT) {

    sz_log_odds = (GRID_WIDTH * GRID_HEIGHT) * sizeof(float);
    gpuErrchk(cudaMalloc((void**)&d_log_odds, sz_log_odds));

    res_log_odds = (float*)malloc(sz_log_odds);
}

void alloc_init_log_odds_free_vars(int GRID_WIDTH, int GRID_HEIGHT) {

    sz_free_map_idx = 2 * sizeof(int);
    sz_free_map_2d = GRID_WIDTH * GRID_HEIGHT * sizeof(uint8_t);
    sz_free_unique_counter = 1 * sizeof(int);
    sz_free_unique_counter_col = (GRID_WIDTH + 1) * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_free_map_2d, sz_free_map_2d));
    gpuErrchk(cudaMalloc((void**)&d_free_unique_counter, sz_free_unique_counter));
    gpuErrchk(cudaMalloc((void**)&d_free_unique_counter_col, sz_free_unique_counter_col));
    gpuErrchk(cudaMalloc((void**)&d_free_map_idx, sz_free_map_idx));

    res_free_unique_counter = (int*)malloc(sz_free_unique_counter);
}

void alloc_init_log_odds_occupied_vars(int GRID_WIDTH, int GRID_HEIGHT) {

    sz_occupied_map_idx = 2 * sizeof(int);
    sz_occupied_map_2d = GRID_WIDTH * GRID_HEIGHT * sizeof(uint8_t);
    sz_occupied_unique_counter = 1 * sizeof(int);
    sz_occupied_unique_counter_col = (GRID_WIDTH + 1) * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_occupied_map_2d, sz_occupied_map_2d));
    gpuErrchk(cudaMalloc((void**)&d_occupied_unique_counter, sz_occupied_unique_counter));
    gpuErrchk(cudaMalloc((void**)&d_occupied_unique_counter_col, sz_occupied_unique_counter_col));
    gpuErrchk(cudaMalloc((void**)&d_occupied_map_idx, sz_occupied_map_idx));

    res_occupied_unique_counter = (int*)malloc(sz_occupied_unique_counter);
}

void init_log_odds_vars(float* h_log_odds, int PARTICLES_OCCUPIED_LEN) {

    h_occupied_map_idx[1] = PARTICLES_OCCUPIED_LEN;
    h_free_map_idx[1] = 0;

    memset(res_log_odds, 0, sz_log_odds);

    gpuErrchk(cudaMemcpy(d_occupied_map_idx, h_occupied_map_idx, sz_occupied_map_idx, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_free_map_idx, h_free_map_idx, sz_free_map_idx, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_log_odds, h_log_odds, sz_log_odds, cudaMemcpyHostToDevice));
}



void exec_world_to_image_transform_step_1(int xmin, int ymax, float res, int LIDAR_COORDS_LEN) {

    kernel_matrix_mul_3x3 << < 1, 1 >> > (d_transition_single_world_body, d_transition_body_lidar, d_transition_single_world_lidar);
    cudaDeviceSynchronize();

    threadsPerBlock = 1;
    blocksPerGrid = LIDAR_COORDS_LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (d_particles_world_x, d_particles_world_y, SEP,
        d_transition_single_world_lidar, d_lidar_coords, LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    kernel_position_to_image << < 1, 1 >> > (d_position_image_body, SEP, d_transition_single_world_lidar, res, xmin, ymax);
    cudaDeviceSynchronize();
}

void exec_map_extend(int& xmin, int& xmax, int& ymin, int& ymax, float res, int LIDAR_COORDS_LEN, int& GRID_WIDTH, int& GRID_HEIGHT) {

    int xmin_pre = xmin;
    int ymax_pre = ymax;

    sz_should_extend = 4 * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_should_extend, sz_should_extend));
    res_should_extend = (int*)malloc(sz_should_extend);
    memset(res_should_extend, 0, sz_should_extend);
    gpuErrchk(cudaMemset(d_should_extend, 0, sz_should_extend));

    threadsPerBlock = 256;
    blocksPerGrid = (LIDAR_COORDS_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_check_map_extend_less << <blocksPerGrid, threadsPerBlock >> > (d_should_extend, SEP, d_particles_world_x, xmin, 0, LIDAR_COORDS_LEN);
    kernel_check_map_extend_less << <blocksPerGrid, threadsPerBlock >> > (d_should_extend, SEP, d_particles_world_y, ymin, 1, LIDAR_COORDS_LEN);

    kernel_check_map_extend_greater << <blocksPerGrid, threadsPerBlock >> > (d_should_extend, SEP, d_particles_world_x, xmax, 2, LIDAR_COORDS_LEN);
    kernel_check_map_extend_greater << <blocksPerGrid, threadsPerBlock >> > (d_should_extend, SEP, d_particles_world_y, ymax, 3, LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();
    gpuErrchk(cudaMemcpy(res_should_extend, d_should_extend, sz_should_extend, cudaMemcpyDeviceToHost));

    bool EXTEND = false;
    if (res_should_extend[0] != 0) {
        EXTEND = true;
        xmin = xmin * 2;
    }
    else if (res_should_extend[2] != 0) {
        EXTEND = true;
        xmax = xmax * 2;
    }
    else if (res_should_extend[1] != 0) {
        EXTEND = true;
        ymin = ymin * 2;
    }
    else if (res_should_extend[3] != 0) {
        EXTEND = true;
        ymax = ymax * 2;
    }

    //printf("EXTEND = %d\n", EXTEND);

    if (EXTEND == true) {

        sz_coord = 2 * sizeof(int);
        gpuErrchk(cudaMalloc((void**)&d_coord, sz_coord));
        res_coord = (int*)malloc(sz_coord);

        kernel_position_to_image << <1, 1 >> > (d_coord, SEP, xmin_pre, ymax_pre, res, xmin, ymax);
        cudaDeviceSynchronize();

        gpuErrchk(cudaMemcpy(res_coord, d_coord, sz_coord, cudaMemcpyDeviceToHost));

        int* dc_grid_map = NULL;
        gpuErrchk(cudaMalloc((void**)&dc_grid_map, sz_grid_map));
        gpuErrchk(cudaMemcpy(dc_grid_map, d_grid_map, sz_grid_map, cudaMemcpyDeviceToDevice));

        float* dc_log_odds = NULL;
        gpuErrchk(cudaMalloc((void**)&dc_log_odds, sz_log_odds));
        gpuErrchk(cudaMemcpy(dc_log_odds, d_log_odds, sz_log_odds, cudaMemcpyDeviceToDevice));

        const int PRE_GRID_WIDTH = GRID_WIDTH;
        const int PRE_GRID_HEIGHT = GRID_HEIGHT;
        GRID_WIDTH = ceil((ymax - ymin) / res + 1);
        GRID_HEIGHT = ceil((xmax - xmin) / res + 1);
        //printf("GRID_WIDTH=%d, GRID_HEIGHT=%d, PRE_GRID_WIDTH=%d, PRE_GRID_HEIGHT=%d\n", GRID_WIDTH, GRID_HEIGHT, PRE_GRID_WIDTH, PRE_GRID_HEIGHT);
        //assert(GRID_WIDTH == AF_GRID_WIDTH);
        //assert(GRID_HEIGHT == AF_GRID_HEIGHT);

        const int PRE_GRID_SIZE = PRE_GRID_WIDTH * PRE_GRID_HEIGHT;
        const int NEW_GRID_SIZE = GRID_WIDTH * GRID_HEIGHT;

        //gpuErrchk(cudaFree(d_grid_map));

        sz_grid_map = GRID_WIDTH * GRID_HEIGHT * sizeof(int);
        gpuErrchk(cudaMalloc((void**)&d_grid_map, sz_grid_map));
        gpuErrchk(cudaMemset(d_grid_map, 0, sz_grid_map));

        sz_log_odds = GRID_WIDTH * GRID_HEIGHT * sizeof(float);
        gpuErrchk(cudaMalloc((void**)&d_log_odds, sz_log_odds));

        threadsPerBlock = 256;
        blocksPerGrid = (NEW_GRID_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        kernel_index_arr_const << <blocksPerGrid, threadsPerBlock >> > (d_log_odds, LOG_ODD_PRIOR, NEW_GRID_SIZE);
        cudaDeviceSynchronize();

        res_grid_map = (int*)malloc(sz_grid_map);
        res_log_odds = (float*)malloc(sz_log_odds);

        threadsPerBlock = 256;
        blocksPerGrid = (PRE_GRID_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        kernel_2d_copy_with_offset << <blocksPerGrid, threadsPerBlock >> > (d_grid_map, d_log_odds, SEP,
            dc_grid_map, dc_log_odds, res_coord[0], res_coord[1],
            PRE_GRID_HEIGHT, GRID_HEIGHT, PRE_GRID_SIZE);
        cudaDeviceSynchronize();

        sz_free_unique_counter_col = (GRID_WIDTH + 1) * sizeof(int);
        gpuErrchk(cudaMalloc((void**)&d_free_unique_counter_col, sz_free_unique_counter_col));
        res_free_unique_counter_col = (int*)malloc(sz_free_unique_counter_col);

        sz_free_map_2d = GRID_WIDTH * GRID_HEIGHT * sizeof(uint8_t);
        gpuErrchk(cudaMalloc((void**)&d_free_map_2d, sz_free_map_2d));


        sz_occupied_unique_counter_col = (GRID_WIDTH + 1) * sizeof(int);
        gpuErrchk(cudaMalloc((void**)&d_occupied_unique_counter_col, sz_occupied_unique_counter_col));
        res_occupied_unique_counter_col = (int*)malloc(sz_occupied_unique_counter_col);

        sz_occupied_map_2d = GRID_WIDTH * GRID_HEIGHT * sizeof(uint8_t);
        gpuErrchk(cudaMalloc((void**)&d_occupied_map_2d, sz_occupied_map_2d));

        gpuErrchk(cudaMemcpy(res_grid_map, d_grid_map, sz_grid_map, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(res_log_odds, d_log_odds, sz_log_odds, cudaMemcpyDeviceToHost));
    }
}

void exec_world_to_image_transform_step_2(int xmin, int ymax, float res, int LIDAR_COORDS_LEN) {

    threadsPerBlock = 1;
    blocksPerGrid = LIDAR_COORDS_LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (d_particles_occupied_x, d_particles_occupied_y, SEP,
        d_transition_single_world_lidar, d_lidar_coords, res, xmin, ymax, LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    kernel_position_to_image << < 1, 1 >> > (d_position_image_body, SEP, d_transition_single_world_lidar, res, xmin, ymax);
    cudaDeviceSynchronize();
}

void exec_bresenham(int PARTICLES_OCCUPIED_LEN, int& PARTICLES_FREE_LEN, int PARTICLE_UNIQUE_COUNTER, int MAX_DIST_IN_MAP) {

    threadsPerBlock = 256;
    blocksPerGrid = (PARTICLES_OCCUPIED_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_bresenham << <blocksPerGrid, threadsPerBlock >> > (d_particles_free_x_max, d_particles_free_y_max, d_particles_free_counter, SEP,
        d_particles_occupied_x, d_particles_occupied_y, d_position_image_body,
        PARTICLES_OCCUPIED_LEN, MAX_DIST_IN_MAP);
    cudaDeviceSynchronize();
    thrust::exclusive_scan(thrust::device, d_particles_free_counter, d_particles_free_counter + PARTICLE_UNIQUE_COUNTER, d_particles_free_counter, 0);

    gpuErrchk(cudaMemcpy(res_particles_free_counter, d_particles_free_counter, sz_particles_free_counter, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(d_particles_free_idx, d_particles_free_counter, sz_particles_free_idx, cudaMemcpyDeviceToDevice));

    PARTICLES_FREE_LEN = res_particles_free_counter[PARTICLE_UNIQUE_COUNTER - 1];
    sz_particles_free_pos = PARTICLES_FREE_LEN * sizeof(int);
    gpuErrchk(cudaMalloc((void**)&d_particles_free_x, sz_particles_free_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_y, sz_particles_free_pos));

    kernel_bresenham_rearrange << <blocksPerGrid, threadsPerBlock >> > (d_particles_free_x, d_particles_free_y, SEP,
        d_particles_free_x_max, d_particles_free_y_max,
        d_particles_free_counter, MAX_DIST_IN_MAP, PARTICLES_OCCUPIED_LEN);
    cudaDeviceSynchronize();
}

void reinit_map_idx_vars(int PARTICLES_OCCUPIED_LEN, int PARTICLES_FREE_LEN) {

    h_occupied_map_idx[1] = PARTICLES_OCCUPIED_LEN;
    h_free_map_idx[1] = PARTICLES_FREE_LEN;

    gpuErrchk(cudaMemcpy(d_occupied_map_idx, h_occupied_map_idx, sz_occupied_map_idx, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_free_map_idx, h_free_map_idx, sz_free_map_idx, cudaMemcpyHostToDevice));
}

void exec_create_map(int PARTICLES_OCCUPIED_LEN, int PARTICLES_FREE_LEN, int GRID_WIDTH, int GRID_HEIGHT) {

    threadsPerBlock = 256;
    blocksPerGrid = 1;
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (d_occupied_map_2d, SEP,
        d_particles_occupied_x, d_particles_occupied_y, d_occupied_map_idx,
        PARTICLES_OCCUPIED_LEN, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (d_free_map_2d, SEP,
        d_particles_free_x, d_particles_free_y, d_free_map_idx,
        PARTICLES_FREE_LEN, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();

    threadsPerBlock = GRID_WIDTH;
    blocksPerGrid = 1;
    kernel_2d_map_counter << <blocksPerGrid, threadsPerBlock >> > (d_occupied_unique_counter, d_occupied_unique_counter_col, SEP,
        d_occupied_map_2d, GRID_WIDTH, GRID_HEIGHT);
    kernel_2d_map_counter << <blocksPerGrid, threadsPerBlock >> > (d_free_unique_counter, d_free_unique_counter_col, SEP,
        d_free_map_2d, GRID_WIDTH, GRID_HEIGHT);
    cudaDeviceSynchronize();

    threadsPerBlock = 1;
    blocksPerGrid = 1;
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (d_occupied_unique_counter_col, GRID_WIDTH);
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (d_free_unique_counter_col, GRID_WIDTH);
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(res_occupied_unique_counter, d_occupied_unique_counter, sz_occupied_unique_counter, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_free_unique_counter, d_free_unique_counter, sz_free_unique_counter, cudaMemcpyDeviceToHost));
}

void reinit_map_vars(int& PARTICLES_OCCUPIED_UNIQUE_LEN, int& PARTICLES_FREE_UNIQUE_LEN, int GRID_WIDTH, int GRID_HEIGHT) {

    PARTICLES_OCCUPIED_UNIQUE_LEN = res_occupied_unique_counter[0];
    PARTICLES_FREE_UNIQUE_LEN = res_free_unique_counter[0];

    //gpuErrchk(cudaFree(d_particles_occupied_x));
    //gpuErrchk(cudaFree(d_particles_occupied_y));
    //gpuErrchk(cudaFree(d_particles_free_x));
    //gpuErrchk(cudaFree(d_particles_free_y));

    sz_particles_occupied_pos = PARTICLES_OCCUPIED_UNIQUE_LEN * sizeof(int);
    sz_particles_free_pos = PARTICLES_FREE_UNIQUE_LEN * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_particles_occupied_x, sz_particles_occupied_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_occupied_y, sz_particles_occupied_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_x, sz_particles_free_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_y, sz_particles_free_pos));

    threadsPerBlock = GRID_WIDTH;
    blocksPerGrid = 1;
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (d_particles_occupied_x, d_particles_occupied_y, SEP,
        d_occupied_map_2d, d_occupied_unique_counter_col, GRID_WIDTH, GRID_HEIGHT);
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (d_particles_free_x, d_particles_free_y, SEP,
        d_free_map_2d, d_free_unique_counter_col, GRID_WIDTH, GRID_HEIGHT);
    cudaDeviceSynchronize();
}

void exec_log_odds(float log_t, int PARTICLES_OCCUPIED_UNIQUE_LEN, int PARTICLES_FREE_UNIQUE_LEN,
    int GRID_WIDTH, int GRID_HEIGHT) {

    threadsPerBlock = 256;
    blocksPerGrid = (PARTICLES_OCCUPIED_UNIQUE_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_update_log_odds << <blocksPerGrid, threadsPerBlock >> > (d_log_odds, SEP,
        d_particles_occupied_x, d_particles_occupied_y,
        2 * log_t, GRID_WIDTH, GRID_HEIGHT, PARTICLES_OCCUPIED_UNIQUE_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = 256;
    blocksPerGrid = (PARTICLES_FREE_UNIQUE_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_update_log_odds << <blocksPerGrid, threadsPerBlock >> > (d_log_odds, SEP,
        d_particles_free_x, d_particles_free_y, (-1) * log_t, GRID_WIDTH, GRID_HEIGHT, PARTICLES_FREE_UNIQUE_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = 256;
    blocksPerGrid = ((GRID_WIDTH * GRID_HEIGHT) + threadsPerBlock - 1) / threadsPerBlock;
    kernel_update_map << <blocksPerGrid, threadsPerBlock >> > (d_grid_map, SEP,
        d_log_odds, LOG_ODD_PRIOR, WALL, FREE, GRID_WIDTH * GRID_HEIGHT);
    cudaDeviceSynchronize();
}

void assert_map_results(float* h_log_odds, int* h_grid_map,
    float* h_post_log_odds, int* h_post_grid_map, int PARTICLES_FREE_LEN,
    int PARTICLES_OCCUPIED_UNIQUE_LEN, int PARTICLES_FREE_UNIQUE_LEN, int GRID_WIDTH, int GRID_HEIGHT) {

    printf("\n");
    printf("--> Occupied Unique: \t\t%d\n", PARTICLES_OCCUPIED_UNIQUE_LEN);
    printf("--> Free Unique: \t\t%d\n", PARTICLES_FREE_UNIQUE_LEN);
    printf("~~$ PARTICLES_FREE_LEN: \t%d\n", PARTICLES_FREE_LEN);
    printf("~~$ sz_log_odds: \t\t%d\n", sz_log_odds / sizeof(float));

    gpuErrchk(cudaMemcpy(res_log_odds, d_log_odds, sz_log_odds, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_grid_map, d_grid_map, sz_grid_map, cudaMemcpyDeviceToHost));

    ASSERT_log_odds(res_log_odds, h_log_odds, h_post_log_odds, (GRID_WIDTH * GRID_HEIGHT), true);
    ASSERT_log_odds_maps(res_grid_map, h_grid_map, h_post_grid_map, (GRID_WIDTH * GRID_HEIGHT), true);

    printf("\n~~$ Verification All Passed\n");
}


///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

void alloc_init_movement_vars(float* h_rnds_encoder_counts, float* h_rnds_yaws) {

    gpuErrchk(cudaMalloc((void**)&d_rnds_encoder_counts, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&d_rnds_yaws, sz_states_pos));

    gpuErrchk(cudaMemcpy(d_rnds_encoder_counts, h_rnds_encoder_counts, sz_states_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_rnds_yaws, h_rnds_yaws, sz_states_pos, cudaMemcpyHostToDevice));
}

void exec_robot_advance(float encoder_counts, float yaw, float dt, float nv, float nw) {

    int threadsPerBlock = NUM_PARTICLES;
    int blocksPerGrid = 1;
    kernel_robot_advance << <blocksPerGrid, threadsPerBlock >> > (d_states_x, d_states_y, d_states_theta, SEP,
        d_rnds_encoder_counts, d_rnds_yaws,
        encoder_counts, yaw, dt, nv, nw, NUM_PARTICLES);
    cudaDeviceSynchronize();
}

void assert_robot_move_results(float* post_states_x, float* post_states_y, float* post_states_theta) {

    res_states_x = (float*)malloc(sz_states_pos);
    res_states_y = (float*)malloc(sz_states_pos);
    res_states_theta = (float*)malloc(sz_states_pos);

    gpuErrchk(cudaMemcpy(res_states_x, d_states_x, sz_states_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_states_y, d_states_y, sz_states_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_states_theta, d_states_theta, sz_states_pos, cudaMemcpyDeviceToHost));

    for (int i = 0; i < NUM_PARTICLES; i++) {

        if (abs(res_states_x[i] - post_states_x[i]) > 1e-4)
            printf("i=%d, x=%f, %f\n", i, res_states_x[i], post_states_x[i]);
        if (abs(res_states_y[i] - post_states_y[i]) > 1e-4)
            printf("i=%d, y=%f, %f\n", i, res_states_y[i], post_states_y[i]);
        if (abs(res_states_theta[i] - post_states_theta[i]) > 1e-4)
            printf("i=%d, theta=%f, %f\n", i, res_states_theta[i], post_states_theta[i]);
    }
    printf("~~$ Robot Move Check Finished\n");
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

void test_setup(int& xmin, int& xmax, int& ymin, int& ymax, float& res, float& log_t,
    int& LIDAR_COORDS_LEN, int& MEASURE_LEN, int& PARTICLES_ITEMS_LEN, 
    int& PARTICLES_OCCUPIED_LEN, const int PARTICLES_FREE_LEN, int& PARTICLE_UNIQUE_COUNTER, int& MAX_DIST_IN_MAP,
    int& GRID_WIDTH, int& GRID_HEIGHT) {

    const int data_len = 6;
    int data[data_len] = { 1, 0, 2, 2, 1, 3 };
    int* d_data = NULL;
    gpuErrchk(cudaMalloc((void**)&d_data, data_len * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_data, data, data_len * sizeof(int), cudaMemcpyHostToDevice));

    thrust::exclusive_scan(thrust::host, data, data + 6, data, 0);
    thrust::exclusive_scan(thrust::device, d_data, d_data + 6, d_data, 0);

    GRID_WIDTH = NEW_GRID_WIDTH;
    GRID_HEIGHT = NEW_GRID_HEIGHT;
    LIDAR_COORDS_LEN = NEW_LIDAR_COORDS_LEN;

    PARTICLES_ITEMS_LEN = EXTRA_PARTICLES_ITEMS_LEN;
    MEASURE_LEN = NUM_PARTICLES * LIDAR_COORDS_LEN;

    PARTICLES_OCCUPIED_LEN = NEW_LIDAR_COORDS_LEN;
    MAX_DIST_IN_MAP = sqrt(pow(GRID_WIDTH, 2) + pow(GRID_HEIGHT, 2));
    PARTICLE_UNIQUE_COUNTER = PARTICLES_OCCUPIED_LEN + 1;

    xmin = extra_xmin;
    xmax = extra_xmax;;
    ymin = extra_ymin;
    ymax = extra_ymax;

    res = extra_res;
    log_t = extra_log_t;

#ifdef ADD_HEADER_DATA
    assert(GRID_WIDTH == ST_GRID_WIDTH);
    assert(GRID_HEIGHT == ST_GRID_HEIGHT);
    assert(LIDAR_COORDS_LEN == ST_LIDAR_COORDS_LEN);

    assert(PARTICLES_ITEMS_LEN == ST_PARTICLES_ITEMS_LEN);
    assert(MEASURE_LEN == NUM_PARTICLES * LIDAR_COORDS_LEN);

    assert(PARTICLES_OCCUPIED_LEN == ST_LIDAR_COORDS_LEN);
    assert(PARTICLE_UNIQUE_COUNTER == PARTICLES_OCCUPIED_LEN + 1);

    assert(xmin == ST_xmin);
    assert(xmax == ST_xmax);
    assert(ymin == ST_ymin);
    assert(ymax == ST_ymax);

    assert(abs(res - ST_res) < 1e-4);
    assert(abs(log_t == ST_log_t) < 1e-4);
#endif

    h_occupied_map_idx[1] = PARTICLES_OCCUPIED_LEN;
    h_free_map_idx[1] = PARTICLES_FREE_LEN;
}

void test_allocation_initialization(const int LIDAR_COORDS_LEN, const int MEASURE_LEN, 
    const int PARTICLES_ITEMS_LEN, const int PARTICLES_OCCUPIED_LEN, const int PARTICLE_UNIQUE_COUNTER,
    const int MAX_DIST_IN_MAP, const int GRID_WIDTH, const int GRID_HEIGHT) {

#ifdef VERBOSE_BANNER
    printf("/****************** ALLOCATIONS & INITIALIZATIONS  ******************/\n");
#endif

#ifdef VERBOSE_TOTAL_INFO
    printf("~~$ GRID_WIDTH: \t\t%d\n", GRID_WIDTH);
    printf("~~$ GRID_HEIGHT: \t\t%d\n", GRID_HEIGHT);

    printf("~~$ PARTICLES_OCCUPIED_LEN: \t%d\n", PARTICLES_OCCUPIED_LEN);
    printf("~~$ PARTICLE_UNIQUE_COUNTER: \t%d\n", PARTICLE_UNIQUE_COUNTER);
    printf("~~$ MAX_DIST_IN_MAP: \t\t%d\n", MAX_DIST_IN_MAP);
    printf("~~$ LIDAR_COORDS_LEN: \t\t%d\n", LIDAR_COORDS_LEN);
    printf("~~$ PARTICLES_ITEMS_LEN: \t%d\n", PARTICLES_ITEMS_LEN);
#endif

    alloc_init_state_vars(vec_states_x.data(), vec_states_y.data(), vec_states_theta.data());
    alloc_init_movement_vars(vec_rnds_encoder_counts.data(), vec_rnds_yaws.data());


    auto start_robot_particles_alloc = std::chrono::high_resolution_clock::now();
    alloc_init_lidar_coords_var(vec_lidar_coords.data(), NEW_LIDAR_COORDS_LEN);
    alloc_init_grid_map(extra_grid_map.data(), GRID_WIDTH, GRID_HEIGHT);
    alloc_init_particles_vars(extra_particles_x.data(), extra_particles_y.data(), extra_particles_idx.data(),
        extra_particles_weight_pre.data(), PARTICLES_ITEMS_LEN);
    alloc_extended_idx(PARTICLES_ITEMS_LEN);
    alloc_states_copy_vars();
    alloc_correlation_vars();
    alloc_init_transition_vars(h_transition_body_lidar);
    alloc_init_processed_measurement_vars(NEW_LIDAR_COORDS_LEN);
    alloc_map_2d_var(GRID_WIDTH, GRID_HEIGHT);
    alloc_map_2d_unique_counter_vars(UNIQUE_COUNTER_LEN, GRID_WIDTH);
    alloc_correlation_weights_vars();
    alloc_resampling_vars(vec_rnds.data());
    auto stop_robot_particles_alloc = std::chrono::high_resolution_clock::now();

    //auto start_mapping_alloc = std::chrono::high_resolution_clock::now();
    //alloc_init_transition_vars(h_transition_body_lidar, extra_transition_single_world_body.data());
    //alloc_particles_world_vars(LIDAR_COORDS_LEN);
    //alloc_particles_free_vars(PARTICLES_OCCUPIED_LEN, PARTICLE_UNIQUE_COUNTER, MAX_DIST_IN_MAP);
    //alloc_particles_occupied_vars(LIDAR_COORDS_LEN);
    //alloc_bresenham_vars();
    //alloc_init_map_vars(extra_grid_map.data(), GRID_WIDTH, GRID_HEIGHT);
    ////alloc_init_map_vars(); //????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    //alloc_log_odds_vars(GRID_WIDTH, GRID_HEIGHT);
    //alloc_init_log_odds_free_vars(GRID_WIDTH, GRID_HEIGHT);
    //alloc_init_log_odds_occupied_vars(GRID_WIDTH, GRID_HEIGHT);
    //init_log_odds_vars(extra_log_odds.data(), PARTICLES_OCCUPIED_LEN);
    //auto stop_mapping_alloc = std::chrono::high_resolution_clock::now();

    alloc_init_transition_vars(h_transition_body_lidar, extra_transition_single_world_body.data());
    alloc_init_lidar_coords_var(vec_lidar_coords.data(), LIDAR_COORDS_LEN);
    alloc_particles_world_vars(LIDAR_COORDS_LEN);
    alloc_particles_free_vars(PARTICLES_OCCUPIED_LEN, PARTICLE_UNIQUE_COUNTER, MAX_DIST_IN_MAP);
    alloc_particles_occupied_vars(LIDAR_COORDS_LEN);
    alloc_bresenham_vars();
    alloc_init_map_vars(extra_grid_map.data(), GRID_WIDTH, GRID_HEIGHT);
    alloc_log_odds_vars(GRID_WIDTH, GRID_HEIGHT);
    alloc_init_log_odds_free_vars(GRID_WIDTH, GRID_HEIGHT);
    alloc_init_log_odds_occupied_vars(GRID_WIDTH, GRID_HEIGHT);
    init_log_odds_vars(extra_log_odds.data(), PARTICLES_OCCUPIED_LEN);


#ifdef VERBOSE_EXECUTION_TIME
    auto duration_robot_particles_alloc = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_particles_alloc - start_robot_particles_alloc);
    auto duration_mapping_alloc = std::chrono::duration_cast<std::chrono::microseconds>(stop_mapping_alloc - start_mapping_alloc);

    std::cout << "Time taken by function (Particles Allocation): " << duration_robot_particles_alloc.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Mapping Allocation): " << duration_mapping_alloc.count() << " microseconds" << std::endl;
    std::cout << std::endl;
#endif
}

void test_robot_move(bool check_result, float encoder_counts, float yaw, float dt) {

#ifdef VERBOSE_BANNER
    printf("/**************************** ROBOT MOVE ****************************/\n");
    std::cout << "Start Robot Move" << std::endl;
#endif

    auto start_robot_advance_kernel = std::chrono::high_resolution_clock::now();
    exec_robot_advance(encoder_counts, yaw, dt, ST_nv, ST_nw);
    auto stop_robot_advance_kernel = std::chrono::high_resolution_clock::now();

    if(check_result == true)
        assert_robot_move_results(extra_states_x.data(), extra_states_y.data(), extra_states_theta.data());

#ifdef VERBOSE_EXECUTION_TIME
    auto duration_robot_advance_total = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_advance_kernel - start_robot_advance_kernel);
    std::cout << std::endl;
    std::cout << "Time taken by function (Robot Advance Kernel): " << duration_robot_advance_total.count() << " microseconds" << std::endl;
    std::cout << std::endl;
#endif
}

void test_robot(bool check_result, const int xmin, const int ymax, const float res, const float log_t,
    const int LIDAR_COORDS_LEN, const int MEASURE_LEN, 
    int& PARTICLES_ITEMS_LEN, int& C_PARTICLES_ITEMS_LEN,
    const int GRID_WIDTH, const int GRID_HEIGHT) {

#ifdef VERBOSE_BANNER
    printf("/****************************** ROBOT *******************************/\n");
#endif

    res_particles_x = (int*)malloc(sz_particles_pos);
    res_particles_y = (int*)malloc(sz_particles_pos);

    gpuErrchk(cudaMemcpy(res_particles_x, d_particles_x, sz_particles_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_y, d_particles_y, sz_particles_pos, cudaMemcpyDeviceToHost));

#ifdef VERBOSE_BORDER_LINE_COUNTER
    int negative_before_counter = getNegativeCounter(res_particles_x, res_particles_y, PARTICLES_ITEMS_LEN);
    int count_bigger_than_height = getGreaterThanCounter(res_particles_y, GRID_HEIGHT, PARTICLES_ITEMS_LEN);

    printf("~~$ negative_before_counter: \t%d\n", negative_before_counter);
    printf("~~$ count_bigger_than_height: \t%d\n", count_bigger_than_height);
#endif

    auto start_robot_particles_kernel = std::chrono::high_resolution_clock::now();
    exec_calc_transition();
    exec_process_measurements(res, xmin, ymax, LIDAR_COORDS_LEN);
    exec_create_2d_map(PARTICLES_ITEMS_LEN, GRID_WIDTH, GRID_HEIGHT);
    exec_update_map(MEASURE_LEN, GRID_WIDTH, GRID_HEIGHT);
    exec_particle_unique_cum_sum(PARTICLES_ITEMS_LEN, C_PARTICLES_ITEMS_LEN, GRID_WIDTH);
    reinit_map_vars(PARTICLES_ITEMS_LEN);
    exec_map_restructure(GRID_WIDTH, GRID_HEIGHT);
    exec_index_expansion(PARTICLES_ITEMS_LEN);
    exec_correlation(PARTICLES_ITEMS_LEN, GRID_WIDTH, GRID_HEIGHT);
    exec_update_weights();
    exec_resampling();
    reinit_particles_vars(PARTICLES_ITEMS_LEN);
    exec_rearrangement(PARTICLES_ITEMS_LEN, C_PARTICLES_ITEMS_LEN, GRID_WIDTH, GRID_HEIGHT);
    exec_update_states();
    auto stop_robot_particles_kernel = std::chrono::high_resolution_clock::now();

    if(check_result == true)
        assert_robot_results(extra_new_weights.data(), vec_particles_weight_post.data(), 
            vec_robot_transition_world_body.data(), vec_robot_state.data());

#ifdef VERBOSE_EXECUTION_TIME
    auto duration_robot_particles_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_particles_kernel - start_robot_particles_kernel);

    std::cout << std::endl;
    std::cout << "Time taken by function (Kernel): " << duration_robot_particles_kernel.count() << " microseconds" << std::endl;
    std::cout << std::endl;
#endif
}

void test_map(bool check_result, int& xmin, int& xmax, int& ymin, int& ymax, const float res, const float log_t,
    const int PARTICLES_OCCUPIED_LEN, int& PARTICLES_OCCUPIED_UNIQUE_LEN, 
    int& PARTICLES_FREE_LEN, int& PARTICLES_FREE_UNIQUE_LEN, 
    const int PARTICLE_UNIQUE_COUNTER, const int MAX_DIST_IN_MAP,
    const int LIDAR_COORDS_LEN, int& GRID_WIDTH, int& GRID_HEIGHT) {

#ifdef VERBOSE_BANNER
    printf("/****************************** MAP MAIN ****************************/\n");
#endif

    exec_world_to_image_transform_step_1(xmin, ymax, res, LIDAR_COORDS_LEN);
    exec_map_extend(xmin, xmax, ymin, ymax, res, LIDAR_COORDS_LEN, GRID_WIDTH, GRID_HEIGHT);
    exec_world_to_image_transform_step_2(xmin, ymax, res, LIDAR_COORDS_LEN);
    exec_bresenham(PARTICLES_OCCUPIED_LEN, PARTICLES_FREE_LEN, PARTICLE_UNIQUE_COUNTER, MAX_DIST_IN_MAP);
    reinit_map_idx_vars(PARTICLES_OCCUPIED_LEN, PARTICLES_FREE_LEN);

    exec_create_map(PARTICLES_OCCUPIED_LEN, PARTICLES_FREE_LEN, GRID_WIDTH, GRID_HEIGHT);
    reinit_map_vars(PARTICLES_OCCUPIED_UNIQUE_LEN, PARTICLES_FREE_UNIQUE_LEN, GRID_WIDTH, GRID_HEIGHT);

    exec_log_odds(log_t, PARTICLES_OCCUPIED_UNIQUE_LEN, PARTICLES_FREE_UNIQUE_LEN, GRID_WIDTH, GRID_HEIGHT);

    if(check_result == true)
        assert_map_results(extra_log_odds.data(), extra_grid_map.data(),
            vec_log_odds.data(), vec_grid_map.data(), PARTICLES_FREE_LEN,
            PARTICLES_OCCUPIED_UNIQUE_LEN, PARTICLES_FREE_UNIQUE_LEN, GRID_WIDTH, GRID_HEIGHT);

#ifdef VERBOSE_EXECUTION_TIME
    auto duration_mapping_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_mapping_kernel - start_mapping_kernel);

    std::cout << std::endl;
    std::cout << "Time taken by function (Mapping Kernel): " << duration_mapping_kernel.count() << " microseconds" << std::endl;
    std::cout << std::endl;
#endif
}

//void test_iterations() {
//
//    const int ST_FILE_NUMBER = 400;
//
//	read_map_data(ST_FILE_NUMBER);
//	read_robot_data(ST_FILE_NUMBER);
//	read_robot_move_data(ST_FILE_NUMBER);
//    read_robot_extra(ST_FILE_NUMBER);
//
//    std::cout << std::endl << std::endl;
//	std::cout << "Len: " << vec_grid_map.size() << std::endl;
//	std::cout << "Grid Len: " << (NEW_GRID_WIDTH * NEW_GRID_HEIGHT) << std::endl;
//
//	std::cout << "GRID_WIDTH: " << NEW_GRID_WIDTH << std::endl;
//	std::cout << "GRID_HEIGHT: " << NEW_GRID_HEIGHT << std::endl;
//	std::cout << "LIDAR_COORDS_LEN: " << NEW_LIDAR_COORDS_LEN << std::endl;
//    std::cout << std::endl;
//
//    test_setup();
//    test_allocation_initialization();
//
//    test_robot_move(encoder_counts, yaw, dt);
//    test_robot();
//    test_map();
//
//    for (int file_number = ST_FILE_NUMBER + 1; file_number < ST_FILE_NUMBER + 5; file_number++) {
//
//        printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
//        printf("Iteration: %d\n", file_number);
//        printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
//
//        gpuErrchk(cudaMemset(d_correlation_weights_raw, 0, sz_correlation_weights_raw));
//        gpuErrchk(cudaMemset(d_processed_measure_x, 0, sz_processed_measure_pos));
//        gpuErrchk(cudaMemset(d_processed_measure_y, 0, sz_processed_measure_pos));
//        gpuErrchk(cudaMemset(d_processed_measure_idx, 0, sz_processed_measure_idx));
//
//        gpuErrchk(cudaMemset(d_map_2d, 0, sz_map_2d));
//        gpuErrchk(cudaMemset(d_unique_in_particle, 0, sz_unique_in_particle));
//        gpuErrchk(cudaMemset(d_unique_in_particle_col, 0, sz_unique_in_particle_col));
//
//        gpuErrchk(cudaMemset(d_correlation_sum_exp, 0, sz_correlation_sum_exp));
//        gpuErrchk(cudaMemset(d_correlation_weights_max, 0, sz_correlation_weights_max));
//
//        gpuErrchk(cudaMemset(d_resampling_js, 0, sz_resampling_js));
//
//        threadsPerBlock = 256;
//        blocksPerGrid = (NUM_PARTICLES + threadsPerBlock - 1) / threadsPerBlock;
//        kernel_index_arr_const << <blocksPerGrid, threadsPerBlock >> > (d_particles_weight, 1, NUM_PARTICLES);
//        cudaDeviceSynchronize();
//
//        read_robot_move_data(file_number);
//        read_robot_data(file_number);
//        read_map_data(file_number);
//        read_robot_extra(file_number);
//        read_map_extra(file_number);
//
//        LIDAR_COORDS_LEN = NEW_LIDAR_COORDS_LEN;
//
//        alloc_particles_free_vars();
//        alloc_particles_occupied_vars();
//        alloc_init_log_odds_free_vars();
//        alloc_init_log_odds_occupied_vars();
//        gpuErrchk(cudaMemcpy(d_transition_single_world_body,
//            extra_transition_single_world_body.data(), sz_transition_single_frame, cudaMemcpyHostToDevice));
//
//        alloc_init_movement_vars(vec_rnds_encoder_counts.data(), vec_rnds_yaws.data());
//        alloc_init_lidar_coords_var(vec_lidar_coords.data(), NEW_LIDAR_COORDS_LEN);
//        alloc_init_processed_measurement_vars(NEW_LIDAR_COORDS_LEN);
//        alloc_resampling_vars(vec_rnds.data());
//
//        test_robot_move(encoder_counts, yaw, dt);
//        test_robot();
//        test_map();
//    }
//
//}

void check_files_data(bool check, const int ST_FILE_NUMBER,
    int& xmin, int& xmax, int& ymin, int& ymax, float& res, float& log_t,
    int& LIDAR_COORDS_LEN, int& MEASURE_LEN, int& PARTICLES_ITEMS_LEN,
    int& PARTICLES_OCCUPIED_LEN, const int PARTICLES_FREE_LEN, int& PARTICLE_UNIQUE_COUNTER, int& MAX_DIST_IN_MAP,
    int& GRID_WIDTH, int& GRID_HEIGHT) {

#ifdef ADD_HEADER_DATA
    if (check == true) {

        read_map_data(ST_FILE_NUMBER);
        read_robot_data(ST_FILE_NUMBER);
        read_robot_move_data(ST_FILE_NUMBER);
        read_robot_extra(ST_FILE_NUMBER);
        read_map_extra(ST_FILE_NUMBER);

        test_setup(xmin, xmax, ymin, ymax, res, log_t,
            LIDAR_COORDS_LEN, MEASURE_LEN, PARTICLES_ITEMS_LEN,
            PARTICLES_OCCUPIED_LEN, PARTICLES_FREE_LEN, PARTICLE_UNIQUE_COUNTER, MAX_DIST_IN_MAP,
            GRID_WIDTH, GRID_HEIGHT);
        test_allocation_initialization(LIDAR_COORDS_LEN, MEASURE_LEN,
            PARTICLES_ITEMS_LEN, PARTICLES_OCCUPIED_LEN, PARTICLE_UNIQUE_COUNTER,
            MAX_DIST_IN_MAP, GRID_WIDTH, GRID_HEIGHT);

        std::cout << std::endl << std::endl;
        std::cout << "Len: " << vec_grid_map.size() << std::endl;
        //std::cout << "Grid Len: " << (GRID_WIDTH * GRID_HEIGHT) << std::endl;

        //std::cout << "GRID_WIDTH: " << GRID_WIDTH << std::endl;
        //std::cout << "GRID_HEIGHT: " << GRID_HEIGHT << std::endl;
        //std::cout << "LIDAR_COORDS_LEN: " << LIDAR_COORDS_LEN << std::endl;
        std::cout << std::endl;
    }
#endif
}

// [ ] - 
// [ ] - 

void test_map_extend() {

    const int data_len = 6;
    int data[data_len] = { 1, 0, 2, 2, 1, 3 };
    int* d_data = NULL;
    gpuErrchk(cudaMalloc((void**)&d_data, data_len * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_data, data, data_len * sizeof(int), cudaMemcpyHostToDevice));

    thrust::exclusive_scan(thrust::host, data, data + 6, data, 0);
    thrust::exclusive_scan(thrust::device, d_data, d_data + 6, d_data, 0);

    //check_file_data();

    printf("\n");
    printf("/****************************** MAP MAIN ****************************/\n");

    const int ST_FILE_NUMBER = 400;

    for (int file_number = ST_FILE_NUMBER; file_number < ST_FILE_NUMBER + 10; file_number++) {

        read_map_data(file_number, false, false, false);
        read_map_extra(file_number, false, false, false);

        float res = extra_res;
        float log_t = extra_log_t;
        int GRID_WIDTH = EXTRA_GRID_WIDTH;
        int GRID_HEIGHT = EXTRA_GRID_HEIGHT;
        int xmin = extra_xmin;
        int xmax = extra_xmax;;
        int ymin = extra_ymin;
        int ymax = extra_ymax;

        int PARTICLES_OCCUPIED_LEN = NEW_LIDAR_COORDS_LEN;
        int PARTICLES_OCCUPIED_UNIQUE_LEN = 0;
        int PARTICLES_FREE_LEN = 0;
        int PARTICLES_FREE_UNIQUE_LEN = 0;

        int LIDAR_COORDS_LEN = NEW_LIDAR_COORDS_LEN;

        int PARTICLE_UNIQUE_COUNTER = PARTICLES_OCCUPIED_LEN + 1;
        int MAX_DIST_IN_MAP = sqrt(pow(GRID_WIDTH, 2) + pow(GRID_HEIGHT, 2));


        h_occupied_map_idx[1] = PARTICLES_OCCUPIED_LEN;
        h_free_map_idx[1] = PARTICLES_FREE_LEN;

        //printf("~~$ GRID_WIDTH: \t\t%d \tEXTRA_GRID_WIDTH:   \t\t%d\n", GRID_WIDTH, EXTRA_GRID_WIDTH);
        //printf("~~$ GRID_HEIGHT: \t\t%d \tEXTRA_GRID_HEIGHT: \t\t%d\n", GRID_HEIGHT, EXTRA_GRID_HEIGHT);
        //printf("~~$ LIDAR_COORDS_LEN: \t\t%d\n", LIDAR_COORDS_LEN);

        //printf("~~$ PARTICLES_OCCUPIED_LEN = \t%d\n", PARTICLES_OCCUPIED_LEN);
        //printf("~~$ PARTICLE_UNIQUE_COUNTER = \t%d\n", PARTICLE_UNIQUE_COUNTER);
        //printf("~~$ MAX_DIST_IN_MAP = \t\t%d\n", MAX_DIST_IN_MAP);


        auto start_mapping_alloc = std::chrono::high_resolution_clock::now();
        alloc_init_transition_vars(h_transition_body_lidar, extra_transition_single_world_body.data());
        alloc_init_lidar_coords_var(vec_lidar_coords.data(), LIDAR_COORDS_LEN);
        alloc_particles_world_vars(LIDAR_COORDS_LEN);
        alloc_particles_free_vars(PARTICLES_OCCUPIED_LEN, PARTICLE_UNIQUE_COUNTER, MAX_DIST_IN_MAP);
        alloc_particles_occupied_vars(LIDAR_COORDS_LEN);
        alloc_bresenham_vars();
        alloc_init_map_vars(extra_grid_map.data(), GRID_WIDTH, GRID_HEIGHT);
        alloc_log_odds_vars(GRID_WIDTH, GRID_HEIGHT);
        alloc_init_log_odds_free_vars(GRID_WIDTH, GRID_HEIGHT);
        alloc_init_log_odds_occupied_vars(GRID_WIDTH, GRID_HEIGHT);
        init_log_odds_vars(extra_log_odds.data(), PARTICLES_OCCUPIED_LEN);
        auto stop_mapping_alloc = std::chrono::high_resolution_clock::now();


        auto start_mapping_kernel = std::chrono::high_resolution_clock::now();
        exec_world_to_image_transform_step_1(xmin, ymax, res, LIDAR_COORDS_LEN);
        exec_map_extend(xmin, xmax, ymin, ymax, res, LIDAR_COORDS_LEN, GRID_WIDTH, GRID_HEIGHT);
        exec_world_to_image_transform_step_2(xmin, ymax, res, LIDAR_COORDS_LEN);
        exec_bresenham(PARTICLES_OCCUPIED_LEN, PARTICLES_FREE_LEN, PARTICLE_UNIQUE_COUNTER, MAX_DIST_IN_MAP);
        reinit_map_idx_vars(PARTICLES_OCCUPIED_LEN, PARTICLES_FREE_LEN);

        exec_create_map(PARTICLES_OCCUPIED_LEN, PARTICLES_FREE_LEN, GRID_WIDTH, GRID_HEIGHT);
        reinit_map_vars(PARTICLES_OCCUPIED_UNIQUE_LEN, PARTICLES_FREE_UNIQUE_LEN, GRID_WIDTH, GRID_HEIGHT);

        exec_log_odds(log_t, PARTICLES_OCCUPIED_UNIQUE_LEN, PARTICLES_FREE_UNIQUE_LEN, GRID_WIDTH, GRID_HEIGHT);
        auto stop_mapping_kernel = std::chrono::high_resolution_clock::now();

        assert_map_results(extra_log_odds.data(), extra_grid_map.data(),
            vec_log_odds.data(), vec_grid_map.data(), PARTICLES_FREE_LEN,
            PARTICLES_OCCUPIED_UNIQUE_LEN, PARTICLES_FREE_UNIQUE_LEN, GRID_WIDTH, GRID_HEIGHT);

        auto duration_mapping_alloc = std::chrono::duration_cast<std::chrono::microseconds>(stop_mapping_alloc - start_mapping_alloc);
        auto duration_mapping_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_mapping_kernel - start_mapping_kernel);
        auto duration_mapping_total = std::chrono::duration_cast<std::chrono::microseconds>(stop_mapping_kernel - start_mapping_alloc);

        //std::cout << std::endl;
        //std::cout << "Time taken by function (Mapping Allocation): " << duration_mapping_alloc.count() << " microseconds" << std::endl;
        //std::cout << "Time taken by function (Mapping Kernel): " << duration_mapping_kernel.count() << " microseconds" << std::endl;
        //std::cout << "Time taken by function (Mapping Total): " << duration_mapping_total.count() << " microseconds" << std::endl;
        std::cout << std::endl;
    }
}


void test_iterations() {

    //test_map_extend();

    //return;

    int GRID_WIDTH = 0; 
    int GRID_HEIGHT = 0;
    int LIDAR_COORDS_LEN = 0;
    int MEASURE_LEN = 0;

    int PARTICLES_ITEMS_LEN = 0;
    int C_PARTICLES_ITEMS_LEN = 0;

    int xmin = 0;
    int xmax = 0;
    int ymin = 0;
    int ymax = 0;

    int PARTICLES_OCCUPIED_LEN = 0;
    int PARTICLES_OCCUPIED_UNIQUE_LEN = 0;
    int PARTICLES_FREE_LEN = 0;
    int PARTICLES_FREE_UNIQUE_LEN = 0;
    int PARTICLE_UNIQUE_COUNTER = 0;
    int MAX_DIST_IN_MAP = 0;

    float res = 0;
    float log_t = 0;

    const int LOOP_LEN = 31;
    const int ST_FILE_NUMBER = 400;
    //check_files_data(true, ST_FILE_NUMBER,
    //    xmin, xmax, ymin, ymax, res, log_t, LIDAR_COORDS_LEN, MEASURE_LEN,PARTICLES_ITEMS_LEN,
    //    PARTICLES_OCCUPIED_LEN, PARTICLES_FREE_LEN, PARTICLE_UNIQUE_COUNTER, MAX_DIST_IN_MAP, GRID_WIDTH, GRID_HEIGHT);

    bool run_normal = false;
    int num_items = 0;

    for (int file_number = ST_FILE_NUMBER; file_number < ST_FILE_NUMBER + LOOP_LEN; ) {

        printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
        printf("Iteration: %d\n", file_number);
        printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");

        if (file_number <= ST_FILE_NUMBER) {

            auto start_read_file = std::chrono::high_resolution_clock::now();
            read_robot_data(file_number);
            read_robot_move_data(file_number);
            read_robot_extra(file_number);
            read_map_data(file_number);
            read_map_extra(file_number);
            auto stop_read_file = std::chrono::high_resolution_clock::now();

            test_setup(xmin, xmax, ymin, ymax, res, log_t, LIDAR_COORDS_LEN, MEASURE_LEN, PARTICLES_ITEMS_LEN,
                PARTICLES_OCCUPIED_LEN, PARTICLES_FREE_LEN, PARTICLE_UNIQUE_COUNTER, MAX_DIST_IN_MAP, GRID_WIDTH, GRID_HEIGHT);

            test_allocation_initialization(LIDAR_COORDS_LEN, MEASURE_LEN,
                PARTICLES_ITEMS_LEN, PARTICLES_OCCUPIED_LEN, PARTICLE_UNIQUE_COUNTER, MAX_DIST_IN_MAP, GRID_WIDTH, GRID_HEIGHT);

            auto duration_read_file = std::chrono::duration_cast<std::chrono::microseconds>(stop_read_file - start_read_file);
            std::cout << std::endl;
            std::cout << "Time taken by function (Read Data Files): " << duration_read_file.count() << " microseconds" << std::endl;
            std::cout << std::endl;
        }

        bool check_assert = (file_number % 5 == 0);
        test_robot_move(check_assert, encoder_counts, yaw, dt);
        test_robot(check_assert, xmin, ymax, res, log_t, LIDAR_COORDS_LEN, MEASURE_LEN,
            PARTICLES_ITEMS_LEN, C_PARTICLES_ITEMS_LEN, GRID_WIDTH, GRID_HEIGHT);
        test_map(check_assert, xmin, xmax, ymin, ymax, res, log_t, PARTICLES_OCCUPIED_LEN, PARTICLES_OCCUPIED_UNIQUE_LEN,
             PARTICLES_FREE_LEN, PARTICLES_FREE_UNIQUE_LEN, PARTICLE_UNIQUE_COUNTER, MAX_DIST_IN_MAP,
              LIDAR_COORDS_LEN, GRID_WIDTH, GRID_HEIGHT);

        file_number += 1;
       
        if (true) {

            printf("Reach Reset\n");

            //gpuErrchk(cudaMemset(d_correlation_weights_raw, 0, sz_correlation_weights_raw));

            //gpuErrchk(cudaMemset(d_processed_measure_x, 0, sz_processed_measure_pos));
            //gpuErrchk(cudaMemset(d_processed_measure_y, 0, sz_processed_measure_pos));
            //gpuErrchk(cudaMemset(d_processed_measure_idx, 0, sz_processed_measure_idx));

            //gpuErrchk(cudaMemset(d_map_2d, 0, sz_map_2d));
            //gpuErrchk(cudaMemset(d_unique_in_particle, 0, sz_unique_in_particle));
            //gpuErrchk(cudaMemset(d_unique_in_particle_col, 0, sz_unique_in_particle_col));

            //gpuErrchk(cudaMemset(d_correlation_sum_exp, 0, sz_correlation_sum_exp));
            //gpuErrchk(cudaMemset(d_correlation_weights_max, 0, sz_correlation_weights_max));
            //gpuErrchk(cudaMemset(d_resampling_js, 0, sz_resampling_js));


            num_items = 25 * NUM_PARTICLES;
            threadsPerBlock = 256;
            blocksPerGrid = (num_items + threadsPerBlock - 1) / threadsPerBlock;
            kernel_index_arr_const << <blocksPerGrid, threadsPerBlock >> > (d_correlation_weights_raw, 0, num_items);

            num_items = NUM_PARTICLES * LIDAR_COORDS_LEN;
            threadsPerBlock = 256;
            blocksPerGrid = (num_items + threadsPerBlock - 1) / threadsPerBlock;
            kernel_index_arr_const << <blocksPerGrid, threadsPerBlock >> > (d_processed_measure_x, 0, num_items);
            kernel_index_arr_const << <blocksPerGrid, threadsPerBlock >> > (d_processed_measure_y, 0, num_items);
            kernel_index_arr_const << <blocksPerGrid, threadsPerBlock >> > (d_processed_measure_idx, 0, num_items);

            num_items = GRID_WIDTH * GRID_HEIGHT * NUM_PARTICLES;
            threadsPerBlock = 256;
            blocksPerGrid = (num_items + threadsPerBlock - 1) / threadsPerBlock;
            kernel_index_arr_const << <blocksPerGrid, threadsPerBlock >> > (d_map_2d, 0, num_items);

            num_items = UNIQUE_COUNTER_LEN;
            threadsPerBlock = 256;
            blocksPerGrid = (num_items + threadsPerBlock - 1) / threadsPerBlock;
            kernel_index_arr_const << <blocksPerGrid, threadsPerBlock >> > (d_unique_in_particle, 0, num_items);

            num_items = UNIQUE_COUNTER_LEN * GRID_WIDTH;
            threadsPerBlock = 256;
            blocksPerGrid = (num_items + threadsPerBlock - 1) / threadsPerBlock;
            kernel_index_arr_const << <blocksPerGrid, threadsPerBlock >> > (d_unique_in_particle_col, 0, num_items);

            num_items = 1;
            threadsPerBlock = 1;
            blocksPerGrid = 1;
            kernel_index_arr_const << <blocksPerGrid, threadsPerBlock >> > (d_correlation_sum_exp, 0, num_items);
            
            num_items = 1;
            threadsPerBlock = 1;
            blocksPerGrid = 1;
            kernel_index_arr_const << <blocksPerGrid, threadsPerBlock >> > (d_correlation_weights_max, 0, num_items);

            num_items = NUM_PARTICLES;
            threadsPerBlock = 256;
            blocksPerGrid = (num_items + threadsPerBlock - 1) / threadsPerBlock;
            kernel_index_arr_const << <blocksPerGrid, threadsPerBlock >> > (d_resampling_js, 0, num_items);
            cudaDeviceSynchronize();


            num_items = NUM_PARTICLES;
            threadsPerBlock = 256;
            blocksPerGrid = (num_items + threadsPerBlock - 1) / threadsPerBlock;
            kernel_index_arr_const << <blocksPerGrid, threadsPerBlock >> > (d_particles_weight, 1, num_items);
            cudaDeviceSynchronize();

            read_robot_move_data(file_number);
            read_robot_data(file_number);
            read_map_data(file_number);
            read_robot_extra(file_number);
            read_map_extra(file_number);

            LIDAR_COORDS_LEN = NEW_LIDAR_COORDS_LEN;
            MEASURE_LEN = NUM_PARTICLES * LIDAR_COORDS_LEN;
            PARTICLES_OCCUPIED_LEN = LIDAR_COORDS_LEN;
            PARTICLE_UNIQUE_COUNTER = PARTICLES_OCCUPIED_LEN + 1;

            alloc_init_transition_vars(h_transition_body_lidar, extra_transition_single_world_body.data());

            alloc_particles_free_vars(PARTICLES_OCCUPIED_LEN, PARTICLE_UNIQUE_COUNTER, MAX_DIST_IN_MAP);
            alloc_particles_occupied_vars(LIDAR_COORDS_LEN);
            alloc_init_log_odds_free_vars(GRID_WIDTH, GRID_HEIGHT);
            alloc_init_log_odds_occupied_vars(GRID_WIDTH, GRID_HEIGHT);
            
            //gpuErrchk(cudaMemcpy(d_transition_single_world_body,
            //    extra_transition_single_world_body.data(), sz_transition_single_frame, cudaMemcpyHostToDevice));


            ////alloc_init_transition_vars(h_transition_body_lidar, extra_transition_single_world_body.data());
            ////alloc_init_lidar_coords_var(vec_lidar_coords.data(), LIDAR_COORDS_LEN);
            ////alloc_particles_world_vars(LIDAR_COORDS_LEN);
            ////alloc_particles_free_vars(PARTICLES_OCCUPIED_LEN, PARTICLE_UNIQUE_COUNTER, MAX_DIST_IN_MAP);
            ////alloc_particles_occupied_vars(LIDAR_COORDS_LEN);
            ////alloc_bresenham_vars();
            ////alloc_init_grid_map(extra_grid_map.data(), GRID_WIDTH, GRID_HEIGHT);
            ////alloc_log_odds_vars(GRID_WIDTH, GRID_HEIGHT);
            ////alloc_init_log_odds_free_vars(GRID_WIDTH, GRID_HEIGHT);
            ////alloc_init_log_odds_occupied_vars(GRID_WIDTH, GRID_HEIGHT);
            //init_log_odds_vars(extra_log_odds.data(), PARTICLES_OCCUPIED_LEN);



            alloc_init_movement_vars(vec_rnds_encoder_counts.data(), vec_rnds_yaws.data());
            alloc_init_lidar_coords_var(vec_lidar_coords.data(), LIDAR_COORDS_LEN);
            alloc_init_processed_measurement_vars(LIDAR_COORDS_LEN);
            alloc_resampling_vars(vec_rnds.data());

            //test_robot_move(encoder_counts, yaw, dt);
            //test_robot();
            //test_map();
        }
    }
}

#endif
