#ifndef _TEST_ROBOT_EXTEND_H_
#define _TEST_ROBOT_EXTEND_H_

#include "headers.h"
#include "host_asserts.h"
#include "host_utils.h"
#include "kernels.cuh"
#include "kernels_robot.cuh"
#include "kernels_utils.cuh"

//#define ADD_ROBOT_MOVE
//#define ADD_ROBOT

#ifdef ADD_ROBOT_MOVE
#include "data/robot_advance/600.h"
#endif

#ifdef ADD_ROBOT
#include "data/robot_iteration/600.h"
#endif

const int UNIQUE_COUNTER_LEN = NUM_PARTICLES + 1;


int GRID_WIDTH = 0;
int GRID_HEIGHT = 0;
int xmin = 0;
int ymax = 0;
float res = 0;
int LIDAR_COORDS_LEN = 0;


int MEASURE_LEN = 0;

int PARTICLES_ITEMS_LEN = 0;
int C_PARTICLES_ITEMS_LEN = 0;

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

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

vector<float> vec_robot_transition_world_body;
vector<float> vec_robot_state;
vector<float> vec_particles_weight_post;
vector<float> vec_rnds;

int EXTRA_GRID_WIDTH = 0;
int EXTRA_GRID_HEIGHT = 0;
int EXTRA_PARTICLES_ITEMS_LEN = 0;
int EXTRA_LIDAR_COORDS_LEN = 0;

float extra_res = 0;
int extra_xmin = 0;
int extra_ymax = 0;

vector<int> extra_grid_map;
vector<float> extra_lidar_coords;
vector<int> extra_particles_x;
vector<int> extra_particles_y;
vector<int> extra_particles_idx;
vector<float> extra_states_x;
vector<float> extra_states_y;
vector<float> extra_states_theta;
vector<float> extra_new_weights;
vector<float> extra_particles_weight_pre;

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////


void read_robot_data(int file_number, bool check_robot_transition = false, bool check_state = false,
    bool check_particles_weight = false, bool check_rnds = false) {

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

    int num_equals = 0;

#ifdef ADD_ROBOT
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

void read_robot_extra(int file_number, bool check_grid_map = false, bool check_particles = false,
    bool check_states = false, bool check_weights = false) {

    string file_name = std::to_string(file_number);

    const int SCALAR_VALUES = 1;
    const int GRID_MAP_VALUES = 2;
    const int LIDAR_COORDS_VALUES = 12;
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
    string str_lidar_coords = "";
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
            else if (line == "LIDAR_COORDS_LEN") {
                getline(data, line);
                EXTRA_LIDAR_COORDS_LEN = std::stoi(line);
            }
            else if (line == "res") {
                getline(data, line);
                extra_res = std::stof(line);
            }
            else if (line == "xmin") {
                getline(data, line);
                extra_xmin = std::stoi(line);
            }
            else if (line == "ymax") {
                getline(data, line);
                extra_ymax = std::stoi(line);
            }
        }

        if (line == "grid_map") {
            curr_state = GRID_MAP_VALUES;
            continue;
        }
        else if (line == "lidar_coords") {
            curr_state = LIDAR_COORDS_VALUES;
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
        else if (curr_state == LIDAR_COORDS_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_lidar_coords += line;
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
    stringstream stream_lidar_coords(str_lidar_coords);
    extra_lidar_coords.resize(GRID_SIZE);
    for (int i = 0; std::getline(stream_lidar_coords, segment, ','); i++) {
        extra_lidar_coords[i] = std::stof(segment);
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

    int num_equals = 0;

#ifdef ADD_ROBOT
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

#ifdef ADD_ROBOT_MOVE
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

void alloc_init_lidar_coords_var(float* h_lidar_coords, const int LIDAR_COORDS_LEN) {

    sz_lidar_coords = 2 * LIDAR_COORDS_LEN * sizeof(float);
    gpuErrchk(cudaMalloc((void**)&d_lidar_coords, sz_lidar_coords));

    gpuErrchk(cudaMemcpy(d_lidar_coords, h_lidar_coords, sz_lidar_coords, cudaMemcpyHostToDevice));
}

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

void alloc_extended_idx() {

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

void exec_process_measurements(float res) {

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

void exec_create_2d_map() {

    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (d_map_2d, d_unique_in_particle, d_unique_in_particle_col, SEP,
        d_particles_x, d_particles_y, d_particles_idx, PARTICLES_ITEMS_LEN, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();
}

void exec_update_map() {

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;

    kernel_update_2d_map_with_measure << <blocksPerGrid, threadsPerBlock >> > (d_map_2d, d_unique_in_particle, d_unique_in_particle_col, SEP,
        d_processed_measure_x, d_processed_measure_y, d_processed_measure_idx,
        MEASURE_LEN, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();
}

void exec_particle_unique_cum_sum() {

    threadsPerBlock = UNIQUE_COUNTER_LEN;
    blocksPerGrid = 1;
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (d_unique_in_particle_col, GRID_WIDTH);
    thrust::exclusive_scan(thrust::device, d_unique_in_particle, d_unique_in_particle + UNIQUE_COUNTER_LEN, d_unique_in_particle, 0);
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(res_unique_in_particle, d_unique_in_particle, sz_unique_in_particle, cudaMemcpyDeviceToHost));

    PARTICLES_ITEMS_LEN = res_unique_in_particle[UNIQUE_COUNTER_LEN - 1];
    C_PARTICLES_ITEMS_LEN = 0;
}

void reinit_map_vars() {

    //gpuErrchk(cudaFree(d_particles_x));
    //gpuErrchk(cudaFree(d_particles_y));
    //gpuErrchk(cudaFree(d_extended_idx));

    sz_particles_pos = PARTICLES_ITEMS_LEN * sizeof(int);
    sz_extended_idx = PARTICLES_ITEMS_LEN * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_particles_x, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_y, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_extended_idx, sz_extended_idx));
}

void exec_map_restructure() {

    threadsPerBlock = GRID_WIDTH;
    blocksPerGrid = NUM_PARTICLES;

    gpuErrchk(cudaMemset(d_particles_idx, 0, sz_particles_idx));

    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (d_particles_x, d_particles_y, d_particles_idx, SEP,
        d_map_2d, d_unique_in_particle, d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, d_particles_idx, d_particles_idx + NUM_PARTICLES, d_particles_idx, 0);
}

void exec_index_expansion() {

    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;
    kernel_index_expansion << <blocksPerGrid, threadsPerBlock >> > (d_extended_idx, d_particles_idx, PARTICLES_ITEMS_LEN);
    cudaDeviceSynchronize();

    res_extended_idx = (int*)malloc(sz_extended_idx);
    gpuErrchk(cudaMemcpy(res_extended_idx, d_extended_idx, sz_extended_idx, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToHost));
}

void exec_correlation() {

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

void reinit_particles_vars() {

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

void exec_rearrangement() {

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

void assertResults(float* h_new_weights, float* h_particles_weight_post, 
    float* h_robot_transition_world_body, float* h_robot_state) {

    gpuErrchk(cudaMemcpy(res_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToHost));

    res_correlation_weights = (float*)malloc(sz_correlation_weights);
    res_particles_weight = (float*)malloc(sz_particles_weight);

    gpuErrchk(cudaMemcpy(res_correlation_weights, d_correlation_weights, sz_correlation_weights, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_weight, d_particles_weight, sz_particles_weight, cudaMemcpyDeviceToHost));

    ASSERT_update_particle_weights(res_correlation_weights, h_new_weights, NUM_PARTICLES, "weights", false, true, true);
    ASSERT_update_particle_weights(res_particles_weight, h_particles_weight_post, NUM_PARTICLES, "particles weight", false, false, true);

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

void test_robot_extend() {

    const int data_len = 6;
    int data[data_len] = { 1, 0, 2, 2, 1, 3 };
    int* d_data = NULL;
    gpuErrchk(cudaMalloc((void**)&d_data, data_len * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_data, data, data_len * sizeof(int), cudaMemcpyHostToDevice));

    thrust::exclusive_scan(thrust::host, data, data + 6, data, 0);
    thrust::exclusive_scan(thrust::device, d_data, d_data + 6, d_data, 0);

    printf("\n");
    printf("/****************************** ROBOT  ******************************/\n");

#ifdef ADD_ROBOT
    res = ST_res;
    GRID_WIDTH = ST_GRID_WIDTH;
    GRID_HEIGHT = ST_GRID_HEIGHT;
    xmin = ST_xmin;
    ymax = ST_ymax;
    LIDAR_COORDS_LEN = ST_LIDAR_COORDS_LEN;
    PARTICLES_ITEMS_LEN = ST_PARTICLES_ITEMS_LEN;
    MEASURE_LEN = NUM_PARTICLES * LIDAR_COORDS_LEN;
#endif

    int file_number = 620;
    read_robot_data(file_number, true, true, true, true);
    read_robot_extra(file_number, true, true, true, true);

    int negative_before_counter = getNegativeCounter(extra_particles_x.data(), extra_particles_y.data(), PARTICLES_ITEMS_LEN);
    int count_bigger_than_height = getGreaterThanCounter(extra_particles_y.data(), EXTRA_GRID_HEIGHT, PARTICLES_ITEMS_LEN);

    printf("~~$ GRID_WIDTH: \t\t%d \tEXTRA_GRID_WIDTH:   \t\t%d\n", GRID_WIDTH, EXTRA_GRID_WIDTH);
    printf("~~$ GRID_HEIGHT: \t\t%d \tEXTRA_GRID_HEIGHT: \t\t%d\n", GRID_HEIGHT, EXTRA_GRID_HEIGHT);
    printf("~~$ PARTICLES_ITEMS_LEN: \t%d \tEXTRA_PARTICLES_ITEMS_LEN: \t%d\n", PARTICLES_ITEMS_LEN, EXTRA_PARTICLES_ITEMS_LEN);
    printf("~~$ LIDAR_COORDS_LEN: \t\t%d \tEXTRA_LIDAR_COORDS_LEN: \t%d\n", LIDAR_COORDS_LEN, EXTRA_LIDAR_COORDS_LEN);
    printf("~~$ negative_before_counter: \t%d\n", negative_before_counter);
    printf("~~$ count_bigger_than_height: \t%d\n", count_bigger_than_height);

#ifdef ADD_ROBOT
    assert(GRID_WIDTH == EXTRA_GRID_WIDTH);
    assert(GRID_HEIGHT == EXTRA_GRID_HEIGHT);
    assert(LIDAR_COORDS_LEN == EXTRA_LIDAR_COORDS_LEN);
    assert(PARTICLES_ITEMS_LEN == EXTRA_PARTICLES_ITEMS_LEN);
#endif

    GRID_WIDTH = EXTRA_GRID_WIDTH;
    GRID_HEIGHT = EXTRA_GRID_HEIGHT;
    LIDAR_COORDS_LEN = EXTRA_LIDAR_COORDS_LEN;
    PARTICLES_ITEMS_LEN = EXTRA_PARTICLES_ITEMS_LEN;
    res = extra_res;
    xmin = extra_xmin;
    ymax = extra_ymax;
    MEASURE_LEN  = NUM_PARTICLES * EXTRA_LIDAR_COORDS_LEN;

    auto start_robot_particles_alloc = std::chrono::high_resolution_clock::now();

    alloc_init_state_vars(extra_states_x.data(), extra_states_y.data(), extra_states_theta.data());
    alloc_init_lidar_coords_var(extra_lidar_coords.data(), EXTRA_LIDAR_COORDS_LEN);
    alloc_init_grid_map(extra_grid_map.data(), EXTRA_GRID_WIDTH, EXTRA_GRID_HEIGHT);
    alloc_init_particles_vars(extra_particles_x.data(), extra_particles_y.data(), extra_particles_idx.data(), 
        extra_particles_weight_pre.data(), EXTRA_PARTICLES_ITEMS_LEN);

    alloc_extended_idx();
    alloc_states_copy_vars();
    alloc_correlation_vars();
    alloc_init_transition_vars(h_transition_body_lidar);
    alloc_init_processed_measurement_vars(EXTRA_LIDAR_COORDS_LEN);
    alloc_map_2d_var(EXTRA_GRID_WIDTH, EXTRA_GRID_HEIGHT);
    alloc_map_2d_unique_counter_vars(UNIQUE_COUNTER_LEN, EXTRA_GRID_WIDTH);
    alloc_correlation_weights_vars();
    alloc_resampling_vars(vec_rnds.data());
    auto stop_robot_particles_alloc = std::chrono::high_resolution_clock::now();


    auto start_robot_particles_kernel = std::chrono::high_resolution_clock::now();
    exec_calc_transition();
    exec_process_measurements(res);
    exec_create_2d_map();
    exec_update_map();
    exec_particle_unique_cum_sum();
    reinit_map_vars();
    exec_map_restructure();
    exec_index_expansion();
    exec_correlation();
    exec_update_weights();
    exec_resampling();
    reinit_particles_vars();
    exec_rearrangement();
    exec_update_states();
    auto stop_robot_particles_kernel = std::chrono::high_resolution_clock::now();

    assertResults(extra_new_weights.data(), vec_particles_weight_post.data(), 
        vec_robot_transition_world_body.data(), vec_robot_state.data());

    auto duration_robot_particles_alloc = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_particles_alloc - start_robot_particles_alloc);
    auto duration_robot_particles_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_particles_kernel - start_robot_particles_kernel);
    auto duration_robot_particles_total = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_particles_kernel - start_robot_particles_alloc);

    std::cout << std::endl;
    std::cout << "Time taken by function (Allocation): " << duration_robot_particles_alloc.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Kernel): " << duration_robot_particles_kernel.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Total): " << duration_robot_particles_total.count() << " microseconds" << std::endl;
}



#endif
