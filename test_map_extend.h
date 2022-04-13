#ifndef _TEST_MAP_EXTEND_H_
#define _TEST_MAP_EXTEND_H_


#include "headers.h"
#include "host_asserts.h"
#include "host_utils.h"
#include "kernels.cuh"
#include "kernels_map.cuh"
#include "kernels_utils.cuh"
#include "kernels_robot.cuh"

#define ENABLE_MAP_DATA

//#include "data/map/1900.h"
//#include "data/robot_advance/400.h"
//#include "data/robot_iteration/400.h"

#ifdef ENABLE_MAP_DATA
#include "data/map_iteration/400.h"
#endif

int LIDAR_COORDS_LEN = 0;
int GRID_WIDTH = 0;
int GRID_HEIGHT = 0;

float res = 0;
float log_t = 0;

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

int threadsPerBlock = 1;
int blocksPerGrid = 1;

/********************* IMAGE TRANSFORM VARIABLES ********************/
size_t sz_transition_single_frame = 0;
size_t sz_transition_body_lidar = 0;

float* d_transition_single_world_body = NULL;
float* d_transition_single_world_lidar = NULL;
float* d_transition_body_lidar = NULL;

/*------------------------ RESULT VARIABLES -----------------------*/
float* res_transition_world_lidar = NULL;

/*********************** MEASUREMENT VARIABLES **********************/
size_t sz_lidar_coords = 0;

float* d_lidar_coords = NULL;

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
size_t sz_grid_map = 0;
int* d_grid_map = NULL;
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

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

int NEW_GRID_WIDTH = 0;
int NEW_GRID_HEIGHT = 0;
int NEW_LIDAR_COORDS_LEN = 0;

vector<int> vec_grid_map;
vector<float> vec_log_odds;
vector<float> vec_lidar_coords;

int EXTRA_GRID_WIDTH = 0;
int EXTRA_GRID_HEIGHT = 0;
int extra_xmin = 0;
int extra_xmax = 0;
int extra_ymin = 0;
int extra_ymax = 0;
float extra_res = 0;
float extra_log_t = 0;

vector<int> extra_grid_map;
vector<float> extra_log_odds;
vector<float> extra_transition_single_world_body;


void read_map_data(int file_number, bool check_grid_map = false, bool check_log_odds = false, bool check_lidar_coords = false) {

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

#ifdef ENABLE_MAP_DATA
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

void read_map_extra(int file_number, bool check_grid_map = false, bool check_log_odds = false, bool check_transition = false) {

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

#ifdef ENABLE_MAP_DATA
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

void alloc_particles_free_vars() {

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

void alloc_particles_occupied_vars() {

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

void alloc_log_odds_vars() {

    sz_log_odds = (GRID_WIDTH * GRID_HEIGHT) * sizeof(float);
    gpuErrchk(cudaMalloc((void**)&d_log_odds, sz_log_odds));

    res_log_odds = (float*)malloc(sz_log_odds);
}

void alloc_init_log_odds_free_vars() {

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

void alloc_init_log_odds_occupied_vars() {

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

void init_log_odds_vars(float* h_log_odds) {

    h_occupied_map_idx[1] = PARTICLES_OCCUPIED_LEN;
    h_free_map_idx[1] = 0;

    memset(res_log_odds, 0, sz_log_odds);

    gpuErrchk(cudaMemcpy(d_occupied_map_idx, h_occupied_map_idx, sz_occupied_map_idx, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_free_map_idx, h_free_map_idx, sz_free_map_idx, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_log_odds, h_log_odds, sz_log_odds, cudaMemcpyHostToDevice));
}



void exec_world_to_image_transform_step_1() {

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

void exec_map_extend() {

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

void exec_world_to_image_transform_step_2() {

    threadsPerBlock = 1;
    blocksPerGrid = LIDAR_COORDS_LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (d_particles_occupied_x, d_particles_occupied_y, SEP,
        d_transition_single_world_lidar, d_lidar_coords, res, xmin, ymax, LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    kernel_position_to_image << < 1, 1 >> > (d_position_image_body, SEP, d_transition_single_world_lidar, res, xmin, ymax);
    cudaDeviceSynchronize();
}

void exec_bresenham() {

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

void reinit_map_idx_vars() {

    h_occupied_map_idx[1] = PARTICLES_OCCUPIED_LEN;
    h_free_map_idx[1] = PARTICLES_FREE_LEN;

    gpuErrchk(cudaMemcpy(d_occupied_map_idx, h_occupied_map_idx, sz_occupied_map_idx, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_free_map_idx, h_free_map_idx, sz_free_map_idx, cudaMemcpyHostToDevice));
}

void exec_create_map() {

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

void reinit_map_vars() {

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

void exec_log_odds(float log_t) {

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

void assertResults(float* h_post_log_odds, int* h_post_grid_map) {

    printf("\n--> Occupied Unique: %d\n", PARTICLES_OCCUPIED_UNIQUE_LEN);
    printf("\n--> Free Unique: %d\n", PARTICLES_FREE_UNIQUE_LEN);

    printf("~~$ PARTICLES_FREE_LEN=%d\n", PARTICLES_FREE_LEN);
    printf("~~$ sz_log_odds=%d\n", sz_log_odds / sizeof(float));

    gpuErrchk(cudaMemcpy(res_log_odds, d_log_odds, sz_log_odds, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_grid_map, d_grid_map, sz_grid_map, cudaMemcpyDeviceToHost));

    ASSERT_log_odds(res_log_odds, h_log_odds, h_post_log_odds, (GRID_WIDTH * GRID_HEIGHT));
    ASSERT_log_odds_maps(res_grid_map, h_grid_map, h_post_grid_map, (GRID_WIDTH * GRID_HEIGHT));

    printf("\n~~$ Verification All Passed\n");
}


void test_map_extend() {

    const int data_len = 6;
    int data[data_len] = { 1, 0, 2, 2, 1, 3 };
    int* d_data = NULL;
    gpuErrchk(cudaMalloc((void**)&d_data, data_len * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_data, data, data_len * sizeof(int), cudaMemcpyHostToDevice));

    thrust::exclusive_scan(thrust::host, data, data + 6, data, 0);
    thrust::exclusive_scan(thrust::device, d_data, d_data + 6, d_data, 0);

    printf("\n");
    printf("/****************************** MAP MAIN ****************************/\n");

    read_map_data(400, true, true, true);
    read_map_extra(400, true, true, true);

    //res = ST_res;
    //log_t = ST_log_t;
    //GRID_WIDTH = NEW_GRID_WIDTH;
    //GRID_HEIGHT = NEW_GRID_HEIGHT;
    //LIDAR_COORDS_LEN = NEW_LIDAR_COORDS_LEN;
    //PARTICLES_OCCUPIED_LEN = NEW_LIDAR_COORDS_LEN;
    //xmin = ST_xmin;
    //xmax = ST_xmax;;
    //ymin = ST_ymin;
    //ymax = ST_ymax;

    res = extra_res;
    log_t = extra_log_t;
    GRID_WIDTH = EXTRA_GRID_WIDTH;
    GRID_HEIGHT = EXTRA_GRID_HEIGHT;
    xmin = extra_xmin;
    xmax = extra_xmax;;
    ymin = extra_ymin;
    ymax = extra_ymax;

    LIDAR_COORDS_LEN = NEW_LIDAR_COORDS_LEN;
    PARTICLES_OCCUPIED_LEN = NEW_LIDAR_COORDS_LEN;

    MAX_DIST_IN_MAP = sqrt(pow(GRID_WIDTH, 2) + pow(GRID_HEIGHT, 2));
    PARTICLE_UNIQUE_COUNTER = PARTICLES_OCCUPIED_LEN + 1;


    h_occupied_map_idx[1] = PARTICLES_OCCUPIED_LEN;
    h_free_map_idx[1] = PARTICLES_FREE_LEN;

    printf("~~$ GRID_WIDTH: \t\t%d \tEXTRA_GRID_WIDTH:   \t\t%d\n", GRID_WIDTH, EXTRA_GRID_WIDTH);
    printf("~~$ GRID_HEIGHT: \t\t%d \tEXTRA_GRID_HEIGHT: \t\t%d\n", GRID_HEIGHT, EXTRA_GRID_HEIGHT);
    printf("~~$ LIDAR_COORDS_LEN: \t\t%d\n", LIDAR_COORDS_LEN);
    printf("log_t = %f\n", log_t);

    printf("~~$ PARTICLES_OCCUPIED_LEN = \t%d\n",   PARTICLES_OCCUPIED_LEN);
    printf("~~$ PARTICLE_UNIQUE_COUNTER = \t%d\n",  PARTICLE_UNIQUE_COUNTER);
    printf("~~$ MAX_DIST_IN_MAP = \t\t%d\n",        MAX_DIST_IN_MAP);


    auto start_mapping_alloc = std::chrono::high_resolution_clock::now();
    alloc_init_transition_vars(h_transition_body_lidar, extra_transition_single_world_body.data());
    alloc_init_lidar_coords_var(vec_lidar_coords.data(), LIDAR_COORDS_LEN);
    alloc_particles_world_vars(LIDAR_COORDS_LEN);
    alloc_particles_free_vars();
    alloc_particles_occupied_vars();
    alloc_bresenham_vars();
    alloc_init_map_vars(extra_grid_map.data(), GRID_WIDTH, GRID_HEIGHT);
    alloc_log_odds_vars();
    alloc_init_log_odds_free_vars();
    alloc_init_log_odds_occupied_vars();
    init_log_odds_vars(extra_log_odds.data());
    auto stop_mapping_alloc = std::chrono::high_resolution_clock::now();


    auto start_mapping_kernel = std::chrono::high_resolution_clock::now();
    exec_world_to_image_transform_step_1();
    exec_map_extend();
    exec_world_to_image_transform_step_2();
    exec_bresenham();
    reinit_map_idx_vars();

    exec_create_map();
    reinit_map_vars();

    exec_log_odds(log_t);
    auto stop_mapping_kernel = std::chrono::high_resolution_clock::now();

    assertResults(vec_log_odds.data(), vec_grid_map.data());

    auto duration_mapping_alloc = std::chrono::duration_cast<std::chrono::microseconds>(stop_mapping_alloc - start_mapping_alloc);
    auto duration_mapping_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_mapping_kernel - start_mapping_kernel);
    auto duration_mapping_total = std::chrono::duration_cast<std::chrono::microseconds>(stop_mapping_kernel - start_mapping_alloc);

    std::cout << std::endl;
    std::cout << "Time taken by function (Mapping Allocation): " << duration_mapping_alloc.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Mapping Kernel): " << duration_mapping_kernel.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Mapping Total): " << duration_mapping_total.count() << " microseconds" << std::endl;
    std::cout << std::endl;
}


#endif
