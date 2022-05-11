#ifndef _RUN_KERNELS_H_
#define _RUN_KERNELS_H_

#include "headers.h"
#include "host_utils.h"
#include "kernels.cuh"
#include "kernels_robot.cuh"
#include "kernels_map.cuh"
#include "kernels_utils.cuh"
#include "draw_utils.h"

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

bool map_size_changed = false;

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

Window mainWindow;
vector<Shader> shader_list;
Camera camera;

vector<Mesh*> freeList;
vector<Mesh*> wallList;

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

void alloc_resampling_vars(float* h_resampling_rnds, bool should_mem_allocate) {

    if (should_mem_allocate == true) {

        sz_resampling_js = NUM_PARTICLES * sizeof(int);
        sz_resampling_rnd = NUM_PARTICLES * sizeof(float);

        gpuErrchk(cudaMalloc((void**)&d_resampling_js, sz_resampling_js));
        gpuErrchk(cudaMalloc((void**)&d_resampling_rnd, sz_resampling_rnd));
    }

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

    gpuErrchk(cudaMemcpy(d_transition_single_world_body, res_robot_world_body, sz_transition_single_frame, cudaMemcpyHostToDevice));

    res_robot_state[0] = std::get<0>(key); res_robot_state[1] = std::get<1>(key); res_robot_state[2] = std::get<2>(key);
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

void alloc_init_log_odds_free_vars(bool should_mem_allocate, int GRID_WIDTH, int GRID_HEIGHT) {

    if (should_mem_allocate == true) {

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

    gpuErrchk(cudaMemset(d_free_map_2d, 0, sz_free_map_2d));
    gpuErrchk(cudaMemset(d_free_unique_counter, 0, sz_free_unique_counter));
    gpuErrchk(cudaMemset(d_free_unique_counter_col, 0, sz_free_unique_counter_col));
    gpuErrchk(cudaMemset(d_free_map_idx, 0, sz_free_map_idx));
}

void alloc_init_log_odds_occupied_vars(bool should_mem_allocate, int GRID_WIDTH, int GRID_HEIGHT) {

    if (should_mem_allocate == true) {

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

    gpuErrchk(cudaMemset(d_occupied_map_2d, 0, sz_occupied_map_2d));
    gpuErrchk(cudaMemset(d_occupied_unique_counter, 0, sz_occupied_unique_counter));
    gpuErrchk(cudaMemset(d_occupied_unique_counter_col, 0, sz_occupied_unique_counter_col));
    gpuErrchk(cudaMemset(d_occupied_map_idx, 0, sz_occupied_map_idx));
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

        map_size_changed = true;

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

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

void alloc_init_movement_vars(float* h_rnds_encoder_counts, float* h_rnds_yaws, bool should_mem_allocate) {

    if (should_mem_allocate == true) {

        gpuErrchk(cudaMalloc((void**)&d_rnds_encoder_counts, sz_states_pos));
        gpuErrchk(cudaMalloc((void**)&d_rnds_yaws, sz_states_pos));
    }

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

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

void test_setup(int& xmin, int& xmax, int& ymin, int& ymax, float& res, float& log_t,
    int& LIDAR_COORDS_LEN, int& MEASURE_LEN, int& PARTICLES_ITEMS_LEN,
    int& PARTICLES_OCCUPIED_LEN, const int PARTICLES_FREE_LEN, int& PARTICLE_UNIQUE_COUNTER, int& MAX_DIST_IN_MAP,
    int& GRID_WIDTH, int& GRID_HEIGHT,
    const int NEW_GRID_WIDTH, const int NEW_GRID_HEIGHT, const int NEW_LIDAR_COORDS_LEN,
    const int extra_xmin, const int extra_xmax, const int extra_ymin, const int extra_ymax,
    const float extra_res, const float extra_log_t, const int EXTRA_PARTICLES_ITEMS_LEN) {

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

    h_occupied_map_idx[1] = PARTICLES_OCCUPIED_LEN;
    h_free_map_idx[1] = PARTICLES_FREE_LEN;
}

void test_allocation_initialization(float* h_states_x, float* h_states_y, float* h_states_theta,
    float* h_rnds_encoder_counts, float* h_rnds_yaws, float* h_particles_weight,
    float* h_lidar_coords, int* h_grid_map, float* h_log_odds,
    int* h_particles_x, int* h_particles_y, int* h_particles_idx,
    float* h_rnds, float* h_transition_single_world_body,
    const int LIDAR_COORDS_LEN, const int MEASURE_LEN,
    const int PARTICLES_ITEMS_LEN, const int PARTICLES_OCCUPIED_LEN, const int PARTICLE_UNIQUE_COUNTER,
    const int MAX_DIST_IN_MAP, const int GRID_WIDTH, const int GRID_HEIGHT) {

    alloc_init_state_vars(h_states_x, h_states_y, h_states_theta);
    alloc_init_movement_vars(h_rnds_encoder_counts, h_rnds_yaws, true);

    alloc_init_lidar_coords_var(h_lidar_coords, LIDAR_COORDS_LEN);
    alloc_init_grid_map(h_grid_map, GRID_WIDTH, GRID_HEIGHT);
    alloc_init_particles_vars(h_particles_x, h_particles_y, h_particles_idx,
        h_particles_weight, PARTICLES_ITEMS_LEN);
    alloc_extended_idx(PARTICLES_ITEMS_LEN);
    alloc_states_copy_vars();
    alloc_correlation_vars();
    alloc_init_transition_vars(h_transition_body_lidar);
    alloc_init_processed_measurement_vars(LIDAR_COORDS_LEN);
    alloc_map_2d_var(GRID_WIDTH, GRID_HEIGHT);
    alloc_map_2d_unique_counter_vars(UNIQUE_COUNTER_LEN, GRID_WIDTH);
    alloc_correlation_weights_vars();
    alloc_resampling_vars(h_rnds, true);

    alloc_init_transition_vars(h_transition_body_lidar, h_transition_single_world_body);
    alloc_init_lidar_coords_var(h_lidar_coords, LIDAR_COORDS_LEN);
    alloc_particles_world_vars(LIDAR_COORDS_LEN);
    alloc_particles_free_vars(PARTICLES_OCCUPIED_LEN, PARTICLE_UNIQUE_COUNTER, MAX_DIST_IN_MAP);
    alloc_particles_occupied_vars(LIDAR_COORDS_LEN);
    alloc_bresenham_vars();
    alloc_init_map_vars(h_grid_map, GRID_WIDTH, GRID_HEIGHT);
    alloc_log_odds_vars(GRID_WIDTH, GRID_HEIGHT);
    alloc_init_log_odds_free_vars(true, GRID_WIDTH, GRID_HEIGHT);
    alloc_init_log_odds_occupied_vars(true, GRID_WIDTH, GRID_HEIGHT);
    init_log_odds_vars(h_log_odds, PARTICLES_OCCUPIED_LEN);
}

void test_robot_move(float* extra_states_x, float* extra_states_y, float* extra_states_theta,
                    bool check_result, float encoder_counts, float yaw, float dt) {

    exec_robot_advance(encoder_counts, yaw, dt, ST_nv, ST_nw);
}

void test_robot(float* extra_new_weights, float* vec_robot_transition_world_body,
    float* vec_robot_state, float* vec_particles_weight_post,
    bool check_result, const int xmin, const int ymax, const float res, const float log_t,
    const int LIDAR_COORDS_LEN, const int MEASURE_LEN,
    int& PARTICLES_ITEMS_LEN, int& C_PARTICLES_ITEMS_LEN,
    const int GRID_WIDTH, const int GRID_HEIGHT) {

    res_particles_x = (int*)malloc(sz_particles_pos);
    res_particles_y = (int*)malloc(sz_particles_pos);

    gpuErrchk(cudaMemcpy(res_particles_x, d_particles_x, sz_particles_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_y, d_particles_y, sz_particles_pos, cudaMemcpyDeviceToHost));

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
}

void test_map(int* extra_grid_map, float* extra_log_odds,
    int* vec_grid_map, float* vec_log_odds,
    bool check_result, int& xmin, int& xmax, int& ymin, int& ymax, const float res, const float log_t,
    const int PARTICLES_OCCUPIED_LEN, int& PARTICLES_OCCUPIED_UNIQUE_LEN,
    int& PARTICLES_FREE_LEN, int& PARTICLES_FREE_UNIQUE_LEN,
    const int PARTICLE_UNIQUE_COUNTER, const int MAX_DIST_IN_MAP,
    const int LIDAR_COORDS_LEN, int& GRID_WIDTH, int& GRID_HEIGHT) {

    exec_world_to_image_transform_step_1(xmin, ymax, res, LIDAR_COORDS_LEN);
    exec_map_extend(xmin, xmax, ymin, ymax, res, LIDAR_COORDS_LEN, GRID_WIDTH, GRID_HEIGHT);
    exec_world_to_image_transform_step_2(xmin, ymax, res, LIDAR_COORDS_LEN);
    exec_bresenham(PARTICLES_OCCUPIED_LEN, PARTICLES_FREE_LEN, PARTICLE_UNIQUE_COUNTER, MAX_DIST_IN_MAP);
    reinit_map_idx_vars(PARTICLES_OCCUPIED_LEN, PARTICLES_FREE_LEN);

    exec_create_map(PARTICLES_OCCUPIED_LEN, PARTICLES_FREE_LEN, GRID_WIDTH, GRID_HEIGHT);
    reinit_map_vars(PARTICLES_OCCUPIED_UNIQUE_LEN, PARTICLES_FREE_UNIQUE_LEN, GRID_WIDTH, GRID_HEIGHT);

    exec_log_odds(log_t, PARTICLES_OCCUPIED_UNIQUE_LEN, PARTICLES_FREE_UNIQUE_LEN, GRID_WIDTH, GRID_HEIGHT);
}

void resetMiddleVariables(int LIDAR_COORDS_LEN, int GRID_WIDTH, int GRID_HEIGHT) {

    int num_items = 25 * NUM_PARTICLES;
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
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

// [ ] - Define THR_GRID_WIDTH, THR_GRID_HEIGHT
// [ ] - Define CURR_GRID_WIDTH, CURR_GRID_HEIGHT
// [ ] - Define curr_grid_map
// [ ] - Define draw thread function
// [ ] - Define mutex for synchronization

int THR_GRID_WIDTH = 0;
int THR_GRID_HEIGHT = 0;

timed_mutex timed_mutex_draw;

void thread_draw() {

    GLfloat delta_time = 0.0f;
    GLfloat last_time = 0.0f;

    Light main_light;

    int CURR_GRID_WIDTH = 0;
    int CURR_GRID_HEIGHT = 0;

    while (timed_mutex_draw.try_lock_until(std::chrono::steady_clock::now() + std::chrono::seconds(1)) == false);
    printf("Draw Thread Started ...\n");

    CURR_GRID_WIDTH = THR_GRID_WIDTH;
    CURR_GRID_HEIGHT = THR_GRID_HEIGHT;
    printf("CURR_GRID_WIDTH: %d, CURR_GRID_HEIGHT: %d\n", CURR_GRID_WIDTH, CURR_GRID_HEIGHT);

    // Vertex Shader
    static const char* vShader = "Shaders/shader.vert";

    // Fragment Shader
    static const char* fShader = "Shaders/shader.frag";

    mainWindow.initialize();

    gpuErrchk(cudaMemcpy(res_grid_map, d_grid_map, sz_grid_map, cudaMemcpyDeviceToHost));

    CreateObjects(freeList, res_grid_map, CURR_GRID_WIDTH * CURR_GRID_HEIGHT, 1, 0.1f, CURR_GRID_WIDTH);
    CreateObjects(wallList, res_grid_map, CURR_GRID_WIDTH * CURR_GRID_HEIGHT, 2, 0.5f, CURR_GRID_WIDTH);
    CreateShaders(shader_list, vShader, fShader);

    camera = Camera(glm::vec3(-2.0f, 4.0f, 12.0f), glm::vec3(0.0f, 1.0f, 0.0f), -90.0f, -45.0f, 5.0f, 0.1f);

    main_light = Light(1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    GLuint uniformModel = 0, uniformProjection = 0, uniformView = 0, uniformColor = 0,
        uniformAmbientIntensity = 0, uniformAmbientColor = 0;
    glm::mat4 projection = glm::perspective(45.0f,
        (GLfloat)mainWindow.getBufferWidth() / (GLfloat)mainWindow.getBufferHeight(), 0.1f, 90.0f);


    // Loop until windows closed
    while (!mainWindow.getShouldClose()) {

        if (timed_mutex_draw.try_lock_until(std::chrono::steady_clock::now() + std::chrono::milliseconds(10)) == true) {
            
            CURR_GRID_WIDTH = THR_GRID_WIDTH;
            CURR_GRID_HEIGHT = THR_GRID_HEIGHT;
            printf("CURR_GRID_WIDTH: %d, CURR_GRID_HEIGHT: %d\n", CURR_GRID_WIDTH, CURR_GRID_HEIGHT);

            gpuErrchk(cudaMemcpy(res_grid_map, d_grid_map, sz_grid_map, cudaMemcpyDeviceToHost));

            freeList.clear();
            wallList.clear();

            CreateObjects(freeList, res_grid_map, CURR_GRID_WIDTH * CURR_GRID_HEIGHT, 1, 0.1f, CURR_GRID_HEIGHT);
            CreateObjects(wallList, res_grid_map, CURR_GRID_WIDTH * CURR_GRID_HEIGHT, 2, 0.5f, CURR_GRID_HEIGHT);
            //CreateShaders(shader_list, vShader, fShader);
        }

        GLfloat now = glfwGetTime();
        delta_time = now - last_time;
        last_time = now;

        // Get+Handle user inputs
        glfwPollEvents();

        camera.keyControl(mainWindow.getskeys(), delta_time);
        camera.mouseControl(mainWindow.getXChange(), mainWindow.getYChange(), delta_time);

        // Clear window
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        shader_list[0].UseShader();
        uniformModel = shader_list[0].GetModelLocation();
        uniformProjection = shader_list[0].GetProjectionLocation();
        uniformView = shader_list[0].GetViewLocation();
        uniformColor = shader_list[0].GetColorLocation();
        uniformAmbientColor = shader_list[0].GetAmbientColorLocation();
        uniformAmbientIntensity = shader_list[0].GetAmbientIntensityLocation();

        main_light.UseLight(uniformAmbientIntensity, uniformAmbientColor, 0.0f, 0.0f);

        glm::mat4 model = glm::identity<glm::mat4>();

        model = glm::translate(model, glm::vec3(0.0f, 0.0f, -5.0f));
        //model = glm::scale(model, glm::vec3(0.4f, 0.4f, 0.4f));
        glUniformMatrix4fv(uniformModel, 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(uniformProjection, 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(uniformView, 1, GL_FALSE, glm::value_ptr(camera.calculateViewMatrix()));

        glUniform4f(uniformColor, 0.8f, 0.8f, 0.8f, 1.0f);
        for (int i = 0; i < freeList.size(); i++) {
            freeList[i]->RenderMesh();
        }

        glUniform4f(uniformColor, 0.0f, 0.2f, 0.9f, 1.0f);
        for (int i = 0; i < wallList.size(); i++) {
            wallList[i]->RenderMesh();
        }

        glUseProgram(0);

        mainWindow.swapBuffers();
    }
}

void run_main() {

    std::cout << "Run Application" << std::endl;

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

    int NEW_GRID_WIDTH = 0;
    int NEW_GRID_HEIGHT = 0;
    int NEW_LIDAR_COORDS_LEN = 0;
    float encoder_counts = 0;
    float yaw = 0;
    float dt = 0;

    int EXTRA_GRID_WIDTH = 0;
    int EXTRA_GRID_HEIGHT = 0;
    int EXTRA_PARTICLES_ITEMS_LEN = 0;

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

    vector<int> extra_grid_map;
    vector<int> extra_particles_x;
    vector<int> extra_particles_y;
    vector<int> extra_particles_idx;
    vector<float> extra_states_x;
    vector<float> extra_states_y;
    vector<float> extra_states_theta;
    vector<float> extra_new_weights;
    vector<float> extra_particles_weight_pre;
    vector<float> extra_log_odds;
    vector<float> extra_transition_single_world_body;

    int extra_xmin = 0;
    int extra_xmax = 0;
    int extra_ymin = 0;
    int extra_ymax = 0;
    float extra_res = 0;
    float extra_log_t = 0;

    vector<float> vec_encoder_counts;
    vector<vector<float>> vec_arr_rnds_encoder_counts;
    vector<vector<float>> vec_arr_lidar_coords;
    vector<vector<float>> vec_arr_rnds;
    vector<vector<float>> vec_arr_transition;
    vector<float> vec_yaws;
    vector<vector<float>> vec_arr_rnds_yaws;
    vector<float> vec_dt;

    printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
    printf("Reading Data Files\n");
    printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");

    read_small_steps_vec("encoder_counts", vec_encoder_counts);
    read_small_steps_vec_arr("rnds_encoder_counts", vec_arr_rnds_encoder_counts);
    read_small_steps_vec_arr("lidar_coords", vec_arr_lidar_coords);
    read_small_steps_vec_arr("rnds", vec_arr_rnds);
    read_small_steps_vec_arr("transition", vec_arr_transition);
    read_small_steps_vec("yaws", vec_yaws);
    read_small_steps_vec_arr("rnds_yaws", vec_arr_rnds_yaws);
    read_small_steps_vec("dt", vec_dt);

    const int OFFSET = 200;
    const int ST_FILE_NUMBER = 2800 + OFFSET;
    const int LOOP_LEN = 500;
    const int CHECK_STEP = 10;

    bool check_assert = false;
    std::thread t(thread_draw);
    lock_guard<timed_mutex> l(timed_mutex_draw);

    for (int file_number = ST_FILE_NUMBER; file_number < ST_FILE_NUMBER + LOOP_LEN; file_number += 1) {

        auto start_run_step = std::chrono::high_resolution_clock::now();
        int vec_index = file_number - ST_FILE_NUMBER + OFFSET;

        if (file_number == ST_FILE_NUMBER) {

            auto start_read_file = std::chrono::high_resolution_clock::now();
            read_robot_data(file_number, vec_robot_transition_world_body, vec_robot_state,
                vec_particles_weight_post, vec_rnds);
            read_robot_move_data(file_number, vec_rnds_encoder_counts, vec_rnds_yaws,
                vec_states_x, vec_states_y, vec_states_theta, encoder_counts, yaw, dt);
            read_robot_extra(file_number,
                extra_grid_map, extra_particles_x, extra_particles_y, extra_particles_idx,
                extra_states_x, extra_states_y, extra_states_theta,
                extra_new_weights, extra_particles_weight_pre,
                EXTRA_GRID_WIDTH, EXTRA_GRID_HEIGHT, EXTRA_PARTICLES_ITEMS_LEN);
            read_map_data(file_number, vec_grid_map, vec_log_odds, vec_lidar_coords,
                NEW_GRID_WIDTH, NEW_GRID_HEIGHT, NEW_LIDAR_COORDS_LEN);
            read_map_extra(file_number, extra_grid_map, extra_log_odds, extra_transition_single_world_body,
                extra_xmin, extra_xmax, extra_ymin, extra_ymax, extra_res, extra_log_t,
                EXTRA_GRID_WIDTH, EXTRA_GRID_HEIGHT,
                NEW_GRID_WIDTH, NEW_GRID_HEIGHT);
            auto stop_read_file = std::chrono::high_resolution_clock::now();

            test_setup(xmin, xmax, ymin, ymax, res, log_t,
                LIDAR_COORDS_LEN, MEASURE_LEN, PARTICLES_ITEMS_LEN,
                PARTICLES_OCCUPIED_LEN, PARTICLES_FREE_LEN, PARTICLE_UNIQUE_COUNTER, MAX_DIST_IN_MAP,
                GRID_WIDTH, GRID_HEIGHT,
                NEW_GRID_WIDTH, NEW_GRID_HEIGHT, NEW_LIDAR_COORDS_LEN,
                extra_xmin, extra_xmax, extra_ymin, extra_ymax,
                extra_res, extra_log_t, EXTRA_PARTICLES_ITEMS_LEN);

            test_allocation_initialization(vec_states_x.data(), vec_states_y.data(), vec_states_theta.data(),
                vec_rnds_encoder_counts.data(), vec_rnds_yaws.data(), extra_particles_weight_pre.data(),
                vec_lidar_coords.data(), extra_grid_map.data(), extra_log_odds.data(),
                extra_particles_x.data(), extra_particles_y.data(), extra_particles_idx.data(),
                vec_rnds.data(), extra_transition_single_world_body.data(),
                LIDAR_COORDS_LEN, MEASURE_LEN,
                PARTICLES_ITEMS_LEN, PARTICLES_OCCUPIED_LEN, PARTICLE_UNIQUE_COUNTER,
                MAX_DIST_IN_MAP, GRID_WIDTH, GRID_HEIGHT);

            auto duration_read_file = std::chrono::duration_cast<std::chrono::microseconds>(stop_read_file - start_read_file);
            std::cout << std::endl;
            std::cout << "Time taken by function (Read Data Files): " << duration_read_file.count() << " microseconds" << std::endl;
            std::cout << std::endl;
        }
        else {

            resetMiddleVariables(LIDAR_COORDS_LEN, GRID_WIDTH, GRID_HEIGHT);

            LIDAR_COORDS_LEN = vec_arr_lidar_coords[vec_index].size() / 2;
            MEASURE_LEN = NUM_PARTICLES * LIDAR_COORDS_LEN;
            PARTICLES_OCCUPIED_LEN = LIDAR_COORDS_LEN;
            PARTICLE_UNIQUE_COUNTER = PARTICLES_OCCUPIED_LEN + 1;

#if defined(GET_EXTRA_TRANSITION_WORLD_BODY)
            gpuErrchk(cudaMemcpy(d_transition_single_world_body, vec_arr_transition[vec_index].data(),
                sz_transition_single_frame, cudaMemcpyHostToDevice));
#endif

            alloc_particles_free_vars(PARTICLES_OCCUPIED_LEN, PARTICLE_UNIQUE_COUNTER, MAX_DIST_IN_MAP);
            alloc_particles_occupied_vars(LIDAR_COORDS_LEN);
            alloc_init_log_odds_free_vars(map_size_changed, GRID_WIDTH, GRID_HEIGHT);
            alloc_init_log_odds_occupied_vars(map_size_changed, GRID_WIDTH, GRID_HEIGHT);

            alloc_init_movement_vars(vec_arr_rnds_encoder_counts[vec_index].data(),
                vec_arr_rnds_yaws[vec_index].data(), false);
            alloc_init_lidar_coords_var(vec_arr_lidar_coords[vec_index].data(), LIDAR_COORDS_LEN);
            alloc_init_processed_measurement_vars(LIDAR_COORDS_LEN);
            alloc_resampling_vars(vec_arr_rnds[vec_index].data(), false);

            map_size_changed = false;
        }

        //check_assert = (file_number % CHECK_STEP == 0);
        test_robot_move(extra_states_x.data(), extra_states_y.data(), extra_states_theta.data(), check_assert,
            vec_encoder_counts[vec_index], vec_yaws[vec_index], vec_dt[vec_index]);
        test_robot(extra_new_weights.data(), vec_robot_transition_world_body.data(),
            vec_robot_state.data(), vec_particles_weight_post.data(),
            check_assert, xmin, ymax, res, log_t, LIDAR_COORDS_LEN, MEASURE_LEN,
            PARTICLES_ITEMS_LEN, C_PARTICLES_ITEMS_LEN, GRID_WIDTH, GRID_HEIGHT);
        test_map(extra_grid_map.data(), extra_log_odds.data(), vec_grid_map.data(), vec_log_odds.data(),
            check_assert, xmin, xmax, ymin, ymax, res, log_t, PARTICLES_OCCUPIED_LEN, PARTICLES_OCCUPIED_UNIQUE_LEN,
            PARTICLES_FREE_LEN, PARTICLES_FREE_UNIQUE_LEN, PARTICLE_UNIQUE_COUNTER, MAX_DIST_IN_MAP,
            LIDAR_COORDS_LEN, GRID_WIDTH, GRID_HEIGHT);

        if ((file_number + 1) % CHECK_STEP == 0) {

            auto stop_run_step = std::chrono::high_resolution_clock::now();

            printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
            printf("Iteration: %d\n", file_number);
            printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");

            auto duration_run_step = std::chrono::duration_cast<std::chrono::microseconds>(stop_run_step - start_run_step);
            std::cout << std::endl;
            std::cout << "Time taken by function (Run Step): " << duration_run_step.count() << " microseconds" << std::endl;
            std::cout << std::endl;

            THR_GRID_WIDTH = GRID_WIDTH;
            THR_GRID_HEIGHT = GRID_HEIGHT;
            timed_mutex_draw.unlock();
        }
    }

    t.join();
}

#endif

