#ifndef _TEST_RUN_H_
#define _TEST_RUN_H_

#include "headers.h"
#include "host_asserts.h"
#include "host_utils.h"
#include "kernels.cuh"
#include "kernels_robot.cuh"
#include "kernels_map.cuh"
#include "kernels_utils.cuh"

#include "data/robot_update/1900.h"

#define ST_log_t	0

const int UNIQUE_COUNTER_LEN = NUM_PARTICLES + 1;
float res = ST_res;
float log_t = ST_log_t;

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

int threadsPerBlock = 1;
int blocksPerGrid = 1;

/********************************************************************/
/********************* IMAGE TRANSFORM VARIABLES ********************/
/********************************************************************/
size_t sz_transition_world_frame = 0;
size_t sz_transition_body_lidar = 0;

float* d_transition_world_body = NULL;
float* d_transition_body_lidar = NULL;
float* d_transition_world_lidar = NULL;

/*------------------------ RESULT VARIABLES -----------------------*/
float* res_transition_world_body = NULL;
//float* res_transition_world_lidar = NULL;

/********************************************************************/
/************************* STATES VARIABLES *************************/
/********************************************************************/
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


/********************************************************************/
/************************ PARTICLES VARIABLES ***********************/
/********************************************************************/
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

/********************************************************************/
/*********************** MEASUREMENT VARIABLES **********************/
/********************************************************************/
size_t sz_lidar_coords = 0;
float* d_lidar_coords = NULL;

/********************************************************************/
/**************** PROCESSED MEASUREMENTS VARIABLES ******************/
/********************************************************************/
size_t sz_processed_measure_pos = 0;
size_t sz_processed_measure_idx = 0;

int* d_processed_measure_x = NULL;
int* d_processed_measure_y = NULL;
int* d_processed_measure_idx = NULL;

/*------------------------ RESULT VARIABLES -----------------------*/
int* res_processed_measure_x = NULL;
int* res_processed_measure_y = NULL;
int* res_processed_measure_idx = NULL;


/********************************************************************/
/************************ WEIGHTS VARIABLES *************************/
/********************************************************************/
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

/********************************************************************/
/*********************** RESAMPLING VARIABLES ***********************/
/********************************************************************/
size_t sz_resampling_js = 0;
size_t sz_resampling_rnd = 0;

int* d_resampling_js = NULL;
float* d_resampling_rnd = NULL;

int* res_js = NULL;

/********************************************************************/
/**************************** MAP VARIABLES *************************/
/********************************************************************/
size_t sz_grid_map = 0;
size_t sz_extended_idx = 0;

int* d_grid_map = NULL;
int* d_extended_idx = NULL;

/*------------------------ RESULT VARIABLES -----------------------*/
int* res_extended_idx = NULL;

/********************************************************************/
/************************** 2D MAP VARIABLES ************************/
/********************************************************************/
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


/********************************************************************/
/********************* RESIZE PARTICLES VARIABLES *******************/
/********************************************************************/
size_t sz_last_len = 0;
int* d_last_len = NULL;
int* res_last_len = NULL;

size_t sz_particles_weight = 0;
float* d_particles_weight = NULL;
float* res_particles_weight = NULL;

float* res_robot_state = NULL;
float* res_robot_world_body = NULL;

/********************************************************************/
/********************* UPDATE STATES VARIABLES **********************/
/********************************************************************/
std::vector<float> std_vec_states_x;
std::vector<float> std_vec_states_y;
std::vector<float> std_vec_states_theta;

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

/********************************************************************/
/********************* IMAGE TRANSFORM VARIABLES ********************/
/********************************************************************/
size_t sz_transition_single_frame = 0;

float* d_transition_single_world_body = NULL;
float* d_transition_single_world_lidar = NULL;
//float* d_transition_body_lidar = NULL;

/*------------------------ RESULT VARIABLES -----------------------*/
//float* res_transition_world_lidar = NULL;

/********************************************************************/
/*********************** MEASUREMENT VARIABLES **********************/
/********************************************************************/
//size_t sz_lidar_coords = 0;

//float* d_lidar_coords = NULL;

/********************************************************************/
/**************** PROCESSED MEASUREMENTS VARIABLES ******************/
/********************************************************************/
//size_t sz_processed_measure_pos = 0;

//int* d_processed_measure_x = NULL;
//int* d_processed_measure_y = NULL;

/*------------------------ RESULT VARIABLES -----------------------*/
//int* res_processed_measure_x = NULL;
//int* res_processed_measure_y = NULL;


/********************************************************************/
/******************* OCCUPIED PARTICLES VARIABLES *******************/
/********************************************************************/
size_t sz_particles_occupied_pos = 0;

int* d_particles_occupied_x = NULL;
int* d_particles_occupied_y = NULL;

/*------------------------ RESULT VARIABLES -----------------------*/
int* res_particles_occupied_x = NULL;
int* res_particles_occupied_y = NULL;


/********************************************************************/
/********************** FREE PARTICLES VARIABLES ********************/
/********************************************************************/
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

/*------------------------ RESULT VARIABLES -----------------------*/
int* res_particles_free_x = NULL;
int* res_particles_free_y = NULL;
int* res_particles_free_counter = NULL;


/********************************************************************/
/**************************** MAP VARIABLES *************************/
/********************************************************************/
//size_t sz_grid_map = 0;
//int* d_grid_map = NULL;
int* res_grid_map = NULL;

size_t sz_map_idx = 0;
int* d_map_occupied_idx = NULL;
int* d_map_free_idx = NULL;

/********************************************************************/
/************************* 2D MAP VARIABLES *************************/
/********************************************************************/
//size_t sz_map_2d = 0;
size_t sz_unique_counter = 0;
size_t sz_unique_counter_col = 0;

uint8_t* d_map_occupied_2d = NULL;
uint8_t* d_map_free_2d = NULL;

int* d_occupied_unique_counter = NULL;
int* d_occupied_unique_counter_col = NULL;
int* d_free_unique_counter = NULL;
int* d_free_unique_counter_col = NULL;

/*------------------------ RESULT VARIABLES -----------------------*/
int* res_occupied_unique_counter = NULL;
int* res_occupied_unique_counter_col = NULL;
int* res_free_unique_counter = NULL;
int* res_free_unique_counter_col = NULL;

/********************************************************************/
/************************* LOG-ODDS VARIABLES ***********************/
/********************************************************************/
size_t sz_log_odds = 0;
float* d_log_odds = NULL;

/*------------------------ RESULT VARIABLES -----------------------*/
float* res_log_odds = NULL;


/********************************************************************/
/********************************************************************/
size_t sz_position_image_body = 0;
size_t sz_particles_world_pos = 0;
size_t sz_position_image = 0;

float* d_particles_world_x = NULL;
float* d_particles_world_y = NULL;
int* d_position_image_body = NULL;

float* res_particles_world_x = NULL;
float* res_particles_world_y = NULL;
int* res_position_image_body = NULL;

int h_map_occupied_idx[] = { 0, 0 };
int h_map_free_idx[] = { 0, 0 };


size_t sz_should_extend = 0;
size_t sz_coord = 0;

int* d_should_extend = NULL;
int* d_coord = NULL;

int* res_should_extend = NULL;
int* res_coord = NULL;

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

/**
 * Allocate Memory For State
 *
 * @param[in] NUM_PARTICLES
 * @param[in] h_states_x
 * @param[in] h_states_y
 * @param[in] h_states_theta
 * @param[out] sz_states_pos
 * @param[out] d_states_x
 * @param[out] d_states_y
 * @param[out] d_states_theta
 * @param[out] res_robot_state
 * @return None
 */
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

/**
 * Allocate Memory For 'Lidar Coords'
 *
 * @param[in] h_lidar_coords
 * @param[in] LIDAR_COORDS_LEN
 * @param[out] sz_lidar_coords
 * @param[out] d_lidar_coords
 * @return None
 */
void alloc_init_lidar_coords_var(float* h_lidar_coords, const int LIDAR_COORDS_LEN) {

    sz_lidar_coords = 2 * LIDAR_COORDS_LEN * sizeof(float);
    gpuErrchk(cudaMalloc((void**)&d_lidar_coords, sz_lidar_coords));

    gpuErrchk(cudaMemcpy(d_lidar_coords, h_lidar_coords, sz_lidar_coords, cudaMemcpyHostToDevice));
}


/**
 * Allocate Memory & Initialize For 'Grid Map'
 *
 * @param[in] h_grid_map
 * @param[in] GRID_WIDTH
 * @param[in] GRID_HEIGHT
 * @param[out] sz_grid_map
 * @param[out] d_grid_map 
 * @return None
 */
void alloc_init_grid_map(int* h_grid_map, const int GRID_WIDTH, const int GRID_HEIGHT) {

    sz_grid_map = GRID_WIDTH * GRID_HEIGHT * sizeof(int);
    gpuErrchk(cudaMalloc((void**)&d_grid_map, sz_grid_map));

    gpuErrchk(cudaMemcpy(d_grid_map, h_grid_map, sz_grid_map, cudaMemcpyHostToDevice));
}

/**
 * Allocate Memory & Initialize For 'Particles State'
 *
 * @param[in] NUM_PARTICLES
 * @param[in] PARTICLES_ITEMS_LEN
 * @param[in] h_particles_x
 * @param[in] h_particles_y
 * @param[in] h_particles_idx
 * @param[in] h_particles_weight
 * @param[out] sz_particles_pos
 * @param[out] sz_particles_idx
 * @param[out] sz_particles_weight
 * @param[out] d_particles_x
 * @param[out] d_particles_y
 * @param[out] d_particles_idx
 * @param[out] d_particles_weight
 * @param[out] res_particles_idx
 * @return None
 */
void alloc_init_particles_vars(int* h_particles_x, int* h_particles_y, int* h_particles_idx, float* h_particles_weight, const int PARTICLES_ITEMS_LEN) {

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

/**
 * Allocate Memory & Initialize For 'Extended Index'
 *
 * @param[in] PARTICLES_ITEMS_LEN
 * @param[out] sz_extended_idx
 * @param[out] d_extended_idx
 * @return None
 */
void alloc_extended_idx() {

    sz_extended_idx = PARTICLES_ITEMS_LEN * sizeof(int);
    gpuErrchk(cudaMalloc((void**)&d_extended_idx, sz_extended_idx));

    res_extended_idx = (int*)malloc(sz_extended_idx);
}


/**
 * Allocate Memory For Copy of 'State'
 *
 * @param[in] sz_states_pos
 * @param[out] dc_states_x
 * @param[out] dc_states_y
 * @param[out] dc_states_theta
 * @return None
 */
void alloc_states_copy_vars() {

    gpuErrchk(cudaMalloc((void**)&dc_states_x, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&dc_states_y, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&dc_states_theta, sz_states_pos));
}

/**
 * Allocate Memory For 'Correlation'
 *
 * @param[in] NUM_PARTICLES
 * @param[out] sz_weights
 * @param[out] sz_weights_raw
 * @param[out] d_weights
 * @param[out] d_weights_raw
 * @param[out] res_weights
 * @return None
 */
void alloc_correlation_vars() {

    sz_correlation_weights = NUM_PARTICLES * sizeof(float);
    sz_correlation_weights_raw = 25 * sz_correlation_weights;

    gpuErrchk(cudaMalloc((void**)&d_correlation_weights, sz_correlation_weights));
    gpuErrchk(cudaMalloc((void**)&d_correlation_weights_raw, sz_correlation_weights_raw));
    gpuErrchk(cudaMemset(d_correlation_weights_raw, 0, sz_correlation_weights_raw));

    res_correlation_weights = (float*)malloc(sz_correlation_weights);
    memset(res_correlation_weights, 0, sz_correlation_weights);
}

/**
 * Allocate Memory & Initialize For 'Transition'
 *
 * @param[in] NUM_PARTICLES
 * @param[out] sz_transition_world_frame
 * @param[out] sz_transition_body_lidar
 * @param[out] d_transition_world_body
 * @param[out] d_transition_body_lidar
 * @param[out] d_transition_world_lidar
 * @param[out] res_transition_world_body
 * @param[out] res_robot_world_body
 * @return None
 */
void alloc_init_transition_vars(float* h_transition_body_lidar) {

    sz_transition_world_frame = 9 * NUM_PARTICLES * sizeof(float);
    sz_transition_body_lidar = 9 * sizeof(float);
    sz_transition_single_frame = 9 * sizeof(float);

    gpuErrchk(cudaMalloc((void**)&d_transition_world_body, sz_transition_world_frame));
    gpuErrchk(cudaMalloc((void**)&d_transition_world_lidar, sz_transition_world_frame));
    gpuErrchk(cudaMalloc((void**)&d_transition_body_lidar, sz_transition_body_lidar));

    gpuErrchk(cudaMalloc((void**)&d_transition_single_world_lidar, sz_transition_single_frame));
    gpuErrchk(cudaMalloc((void**)&d_transition_single_world_body, sz_transition_single_frame));

    gpuErrchk(cudaMemcpy(d_transition_body_lidar, h_transition_body_lidar, sz_transition_body_lidar, cudaMemcpyHostToDevice));

    //gpuErrchk(cudaMemcpy(d_transition_body_lidar, h_transition_body_lidar, sz_transition_single_frame, cudaMemcpyHostToDevice));
    //gpuErrchk(cudaMemcpy(d_transition_world_body, h_transition_world_body, sz_transition_single_frame, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemset(d_transition_world_body, 0, sz_transition_world_frame));
    gpuErrchk(cudaMemset(d_transition_world_lidar, 0, sz_transition_world_frame));

    res_transition_world_body = (float*)malloc(sz_transition_world_frame);
    res_robot_world_body = (float*)malloc(sz_transition_world_frame);
}

/**
 * Allocate Memory For 'Processed Measurement'
 *
 * @param[in] NUM_PARTICLES
 * @param[in] LIDAR_COORDS_LEN
 * @param[out] sz_processed_measure_pos
 * @param[out] sz_processed_measure_idx
 * @param[out] d_processed_measure_x
 * @param[out] d_processed_measure_y
 * @param[out] d_processed_measure_idx
 * @return None
 */
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


/**
 * Allocate Memory For '2D Grid Map'
 *
 * @param[in] NUM_PARTICLES
 * @param[in] GRID_WIDTH
 * @param[in] GRID_HEIGHT
 * @param[out] sz_map_2d
 * @param[out] d_map_2d
 * @return None
 */
void alloc_map_2d_var(const int GRID_WIDTH, const int GRID_HEIGHT) {

    sz_map_2d = GRID_WIDTH * GRID_HEIGHT * NUM_PARTICLES * sizeof(uint8_t);

    gpuErrchk(cudaMalloc((void**)&d_map_2d, sz_map_2d));
    gpuErrchk(cudaMemset(d_map_2d, 0, sz_map_2d));
}

/**
 * Allocate Memory For 'Unique Counter'
 *
 * @param[in] NUM_PARTICLES
 * @param[in] UNIQUE_COUNTER_LEN
 * @param[in] GRID_WIDTH
 * @param[out] sz_unique_in_particle
 * @param[out] sz_unique_in_particle_col
 * @param[out] d_unique_in_particle
 * @param[out] d_unique_in_particle_col
 * @param[out] res_unique_in_particle
 * @return None
 */
void alloc_map_2d_unique_counter_vars(const int UNIQUE_COUNTER_LEN, const int GRID_WIDTH) {

    sz_unique_in_particle = UNIQUE_COUNTER_LEN * sizeof(int);
    sz_unique_in_particle_col = UNIQUE_COUNTER_LEN * GRID_WIDTH * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_unique_in_particle, sz_unique_in_particle));
    gpuErrchk(cudaMalloc((void**)&d_unique_in_particle_col, sz_unique_in_particle_col));

    gpuErrchk(cudaMemset(d_unique_in_particle, 0, sz_unique_in_particle));
    gpuErrchk(cudaMemset(d_unique_in_particle_col, 0, sz_unique_in_particle_col));

    res_unique_in_particle = (int*)malloc(sz_unique_in_particle);
}

/**
 * Allocate Memory For 'Correlation Weights'
 *
 * @param[out] sz_correlation_weights_max
 * @param[out] sz_correlation_sum_exp
 * @param[out] d_correlation_weights_max
 * @param[out] d_correlation_sum_exp
 * @param[out] res_correlation_weights_max
 * @param[out] res_correlation_sum_exp
 * @return None
 */
void alloc_correlation_weights_vars() {

    sz_correlation_weights_max = sizeof(float);
    sz_correlation_sum_exp = sizeof(double);

    gpuErrchk(cudaMalloc((void**)&d_correlation_weights_max, sz_correlation_weights_max));
    gpuErrchk(cudaMalloc((void**)&d_correlation_sum_exp, sz_correlation_sum_exp));

    gpuErrchk(cudaMemset(d_correlation_weights_max, 0, sz_correlation_weights_max));
    gpuErrchk(cudaMemset(d_correlation_sum_exp, 0, sz_correlation_sum_exp));

    res_correlation_weights_max = (float*)malloc(sz_correlation_weights_max);
    res_correlation_sum_exp = (double*)malloc(sz_correlation_sum_exp);
}

/**
 * Allocate Memory For 'Resampling'
 *
 * @param[out] sz_resampling_js
 * @param[out] sz_resampling_rnd
 * @param[out] d_resampling_js
 * @param[out] d_resampling_rnd
 * @return None
 */
void alloc_resampling_vars(float* h_resampling_rnds) {

    sz_resampling_js = NUM_PARTICLES * sizeof(int);
    sz_resampling_rnd = NUM_PARTICLES * sizeof(float);

    gpuErrchk(cudaMalloc((void**)&d_resampling_js, sz_resampling_js));
    gpuErrchk(cudaMalloc((void**)&d_resampling_rnd, sz_resampling_rnd));

    gpuErrchk(cudaMemcpy(d_resampling_rnd, h_resampling_rnds, sz_resampling_rnd, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(d_resampling_js, 0, sz_resampling_js));
}

/**
 * Execution of Transition
 *
 * @param[in] NUM_PARTICLES
 * @param[in] d_states_x
 * @param[in] d_states_y
 * @param[in] d_states_theta
 * @param[in] d_transition_body_lidar
 * @param[out] d_transition_world_lidar
 * @param[out] res_transition_world_body 
 * @return None
 */
void exec_calc_transition() {

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_update_particles_states << <blocksPerGrid, threadsPerBlock >> > (d_transition_world_body, d_transition_world_lidar, SEP,
        d_states_x, d_states_y, d_states_theta,
        d_transition_body_lidar, NUM_PARTICLES);
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(res_transition_world_body, d_transition_world_body, sz_transition_world_frame, cudaMemcpyDeviceToHost));
}

/**
 * Process Measurements
 *
 * @param[in] NUM_PARTICLES
 * @param[in] LIDAR_COORDS_LEN
 * @param[in] d_transition_world_lidar
 * @param[in] d_lidar_coords
 * @param[in] res
 * @param[in] xmin
 * @param[in] ymax
 * @param[out] d_processed_measure_x
 * @param[out] d_processed_measure_y
 * @param[out] d_processed_measure_idx
 * @return None
 */
void exec_process_measurements() {

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = LIDAR_COORDS_LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (d_processed_measure_x, d_processed_measure_y, SEP,
        d_transition_world_lidar, d_lidar_coords, res, xmin, ymax, LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_index_init_const << < blocksPerGrid, threadsPerBlock >> > (d_processed_measure_idx, LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, d_processed_measure_idx, d_processed_measure_idx + NUM_PARTICLES, d_processed_measure_idx, 0);
}

/**
 * Create 2D MAP
 *
 * @param[in] NUM_PARTICLES
 * @param[in] PARTICLES_ITEMS_LEN
 * @param[in] GRID_WIDTH
 * @param[in] GRID_HEIGHT
 * @param[in] d_particles_x
 * @param[in] d_particles_y
 * @param[in] d_particles_idx
 * @param[out] d_map_2d
 * @param[out] d_unique_in_particle
 * @param[out] d_unique_in_particle_col
 * @return None
 */
void exec_create_map_2d() {

    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (d_map_2d, d_unique_in_particle, d_unique_in_particle_col, SEP,
        d_particles_x, d_particles_y, d_particles_idx, PARTICLES_ITEMS_LEN, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();
}

/**
 * Update 2D MAP
 *
 * @param[in] NUM_PARTICLES
 * @param[in] MEASURE_LEN
 * @param[in] GRID_WIDTH
 * @param[in] GRID_HEIGHT
 * @param[in] d_processed_measure_x
 * @param[in] d_processed_measure_y
 * @param[in] d_processed_measure_idx
 * @param[out] d_map_2d
 * @param[out] d_unique_in_particle
 * @param[out] d_unique_in_particle_col
 * @return None
 */
void exec_update_map_2d() {

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;

    kernel_update_2d_map_with_measure << <blocksPerGrid, threadsPerBlock >> > (d_map_2d, d_unique_in_particle, d_unique_in_particle_col, SEP,
        d_processed_measure_x, d_processed_measure_y, d_processed_measure_idx,
        MEASURE_LEN, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();
}

/**
 * Particle Unique Cumulative Sum
 *
 * @param[in] UNIQUE_COUNTER_LEN
 * @param[in] GRID_WIDTH
 * @param[in] sz_unique_in_particle
 * @param[out] d_unique_in_particle
 * @param[out] d_unique_in_particle_col
 * @param[out] res_unique_in_particle
 * @param[out] PARTICLES_ITEMS_LEN
 * @param[out] C_PARTICLES_ITEMS_LEN
 * @return None
 */
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

/**
 * Reinitialize Particles
 *
 * @param[in] PARTICLES_ITEMS_LEN
 * @param[out] sz_particles_pos
 * @param[out] sz_extended_idx
 * @param[out] d_particles_x
 * @param[out] d_particles_y
 * @param[out] d_extended_idx
 * @return None
 */
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

/**
 * MAP Restructure
 *
 * @param[in] NUM_PARTICLES
 * @param[in] GRID_WIDTH
 * @param[in] GRID_HEIGHT
 * @param[in] d_map_2d
 * @param[in] d_unique_in_particle
 * @param[in] d_unique_in_particle_col
 * @param[out] d_particles_x
 * @param[out] d_particles_y
 * @param[out] d_extended_idx
 * @return None
 */
void exec_map_restructure() {

    threadsPerBlock = GRID_WIDTH;
    blocksPerGrid = NUM_PARTICLES;

    cudaMemset(d_particles_idx, 0, sz_particles_idx);
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (d_particles_x, d_particles_y, d_particles_idx, SEP,
        d_map_2d, d_unique_in_particle, d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, d_particles_idx, d_particles_idx + NUM_PARTICLES, d_particles_idx, 0);
}

/**
 * Index Expansion
 *
 * @param[in] NUM_PARTICLES
 * @param[in] PARTICLES_ITEMS_LEN
 * @param[in] d_particles_idx
 * @param[in] sz_extended_idx
 * @param[in] sz_particles_idx
 * @param[out] d_extended_idx
 * @param[out] res_extended_idx
 * @param[out] res_particles_idx
 * @return None
 */
void exec_index_expansion() {

    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;
    kernel_index_expansion << <blocksPerGrid, threadsPerBlock >> > (d_extended_idx, d_particles_idx, PARTICLES_ITEMS_LEN);
    cudaDeviceSynchronize();

    res_extended_idx = (int*)malloc(sz_extended_idx);
    gpuErrchk(cudaMemcpy(res_extended_idx, d_extended_idx, sz_extended_idx, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToHost));
}

/**
 * Correlation Execution
 *
 * @param[in] NUM_PARTICLES
 * @param[in] PARTICLES_ITEMS_LEN
 * @param[in] GRID_WIDTH
 * @param[in] GRID_HEIGHT
 * @param[in] d_grid_map
 * @param[in] d_particles_x
 * @param[in] d_particles_y
 * @param[in] d_extended_idx
 * @param[out] d_correlation_weights_raw
 * @param[out] d_correlation_weights
 * @return None
 */
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
}

/**
 * Update Weights
 *
 * @param[in] NUM_PARTICLES
 * @param[in] sz_weights_max
 * @param[in] sz_sum_exp
 * @param[in] d_weights_max
 * @param[out] d_weights
 * @param[out] d_particles_weight
 * @return None
 */
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

/**
 * Resampling
 *
 * @param[in] NUM_PARTICLES
 * @param[in] d_weights
 * @param[in] d_rnd
 * @param[out] d_js
 * @return None
 */
void exec_resampling() {

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_resampling << <blocksPerGrid, threadsPerBlock >> > (d_resampling_js, d_correlation_weights, d_resampling_rnd, NUM_PARTICLES);
    cudaDeviceSynchronize();
}


/**
 * Reinitializing Particles
 *
 * @param[in] NUM_PARTICLES
 * @param[in] sz_last_len
 * @param[in] sz_particles_pos
 * @param[in] sz_particles_idx
 * @param[in] sz_states_pos
 * @param[in] d_particles_x
 * @param[in] d_particles_y
 * @param[in] d_particles_idx
 * @param[in] d_states_x
 * @param[in] d_states_y
 * @param[in] d_states_theta
 * @param[out] dc_particles_x
 * @param[out] dc_particles_y
 * @param[out] dc_particles_idx
 * @param[out] dc_states_x
 * @param[out] dc_states_y
 * @param[out] dc_states_theta
 * @param[out] d_last_len
 * @return None
 */
void reinit_particles_vars() {

    sz_last_len = sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_last_len, sz_last_len));
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
        dc_particles_idx, d_resampling_js,  PARTICLES_ITEMS_LEN);
    cudaDeviceSynchronize();

    res_last_len = (int*)malloc(sz_last_len);

    gpuErrchk(cudaMemcpy(res_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_last_len, d_last_len, sz_last_len, cudaMemcpyDeviceToHost));
}

/**
 * Rearrangement
 *
 * @param[in] NUM_PARTICLES
 * @param[in] GRID_WIDTH  &  GRID_HEIGHT
 * @param[in] res_last_len
 * @param[in] dc_states_x  &  dc_states_y  &  dc_states_theta
 * @param[in] d_js
 * @param[out] C_PARTICLES_ITEMS_LEN  &  PARTICLES_ITEMS_LEN
 * @param[out] d_particles_x  &  d_particles_y
 * @param[out] d_states_x  &  d_states_y  &  d_states_theta
 * @return None
 */
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

/**
 * Rearrangement
 *
 * @param[in] NUM_PARTICLES
 * @param[in] d_states_x
 * @param[in] d_states_y
 * @param[in] d_states_theta
 * @param[out] res_robot_world_body
 * @param[out] res_robot_state
 * @return None
 */
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

void assertResults() {

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

void test_robot_particles_main() {

    const int data_len = 6;
    int data[data_len] = { 1, 0, 2, 2, 1, 3 };
    int* d_data = NULL;
    gpuErrchk(cudaMalloc((void**)&d_data, data_len * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_data, data, data_len * sizeof(int), cudaMemcpyHostToDevice));

    thrust::exclusive_scan(thrust::host, data, data + 6, data, 0);
    thrust::exclusive_scan(thrust::device, d_data, d_data + 6, d_data, 0);


    printf("/********************************************************************/\n");
    printf("/****************************** ROBOT  ******************************/\n");
    printf("/********************************************************************/\n");

    GRID_WIDTH = ST_GRID_WIDTH;
    GRID_HEIGHT = ST_GRID_HEIGHT;
    xmin = ST_xmin;
    ymax = ST_ymax;
    LIDAR_COORDS_LEN = ST_LIDAR_COORDS_LEN;

    LIDAR_COORDS_LEN = ST_LIDAR_COORDS_LEN;
    PARTICLES_ITEMS_LEN = ST_PARTICLES_ITEMS_LEN;
    MEASURE_LEN = NUM_PARTICLES * LIDAR_COORDS_LEN;

    int negative_before_counter = getNegativeCounter(h_particles_x, h_particles_y, PARTICLES_ITEMS_LEN);
    int count_bigger_than_height = getGreaterThanCounter(h_particles_y, GRID_HEIGHT, PARTICLES_ITEMS_LEN);
    int negative_after_counter = getNegativeCounter(h_particles_x_after_resampling, h_particles_y_after_resampling, AF_PARTICLES_ITEMS_LEN_RESAMPLING);;

    printf("~~$ GRID_WIDTH: \t\t%d\n", GRID_WIDTH);
    printf("~~$ GRID_HEIGHT: \t\t%d\n", GRID_HEIGHT);
    printf("~~$ negative_before_counter: \t%d\n", negative_before_counter);
    printf("~~$ negative_after_counter: \t%d\n", negative_after_counter);
    printf("~~$ count_bigger_than_height: \t%d\n", count_bigger_than_height);

    auto start_robot_particles_alloc = std::chrono::high_resolution_clock::now();
    alloc_init_state_vars(h_states_x, h_states_y, h_states_theta);
    alloc_init_lidar_coords_var(h_lidar_coords, LIDAR_COORDS_LEN);
    alloc_init_grid_map(h_grid_map, GRID_WIDTH, GRID_HEIGHT);
    alloc_init_particles_vars(h_particles_x, h_particles_y, h_particles_idx, particles_weight_pre, PARTICLES_ITEMS_LEN);
    alloc_extended_idx();
    alloc_states_copy_vars();
    alloc_correlation_vars();
    alloc_init_transition_vars(h_transition_body_lidar);
    alloc_init_processed_measurement_vars(LIDAR_COORDS_LEN);
    alloc_map_2d_var(GRID_WIDTH, GRID_HEIGHT);
    alloc_map_2d_unique_counter_vars(UNIQUE_COUNTER_LEN, GRID_WIDTH);
    alloc_correlation_weights_vars();
    alloc_resampling_vars(h_rnds);
    auto stop_robot_particles_alloc = std::chrono::high_resolution_clock::now();

    auto start_robot_particles_init = std::chrono::high_resolution_clock::now();
    //init_state_vars();
    //init_particles_vars();
    //init_transition_vars();
    auto stop_robot_particles_init = std::chrono::high_resolution_clock::now();

    auto start_robot_particles_kernel = std::chrono::high_resolution_clock::now();
    exec_calc_transition();
    exec_process_measurements();
    exec_create_map_2d();
    exec_update_map_2d();
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

    assertResults();

    auto duration_robot_particles_alloc = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_particles_alloc - start_robot_particles_alloc);
    auto duration_robot_particles_init = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_particles_init - start_robot_particles_init);
    auto duration_robot_particles_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_particles_kernel - start_robot_particles_kernel);
    auto duration_robot_particles_total = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_particles_kernel - start_robot_particles_alloc);

    std::cout << std::endl;
    std::cout << "Time taken by function (Allocation): " << duration_robot_particles_alloc.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Initialization): " << duration_robot_particles_init.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Kernel): " << duration_robot_particles_kernel.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Total): " << duration_robot_particles_total.count() << " microseconds" << std::endl;
}

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

void alloc_image_transform_vars(int LIDAR_COORDS_LEN) {

    sz_particles_world_pos = LIDAR_COORDS_LEN * sizeof(float);

    //gpuErrchk(cudaMalloc((void**)&d_lidar_coords, sz_lidar_coords));
    //gpuErrchk(cudaMalloc((void**)&d_transition_body_lidar, sz_transition_single_frame));
    sz_position_image = 2 * sizeof(int);
    sz_position_image_body = 2 * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_position_image_body, sz_position_image_body));

    gpuErrchk(cudaMalloc((void**)&d_particles_world_x, sz_particles_world_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_world_y, sz_particles_world_pos));
}

void init_image_transform_vars() {

    //gpuErrchk(cudaMemcpy(d_lidar_coords, h_lidar_coords, sz_lidar_coords, cudaMemcpyHostToDevice));
}

void alloc_particles_free_vars() {

    sz_particles_free_pos = 0;
    sz_particles_free_pos_max = PARTICLES_OCCUPIED_LEN * MAX_DIST_IN_MAP * sizeof(int);
    sz_particles_free_idx = PARTICLES_OCCUPIED_LEN * sizeof(int);
    sz_particles_free_counter = PARTICLE_UNIQUE_COUNTER * sizeof(int);
    
    gpuErrchk(cudaMalloc((void**)&d_particles_free_x_max, sz_particles_free_pos_max));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_y_max, sz_particles_free_pos_max));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_counter, sz_particles_free_counter));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_idx, sz_particles_free_idx));

    gpuErrchk(cudaMemset(d_particles_free_x, 0, sz_particles_free_pos));
    gpuErrchk(cudaMemset(d_particles_free_y, 0, sz_particles_free_pos));
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

//void alloc_map_vars() {
//
//    sz_grid_map = (GRID_WIDTH * GRID_HEIGHT) * sizeof(int);
//    gpuErrchk(cudaMalloc((void**)&d_grid_map, sz_grid_map));
//}
//
//void init_map_vars() {
//
//    gpuErrchk(cudaMemcpy(d_grid_map, h_grid_map, sz_grid_map, cudaMemcpyHostToDevice));
//}

void alloc_log_odds_vars() {

    sz_map_2d = GRID_WIDTH * GRID_HEIGHT * sizeof(uint8_t);
    sz_unique_counter = 1 * sizeof(int);
    sz_unique_counter_col = (GRID_WIDTH + 1) * sizeof(int);
    sz_log_odds = (GRID_WIDTH * GRID_HEIGHT) * sizeof(float);
    sz_map_idx = 2 * sizeof(int);

    res_occupied_unique_counter = (int*)malloc(sz_unique_counter);
    res_free_unique_counter = (int*)malloc(sz_unique_counter);
    res_grid_map = (int*)malloc(sz_grid_map);
    res_log_odds = (float*)malloc(sz_log_odds);

    gpuErrchk(cudaMalloc((void**)&d_map_occupied_2d, sz_map_2d));
    gpuErrchk(cudaMalloc((void**)&d_map_free_2d, sz_map_2d));
    gpuErrchk(cudaMalloc((void**)&d_occupied_unique_counter, sz_unique_counter));
    gpuErrchk(cudaMalloc((void**)&d_occupied_unique_counter_col, sz_unique_counter_col));
    gpuErrchk(cudaMalloc((void**)&d_free_unique_counter, sz_unique_counter));
    gpuErrchk(cudaMalloc((void**)&d_free_unique_counter_col, sz_unique_counter_col));
    gpuErrchk(cudaMalloc((void**)&d_log_odds, sz_log_odds));

    gpuErrchk(cudaMalloc((void**)&d_map_occupied_idx, sz_map_idx));
    gpuErrchk(cudaMalloc((void**)&d_map_free_idx, sz_map_idx));
}

//void init_log_odds_vars() {
//
//    h_map_occupied_idx[1] = PARTICLES_OCCUPIED_LEN;
//    h_map_free_idx[1] = 0;
//
//    memset(res_log_odds, 0, sz_log_odds);
//
//    gpuErrchk(cudaMemcpy(d_map_occupied_idx, h_map_occupied_idx, sz_map_idx, cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemcpy(d_map_free_idx, h_map_free_idx, sz_map_idx, cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemcpy(d_log_odds, h_log_odds, sz_log_odds, cudaMemcpyHostToDevice));
//}
//
//void exec_world_to_image_transform_step_1() {
//
//    kernel_matrix_mul_3x3 << < 1, 1 >> > (d_transition_world_body, d_transition_body_lidar, d_transition_world_lidar);
//    cudaDeviceSynchronize();
//
//    threadsPerBlock = 1;
//    blocksPerGrid = LIDAR_COORDS_LEN;
//    //kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (d_transition_world_lidar, d_particles_occupied_x, d_particles_occupied_y,
//    //    d_particles_world_x, d_particles_world_y, d_lidar_coords, res, xmin, ymax, LIDAR_COORDS_LEN);
//    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (d_particles_world_x, d_particles_world_y, d_transition_world_lidar,
//        d_lidar_coords, LIDAR_COORDS_LEN);
//    cudaDeviceSynchronize();
//
//    kernel_position_to_image << < 1, 1 >> > (d_position_image_body, d_transition_world_lidar, res, xmin, ymax);
//    cudaDeviceSynchronize();
//}
//
//void exec_map_extend() {
//
//    int xmin_pre = xmin;
//    int ymax_pre = ymax;
//
//    sz_should_extend = 4 * sizeof(int);
//
//    gpuErrchk(cudaMalloc((void**)&d_should_extend, sz_should_extend));
//    res_should_extend = (int*)malloc(sz_should_extend);
//    memset(res_should_extend, 0, sz_should_extend);
//    gpuErrchk(cudaMemset(d_should_extend, 0, sz_should_extend));
//
//    threadsPerBlock = 256;
//    blocksPerGrid = (LIDAR_COORDS_LEN + threadsPerBlock - 1) / threadsPerBlock;
//    kernel_check_map_extend_less << <blocksPerGrid, threadsPerBlock >> > (d_particles_world_x, xmin, d_should_extend, 0, 0, LIDAR_COORDS_LEN);
//    kernel_check_map_extend_less << <blocksPerGrid, threadsPerBlock >> > (d_particles_world_y, ymin, d_should_extend, 1, 0, LIDAR_COORDS_LEN);
//
//    kernel_check_map_extend_greater << <blocksPerGrid, threadsPerBlock >> > (d_particles_world_x, xmax, d_should_extend, 2, 0, LIDAR_COORDS_LEN);
//    kernel_check_map_extend_greater << <blocksPerGrid, threadsPerBlock >> > (d_particles_world_y, ymax, d_should_extend, 3, 0, LIDAR_COORDS_LEN);
//    cudaDeviceSynchronize();
//    gpuErrchk(cudaMemcpy(res_should_extend, d_should_extend, sz_should_extend, cudaMemcpyDeviceToHost));
//
//    bool EXTEND = false;
//    if (res_should_extend[0] != 0) {
//        EXTEND = true;
//        xmin = xmin * 2;
//    }
//    else if (res_should_extend[2] != 0) {
//        EXTEND = true;
//        xmax = xmax * 2;
//    }
//    else if (res_should_extend[1] != 0) {
//        EXTEND = true;
//        ymin = ymin * 2;
//    }
//    else if (res_should_extend[3] != 0) {
//        EXTEND = true;
//        ymax = ymax * 2;
//    }
//
//    //printf("EXTEND = %d\n", EXTEND);
//
//    if (EXTEND == true) {
//
//        sz_coord = 2 * sizeof(int);
//        gpuErrchk(cudaMalloc((void**)&d_coord, sz_coord));
//        res_coord = (int*)malloc(sz_coord);
//
//        kernel_position_to_image << <1, 1 >> > (d_coord, xmin_pre, ymax_pre, res, xmin, ymax);
//        cudaDeviceSynchronize();
//
//        gpuErrchk(cudaMemcpy(res_coord, d_coord, sz_coord, cudaMemcpyDeviceToHost));
//
//        int* dc_grid_map = NULL;
//        gpuErrchk(cudaMalloc((void**)&dc_grid_map, sz_grid_map));
//        gpuErrchk(cudaMemcpy(dc_grid_map, d_grid_map, sz_grid_map, cudaMemcpyDeviceToDevice));
//
//        float* dc_log_odds = NULL;
//        gpuErrchk(cudaMalloc((void**)&dc_log_odds, sz_log_odds));
//        gpuErrchk(cudaMemcpy(dc_log_odds, d_log_odds, sz_log_odds, cudaMemcpyDeviceToDevice));
//
//        const int PRE_GRID_WIDTH = GRID_WIDTH;
//        const int PRE_GRID_HEIGHT = GRID_HEIGHT;
//        GRID_WIDTH = ceil((ymax - ymin) / res + 1);
//        GRID_HEIGHT = ceil((xmax - xmin) / res + 1);
//        //printf("GRID_WIDTH=%d, GRID_HEIGHT=%d, PRE_GRID_WIDTH=%d, PRE_GRID_HEIGHT=%d\n", GRID_WIDTH, GRID_HEIGHT, PRE_GRID_WIDTH, PRE_GRID_HEIGHT);
//        assert(GRID_WIDTH == AF_GRID_WIDTH);
//        assert(GRID_HEIGHT == AF_GRID_HEIGHT);
//
//        const int PRE_GRID_SIZE = PRE_GRID_WIDTH * PRE_GRID_HEIGHT;
//        const int NEW_GRID_SIZE = GRID_WIDTH * GRID_HEIGHT;
//
//        //gpuErrchk(cudaFree(d_grid_map));
//
//        sz_grid_map = GRID_WIDTH * GRID_HEIGHT * sizeof(int);
//        gpuErrchk(cudaMalloc((void**)&d_grid_map, sz_grid_map));
//        gpuErrchk(cudaMemset(d_grid_map, 0, sz_grid_map));
//
//        sz_log_odds = GRID_WIDTH * GRID_HEIGHT * sizeof(float);
//        gpuErrchk(cudaMalloc((void**)&d_log_odds, sz_log_odds));
//
//        threadsPerBlock = 256;
//        blocksPerGrid = (NEW_GRID_SIZE + threadsPerBlock - 1) / threadsPerBlock;
//        kernel_index_arr_const << <blocksPerGrid, threadsPerBlock >> > (d_log_odds, LOG_ODD_PRIOR, NEW_GRID_SIZE);
//        cudaDeviceSynchronize();
//
//        res_grid_map = (int*)malloc(sz_grid_map);
//        res_log_odds = (float*)malloc(sz_log_odds);
//
//        threadsPerBlock = 256;
//        blocksPerGrid = (PRE_GRID_SIZE + threadsPerBlock - 1) / threadsPerBlock;
//        kernel_2d_copy_with_offset << <blocksPerGrid, threadsPerBlock >> > (d_grid_map, d_log_odds, dc_grid_map, dc_log_odds, res_coord[0], res_coord[1],
//            PRE_GRID_HEIGHT, GRID_HEIGHT, PRE_GRID_SIZE);
//        cudaDeviceSynchronize();
//
//        sz_unique_counter_col = (GRID_WIDTH + 1) * sizeof(int);
//        gpuErrchk(cudaMalloc((void**)&d_occupied_unique_counter_col, sz_unique_counter_col));
//        gpuErrchk(cudaMalloc((void**)&d_free_unique_counter_col, sz_unique_counter_col));
//        res_occupied_unique_counter_col = (int*)malloc(sz_unique_counter_col);
//        res_free_unique_counter_col = (int*)malloc(sz_unique_counter_col);
//
//        sz_map_2d = GRID_WIDTH * GRID_HEIGHT * sizeof(uint8_t);
//        gpuErrchk(cudaMalloc((void**)&d_map_occupied_2d, sz_map_2d));
//        gpuErrchk(cudaMalloc((void**)&d_map_free_2d, sz_map_2d));
//
//        gpuErrchk(cudaMemcpy(res_grid_map, d_grid_map, sz_grid_map, cudaMemcpyDeviceToHost));
//        gpuErrchk(cudaMemcpy(res_log_odds, d_log_odds, sz_log_odds, cudaMemcpyDeviceToHost));
//    }
//}
//
//void exec_world_to_image_transform_step_2() {
//
//    threadsPerBlock = 1;
//    blocksPerGrid = LIDAR_COORDS_LEN;
//    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (d_transition_world_lidar, d_particles_occupied_x, d_particles_occupied_y,
//        d_lidar_coords, res, xmin, ymax, LIDAR_COORDS_LEN);
//    cudaDeviceSynchronize();
//
//    kernel_position_to_image << < 1, 1 >> > (d_position_image_body, d_transition_world_lidar, res, xmin, ymax);
//    cudaDeviceSynchronize();
//}

void exec_bresenham() {

    threadsPerBlock = 256;
    blocksPerGrid = (PARTICLES_OCCUPIED_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_bresenham << <blocksPerGrid, threadsPerBlock >> > (d_particles_occupied_x, d_particles_occupied_y, d_position_image_body,
        d_particles_free_x_max, d_particles_free_y_max, d_particles_free_counter, PARTICLES_OCCUPIED_LEN, MAX_DIST_IN_MAP);
    cudaDeviceSynchronize();
    thrust::exclusive_scan(thrust::device, d_particles_free_counter, d_particles_free_counter + PARTICLE_UNIQUE_COUNTER, d_particles_free_counter, 0);

    gpuErrchk(cudaMemcpy(res_particles_free_counter, d_particles_free_counter, sz_particles_free_counter, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(d_particles_free_idx, d_particles_free_counter, sz_particles_free_idx, cudaMemcpyDeviceToDevice));

    PARTICLES_FREE_LEN = res_particles_free_counter[PARTICLE_UNIQUE_COUNTER - 1];
    sz_particles_free_pos = PARTICLES_FREE_LEN * sizeof(int);
    gpuErrchk(cudaMalloc((void**)&d_particles_free_x, sz_particles_free_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_y, sz_particles_free_pos));

    kernel_bresenham_rearrange << <blocksPerGrid, threadsPerBlock >> > (d_particles_free_x, d_particles_free_y, d_particles_free_x_max, d_particles_free_y_max,
        d_particles_free_counter, MAX_DIST_IN_MAP, PARTICLES_OCCUPIED_LEN);
    cudaDeviceSynchronize();
}

//void reinit_map_idx_vars() {
//
//    h_map_occupied_idx[1] = PARTICLES_OCCUPIED_LEN;
//    h_map_free_idx[1] = PARTICLES_FREE_LEN;
//
//    gpuErrchk(cudaMemcpy(d_map_occupied_idx, h_map_occupied_idx, sz_map_idx, cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemcpy(d_map_free_idx, h_map_free_idx, sz_map_idx, cudaMemcpyHostToDevice));
//}
//
//void exec_create_map() {
//
//    threadsPerBlock = 256;
//    blocksPerGrid = 1;
//    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (d_particles_occupied_x, d_particles_occupied_y, d_map_occupied_idx,
//        PARTICLES_OCCUPIED_LEN, d_map_occupied_2d, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
//    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (d_particles_free_x, d_particles_free_y, d_map_free_idx,
//        PARTICLES_FREE_LEN, d_map_free_2d, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
//    cudaDeviceSynchronize();
//
//    threadsPerBlock = GRID_WIDTH;
//    blocksPerGrid = 1;
//    kernel_2d_map_counter << <blocksPerGrid, threadsPerBlock >> > (d_map_occupied_2d, d_occupied_unique_counter, d_occupied_unique_counter_col, GRID_WIDTH, GRID_HEIGHT);
//    kernel_2d_map_counter << <blocksPerGrid, threadsPerBlock >> > (d_map_free_2d, d_free_unique_counter, d_free_unique_counter_col, GRID_WIDTH, GRID_HEIGHT);
//    cudaDeviceSynchronize();
//
//    threadsPerBlock = 1;
//    blocksPerGrid = 1;
//    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (d_occupied_unique_counter_col, GRID_WIDTH);
//    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (d_free_unique_counter_col, GRID_WIDTH);
//    cudaDeviceSynchronize();
//
//    gpuErrchk(cudaMemcpy(res_occupied_unique_counter, d_occupied_unique_counter, sz_unique_counter, cudaMemcpyDeviceToHost));
//    gpuErrchk(cudaMemcpy(res_free_unique_counter, d_free_unique_counter, sz_unique_counter, cudaMemcpyDeviceToHost));
//}
//
//void reinit_map_vars() {
//
//    PARTICLES_OCCUPIED_UNIQUE_LEN = res_occupied_unique_counter[0];
//    PARTICLES_FREE_UNIQUE_LEN = res_free_unique_counter[0];
//
//    //gpuErrchk(cudaFree(d_particles_occupied_x));
//    //gpuErrchk(cudaFree(d_particles_occupied_y));
//    //gpuErrchk(cudaFree(d_particles_free_x));
//    //gpuErrchk(cudaFree(d_particles_free_y));
//
//    sz_particles_occupied_pos = PARTICLES_OCCUPIED_UNIQUE_LEN * sizeof(int);
//    sz_particles_free_pos = PARTICLES_FREE_UNIQUE_LEN * sizeof(int);
//
//    gpuErrchk(cudaMalloc((void**)&d_particles_occupied_x, sz_particles_occupied_pos));
//    gpuErrchk(cudaMalloc((void**)&d_particles_occupied_y, sz_particles_occupied_pos));
//    gpuErrchk(cudaMalloc((void**)&d_particles_free_x, sz_particles_free_pos));
//    gpuErrchk(cudaMalloc((void**)&d_particles_free_y, sz_particles_free_pos));
//
//    threadsPerBlock = GRID_WIDTH;
//    blocksPerGrid = 1;
//    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (d_map_occupied_2d, d_particles_occupied_x, d_particles_occupied_y,
//        d_occupied_unique_counter_col, GRID_WIDTH, GRID_HEIGHT);
//    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (d_map_free_2d, d_particles_free_x, d_particles_free_y,
//        d_free_unique_counter_col, GRID_WIDTH, GRID_HEIGHT);
//    cudaDeviceSynchronize();
//}
//
//void exec_log_odds(float log_t) {
//
//    threadsPerBlock = 256;
//    blocksPerGrid = (PARTICLES_OCCUPIED_UNIQUE_LEN + threadsPerBlock - 1) / threadsPerBlock;
//    kernel_update_log_odds << <blocksPerGrid, threadsPerBlock >> > (d_log_odds, d_particles_occupied_x, d_particles_occupied_y,
//        2 * log_t, GRID_WIDTH, GRID_HEIGHT, PARTICLES_OCCUPIED_UNIQUE_LEN);
//    cudaDeviceSynchronize();
//
//    threadsPerBlock = 256;
//    blocksPerGrid = (PARTICLES_FREE_UNIQUE_LEN + threadsPerBlock - 1) / threadsPerBlock;
//    kernel_update_log_odds << <blocksPerGrid, threadsPerBlock >> > (d_log_odds, d_particles_free_x, d_particles_free_y, (-1) * log_t, GRID_WIDTH, GRID_HEIGHT, PARTICLES_FREE_UNIQUE_LEN);
//    cudaDeviceSynchronize();
//
//    threadsPerBlock = 256;
//    blocksPerGrid = ((GRID_WIDTH * GRID_HEIGHT) + threadsPerBlock - 1) / threadsPerBlock;
//    kernel_update_map << <blocksPerGrid, threadsPerBlock >> > (d_grid_map, d_log_odds, LOG_ODD_PRIOR, WALL, FREE, GRID_WIDTH * GRID_HEIGHT);
//    cudaDeviceSynchronize();
//}
//
//void assertResults() {
//
//    printf("\n--> Occupied Unique: %d, %d\n", PARTICLES_OCCUPIED_UNIQUE_LEN, ST_PARTICLES_OCCUPIED_UNIQUE_LEN);
//    assert(PARTICLES_OCCUPIED_UNIQUE_LEN == ST_PARTICLES_OCCUPIED_UNIQUE_LEN);
//    printf("\n--> Free Unique: %d, %d\n", PARTICLES_FREE_UNIQUE_LEN, ST_PARTICLES_FREE_UNIQUE_LEN);
//    assert(PARTICLES_FREE_UNIQUE_LEN == ST_PARTICLES_FREE_UNIQUE_LEN);
//
//    printf("~~$ PARTICLES_FREE_LEN=%d\n", PARTICLES_FREE_LEN);
//    ASSERT_particles_free_index(res_particles_free_counter, h_particles_free_idx, PARTICLES_OCCUPIED_LEN, false);
//    ASSERT_particles_free_new_len(PARTICLES_FREE_LEN, ST_PARTICLES_FREE_LEN);
//
//    gpuErrchk(cudaMemcpy(res_log_odds, d_log_odds, sz_log_odds, cudaMemcpyDeviceToHost));
//    gpuErrchk(cudaMemcpy(res_grid_map, d_grid_map, sz_grid_map, cudaMemcpyDeviceToHost));
//
//    ASSERT_log_odds(res_log_odds, h_log_odds, post_log_odds, (GRID_WIDTH * GRID_HEIGHT));
//    ASSERT_log_odds_maps(res_grid_map, h_grid_map, post_grid_map, (GRID_WIDTH * GRID_HEIGHT));
//
//    printf("\n~~$ Verification All Passed\n");
//}
//
//
//void test_map_func() {
//
//    printf("/********************************************************************/\n");
//    printf("/****************************** MAP MAIN ****************************/\n");
//    printf("/********************************************************************/\n");
//
//    const int data_len = 6;
//    int data[data_len] = { 1, 0, 2, 2, 1, 3 };
//    int* d_data = NULL;
//    gpuErrchk(cudaMalloc((void**)&d_data, data_len * sizeof(int)));
//    gpuErrchk(cudaMemcpy(d_data, data, data_len * sizeof(int), cudaMemcpyHostToDevice));
//
//    thrust::exclusive_scan(thrust::host, data, data + 6, data, 0);
//    thrust::exclusive_scan(thrust::device, d_data, d_data + 6, d_data, 0);
//
//    GRID_WIDTH = ST_GRID_WIDTH;
//    GRID_HEIGHT = ST_GRID_HEIGHT;
//    log_t = ST_log_t;
//    LIDAR_COORDS_LEN = ST_LIDAR_COORDS_LEN;
//    PARTICLES_OCCUPIED_LEN = ST_PARTICLES_OCCUPIED_LEN;
//    MAX_DIST_IN_MAP = sqrt(pow(GRID_WIDTH, 2) + pow(GRID_HEIGHT, 2));
//    PARTICLE_UNIQUE_COUNTER = PARTICLES_OCCUPIED_LEN + 1;
//
//    xmin = ST_xmin;
//    xmax = ST_xmax;;
//    ymin = ST_ymin;
//    ymax = ST_ymax;
//
//    h_map_occupied_idx[1] = PARTICLES_OCCUPIED_LEN;
//    h_map_free_idx[1] = PARTICLES_FREE_LEN;
//
//    printf("~~$ PARTICLES_OCCUPIED_LEN = \t%d\n", PARTICLES_OCCUPIED_LEN);
//    printf("~~$ PARTICLE_UNIQUE_COUNTER = \t%d\n", PARTICLE_UNIQUE_COUNTER);
//    printf("~~$ MAX_DIST_IN_MAP = \t\t%d\n", MAX_DIST_IN_MAP);
//
//
//    auto start_mapping_alloc = std::chrono::high_resolution_clock::now();
//    alloc_image_transform_vars(LIDAR_COORDS_LEN);
//    alloc_bresenham_vars();
//    alloc_map_vars();
//    alloc_log_odds_vars();
//    auto stop_mapping_alloc = std::chrono::high_resolution_clock::now();
//
//    auto start_mapping_init = std::chrono::high_resolution_clock::now();
//    init_image_transform_vars();
//    init_map_vars();
//    init_log_odds_vars();
//    auto stop_mapping_init = std::chrono::high_resolution_clock::now();
//
//    auto start_mapping_kernel = std::chrono::high_resolution_clock::now();
//    exec_world_to_image_transform_step_1();
//    exec_map_extend();
//    exec_world_to_image_transform_step_2();
//    exec_bresenham();
//    reinit_map_idx_vars();
//
//    exec_create_map();
//    reinit_map_vars();
//
//    exec_log_odds(ST_log_t);
//    auto stop_mapping_kernel = std::chrono::high_resolution_clock::now();
//
//    assertResults();
//
//    auto duration_mapping_alloc = std::chrono::duration_cast<std::chrono::microseconds>(stop_mapping_alloc - start_mapping_alloc);
//    auto duration_mapping_init = std::chrono::duration_cast<std::chrono::microseconds>(stop_mapping_init - start_mapping_init);
//    auto duration_mapping_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_mapping_kernel - start_mapping_kernel);
//    auto duration_mapping_total = std::chrono::duration_cast<std::chrono::microseconds>(stop_mapping_kernel - start_mapping_alloc);
//
//    std::cout << std::endl;
//    std::cout << "Time taken by function (Mapping Allocation): " << duration_mapping_alloc.count() << " microseconds" << std::endl;
//    std::cout << "Time taken by function (Mapping Initialization): " << duration_mapping_init.count() << " microseconds" << std::endl;
//    std::cout << "Time taken by function (Mapping Kernel): " << duration_mapping_kernel.count() << " microseconds" << std::endl;
//    std::cout << "Time taken by function (Mapping Total): " << duration_mapping_total.count() << " microseconds" << std::endl;
//    std::cout << std::endl;
//}
//


#endif
