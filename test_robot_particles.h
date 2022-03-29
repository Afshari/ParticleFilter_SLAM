﻿
#ifndef _TEST_ROBOT_PARTICLES_H_
#define _TEST_ROBOT_PARTICLES_H_

#include "headers.h"
#include "host_asserts.h"
#include "host_utils.h"
#include "kernels.cuh"
#include "kernels_robot.cuh"
#include "kernels_utils.cuh"

/**************************************************/
#include "data/update_func/1500.h"
/**************************************************/

void host_update_loop();                            // Step 1
void host_update_particles();                       // Step 1.1
void host_update_unique();                          // Step 1.2
void host_correlation();                            // Step 1.3
void host_update_particle_weights();                // Step 2
void host_resampling();                             // Step 3
void host_update_state();                           // Step 4
void host_update_func();                            // Step X

void test_robot_particles_main();

static int GRID_WIDTH = ST_GRID_WIDTH;
static int GRID_HEIGHT = ST_GRID_HEIGHT;

static const int UNIQUE_COUNTER_LEN = NUM_PARTICLES + 1;

static int LIDAR_COORDS_LEN = 0;
static int MEASURE_LEN = 0;

static float res = ST_res;
static int xmin = ST_xmin;
static int ymax = ST_ymax;

static int PARTICLES_ITEMS_LEN = 0;
static int C_PARTICLES_ITEMS_LEN = 0;

static int threadsPerBlock = 1;
static int blocksPerGrid = 1;

// 1.  IMAGE TRANSFORM VARIABLES
// 2.  STATES VARIABLES
// 3.  PARTICLES VARIABLES
// 4.  MEASUREMENT VARIABLES
// 5.  PROCESSED MEASUREMENTS VARIABLES
// 6.  WEIGHTS VARIABLES
// 7.  RESAMPLING VARIABLES
// 8.  MAP VARIABLES
// 9.  2D MAP VARIABLES
// 10. RESIZE PARTICLES VARIABLES

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
float* res_transition_world_lidar = NULL;

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
size_t sz_weights = 0;
size_t sz_weights_raw = 0;
size_t sz_weights_max = 0;
size_t sz_sum_exp = 0;

float* d_weights = NULL;
float* d_weights_raw = NULL;
float* d_weights_max = NULL;
double* d_sum_exp = NULL;

/*------------------------ RESULT VARIABLES -----------------------*/
float* res_weights = NULL;
float* res_weights_max = NULL;
double* res_sum_exp = NULL;

/********************************************************************/
/*********************** RESAMPLING VARIABLES ***********************/
/********************************************************************/
size_t sz_js = 0;
size_t sz_rnd = 0;

int* d_js = NULL;
float* d_rnd = NULL;

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

int test_robot_particles_partials_main() {

    thrust::device_vector<float> d_temp(h_states_x, h_states_x + NUM_PARTICLES);
    
    
    host_update_particles();                    // Step 1.1
    host_update_particles();                    // Step 1.1
    host_update_unique();                       // Step 1.2
    host_correlation();                         // Step 1.3
    host_update_loop();                         // Step 1
    host_update_particle_weights();             // Step 2
    host_resampling();                          // Step 3
    host_update_state();                        // Step 4
    host_update_func();                         // Step X

    test_robot_particles_main();

    return 0;
}


// Step 1.1
void host_update_particles() {

    // ✓
    printf("/********************************************************************/\n");
    printf("/************************** UPDATE PARTICLES ************************/\n");
    printf("/********************************************************************/\n");

    LIDAR_COORDS_LEN = ST_LIDAR_COORDS_LEN;
    printf("~~$ LIDAR_COORDS_LEN: \t\t%d \n", LIDAR_COORDS_LEN);

    /********************************************************************/
    /************************* STATES VARIABLES *************************/
    /********************************************************************/
    sz_states_pos = NUM_PARTICLES * sizeof(float);
    sz_lidar_coords = 2 * LIDAR_COORDS_LEN * sizeof(float);

    gpuErrchk(cudaMalloc((void**)&d_states_x, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&d_states_y, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&d_states_theta, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&d_lidar_coords, sz_lidar_coords));

    gpuErrchk(cudaMemcpy(d_states_x, h_states_x, sz_states_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_states_y, h_states_y, sz_states_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_states_theta, h_states_theta, sz_states_pos, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_lidar_coords, h_lidar_coords, sz_lidar_coords, cudaMemcpyHostToDevice));


    /********************************************************************/
    /************************ TRANSFORM VARIABLES ***********************/
    /********************************************************************/
    sz_transition_world_frame = 9 * NUM_PARTICLES * sizeof(float);
    sz_transition_body_lidar = 9 * sizeof(float);

    gpuErrchk(cudaMalloc((void**)&d_transition_world_body, sz_transition_world_frame));
    gpuErrchk(cudaMalloc((void**)&d_transition_body_lidar, sz_transition_body_lidar));
    gpuErrchk(cudaMalloc((void**)&d_transition_world_lidar, sz_transition_world_frame));

    gpuErrchk(cudaMemset(d_transition_world_body, 0, sz_transition_world_frame));
    gpuErrchk(cudaMemset(d_transition_world_lidar, 0, sz_transition_world_frame));

    /*------------------------ RESULT VARIABLES -----------------------*/
    res_transition_world_body = (float*)malloc(sz_transition_world_frame);
    res_transition_world_lidar = (float*)malloc(sz_transition_world_frame);

    memset(res_transition_world_body, 0, sz_transition_world_frame);
    memset(res_transition_world_lidar, 0, sz_transition_world_frame);

    /********************************************************************/
    /********************* PROCESSED MEASURE VARIABLES ******************/
    /********************************************************************/
    sz_processed_measure_pos = NUM_PARTICLES * LIDAR_COORDS_LEN * sizeof(int);
    sz_processed_measure_idx = NUM_PARTICLES * LIDAR_COORDS_LEN * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_processed_measure_x, sz_processed_measure_pos));
    gpuErrchk(cudaMalloc((void**)&d_processed_measure_y, sz_processed_measure_pos));
    gpuErrchk(cudaMalloc((void**)&d_processed_measure_idx, sz_processed_measure_idx));

    /*------------------------ RESULT VARIABLES -----------------------*/
    gpuErrchk(cudaMemset(d_processed_measure_x, 0, sz_processed_measure_pos));
    gpuErrchk(cudaMemset(d_processed_measure_y, 0, sz_processed_measure_pos));

    res_processed_measure_x = (int*)malloc(sz_processed_measure_pos);
    res_processed_measure_y = (int*)malloc(sz_processed_measure_pos);
    res_processed_measure_idx = (int*)malloc(sz_processed_measure_idx);

    memset(res_processed_measure_x, 0, sz_processed_measure_pos);
    memset(res_processed_measure_y, 0, sz_processed_measure_pos);

    gpuErrchk(cudaMemcpy(d_transition_body_lidar, h_transition_body_lidar, sz_transition_body_lidar, cudaMemcpyHostToDevice));

    /********************************************************************/
    /*************************** KERNEL EXEC ****************************/
    /********************************************************************/
    auto start_kernel = std::chrono::high_resolution_clock::now();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;

    kernel_update_particles_states << <blocksPerGrid, threadsPerBlock >> > (d_states_x, d_states_y, d_states_theta, 
        d_transition_world_body, d_transition_body_lidar, d_transition_world_lidar, NUM_PARTICLES);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = LIDAR_COORDS_LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (d_transition_world_lidar, d_processed_measure_x, d_processed_measure_y, 
        d_lidar_coords, res, xmin, ymax, LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_index_init_const << < blocksPerGrid, threadsPerBlock >> > (d_processed_measure_idx, LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = 1;
    blocksPerGrid = 1;
    thrust::exclusive_scan(thrust::device, d_processed_measure_idx, d_processed_measure_idx + NUM_PARTICLES, d_processed_measure_idx, 0);

    auto stop_kernel = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_transition_world_body, d_transition_world_body, sz_transition_world_frame, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_transition_world_lidar, d_transition_world_lidar, sz_transition_world_frame, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_processed_measure_x, d_processed_measure_x, sz_processed_measure_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_processed_measure_y, d_processed_measure_y, sz_processed_measure_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_processed_measure_idx, d_processed_measure_idx, sz_processed_measure_idx, cudaMemcpyDeviceToHost));

    ASSERT_transition_frames(res_transition_world_body, res_transition_world_lidar,
        h_transition_world_body, h_transition_world_lidar, NUM_PARTICLES, false, true, false);

    ASSERT_processed_measurements(res_processed_measure_x, res_processed_measure_y, res_processed_measure_idx,
        h_processed_measure_x, h_processed_measure_y, (NUM_PARTICLES * LIDAR_COORDS_LEN), LIDAR_COORDS_LEN, true, true);

    
    auto duration_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_kernel - start_kernel);
    std::cout << "Time taken by function (Kernel): " << duration_kernel.count() << " microseconds" << std::endl;
    std::cout << std::endl;
}

// Step 1.2
void host_update_unique() {

    printf("/********************************************************************/\n");
    printf("/************************** UPDATE UNIQUE ***************************/\n");
    printf("/********************************************************************/\n");

    LIDAR_COORDS_LEN = ST_LIDAR_COORDS_LEN;
    PARTICLES_ITEMS_LEN = ST_PARTICLES_ITEMS_LEN;
    MEASURE_LEN = NUM_PARTICLES * LIDAR_COORDS_LEN;

    int negative_before_counter = getNegativeCounter(h_particles_x, h_particles_y, PARTICLES_ITEMS_LEN);
    int count_bigger_than_height = getGreaterThanCounter(h_particles_y, GRID_HEIGHT, PARTICLES_ITEMS_LEN);
    int negative_after_counter = getNegativeCounter(h_particles_x_after_unique, h_particles_y_after_unique, AF_PARTICLES_ITEMS_LEN_UNIQUE);;

    printf("~~$ GRID_WIDTH: \t\t%d\n", GRID_WIDTH, GRID_HEIGHT);
    printf("~~$ GRID_HEIGHT: \t\t%d\n", GRID_WIDTH, GRID_HEIGHT);
    printf("~~$ MEASURE_LEN: \t\t%d\n", LIDAR_COORDS_LEN);
    printf("~~$ negative_before_counter: \t%d\n", negative_before_counter);
    printf("~~$ negative_after_counter: \t%d\n", negative_after_counter);
    printf("~~$ count_bigger_than_height: \t%d\n", count_bigger_than_height);

    /********************************************************************/
    /************************** PRIOR VARIABLES *************************/
    /********************************************************************/
    sz_particles_pos = PARTICLES_ITEMS_LEN * sizeof(int);
    sz_particles_idx = NUM_PARTICLES * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_particles_x, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_y, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_idx, sz_particles_idx));

    gpuErrchk(cudaMemcpy(d_particles_x, h_particles_x, sz_particles_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particles_y, h_particles_y, sz_particles_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particles_idx, h_particles_idx, sz_particles_idx, cudaMemcpyHostToDevice));


    /********************************************************************/
    /**************************** MAP VARIABLES *************************/
    /********************************************************************/
    sz_map_2d = GRID_WIDTH * GRID_HEIGHT * NUM_PARTICLES * sizeof(uint8_t);
    sz_unique_in_particle = UNIQUE_COUNTER_LEN * sizeof(int);
    sz_unique_in_particle_col = UNIQUE_COUNTER_LEN * GRID_WIDTH * sizeof(int);

    res_unique_in_particle = (int*)malloc(sz_unique_in_particle);
    res_unique_in_particle_col = (int*)malloc(sz_unique_in_particle_col);
    res_map_2d = (uint8_t*)malloc(sz_map_2d);

    gpuErrchk(cudaMalloc((void**)&d_map_2d, sz_map_2d));
    gpuErrchk(cudaMalloc((void**)&d_unique_in_particle, sz_unique_in_particle));
    gpuErrchk(cudaMalloc((void**)&d_unique_in_particle_col, sz_unique_in_particle_col));

    gpuErrchk(cudaMemset(d_map_2d, 0, sz_map_2d));
    gpuErrchk(cudaMemset(d_unique_in_particle, 0, sz_unique_in_particle));
    gpuErrchk(cudaMemset(d_unique_in_particle_col, 0, sz_unique_in_particle_col));

    /********************************************************************/
    /*********************** MEASUREMENT VARIABLES **********************/
    /********************************************************************/
    sz_processed_measure_pos = NUM_PARTICLES * LIDAR_COORDS_LEN * sizeof(int);
    sz_processed_measure_idx = NUM_PARTICLES * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_processed_measure_x, sz_processed_measure_pos));
    gpuErrchk(cudaMalloc((void**)&d_processed_measure_y, sz_processed_measure_pos));
    gpuErrchk(cudaMalloc((void**)&d_processed_measure_idx, sz_processed_measure_idx));

    gpuErrchk(cudaMemcpy(d_processed_measure_x, h_processed_measure_x, sz_processed_measure_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_processed_measure_y, h_processed_measure_y, sz_processed_measure_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_processed_measure_idx, h_processed_measure_idx, sz_processed_measure_idx, cudaMemcpyHostToDevice));

    /********************************************************************/
    /**************************** CREATE MAP ****************************/
    /********************************************************************/
    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;

    auto start_create_map = std::chrono::high_resolution_clock::now();
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (d_particles_x, d_particles_y, d_particles_idx, 
        PARTICLES_ITEMS_LEN, d_map_2d, d_unique_in_particle,
        d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();
    auto stop_create_map = std::chrono::high_resolution_clock::now();
    gpuErrchk(cudaMemcpy(res_map_2d, d_map_2d, sz_map_2d, cudaMemcpyDeviceToHost));

    ASSERT_create_2d_map_elements(res_map_2d, negative_before_counter, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES, PARTICLES_ITEMS_LEN, true, true);

    /********************************************************************/
    /**************************** UPDATE MAP ****************************/
    /********************************************************************/
    auto start_update_map = std::chrono::high_resolution_clock::now();
    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;

    kernel_update_2d_map_with_measure << <blocksPerGrid, threadsPerBlock >> > (d_processed_measure_x, d_processed_measure_y, d_processed_measure_idx,
        MEASURE_LEN, d_map_2d, d_unique_in_particle,
        d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();
    auto stop_update_map = std::chrono::high_resolution_clock::now();
    gpuErrchk(cudaMemcpy(res_map_2d, d_map_2d, sz_map_2d, cudaMemcpyDeviceToHost));

    /********************************************************************/
    /************************* CUMULATIVE SUM ***************************/
    /********************************************************************/
    threadsPerBlock = UNIQUE_COUNTER_LEN;
    blocksPerGrid = 1;

    auto start_cumulative_sum = std::chrono::high_resolution_clock::now();
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (d_unique_in_particle_col, GRID_WIDTH);
    thrust::exclusive_scan(thrust::device, d_unique_in_particle, d_unique_in_particle + UNIQUE_COUNTER_LEN, d_unique_in_particle, 0);
    cudaDeviceSynchronize();
    auto stop_cumulative_sum = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_map_2d, d_map_2d, sz_map_2d, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_unique_in_particle, d_unique_in_particle, sz_unique_in_particle, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_unique_in_particle_col, d_unique_in_particle_col, sz_unique_in_particle_col, cudaMemcpyDeviceToHost));

    PARTICLES_ITEMS_LEN = res_unique_in_particle[NUM_PARTICLES];
    gpuErrchk(cudaFree(d_particles_x));
    gpuErrchk(cudaFree(d_particles_y));

    sz_particles_pos = PARTICLES_ITEMS_LEN * sizeof(int);
    gpuErrchk(cudaMalloc((void**)&d_particles_x, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_y, sz_particles_pos));

    res_particles_x = (int*)malloc(sz_particles_pos);
    res_particles_y = (int*)malloc(sz_particles_pos);
    res_particles_idx = (int*)malloc(sz_particles_idx);

    /********************************************************************/
    /************************ MAP RESTRUCTURE ***************************/
    /********************************************************************/
    threadsPerBlock = GRID_WIDTH;
    blocksPerGrid = NUM_PARTICLES;

    auto start_map_restructure = std::chrono::high_resolution_clock::now();
    cudaMemset(d_particles_idx, 0, sz_particles_idx);
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (d_map_2d, d_particles_x, d_particles_y, d_particles_idx,
        d_unique_in_particle, d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, d_particles_idx, d_particles_idx + NUM_PARTICLES, d_particles_idx, 0);
    cudaDeviceSynchronize();
    auto stop_map_restructure = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_particles_x, d_particles_x, sz_particles_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_y, d_particles_y, sz_particles_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToHost));

    printf("~~$ PARTICLES_ITEMS_LEN: \t%d\n", PARTICLES_ITEMS_LEN, AF_PARTICLES_ITEMS_LEN_UNIQUE);
    printf("~~$ AF_PARTICLES_ITEMS_LEN: \t%d\n", PARTICLES_ITEMS_LEN, AF_PARTICLES_ITEMS_LEN_UNIQUE);
    printf("~~$ Measurement Length: \t%d\n", MEASURE_LEN);

    ASSERT_particles_pos_unique(res_particles_x, res_particles_y, h_particles_x_after_unique, h_particles_y_after_unique, PARTICLES_ITEMS_LEN, false, true, true);
    ASSERT_particles_idx_unique(res_particles_idx, h_particles_idx_after_unique, negative_after_counter, NUM_PARTICLES, false, true);

    auto duration_create_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_create_map - start_create_map);
    auto duration_update_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_update_map - start_update_map);
    auto duration_cumulative_sum = std::chrono::duration_cast<std::chrono::microseconds>(stop_cumulative_sum - start_cumulative_sum);
    auto duration_map_restructure = std::chrono::duration_cast<std::chrono::microseconds>(stop_map_restructure - start_map_restructure);

    std::cout << "Time taken by function (Create Map): " << duration_create_map.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Update Map): " << duration_update_map.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Cumulative Sum): " << duration_cumulative_sum.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Map Restructure): " << duration_map_restructure.count() << " microseconds" << std::endl;
    std::cout << std::endl;
}

// Step 1.3
void host_correlation() {

    printf("/********************************************************************/\n");
    printf("/**************************** CORRELATION ***************************/\n");
    printf("/********************************************************************/\n");

    PARTICLES_ITEMS_LEN = AF_PARTICLES_ITEMS_LEN_UNIQUE;

    /********************************************************************/
    /************************** PRIOR VARIABLES *************************/
    /********************************************************************/
    const int GRID_MAP_ITEMS_LEN = GRID_WIDTH * GRID_HEIGHT;

    sz_grid_map = GRID_MAP_ITEMS_LEN * sizeof(int);
    sz_particles_pos = PARTICLES_ITEMS_LEN * sizeof(int);
    sz_particles_idx = NUM_PARTICLES * sizeof(int);
    sz_extended_idx = PARTICLES_ITEMS_LEN * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_grid_map, sz_grid_map));

    auto start_memory_copy = std::chrono::high_resolution_clock::now();
    gpuErrchk(cudaMalloc((void**)&d_grid_map, sz_grid_map));
    gpuErrchk(cudaMalloc((void**)&d_particles_x, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_y, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_extended_idx, sz_extended_idx));
    gpuErrchk(cudaMalloc((void**)&d_particles_idx, sz_particles_idx));

    gpuErrchk(cudaMemcpy(d_grid_map, h_grid_map, sz_grid_map, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particles_x, h_particles_x_after_unique, sz_particles_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particles_y, h_particles_y_after_unique, sz_particles_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particles_idx, h_particles_idx_after_unique, sz_particles_idx, cudaMemcpyHostToDevice));

    sz_weights = NUM_PARTICLES * sizeof(float);
    sz_weights_raw = 25 * sz_weights;

    res_weights = (float*)malloc(sz_weights);
    res_extended_idx = (int*)malloc(sz_extended_idx);
        
    memset(res_weights, 0, sz_weights);

    gpuErrchk(cudaMalloc((void**)&d_weights, sz_weights));
    gpuErrchk(cudaMalloc((void**)&d_weights_raw, sz_weights_raw));
    gpuErrchk(cudaMemset(d_weights_raw, 0, sz_weights_raw));

    auto stop_memory_copy = std::chrono::high_resolution_clock::now();


    /********************************************************************/
    /*************************** PRINT SUMMARY **************************/
    /********************************************************************/
    printf("Elements of particles_x: \t%d  \tSize of particles_x: \t%d\n", (int)PARTICLES_ITEMS_LEN, (int)sz_particles_pos);
    printf("Elements of particles_y: \t%d  \tSize of particles_y: \t%d\n", (int)PARTICLES_ITEMS_LEN, (int)sz_particles_pos);
    printf("Elements of particles_idx: \t%d  \tSize of particles_idx: \t%d\n", (int)PARTICLES_ITEMS_LEN, (int)sz_extended_idx);
    printf("\n");
    printf("Elements of Grid_Map: \t\t%d  \tSize of Grid_Map: \t%d\n", (int)GRID_MAP_ITEMS_LEN, (int)sz_grid_map);

    /********************************************************************/
    /************************* INDEX EXPANSION **************************/
    /********************************************************************/
    auto start_index_expansion = std::chrono::high_resolution_clock::now();
    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;
    kernel_index_expansion << <blocksPerGrid, threadsPerBlock >> > (d_particles_idx, d_extended_idx, PARTICLES_ITEMS_LEN);
    cudaDeviceSynchronize();
    auto stop_index_expansion = std::chrono::high_resolution_clock::now();

    /********************************************************************/
    /************************ KERNEL CORRELATION ************************/
    /********************************************************************/

    auto start_kernel = std::chrono::high_resolution_clock::now();
    threadsPerBlock = 256;
    blocksPerGrid = (PARTICLES_ITEMS_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_correlation << <blocksPerGrid, threadsPerBlock >> > (d_grid_map, d_particles_x, d_particles_y, d_extended_idx, d_weights_raw, GRID_WIDTH, GRID_HEIGHT, PARTICLES_ITEMS_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_correlation_max << <blocksPerGrid, threadsPerBlock >> > (d_weights_raw, d_weights, NUM_PARTICLES);

    auto stop_kernel = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_weights, d_weights, sz_weights, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_extended_idx, d_extended_idx, sz_extended_idx, cudaMemcpyDeviceToHost));


    ASSERT_correlation_Equality(res_weights, pre_weights, NUM_PARTICLES, true, true);

    auto duration_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_kernel - start_kernel);
    auto duration_memory_copy = std::chrono::duration_cast<std::chrono::microseconds>(stop_memory_copy - start_memory_copy);
    auto duration_index_expansion = std::chrono::duration_cast<std::chrono::microseconds>(stop_index_expansion - start_index_expansion);
    std::cout << "Time taken by function (Correlation): " << duration_kernel.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Memory Copy): " << duration_memory_copy.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Index Expansion): " << duration_index_expansion.count() << " microseconds" << std::endl;
    std::cout << std::endl;

    gpuErrchk(cudaFree(d_grid_map));
    gpuErrchk(cudaFree(d_particles_x));
    gpuErrchk(cudaFree(d_particles_y));
    gpuErrchk(cudaFree(d_particles_idx));
    gpuErrchk(cudaFree(d_extended_idx));
    gpuErrchk(cudaFree(d_weights_raw));
}

// Step 1
void host_update_loop() {

    printf("/********************************************************************/\n");
    printf("/**************************** UPDATE LOOP ***************************/\n");
    printf("/********************************************************************/\n");

    // ✓
    LIDAR_COORDS_LEN = ST_LIDAR_COORDS_LEN;
    PARTICLES_ITEMS_LEN = ST_PARTICLES_ITEMS_LEN;
    MEASURE_LEN = NUM_PARTICLES * LIDAR_COORDS_LEN;

    int negative_before_counter = getNegativeCounter(h_particles_x, h_particles_y, PARTICLES_ITEMS_LEN);
    int count_bigger_than_height = getGreaterThanCounter(h_particles_y, GRID_HEIGHT, PARTICLES_ITEMS_LEN);
    int negative_after_unique_counter = getNegativeCounter(h_particles_x_after_unique, h_particles_y_after_unique, AF_PARTICLES_ITEMS_LEN_UNIQUE);
    int negative_after_resampling_counter = getNegativeCounter(h_particles_x_after_resampling, h_particles_y_after_resampling, AF_PARTICLES_ITEMS_LEN_RESAMPLING);

    printf("~~$ GRID_WIDTH: \t\t\t%d\n", GRID_WIDTH);
    printf("~~$ GRID_HEIGHT: \t\t\t%d\n", GRID_HEIGHT);
    printf("~~$ LIDAR_COORDS_LEN: \t\t\t%d\n", LIDAR_COORDS_LEN);
    printf("~~$ negative_before_counter: \t\t%d\n", negative_before_counter);
    printf("~~$ negative_after_unique_counter: \t%d\n", negative_after_unique_counter);
    printf("~~$ negative_after_resampling_counter: \t%d\n", negative_after_resampling_counter);
    printf("~~$ count_bigger_than_height: \t\t%d\n", count_bigger_than_height);
    printf("~~$ MEASURE_LEN: \t\t\t%d \n", MEASURE_LEN);

    /********************************************************************/
    /************************** PRIOR VARIABLES *************************/
    /********************************************************************/
    sz_states_pos = NUM_PARTICLES * sizeof(float);
    sz_lidar_coords = 2 * LIDAR_COORDS_LEN * sizeof(float);


    gpuErrchk(cudaMalloc((void**)&d_states_x, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&d_states_y, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&d_states_theta, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&d_lidar_coords, sz_lidar_coords));

    gpuErrchk(cudaMemcpy(d_states_x, h_states_x, sz_states_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_states_y, h_states_y, sz_states_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_states_theta, h_states_theta, sz_states_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_lidar_coords, h_lidar_coords, sz_lidar_coords, cudaMemcpyHostToDevice));

    /********************************************************************/
    /**************************** MAP VARIABLES *************************/
    /********************************************************************/
    sz_particles_pos = PARTICLES_ITEMS_LEN * sizeof(int);
    sz_particles_idx = NUM_PARTICLES * sizeof(int);
    sz_extended_idx = PARTICLES_ITEMS_LEN * sizeof(int);
    sz_grid_map = GRID_WIDTH * GRID_HEIGHT * sizeof(int);

    res_particles_x = (int*)malloc(sz_particles_pos);
    res_particles_y = (int*)malloc(sz_particles_pos);
    res_particles_idx = (int*)malloc(sz_particles_idx);
    res_extended_idx = (int*)malloc(sz_extended_idx);

    gpuErrchk(cudaMalloc((void**)&d_grid_map, sz_grid_map));
    gpuErrchk(cudaMalloc((void**)&d_particles_x, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_y, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_idx, sz_particles_idx));
    gpuErrchk(cudaMalloc((void**)&d_extended_idx, sz_extended_idx));

    gpuErrchk(cudaMemcpy(d_grid_map, h_grid_map, sz_grid_map, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particles_x, h_particles_x, sz_particles_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particles_y, h_particles_y, sz_particles_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particles_idx, h_particles_idx, sz_particles_idx, cudaMemcpyHostToDevice));


    /********************************************************************/
    /********************** CORRELATION VARIABLES ***********************/
    /********************************************************************/
    sz_weights = NUM_PARTICLES * sizeof(float);
    sz_weights_raw = 25 * sz_weights;

    res_weights = (float*)malloc(sz_weights);
    res_extended_idx = (int*)malloc(sz_extended_idx);
    res_weights = (float*)malloc(sz_weights);

    memset(res_weights, 0, sz_weights);

    gpuErrchk(cudaMalloc((void**)&d_weights, sz_weights));
    gpuErrchk(cudaMalloc((void**)&d_weights_raw, sz_weights_raw));
    gpuErrchk(cudaMemset(d_weights_raw, 0, sz_weights_raw));

    /********************************************************************/
    /*********************** TRANSITION VARIABLES ***********************/
    /********************************************************************/
    sz_transition_world_frame = 9 * NUM_PARTICLES * sizeof(float);
    sz_transition_body_lidar = 9 * sizeof(float);
    sz_processed_measure_pos = NUM_PARTICLES * LIDAR_COORDS_LEN * sizeof(int);
    sz_processed_measure_idx = NUM_PARTICLES * sizeof(int);

    res_transition_world_body = (float*)malloc(sz_transition_world_frame);
    res_transition_world_lidar = (float*)malloc(sz_transition_world_frame);
    res_processed_measure_x = (int*)malloc(sz_processed_measure_pos);
    res_processed_measure_y = (int*)malloc(sz_processed_measure_pos);
    res_processed_measure_idx = (int*)malloc(sz_processed_measure_idx);

    gpuErrchk(cudaMalloc((void**)&d_transition_world_body, sz_transition_world_frame));
    gpuErrchk(cudaMalloc((void**)&d_transition_body_lidar, sz_transition_body_lidar));
    gpuErrchk(cudaMalloc((void**)&d_transition_world_lidar, sz_transition_world_frame));
    gpuErrchk(cudaMalloc((void**)&d_processed_measure_x, sz_processed_measure_pos));
    gpuErrchk(cudaMalloc((void**)&d_processed_measure_y, sz_processed_measure_pos));
    gpuErrchk(cudaMalloc((void**)&d_processed_measure_idx, sz_processed_measure_idx));


    gpuErrchk(cudaMemset(d_transition_world_body, 0, sz_transition_world_frame));
    gpuErrchk(cudaMemset(d_transition_world_lidar, 0, sz_transition_world_frame));
    gpuErrchk(cudaMemset(d_processed_measure_x, 0, sz_processed_measure_pos));
    gpuErrchk(cudaMemset(d_processed_measure_y, 0, sz_processed_measure_pos));
    gpuErrchk(cudaMemset(d_processed_measure_idx, 0, sz_processed_measure_idx));

    gpuErrchk(cudaMemcpy(d_transition_body_lidar, h_transition_body_lidar, sz_transition_body_lidar, cudaMemcpyHostToDevice));

    /********************************************************************/
    /**************************** MAP VARIABLES *************************/
    /********************************************************************/
    sz_map_2d = GRID_WIDTH * GRID_HEIGHT * NUM_PARTICLES * sizeof(uint8_t);
    sz_unique_in_particle = UNIQUE_COUNTER_LEN * sizeof(int);
    sz_unique_in_particle_col = UNIQUE_COUNTER_LEN * GRID_WIDTH * sizeof(int);

    res_map_2d = (uint8_t*)malloc(sz_map_2d);
    int* h_unique_in_particle = (int*)malloc(sz_unique_in_particle);

    gpuErrchk(cudaMalloc((void**)&d_map_2d, sz_map_2d));
    gpuErrchk(cudaMalloc((void**)&d_unique_in_particle, sz_unique_in_particle));
    gpuErrchk(cudaMalloc((void**)&d_unique_in_particle_col, sz_unique_in_particle_col));

    gpuErrchk(cudaMemset(d_map_2d, 0, sz_map_2d));
    gpuErrchk(cudaMemset(d_unique_in_particle, 0, sz_unique_in_particle));
    gpuErrchk(cudaMemset(d_unique_in_particle_col, 0, sz_unique_in_particle_col));

    /********************************************************************/
    /************************ TRANSITION KERNEL *************************/
    /********************************************************************/
    auto start_transition_kernel = std::chrono::high_resolution_clock::now();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;

    kernel_update_particles_states << <blocksPerGrid, threadsPerBlock >> > (d_states_x, d_states_y, d_states_theta, d_transition_world_body, d_transition_body_lidar, d_transition_world_lidar, NUM_PARTICLES);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = LIDAR_COORDS_LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (d_transition_world_lidar, d_processed_measure_x, d_processed_measure_y, d_lidar_coords, res, xmin, ymax, LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_index_init_const << < blocksPerGrid, threadsPerBlock >> > (d_processed_measure_idx, LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    //threadsPerBlock = 1;
    //blocksPerGrid = 1;
    thrust::exclusive_scan(thrust::device, d_processed_measure_idx, d_processed_measure_idx + NUM_PARTICLES, d_processed_measure_idx, 0);
    //cudaDeviceSynchronize();

    auto stop_transition_kernel = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_transition_world_body, d_transition_world_body, sz_transition_world_frame, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_transition_world_lidar, d_transition_world_lidar, sz_transition_world_frame, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_processed_measure_x, d_processed_measure_x, sz_processed_measure_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_processed_measure_y, d_processed_measure_y, sz_processed_measure_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_processed_measure_idx, d_processed_measure_idx, sz_processed_measure_idx, cudaMemcpyDeviceToHost));

    
    ASSERT_transition_frames(res_transition_world_body, res_transition_world_lidar, h_transition_world_body, h_transition_world_lidar, 
        NUM_PARTICLES, false, true, false);
    ASSERT_processed_measurements(res_processed_measure_x, res_processed_measure_y, res_processed_measure_idx, 
        h_processed_measure_x, h_processed_measure_y,
        (NUM_PARTICLES * LIDAR_COORDS_LEN), LIDAR_COORDS_LEN, true, false);
    
    /********************************************************************/
    /************************** CREATE 2D MAP ***************************/
    /********************************************************************/
    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;

    auto start_create_map = std::chrono::high_resolution_clock::now();

    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (d_particles_x, d_particles_y, d_particles_idx, PARTICLES_ITEMS_LEN, d_map_2d, d_unique_in_particle,
        d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_create_map = std::chrono::high_resolution_clock::now();

    //cudaError_t err = cudaPeekAtLastError();
    //printf("%s\n", cudaGetErrorString(err));

    gpuErrchk(cudaMemcpy(res_map_2d, d_map_2d, sz_map_2d, cudaMemcpyDeviceToHost));

    ASSERT_create_2d_map_elements(res_map_2d, negative_before_counter, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES, PARTICLES_ITEMS_LEN, true, false);
    
    /********************************************************************/
    /**************************** UPDATE MAP ****************************/
    /********************************************************************/
    auto start_update_map = std::chrono::high_resolution_clock::now();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;

    kernel_update_2d_map_with_measure << <blocksPerGrid, threadsPerBlock >> > (d_processed_measure_x, d_processed_measure_y, d_processed_measure_idx,
        MEASURE_LEN, d_map_2d, d_unique_in_particle,
        d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_update_map = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_processed_measure_idx, d_processed_measure_idx, sz_processed_measure_idx, cudaMemcpyDeviceToHost));
    
    /********************************************************************/
    /************************* CUMULATIVE SUM ***************************/
    /********************************************************************/
    threadsPerBlock = UNIQUE_COUNTER_LEN;
    blocksPerGrid = 1;

    auto start_cumulative_sum = std::chrono::high_resolution_clock::now();

    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (d_unique_in_particle_col, GRID_WIDTH);
    thrust::exclusive_scan(thrust::device, d_unique_in_particle, d_unique_in_particle + UNIQUE_COUNTER_LEN, d_unique_in_particle, 0);
    cudaDeviceSynchronize();

    auto stop_cumulative_sum = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(h_unique_in_particle, d_unique_in_particle, sz_unique_in_particle, cudaMemcpyDeviceToHost));

    PARTICLES_ITEMS_LEN = h_unique_in_particle[UNIQUE_COUNTER_LEN - 1];
    printf("\n~~$ PARTICLES_ITEMS_LEN=%d, AF_PARTICLES_ITEMS_LEN=%d\n", PARTICLES_ITEMS_LEN, AF_PARTICLES_ITEMS_LEN_UNIQUE);
    ASSERT_new_len_calculation(PARTICLES_ITEMS_LEN, AF_PARTICLES_ITEMS_LEN_UNIQUE, negative_after_resampling_counter);

    /********************************************************************/
    /******************* REINITIALIZE MAP VARIABLES *********************/
    /********************************************************************/
    gpuErrchk(cudaFree(d_particles_x));
    gpuErrchk(cudaFree(d_particles_y));
    gpuErrchk(cudaFree(d_extended_idx));
    free(res_particles_x);
    free(res_particles_y);

    sz_particles_pos = PARTICLES_ITEMS_LEN * sizeof(int);
    sz_extended_idx = PARTICLES_ITEMS_LEN * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_particles_x, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_y, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_extended_idx, sz_extended_idx));

    res_particles_x = (int*)malloc(sz_particles_pos);
    res_particles_y = (int*)malloc(sz_particles_pos);


    /********************************************************************/
    /************************ MAP RESTRUCTURE ***************************/
    /********************************************************************/
    threadsPerBlock = GRID_WIDTH;
    blocksPerGrid = NUM_PARTICLES;

    auto start_map_restructure = std::chrono::high_resolution_clock::now();

    cudaMemset(d_particles_idx, 0, sz_particles_idx);
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (d_map_2d, d_particles_x, d_particles_y, d_particles_idx,
        d_unique_in_particle, d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, d_particles_idx, d_particles_idx + NUM_PARTICLES, d_particles_idx, 0);

    auto stop_map_restructure = std::chrono::high_resolution_clock::now();

    auto start_copy_particles_pos = std::chrono::high_resolution_clock::now();
    gpuErrchk(cudaMemcpy(res_particles_x, d_particles_x, sz_particles_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_y, d_particles_y, sz_particles_pos, cudaMemcpyDeviceToHost));
    auto stop_copy_particles_pos = std::chrono::high_resolution_clock::now();

    ASSERT_particles_pos_unique(res_particles_x, res_particles_y, h_particles_x_after_unique, h_particles_y_after_unique, PARTICLES_ITEMS_LEN, false, false, true);

    /********************************************************************/
    /************************* INDEX EXPANSION **************************/
    /********************************************************************/
    auto start_index_expansion = std::chrono::high_resolution_clock::now();

    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;

    kernel_index_expansion << <blocksPerGrid, threadsPerBlock >> > (d_particles_idx, d_extended_idx, PARTICLES_ITEMS_LEN);
    cudaDeviceSynchronize();

    auto stop_index_expansion = std::chrono::high_resolution_clock::now();

    res_extended_idx = (int*)malloc(sz_extended_idx);
    gpuErrchk(cudaMemcpy(res_extended_idx, d_extended_idx, sz_extended_idx, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToHost));


    /********************************************************************/
    /************************ KERNEL CORRELATION ************************/
    /********************************************************************/
    threadsPerBlock = 256;
    blocksPerGrid = (PARTICLES_ITEMS_LEN + threadsPerBlock - 1) / threadsPerBlock;

    auto start_correlation = std::chrono::high_resolution_clock::now();
    kernel_correlation << <blocksPerGrid, threadsPerBlock >> > (d_grid_map, d_particles_x, d_particles_y, d_extended_idx, d_weights_raw, GRID_WIDTH, GRID_HEIGHT, PARTICLES_ITEMS_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_correlation_max << <blocksPerGrid, threadsPerBlock >> > (d_weights_raw, d_weights, NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_correlation = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_weights, d_weights, sz_weights, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_extended_idx, d_extended_idx, sz_extended_idx, cudaMemcpyDeviceToHost));

    ASSERT_correlation_Equality(res_weights, pre_weights, NUM_PARTICLES, false, true);


    auto duration_create_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_create_map - start_create_map);
    auto duration_update_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_update_map - start_update_map);
    auto duration_cumulative_sum = std::chrono::duration_cast<std::chrono::microseconds>(stop_cumulative_sum - start_cumulative_sum);
    auto duration_map_restructure = std::chrono::duration_cast<std::chrono::microseconds>(stop_map_restructure - start_map_restructure);
    auto duration_copy_particles_pos = std::chrono::duration_cast<std::chrono::microseconds>(stop_copy_particles_pos - start_copy_particles_pos);
    auto duration_transition_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_transition_kernel - start_transition_kernel);
    auto duration_correlation = std::chrono::duration_cast<std::chrono::microseconds>(stop_correlation - start_correlation);
    auto duration_sum = duration_create_map + duration_update_map + duration_cumulative_sum + duration_map_restructure + duration_copy_particles_pos +
        duration_transition_kernel + duration_correlation;

    std::cout << "Time taken by function (Create Map): " << duration_create_map.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Update Map): " << duration_update_map.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Cumulative Sum): " << duration_cumulative_sum.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Map Restructure): " << duration_map_restructure.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Copy Particles): " << duration_copy_particles_pos.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Transition Kernel): " << duration_transition_kernel.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Correlation Kernel): " << duration_correlation.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Sum): " << duration_sum.count() << " microseconds" << std::endl;
    std::cout << std::endl;
}

// Step 2
void host_update_particle_weights() {

    printf("/********************************************************************/\n");
    printf("/********************** UPDATE PARTICLE WEIGHTS *********************/\n");
    printf("/********************************************************************/\n");

    /********************************************************************/
    /************************ WEIGHTS VARIABLES *************************/
    /********************************************************************/
    sz_weights = NUM_PARTICLES * sizeof(float);
    sz_weights_max = sizeof(float);
    sz_sum_exp = sizeof(double);

    res_weights = (float*)malloc(sz_weights);
    res_weights_max = (float*)malloc(sz_weights_max);
    res_sum_exp = (double*)malloc(sz_sum_exp);

    gpuErrchk(cudaMalloc((void**)&d_weights, sz_weights));
    gpuErrchk(cudaMalloc((void**)&d_weights_max, sz_weights_max));
    gpuErrchk(cudaMalloc((void**)&d_sum_exp, sz_sum_exp));

    gpuErrchk(cudaMemcpy(d_weights, pre_weights, sz_weights, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(d_weights_max, 0, sz_weights_max));
    gpuErrchk(cudaMemset(d_sum_exp, 0, sz_sum_exp));


    /********************************************************************/
    /********************** UPDATE WEIGHTS KERNEL ***********************/
    /********************************************************************/
    auto start_update_particle_weights = std::chrono::high_resolution_clock::now();

    threadsPerBlock = 1;
    blocksPerGrid = 1;

    kernel_arr_max << < blocksPerGrid, threadsPerBlock >> > (d_weights, d_weights_max, NUM_PARTICLES);
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(res_weights_max, d_weights_max, sz_weights_max, cudaMemcpyDeviceToHost));
    printf("~~$ res_weights_max[0]=%f\n", res_weights_max[0]);

    float norm_value = -res_weights_max[0] + 50;

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_arr_increase << < blocksPerGrid, threadsPerBlock >> > (d_weights, norm_value, 0);
    cudaDeviceSynchronize();

    threadsPerBlock = 1;
    blocksPerGrid = 1;
    kernel_arr_sum_exp << < blocksPerGrid, threadsPerBlock >> > (d_weights, d_sum_exp, NUM_PARTICLES);
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(res_sum_exp, d_sum_exp, sz_sum_exp, cudaMemcpyDeviceToHost));

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_arr_normalize << < blocksPerGrid, threadsPerBlock >> > (d_weights, res_sum_exp[0]);
    cudaDeviceSynchronize();

    auto stop_update_particle_weights = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_weights, d_weights, sz_weights, cudaMemcpyDeviceToHost));

    ASSERT_update_particle_weights(res_weights, new_weights, NUM_PARTICLES, "weights", false, false, true);

    auto duration_update_particle_weights = std::chrono::duration_cast<std::chrono::microseconds>(stop_update_particle_weights - start_update_particle_weights);
    std::cout << "Time taken by function (Update Particle Weights): " << duration_update_particle_weights.count() << " microseconds" << std::endl;
    std::cout << std::endl;
}

// Step 3
void host_resampling() {

    printf("/********************************************************************/\n");
    printf("/***************************** RESAMPLING ***************************/\n");
    printf("/********************************************************************/\n");

    PARTICLES_ITEMS_LEN = ST_PARTICLES_ITEMS_LEN;
    C_PARTICLES_ITEMS_LEN = 0;

    int negative_before_counter = getNegativeCounter(h_particles_x, h_particles_y, PARTICLES_ITEMS_LEN);
    int count_bigger_than_height = getGreaterThanCounter(h_particles_y, GRID_HEIGHT, PARTICLES_ITEMS_LEN);
    int negative_after_counter = getNegativeCounter(h_particles_x_after_resampling, h_particles_y_after_resampling, AF_PARTICLES_ITEMS_LEN_UNIQUE);

    printf("~~$ GRID_WIDTH: \t\t%d\n", GRID_WIDTH);
    printf("~~$ GRID_HEIGHT: \t\t%d\n", GRID_HEIGHT);
    printf("~~$ negative_before_counter: \t%d\n", negative_before_counter);
    printf("~~$ negative_after_counter: \t%d\n", negative_after_counter);
    printf("~~$ count_bigger_than_height: \t%d\n", count_bigger_than_height);

    /********************************************************************/
    /*********************** RESAMPLING VARIABLES ***********************/
    /********************************************************************/
    sz_weights = NUM_PARTICLES * sizeof(float);
    sz_js = NUM_PARTICLES * sizeof(int);
    sz_rnd = NUM_PARTICLES * sizeof(float);

    res_js = (int*)malloc(sz_js);

    gpuErrchk(cudaMalloc((void**)&d_weights, sz_weights));
    gpuErrchk(cudaMalloc((void**)&d_js, sz_js));
    gpuErrchk(cudaMalloc((void**)&d_rnd, sz_rnd));

    cudaMemcpy(d_weights, new_weights, sz_weights, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rnd, h_rnds, sz_rnd, cudaMemcpyHostToDevice);
    gpuErrchk(cudaMemset(d_js, 0, sz_js));

    /********************************************************************/
    /************************ RESAMPLING kerenel ************************/
    /********************************************************************/
    auto start_resampling = std::chrono::high_resolution_clock::now();
    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;

    kernel_resampling << <blocksPerGrid, threadsPerBlock >> > (d_weights, d_js, d_rnd, NUM_PARTICLES);
    cudaDeviceSynchronize();
    auto stop_resampling = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_js, d_js, sz_js, cudaMemcpyDeviceToHost));

    ASSERT_resampling_indices(res_js, h_js, NUM_PARTICLES, false, true, false);
    ASSERT_resampling_states(h_states_x, h_states_y, h_states_theta,
        h_states_x_updated, h_states_y_updated, h_states_theta_updated, res_js, NUM_PARTICLES, false, true, true);

    auto duration_resampling = std::chrono::duration_cast<std::chrono::microseconds>(stop_resampling - start_resampling);
    std::cout << "Time taken by function (Kernel Resampling): " << duration_resampling.count() << " microseconds" << std::endl;
    std::cout << std::endl;
}

// Step 4
void host_update_state() {

    printf("/********************************************************************/\n");
    printf("/**************************** UPDATE STATE **************************/\n");
    printf("/********************************************************************/\n");

    sz_states_pos = NUM_PARTICLES * sizeof(float);

    gpuErrchk(cudaMalloc((void**)&d_states_x, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&d_states_y, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&d_states_theta, sz_states_pos));


    gpuErrchk(cudaMemcpy(d_states_x, h_states_x_updated, sz_states_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_states_y, h_states_y_updated, sz_states_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_states_theta, h_states_theta_updated, sz_states_pos, cudaMemcpyHostToDevice));


    auto start_update_states = std::chrono::high_resolution_clock::now();

    thrust::device_vector<float> d_vec_states_x(d_states_x, d_states_x + NUM_PARTICLES);
    thrust::device_vector<float> d_vec_states_y(d_states_y, d_states_y + NUM_PARTICLES);
    thrust::device_vector<float> d_vec_states_theta(d_states_theta, d_states_theta + NUM_PARTICLES);


    thrust::host_vector<float> h_vec_states_x(d_vec_states_x.begin(), d_vec_states_x.end());
    thrust::host_vector<float> h_vec_states_y(d_vec_states_y.begin(), d_vec_states_y.end());
    thrust::host_vector<float> h_vec_states_theta(d_vec_states_theta.begin(), d_vec_states_theta.end());

    std::vector<float> std_vec_states_x(h_vec_states_x.begin(), h_vec_states_x.end());
    std::vector<float> std_vec_states_y(h_vec_states_y.begin(), h_vec_states_y.end());
    std::vector<float> std_vec_states_theta(h_vec_states_theta.begin(), h_vec_states_theta.end());


    std::map<std::tuple<float, float, float>, int> states;

    for (int i = 0; i < NUM_PARTICLES; i++) {

        if (states.find(std::make_tuple(std_vec_states_x[i], std_vec_states_y[i], std_vec_states_theta[i])) == states.end()) {
            states.insert({ std::make_tuple(std_vec_states_x[i], std_vec_states_y[i], std_vec_states_theta[i]), 1 });
        }
        else {
            states[std::make_tuple(std_vec_states_x[i], std_vec_states_y[i], std_vec_states_theta[i])] += 1;
        }
    }

    std::map<std::tuple<float, float, float>, int>::iterator best
        = std::max_element(states.begin(), states.end(), [](const std::pair<std::tuple<float, float, float>, int>& a,
            const std::pair<std::tuple<float, float, float>, int>& b)->bool { return a.second < b.second; });

    auto key = best->first;
    printf("~~$ Max Weight: %d\n", best->second);

    float theta = std::get<2>(key);
    float res_transition_world_body[] = { cos(theta), -sin(theta), std::get<0>(key),
                        sin(theta),  cos(theta), std::get<1>(key),
                        0, 0, 1 };

    auto stop_update_states = std::chrono::high_resolution_clock::now();

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
    printf("~~$ Robot State (Result): %f, %f, %f\n", std::get<0>(key), std::get<1>(key), std::get<2>(key));
    printf("~~$ Robot State (Host)  : %f, %f, %f\n", h_robot_state[0], h_robot_state[1], h_robot_state[2]);

    std::cout << std::endl;
    auto duration_update_states = std::chrono::duration_cast<std::chrono::microseconds>(stop_update_states - start_update_states);
    std::cout << "Time taken by function (Update States): " << duration_update_states.count() << " microseconds" << std::endl;
    std::cout << std::endl;
}

// Step X
void host_update_func() {

    printf("/********************************************************************/\n");
    printf("/**************************** UPDATE FUNC ***************************/\n");
    printf("/********************************************************************/\n");

    PARTICLES_ITEMS_LEN = ST_PARTICLES_ITEMS_LEN;
    MEASURE_LEN = NUM_PARTICLES * LIDAR_COORDS_LEN;

    thrust::device_vector<float> d_temp(h_states_x, h_states_x + NUM_PARTICLES);

    int negative_before_counter = getNegativeCounter(h_particles_x, h_particles_y, PARTICLES_ITEMS_LEN);
    int count_bigger_than_height = getGreaterThanCounter(h_particles_y, GRID_HEIGHT, PARTICLES_ITEMS_LEN);
    int negative_after_counter = getNegativeCounter(h_particles_x_after_resampling, h_particles_y_after_resampling, AF_PARTICLES_ITEMS_LEN_RESAMPLING);;

    printf("~~$ GRID_WIDTH: \t\t%d\n", GRID_WIDTH);
    printf("~~$ GRID_HEIGHT: \t\t%d\n", GRID_HEIGHT);
    printf("~~$ negative_before_counter: \t%d\n", negative_before_counter);
    printf("~~$ negative_after_counter: \t%d\n", negative_after_counter);
    printf("~~$ count_bigger_than_height: \t%d\n", count_bigger_than_height);
    printf("~~$ MEASURE_LEN: \t\t%d\n", MEASURE_LEN);

    /**************************************************************************************************************************************************/
    /**************************************************************** VARIABLES SCOPE *****************************************************************/
    /**************************************************************************************************************************************************/

    /********************************************************************/
    /************************** STATES VARIABLES ************************/
    /********************************************************************/
    sz_states_pos = NUM_PARTICLES * sizeof(float);
    sz_lidar_coords = 2 * LIDAR_COORDS_LEN * sizeof(float);

    res_states_x = (float*)malloc(sz_states_pos);
    res_states_y = (float*)malloc(sz_states_pos);
    res_states_theta = (float*)malloc(sz_states_pos);

    gpuErrchk(cudaMalloc((void**)&d_states_x, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&d_states_y, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&d_states_theta, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&d_lidar_coords, sz_lidar_coords));

    gpuErrchk(cudaMemcpy(d_states_x, h_states_x, sz_states_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_states_y, h_states_y, sz_states_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_states_theta, h_states_theta, sz_states_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_lidar_coords, h_lidar_coords, sz_lidar_coords, cudaMemcpyHostToDevice));

    /********************************************************************/
    /************************* PARTICLES VARIABLES **********************/
    /********************************************************************/
    sz_particles_pos = PARTICLES_ITEMS_LEN * sizeof(int);
    sz_particles_idx = NUM_PARTICLES * sizeof(int);
    sz_extended_idx = PARTICLES_ITEMS_LEN * sizeof(int);
    sz_grid_map = GRID_WIDTH * GRID_HEIGHT * sizeof(int);

    sz_particles_weight = NUM_PARTICLES * sizeof(float);
    //d_particles_weight = NULL;
    res_particles_weight = (float*)malloc(sz_particles_weight);

    res_particles_x = (int*)malloc(sz_particles_pos);
    res_particles_y = (int*)malloc(sz_particles_pos);
    res_particles_idx = (int*)malloc(sz_particles_idx);
    res_extended_idx = (int*)malloc(sz_extended_idx);

    gpuErrchk(cudaMalloc((void**)&d_grid_map, sz_grid_map));
    gpuErrchk(cudaMalloc((void**)&d_particles_x, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_y, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_idx, sz_particles_idx));
    gpuErrchk(cudaMalloc((void**)&d_particles_weight, sz_particles_weight));
    gpuErrchk(cudaMalloc((void**)&d_extended_idx, sz_extended_idx));

    gpuErrchk(cudaMemcpy(d_grid_map, h_grid_map, sz_grid_map, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particles_x, h_particles_x, sz_particles_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particles_y, h_particles_y, sz_particles_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particles_idx, h_particles_idx, sz_particles_idx, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particles_weight, particles_weight_pre, sz_particles_weight, cudaMemcpyHostToDevice));


    /********************************************************************/
    /******************** PARTICLES COPY VARIABLES **********************/
    /********************************************************************/

    gpuErrchk(cudaMalloc((void**)&dc_states_x, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&dc_states_y, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&dc_states_theta, sz_states_pos));

    /********************************************************************/
    /********************** CORRELATION VARIABLES ***********************/
    /********************************************************************/
    sz_weights = NUM_PARTICLES * sizeof(float);
    sz_weights_raw = 25 * sz_weights;

    res_weights = (float*)malloc(sz_weights);
    int* h_extended_idx = (int*)malloc(sz_extended_idx);
    res_weights = (float*)malloc(sz_weights);
        
    memset(res_weights, 0, sz_weights);

    gpuErrchk(cudaMalloc((void**)&d_weights, sz_weights));
    gpuErrchk(cudaMalloc((void**)&d_weights_raw, sz_weights_raw));
    gpuErrchk(cudaMemset(d_weights_raw, 0, sz_weights_raw));

    /********************************************************************/
    /*********************** TRANSITION VARIABLES ***********************/
    /********************************************************************/
    sz_transition_world_frame = 9 * NUM_PARTICLES * sizeof(float);
    sz_transition_body_lidar = 9 * sizeof(float);
    sz_processed_measure_pos = NUM_PARTICLES * LIDAR_COORDS_LEN * sizeof(int);
    sz_processed_measure_idx = NUM_PARTICLES * LIDAR_COORDS_LEN * sizeof(int);

    res_transition_world_body = (float*)malloc(sz_transition_world_frame);
    res_transition_world_lidar = (float*)malloc(sz_transition_world_frame);
    res_processed_measure_x = (int*)malloc(sz_processed_measure_pos);
    res_processed_measure_y = (int*)malloc(sz_processed_measure_pos);

    gpuErrchk(cudaMalloc((void**)&d_transition_world_body, sz_transition_world_frame));
    gpuErrchk(cudaMalloc((void**)&d_transition_body_lidar, sz_transition_body_lidar));
    gpuErrchk(cudaMalloc((void**)&d_transition_world_lidar, sz_transition_world_frame));
    gpuErrchk(cudaMalloc((void**)&d_processed_measure_x, sz_processed_measure_pos));
    gpuErrchk(cudaMalloc((void**)&d_processed_measure_y, sz_processed_measure_pos));
    gpuErrchk(cudaMalloc((void**)&d_processed_measure_idx, sz_processed_measure_idx));

    gpuErrchk(cudaMemset(d_transition_world_body, 0, sz_transition_world_frame));
    gpuErrchk(cudaMemset(d_transition_world_lidar, 0, sz_transition_world_frame));
    gpuErrchk(cudaMemset(d_processed_measure_x, 0, sz_processed_measure_pos));
    gpuErrchk(cudaMemset(d_processed_measure_y, 0, sz_processed_measure_pos));
    gpuErrchk(cudaMemset(d_processed_measure_idx, 0, sz_processed_measure_idx));

    gpuErrchk(cudaMemcpy(d_transition_body_lidar, h_transition_body_lidar, sz_transition_body_lidar, cudaMemcpyHostToDevice));

    /********************************************************************/
    /**************************** MAP VARIABLES *************************/
    /********************************************************************/
    sz_map_2d = GRID_WIDTH * GRID_HEIGHT * NUM_PARTICLES * sizeof(uint8_t);
    sz_unique_in_particle = UNIQUE_COUNTER_LEN * sizeof(int);
    sz_unique_in_particle_col = UNIQUE_COUNTER_LEN * GRID_WIDTH * sizeof(int);

    res_map_2d = (uint8_t*)malloc(sz_map_2d);
    res_unique_in_particle = (int*)malloc(sz_unique_in_particle);

    gpuErrchk(cudaMalloc((void**)&d_map_2d, sz_map_2d));
    gpuErrchk(cudaMalloc((void**)&d_unique_in_particle, sz_unique_in_particle));
    gpuErrchk(cudaMalloc((void**)&d_unique_in_particle_col, sz_unique_in_particle_col));

    gpuErrchk(cudaMemset(d_map_2d, 0, sz_map_2d));
    gpuErrchk(cudaMemset(d_unique_in_particle, 0, sz_unique_in_particle));
    gpuErrchk(cudaMemset(d_unique_in_particle_col, 0, sz_unique_in_particle_col));

    /********************************************************************/
    /************************ WEIGHTS VARIABLES *************************/
    /********************************************************************/
    sz_weights_max = sizeof(float);
    sz_sum_exp = sizeof(double);

    res_weights_max = (float*)malloc(sz_weights_max);
    res_sum_exp = (double*)malloc(sz_sum_exp);

    gpuErrchk(cudaMalloc((void**)&d_weights_max, sz_weights_max));
    gpuErrchk(cudaMalloc((void**)&d_sum_exp, sz_sum_exp));

    gpuErrchk(cudaMemset(d_weights_max, 0, sz_weights_max));
    gpuErrchk(cudaMemset(d_sum_exp, 0, sz_sum_exp));

    /********************************************************************/
    /*********************** RESAMPLING VARIABLES ***********************/
    /********************************************************************/
    sz_js = NUM_PARTICLES * sizeof(int);
    sz_rnd = NUM_PARTICLES * sizeof(float);

    res_js = (int*)malloc(sz_js);

    gpuErrchk(cudaMalloc((void**)&d_js, sz_js));
    gpuErrchk(cudaMalloc((void**)&d_rnd, sz_rnd));

    gpuErrchk(cudaMemcpy(d_rnd, h_rnds, sz_rnd, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(d_js, 0, sz_js));

    /********************************************************************/
    /********************* REARRANGEMENT VARIABLES **********************/
    /********************************************************************/
    std::vector<float> std_vec_states_x;
    std::vector<float> std_vec_states_y;
    std::vector<float> std_vec_states_theta;

    /**************************************************************************************************************************************************/
    /************************************************************* KERNEL EXECUTION SCOPE *************************************************************/
    /**************************************************************************************************************************************************/

    /********************************************************************/
    /************************ TRANSITION KERNEL *************************/
    /********************************************************************/
    auto start_transition_kernel = std::chrono::high_resolution_clock::now();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;

    kernel_update_particles_states << <blocksPerGrid, threadsPerBlock >> > (d_states_x, d_states_y, d_states_theta, 
        d_transition_world_body, d_transition_body_lidar, d_transition_world_lidar, NUM_PARTICLES);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = LIDAR_COORDS_LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (d_transition_world_lidar, d_processed_measure_x, d_processed_measure_y, 
        d_lidar_coords, res, xmin, ymax, LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_index_init_const << < blocksPerGrid, threadsPerBlock >> > (d_processed_measure_idx, LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, d_processed_measure_idx, d_processed_measure_idx + NUM_PARTICLES, d_processed_measure_idx, 0);

    auto stop_transition_kernel = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_transition_world_body, d_transition_world_body, sz_transition_world_frame, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_transition_world_lidar, d_transition_world_lidar, sz_transition_world_frame, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_processed_measure_x, d_processed_measure_x, sz_processed_measure_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_processed_measure_y, d_processed_measure_y, sz_processed_measure_pos, cudaMemcpyDeviceToHost));

    //ASSERT_transition_frames(res_transition_world_body, res_transition_world_lidar,
    //    h_transition_world_body, h_transition_world_lidar, NUM_PARTICLES, false);
    // ASSERT_processed_measurements(res_processed_measure_x, res_processed_measure_y, processed_measure, NUM_PARTICLES, LIDAR_COORDS_LEN);

    /********************************************************************/
    /************************** CREATE 2D MAP ***************************/
    /********************************************************************/
    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;

    auto start_create_map = std::chrono::high_resolution_clock::now();
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (d_particles_x, d_particles_y, d_particles_idx, 
        PARTICLES_ITEMS_LEN, d_map_2d, d_unique_in_particle,
        d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();
    auto stop_create_map = std::chrono::high_resolution_clock::now();

    //cudaError_t err = cudaPeekAtLastError();
    //printf("%s\n", cudaGetErrorString(err));

    gpuErrchk(cudaMemcpy(res_map_2d, d_map_2d, sz_map_2d, cudaMemcpyDeviceToHost));

    ASSERT_create_2d_map_elements(res_map_2d, negative_before_counter, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES, PARTICLES_ITEMS_LEN, true, true);

    /********************************************************************/
    /**************************** UPDATE MAP ****************************/
    /********************************************************************/
    auto start_update_map = std::chrono::high_resolution_clock::now();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;

    kernel_update_2d_map_with_measure << <blocksPerGrid, threadsPerBlock >> > (d_processed_measure_x, d_processed_measure_y, d_processed_measure_idx,
        MEASURE_LEN, d_map_2d, d_unique_in_particle,
        d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_update_map = std::chrono::high_resolution_clock::now();

    /********************************************************************/
    /************************* CUMULATIVE SUM ***************************/
    /********************************************************************/
    auto start_cumulative_sum = std::chrono::high_resolution_clock::now();

    threadsPerBlock = UNIQUE_COUNTER_LEN;
    blocksPerGrid = 1;
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (d_unique_in_particle_col, GRID_WIDTH);
    thrust::exclusive_scan(thrust::device, d_unique_in_particle, d_unique_in_particle + UNIQUE_COUNTER_LEN, d_unique_in_particle, 0);
    cudaDeviceSynchronize();

    auto stop_cumulative_sum = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_unique_in_particle, d_unique_in_particle, sz_unique_in_particle, cudaMemcpyDeviceToHost));

    PARTICLES_ITEMS_LEN = res_unique_in_particle[UNIQUE_COUNTER_LEN - 1];    
    C_PARTICLES_ITEMS_LEN = 0;
    //ASSERT_new_len_calculation(NEW_LEN, ELEMS_PARTICLES_AFTER, negative_after_counter);


    /*---------------------------------------------------------------------*/
    /*-------------------- REINITIALIZE MAP VARIABLES ---------------------*/
    /*---------------------------------------------------------------------*/
    /*---------------------------------------------------------------------*/
    gpuErrchk(cudaFree(d_particles_x));
    gpuErrchk(cudaFree(d_particles_y));
    gpuErrchk(cudaFree(d_extended_idx));
    free(res_particles_x);
    free(res_particles_y);

    sz_particles_pos = PARTICLES_ITEMS_LEN * sizeof(int);
    sz_extended_idx = PARTICLES_ITEMS_LEN * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_particles_x, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_y, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_extended_idx, sz_extended_idx));

    res_particles_x = (int*)malloc(sz_particles_pos);
    res_particles_y = (int*)malloc(sz_particles_pos);

    /********************************************************************/
    /************************ MAP RESTRUCTURE ***************************/
    /********************************************************************/
    threadsPerBlock = GRID_WIDTH;
    blocksPerGrid = NUM_PARTICLES;

    auto start_map_restructure = std::chrono::high_resolution_clock::now();

    cudaMemset(d_particles_idx, 0, sz_particles_idx);
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (d_map_2d, d_particles_x, d_particles_y, d_particles_idx,
        d_unique_in_particle, d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, d_particles_idx, d_particles_idx + NUM_PARTICLES, d_particles_idx, 0);

    auto stop_map_restructure = std::chrono::high_resolution_clock::now();

    auto start_copy_particles_pos = std::chrono::high_resolution_clock::now();
    gpuErrchk(cudaMemcpy(res_particles_x, d_particles_x, sz_particles_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_y, d_particles_y, sz_particles_pos, cudaMemcpyDeviceToHost));
    auto stop_copy_particles_pos = std::chrono::high_resolution_clock::now();

    // ASSERT_particles_pos_unique(res_particles_x, res_particles_y, h_particles_x_after_unique, h_particles_y_after_unique, NEW_LEN);

    /********************************************************************/
    /************************* INDEX EXPANSION **************************/
    /********************************************************************/
    auto start_index_expansion = std::chrono::high_resolution_clock::now();

    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;
    kernel_index_expansion << <blocksPerGrid, threadsPerBlock >> > (d_particles_idx, d_extended_idx, PARTICLES_ITEMS_LEN);
    cudaDeviceSynchronize();

    auto stop_index_expansion = std::chrono::high_resolution_clock::now();

    res_extended_idx = (int*)malloc(sz_extended_idx);
    gpuErrchk(cudaMemcpy(res_extended_idx, d_extended_idx, sz_extended_idx, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToHost));

    /********************************************************************/
    /************************ KERNEL CORRELATION ************************/
    /********************************************************************/
    threadsPerBlock = 256;
    blocksPerGrid = (PARTICLES_ITEMS_LEN + threadsPerBlock - 1) / threadsPerBlock;

    auto start_correlation = std::chrono::high_resolution_clock::now();
    kernel_correlation << <blocksPerGrid, threadsPerBlock >> > (d_grid_map, d_particles_x, d_particles_y, d_extended_idx, d_weights_raw, GRID_WIDTH, GRID_HEIGHT, PARTICLES_ITEMS_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_correlation_max << <blocksPerGrid, threadsPerBlock >> > (d_weights_raw, d_weights, NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_correlation = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_weights, d_weights, sz_weights, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_extended_idx, d_extended_idx, sz_extended_idx, cudaMemcpyDeviceToHost));

    ASSERT_correlation_Equality(res_weights, pre_weights, NUM_PARTICLES, false, true);

    /********************************************************************/
    /********************** UPDATE WEIGHTS KERNEL ***********************/
    /********************************************************************/
    auto start_update_particle_weights = std::chrono::high_resolution_clock::now();

    threadsPerBlock = 1;
    blocksPerGrid = 1;

    kernel_arr_max << < blocksPerGrid, threadsPerBlock >> > (d_weights, d_weights_max, NUM_PARTICLES);
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(res_weights_max, d_weights_max, sz_weights_max, cudaMemcpyDeviceToHost));

    float norm_value = -res_weights_max[0] + 50;

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_arr_increase << < blocksPerGrid, threadsPerBlock >> > (d_weights, norm_value, 0);
    cudaDeviceSynchronize();

    threadsPerBlock = 1;
    blocksPerGrid = 1;
    kernel_arr_sum_exp << < blocksPerGrid, threadsPerBlock >> > (d_weights, d_sum_exp, NUM_PARTICLES);
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(res_sum_exp, d_sum_exp, sz_sum_exp, cudaMemcpyDeviceToHost));

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_arr_normalize << < blocksPerGrid, threadsPerBlock >> > (d_weights, res_sum_exp[0]);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_arr_mult << < blocksPerGrid, threadsPerBlock >> > (d_particles_weight, d_weights);
    cudaDeviceSynchronize();

    auto stop_update_particle_weights = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_weights, d_weights, sz_weights, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_weight, d_particles_weight, sz_particles_weight, cudaMemcpyDeviceToHost));

    ASSERT_update_particle_weights(res_weights, new_weights, NUM_PARTICLES, "weights", false, false, true);
    ASSERT_update_particle_weights(res_particles_weight, particles_weight_post, NUM_PARTICLES, "particles weight", false, false, true);

    /********************************************************************/
    /************************ RESAMPLING KERNEL *************************/
    /********************************************************************/
    auto start_resampling = std::chrono::high_resolution_clock::now();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_resampling << <blocksPerGrid, threadsPerBlock >> > (d_weights, d_js, d_rnd, NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_resampling = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_js, d_js, sz_js, cudaMemcpyDeviceToHost));

    ASSERT_resampling_indices(res_js, h_js, NUM_PARTICLES, false, false, true);
    ASSERT_resampling_states(h_states_x, h_states_y, h_states_theta, 
        h_states_x_updated, h_states_y_updated, h_states_theta_updated, res_js, NUM_PARTICLES, false, false, true);


    /*---------------------------------------------------------------------*/
    /*----------------- REINITIALIZE PARTICLES VARIABLES ------------------*/
    /*---------------------------------------------------------------------*/
    /*---------------------------------------------------------------------*/
    sz_last_len = sizeof(int);
    //int* d_last_len = NULL;
    res_last_len = (int*)malloc(sizeof(int));

    gpuErrchk(cudaMalloc((void**)&d_last_len, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dc_particles_x, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&dc_particles_y, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&dc_particles_idx, sz_particles_idx));

    auto start_clone_particles = std::chrono::high_resolution_clock::now();
    gpuErrchk(cudaMemcpy(dc_particles_x, d_particles_x, sz_particles_pos, cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(dc_particles_y, d_particles_y, sz_particles_pos, cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(dc_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToDevice));

    gpuErrchk(cudaMemcpy(dc_states_x, d_states_x, sz_states_pos, cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(dc_states_y, d_states_y, sz_states_pos, cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(dc_states_theta, d_states_theta, sz_states_pos, cudaMemcpyDeviceToDevice));
    auto stop_clone_particles = std::chrono::high_resolution_clock::now();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_rearrange_indecies << <blocksPerGrid, threadsPerBlock >> > (d_particles_idx, dc_particles_idx, d_js, d_last_len, PARTICLES_ITEMS_LEN);
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(res_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_last_len, d_last_len, sz_last_len, cudaMemcpyDeviceToHost));

    /********************************************************************/
    /********************** REARRANGEMENT KERNEL ************************/
    /********************************************************************/
    auto start_rearrange_index = std::chrono::high_resolution_clock::now();
    thrust::exclusive_scan(thrust::device, d_particles_idx, d_particles_idx + NUM_PARTICLES, d_particles_idx, 0);
    auto stop_rearrange_index = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToHost));
    C_PARTICLES_ITEMS_LEN = PARTICLES_ITEMS_LEN;
    PARTICLES_ITEMS_LEN = res_particles_idx[NUM_PARTICLES - 1] + res_last_len[0];
    printf("--> PARTICLES_ITEMS_LEN=%d <> ELEMS_PARTICLES_AFTER=%d\n", PARTICLES_ITEMS_LEN, AF_PARTICLES_ITEMS_LEN_RESAMPLING);
    assert(PARTICLES_ITEMS_LEN + negative_after_counter == AF_PARTICLES_ITEMS_LEN_RESAMPLING);

    free(res_particles_x);
    free(res_particles_y);
    sz_particles_pos = PARTICLES_ITEMS_LEN * sizeof(int);
    res_particles_x = (int*)malloc(sz_particles_pos);
    res_particles_y = (int*)malloc(sz_particles_pos);

    ASSERT_resampling_particles_index(h_particles_idx_after_resampling, res_particles_idx, NUM_PARTICLES, false, negative_after_counter);

    auto start_rearrange_particles_states = std::chrono::high_resolution_clock::now();
    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;
    kernel_rearrange_particles << <blocksPerGrid, threadsPerBlock >> > (d_particles_x, d_particles_y, d_particles_idx,
        dc_particles_x, dc_particles_y, dc_particles_idx, d_js,
        GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES, PARTICLES_ITEMS_LEN, C_PARTICLES_ITEMS_LEN);

    kernel_rearrange_states << <blocksPerGrid, threadsPerBlock >> > (d_states_x, d_states_y, d_states_theta,
        dc_states_x, dc_states_y, dc_states_theta, d_js);
    cudaDeviceSynchronize();
    auto stop_rearrange_particles_states = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_particles_x, d_particles_x, sz_particles_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_y, d_particles_y, sz_particles_pos, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaMemcpy(res_states_x, d_states_x, sz_states_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_states_y, d_states_y, sz_states_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_states_theta, d_states_theta, sz_states_pos, cudaMemcpyDeviceToHost));

    ASSERT_rearrange_particles_states(res_particles_x, res_particles_y, res_states_x, res_states_y, res_states_theta,
        h_particles_x_after_resampling, h_particles_y_after_resampling, h_states_x_updated, h_states_y_updated, h_states_theta_updated,
        PARTICLES_ITEMS_LEN, NUM_PARTICLES);


    /********************************************************************/
    /********************** UPDATE STATES KERNEL ************************/
    /********************************************************************/
    auto start_update_states = std::chrono::high_resolution_clock::now();

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
    float res_robot_transition_world_body[] = { cos(theta), -sin(theta), std::get<0>(key),
                        sin(theta),  cos(theta), std::get<1>(key),
                        0, 0, 1 };
    auto stop_update_states = std::chrono::high_resolution_clock::now();

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
    printf("~~$ Robot State (Result): %f, %f, %f\n", std::get<0>(key), std::get<1>(key), std::get<2>(key));
    printf("~~$ Robot State (Host)  : %f, %f, %f\n", h_robot_state[0], h_robot_state[1], h_robot_state[2]);


    /********************************************************************/
    /************************* EXECUTION TIMES **************************/
    /********************************************************************/
    auto duration_create_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_create_map - start_create_map);
    auto duration_update_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_update_map - start_update_map);
    auto duration_cumulative_sum = std::chrono::duration_cast<std::chrono::microseconds>(stop_cumulative_sum - start_cumulative_sum);
    auto duration_map_restructure = std::chrono::duration_cast<std::chrono::microseconds>(stop_map_restructure - start_map_restructure);
    auto duration_copy_particles_pos = std::chrono::duration_cast<std::chrono::microseconds>(stop_copy_particles_pos - start_copy_particles_pos);
    auto duration_transition_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_transition_kernel - start_transition_kernel);
    auto duration_correlation = std::chrono::duration_cast<std::chrono::microseconds>(stop_correlation - start_correlation);
    auto duration_update_particle_weights = std::chrono::duration_cast<std::chrono::microseconds>(stop_update_particle_weights - start_update_particle_weights);
    auto duration_resampling = std::chrono::duration_cast<std::chrono::microseconds>(stop_resampling - start_resampling);
    auto duration_clone_particles = std::chrono::duration_cast<std::chrono::microseconds>(stop_clone_particles - start_clone_particles);
    auto duration_rearrange_particles_states = std::chrono::duration_cast<std::chrono::microseconds>(stop_rearrange_particles_states - start_rearrange_particles_states);
    auto duration_rearrange_index = std::chrono::duration_cast<std::chrono::microseconds>(stop_rearrange_index - start_rearrange_index);
    auto duration_update_states = std::chrono::duration_cast<std::chrono::microseconds>(stop_update_states - start_update_states);

    auto duration_sum = duration_create_map + duration_update_map + duration_cumulative_sum + duration_map_restructure + duration_copy_particles_pos +
        duration_transition_kernel + duration_correlation + duration_update_particle_weights + duration_resampling + duration_clone_particles +
        duration_rearrange_particles_states + duration_rearrange_index + duration_update_states;

    std::cout << std::endl;
    std::cout << "Time taken by function (Create Map): " << duration_create_map.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Update Map): " << duration_update_map.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Cumulative Sum): " << duration_cumulative_sum.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Map Restructure): " << duration_map_restructure.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Copy Particles): " << duration_copy_particles_pos.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Transition Kernel): " << duration_transition_kernel.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Correlation Kernel): " << duration_correlation.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Update Particle Weights): " << duration_update_particle_weights.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Kernel Resampling): " << duration_resampling.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Clone Particles): " << duration_clone_particles.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Rearrange Particles States): " << duration_rearrange_particles_states.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Rearrange Index): " << duration_rearrange_index.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Update States): " << duration_update_states.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Sum): " << duration_sum.count() << " microseconds" << std::endl;

    printf("\nFinished All\n");
}


void alloc_state_vars() {

    sz_states_pos = NUM_PARTICLES * sizeof(float);
    sz_lidar_coords = 2 * LIDAR_COORDS_LEN * sizeof(float);

    gpuErrchk(cudaMalloc((void**)&d_states_x, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&d_states_y, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&d_states_theta, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&d_lidar_coords, sz_lidar_coords));

    res_robot_state = (float*)malloc(3 * sizeof(float));
}

void init_state_vars() {

    gpuErrchk(cudaMemcpy(d_states_x, h_states_x, sz_states_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_states_y, h_states_y, sz_states_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_states_theta, h_states_theta, sz_states_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_lidar_coords, h_lidar_coords, sz_lidar_coords, cudaMemcpyHostToDevice));
}

void alloc_particles_vars() {

    sz_particles_pos = PARTICLES_ITEMS_LEN * sizeof(int);
    sz_particles_idx = NUM_PARTICLES * sizeof(int);
    sz_particles_weight = NUM_PARTICLES * sizeof(float);
    sz_extended_idx = PARTICLES_ITEMS_LEN * sizeof(int);
    sz_grid_map = GRID_WIDTH * GRID_HEIGHT * sizeof(int);

    res_particles_idx = (int*)malloc(sz_particles_idx);
    res_extended_idx = (int*)malloc(sz_extended_idx);

    gpuErrchk(cudaMalloc((void**)&d_grid_map, sz_grid_map));
    gpuErrchk(cudaMalloc((void**)&d_particles_x, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_y, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_idx, sz_particles_idx));
    gpuErrchk(cudaMalloc((void**)&d_particles_weight, sz_particles_weight));
    gpuErrchk(cudaMalloc((void**)&d_extended_idx, sz_extended_idx));
}

void init_particles_vars() {

    gpuErrchk(cudaMemcpy(d_grid_map, h_grid_map, sz_grid_map, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particles_x, h_particles_x, sz_particles_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particles_y, h_particles_y, sz_particles_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particles_idx, h_particles_idx, sz_particles_idx, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particles_weight, particles_weight_pre, sz_particles_weight, cudaMemcpyHostToDevice));
}

void alloc_particles_copy_vars() {

    gpuErrchk(cudaMalloc((void**)&dc_states_x, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&dc_states_y, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&dc_states_theta, sz_states_pos));
}

void alloc_correlation_vars() {

    sz_weights = NUM_PARTICLES * sizeof(float);
    sz_weights_raw = 25 * sz_weights;

    res_weights = (float*)malloc(sz_weights);
    res_extended_idx = (int*)malloc(sz_extended_idx);
    d_weights = NULL;
    d_weights_raw = NULL;

    memset(res_weights, 0, sz_weights);

    gpuErrchk(cudaMalloc((void**)&d_weights, sz_weights));
    gpuErrchk(cudaMalloc((void**)&d_weights_raw, sz_weights_raw));
    gpuErrchk(cudaMemset(d_weights_raw, 0, sz_weights_raw));
}

void alloc_transition_vars() {

    sz_transition_world_frame = 9 * NUM_PARTICLES * sizeof(float);
    sz_transition_body_lidar = 9 * sizeof(float);
    //sz_transition_world_lidar = 9 * NUM_PARTICLES * sizeof(float);
    sz_processed_measure_pos = NUM_PARTICLES * LIDAR_COORDS_LEN * sizeof(int);
    sz_processed_measure_idx = NUM_PARTICLES * LIDAR_COORDS_LEN * sizeof(int);

    d_transition_world_body = NULL;
    d_transition_body_lidar = NULL;
    d_transition_world_lidar = NULL;
    d_processed_measure_x = NULL;
    d_processed_measure_y = NULL;
    d_processed_measure_idx = NULL;

    res_transition_world_body = (float*)malloc(sz_transition_world_frame);
    res_transition_world_lidar = (float*)malloc(sz_transition_world_frame);
    res_robot_world_body = (float*)malloc(sz_transition_world_frame);

    gpuErrchk(cudaMalloc((void**)&d_transition_world_body, sz_transition_world_frame));
    gpuErrchk(cudaMalloc((void**)&d_transition_body_lidar, sz_transition_body_lidar));
    gpuErrchk(cudaMalloc((void**)&d_transition_world_lidar, sz_transition_world_frame));
    gpuErrchk(cudaMalloc((void**)&d_processed_measure_x, sz_processed_measure_pos));
    gpuErrchk(cudaMalloc((void**)&d_processed_measure_y, sz_processed_measure_pos));
    gpuErrchk(cudaMalloc((void**)&d_processed_measure_idx, sz_processed_measure_idx));
}

void init_transition_vars() {

    gpuErrchk(cudaMemcpy(d_transition_body_lidar, h_transition_body_lidar, sz_transition_body_lidar, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemset(d_transition_world_body, 0, sz_transition_world_frame));
    gpuErrchk(cudaMemset(d_transition_world_lidar, 0, sz_transition_world_frame));
    gpuErrchk(cudaMemset(d_processed_measure_x, 0, sz_processed_measure_pos));
    gpuErrchk(cudaMemset(d_processed_measure_y, 0, sz_processed_measure_pos));
    gpuErrchk(cudaMemset(d_processed_measure_idx, 0, sz_processed_measure_idx));
}

void alloc_map_vars() {

    sz_map_2d = GRID_WIDTH * GRID_HEIGHT * NUM_PARTICLES * sizeof(uint8_t);
    sz_unique_in_particle = UNIQUE_COUNTER_LEN * sizeof(int);
    sz_unique_in_particle_col = UNIQUE_COUNTER_LEN * GRID_WIDTH * sizeof(int);

    res_unique_in_particle = (int*)malloc(sz_unique_in_particle);

    gpuErrchk(cudaMalloc((void**)&d_map_2d, sz_map_2d));
    gpuErrchk(cudaMalloc((void**)&d_unique_in_particle, sz_unique_in_particle));
    gpuErrchk(cudaMalloc((void**)&d_unique_in_particle_col, sz_unique_in_particle_col));

    gpuErrchk(cudaMemset(d_map_2d, 0, sz_map_2d));
    gpuErrchk(cudaMemset(d_unique_in_particle, 0, sz_unique_in_particle));
    gpuErrchk(cudaMemset(d_unique_in_particle_col, 0, sz_unique_in_particle_col));
}

void alloc_weights_vars() {

    sz_weights_max = sizeof(float);
    sz_sum_exp = sizeof(double);

    res_weights_max = (float*)malloc(sz_weights_max);
    res_sum_exp = (double*)malloc(sz_sum_exp);

    gpuErrchk(cudaMalloc((void**)&d_weights_max, sz_weights_max));
    gpuErrchk(cudaMalloc((void**)&d_sum_exp, sz_sum_exp));

    gpuErrchk(cudaMemset(d_weights_max, 0, sz_weights_max));
    gpuErrchk(cudaMemset(d_sum_exp, 0, sz_sum_exp));
}

void alloc_resampling_vars() {

    sz_js = NUM_PARTICLES * sizeof(int);
    sz_rnd = NUM_PARTICLES * sizeof(float);

    d_js = NULL;
    d_rnd = NULL;

    gpuErrchk(cudaMalloc((void**)&d_js, sz_js));
    gpuErrchk(cudaMalloc((void**)&d_rnd, sz_rnd));

    gpuErrchk(cudaMemcpy(d_rnd, h_rnds, sz_rnd, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(d_js, 0, sz_js));
}


void exec_transition() {

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_update_particles_states << <blocksPerGrid, threadsPerBlock >> > (d_states_x, d_states_y, d_states_theta, d_transition_world_body, d_transition_body_lidar, d_transition_world_lidar, NUM_PARTICLES);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = LIDAR_COORDS_LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (d_transition_world_lidar, d_processed_measure_x, d_processed_measure_y, d_lidar_coords, res, xmin, ymax, LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_index_init_const << < blocksPerGrid, threadsPerBlock >> > (d_processed_measure_idx, LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, d_processed_measure_idx, d_processed_measure_idx + NUM_PARTICLES, d_processed_measure_idx, 0);

    gpuErrchk(cudaMemcpy(res_transition_world_body, d_transition_world_body, sz_transition_world_frame, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_transition_world_lidar, d_transition_world_lidar, sz_transition_world_frame, cudaMemcpyDeviceToHost));
}

void exec_create_2d_map() {

    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (d_particles_x, d_particles_y, d_particles_idx, PARTICLES_ITEMS_LEN, d_map_2d, d_unique_in_particle,
        d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();
}

void exec_update_map() {

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;

    kernel_update_2d_map_with_measure << <blocksPerGrid, threadsPerBlock >> > (d_processed_measure_x, d_processed_measure_y, d_processed_measure_idx,
        MEASURE_LEN, d_map_2d, d_unique_in_particle,
        d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();
}

void exec_cum_sum() {

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

    gpuErrchk(cudaFree(d_particles_x));
    gpuErrchk(cudaFree(d_particles_y));
    gpuErrchk(cudaFree(d_extended_idx));

    sz_particles_pos = PARTICLES_ITEMS_LEN * sizeof(int);
    sz_extended_idx = PARTICLES_ITEMS_LEN * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_particles_x, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_y, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_extended_idx, sz_extended_idx));
}

void exec_map_restructure() {

    threadsPerBlock = GRID_WIDTH;
    blocksPerGrid = NUM_PARTICLES;

    cudaMemset(d_particles_idx, 0, sz_particles_idx);
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (d_map_2d, d_particles_x, d_particles_y, d_particles_idx,
        d_unique_in_particle, d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, d_particles_idx, d_particles_idx + NUM_PARTICLES, d_particles_idx, 0);
}

void exec_index_expansion() {

    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;
    kernel_index_expansion << <blocksPerGrid, threadsPerBlock >> > (d_particles_idx, d_extended_idx, PARTICLES_ITEMS_LEN);
    cudaDeviceSynchronize();

    res_extended_idx = (int*)malloc(sz_extended_idx);
    gpuErrchk(cudaMemcpy(res_extended_idx, d_extended_idx, sz_extended_idx, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToHost));
}

void exec_correlation() {

    threadsPerBlock = 256;
    blocksPerGrid = (PARTICLES_ITEMS_LEN + threadsPerBlock - 1) / threadsPerBlock;

    kernel_correlation << <blocksPerGrid, threadsPerBlock >> > (d_grid_map, d_particles_x, d_particles_y,
        d_extended_idx, d_weights_raw, GRID_WIDTH, GRID_HEIGHT, PARTICLES_ITEMS_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_correlation_max << <blocksPerGrid, threadsPerBlock >> > (d_weights_raw, d_weights, NUM_PARTICLES);
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(res_extended_idx, d_extended_idx, sz_extended_idx, cudaMemcpyDeviceToHost));
}

void exec_update_weights() {

    threadsPerBlock = 1;
    blocksPerGrid = 1;

    kernel_arr_max << < blocksPerGrid, threadsPerBlock >> > (d_weights, d_weights_max, NUM_PARTICLES);
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(res_weights_max, d_weights_max, sz_weights_max, cudaMemcpyDeviceToHost));

    float norm_value = -res_weights_max[0] + 50;

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_arr_increase << < blocksPerGrid, threadsPerBlock >> > (d_weights, norm_value, 0);
    cudaDeviceSynchronize();

    threadsPerBlock = 1;
    blocksPerGrid = 1;
    kernel_arr_sum_exp << < blocksPerGrid, threadsPerBlock >> > (d_weights, d_sum_exp, NUM_PARTICLES);
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(res_sum_exp, d_sum_exp, sz_sum_exp, cudaMemcpyDeviceToHost));

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_arr_normalize << < blocksPerGrid, threadsPerBlock >> > (d_weights, res_sum_exp[0]);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_arr_mult << < blocksPerGrid, threadsPerBlock >> > (d_particles_weight, d_weights);
    cudaDeviceSynchronize();
}

void exec_resampling() {

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_resampling << <blocksPerGrid, threadsPerBlock >> > (d_weights, d_js, d_rnd, NUM_PARTICLES);
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
    kernel_rearrange_indecies << <blocksPerGrid, threadsPerBlock >> > (d_particles_idx, dc_particles_idx, d_js, d_last_len, PARTICLES_ITEMS_LEN);
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
    kernel_rearrange_particles << <blocksPerGrid, threadsPerBlock >> > (d_particles_x, d_particles_y, d_particles_idx,
        dc_particles_x, dc_particles_y, dc_particles_idx, d_js,
        GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES, PARTICLES_ITEMS_LEN, C_PARTICLES_ITEMS_LEN);

    kernel_rearrange_states << <blocksPerGrid, threadsPerBlock >> > (d_states_x, d_states_y, d_states_theta,
        dc_states_x, dc_states_y, dc_states_theta, d_js);
    cudaDeviceSynchronize();
}

/********************************************************************/
/********************* UPDATE STATES VARIABLES **********************/
/********************************************************************/
std::vector<float> std_vec_states_x;
std::vector<float> std_vec_states_y;
std::vector<float> std_vec_states_theta;

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

    printf("/********************************************************************/\n");
    printf("/****************************** ROBOT  ******************************/\n");
    printf("/********************************************************************/\n");

    thrust::device_vector<float> d_temp(h_states_x, h_states_x + NUM_PARTICLES);

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
    alloc_state_vars();
    alloc_particles_vars();
    alloc_particles_copy_vars();
    alloc_correlation_vars();
    alloc_transition_vars();
    alloc_map_vars();
    alloc_weights_vars();
    alloc_resampling_vars();
    auto stop_robot_particles_alloc = std::chrono::high_resolution_clock::now();

    auto start_robot_particles_init = std::chrono::high_resolution_clock::now();
    init_state_vars();
    init_particles_vars();
    init_transition_vars();
    auto stop_robot_particles_init = std::chrono::high_resolution_clock::now();

    auto start_robot_particles_kernel = std::chrono::high_resolution_clock::now();
    exec_transition();
    exec_create_2d_map();
    exec_update_map();
    exec_cum_sum();
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

#endif
