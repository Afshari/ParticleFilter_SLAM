#ifndef _TEST_MAP_H_
#define _TEST_MAP_H_

#include "headers.h"
#include "host_asserts.h"
#include "host_utils.h"
#include "kernels.cuh"
#include "kernels_map.cuh"
#include "kernels_utils.cuh"
#include "kernels_robot.cuh"

//#include "data/map/721.h"
//#include "data/map/2789.h"

#include "data/map/400.h"

void host_update_map_init();                    // Step 1
void host_bresenham();                          // Step 2
void host_update_map();                         // Step 3
void host_map();                                // Step Y

void test_map_func();

int LIDAR_COORDS_LEN = 0;
int GRID_WIDTH = 0;
int GRID_HEIGHT = 0;

float res = ST_res;
float log_t = ST_log_t;

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


int test_map_main() {

    const int data_len = 6;
    int data[data_len] = { 1, 0, 2, 2, 1, 3 };
    int* d_data = NULL;
    gpuErrchk(cudaMalloc((void**)&d_data, data_len * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_data, data, data_len * sizeof(int), cudaMemcpyHostToDevice));

    thrust::exclusive_scan(thrust::host, data, data + 6, data, 0);
    thrust::exclusive_scan(thrust::device, d_data, d_data + 6, d_data, 0);

    //host_update_map_init();
    //host_bresenham();
    //host_update_map();
    //host_map();

    //test_map_func();
    test_map_func();

    return 0;
}

void host_update_map_init() {

    printf("/************************** UPDATE MAP INIT *************************/\n");

    xmin = ST_xmin;
    xmax = ST_xmax;
    ymin = ST_ymin;
    ymax = ST_ymax;

    int xmin_pre = xmin;
    int ymax_pre = ymax;

    GRID_WIDTH = ST_GRID_WIDTH;
    GRID_HEIGHT = ST_GRID_HEIGHT;
    LIDAR_COORDS_LEN = ST_LIDAR_COORDS_LEN;

    /********************* IMAGE TRANSFORM VARIABLES ********************/
    sz_transition_single_frame = 9 * sizeof(float);
    sz_transition_body_lidar = 9 * sizeof(float);
    sz_lidar_coords = 2 * LIDAR_COORDS_LEN * sizeof(float);
    sz_processed_single_measure_pos = LIDAR_COORDS_LEN * sizeof(int);
    sz_particles_world_pos = LIDAR_COORDS_LEN * sizeof(float);
    sz_position_image_body = 2 * sizeof(int);

    res_transition_world_lidar = (float*)malloc(sz_transition_single_frame);
    res_processed_single_measure_x = (int*)malloc(sz_processed_single_measure_pos);
    res_processed_single_measure_y = (int*)malloc(sz_processed_single_measure_pos);
    res_particles_world_x = (float*)malloc(sz_particles_world_pos);
    res_particles_world_y = (float*)malloc(sz_particles_world_pos);
    res_position_image_body = (int*)malloc(sz_position_image_body);

    gpuErrchk(cudaMalloc((void**)&d_lidar_coords, sz_lidar_coords));
    gpuErrchk(cudaMalloc((void**)&d_transition_body_lidar, sz_transition_body_lidar));
    gpuErrchk(cudaMalloc((void**)&d_transition_single_world_body, sz_transition_single_frame));
    gpuErrchk(cudaMalloc((void**)&d_transition_single_world_lidar, sz_transition_single_frame));
    gpuErrchk(cudaMalloc((void**)&d_processed_single_measure_x, sz_processed_single_measure_pos));
    gpuErrchk(cudaMalloc((void**)&d_processed_single_measure_y, sz_processed_single_measure_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_world_x, sz_particles_world_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_world_y, sz_particles_world_pos));
    gpuErrchk(cudaMalloc((void**)&d_position_image_body, sz_position_image_body));

    gpuErrchk(cudaMemcpy(d_lidar_coords, h_lidar_coords, sz_lidar_coords, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_transition_body_lidar, h_transition_body_lidar, sz_transition_body_lidar, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_transition_single_world_body, h_transition_world_body, sz_transition_single_frame, cudaMemcpyHostToDevice));


    sz_lidar_coords = 2 * LIDAR_COORDS_LEN * sizeof(float);
    sz_should_extend = 4 * sizeof(int);
    sz_grid_map = GRID_WIDTH * GRID_HEIGHT * sizeof(int);
    sz_log_odds = GRID_WIDTH * GRID_HEIGHT * sizeof(float);

    gpuErrchk(cudaMalloc((void**)&d_lidar_coords, sz_lidar_coords));
    gpuErrchk(cudaMalloc((void**)&d_should_extend, sz_should_extend));
    gpuErrchk(cudaMalloc((void**)&d_grid_map, sz_grid_map));
    gpuErrchk(cudaMalloc((void**)&d_log_odds, sz_log_odds));

    res_should_extend = (int*)malloc(sz_should_extend);

    gpuErrchk(cudaMemcpy(d_lidar_coords, h_lidar_coords, sz_lidar_coords, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_grid_map, h_grid_map, sz_grid_map, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_log_odds, h_log_odds, sz_log_odds, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(d_should_extend, 0, sz_should_extend));

    /********************** IMAGE TRANSFORM KERNEL **********************/
    auto start_world_to_image_transform_1 = std::chrono::high_resolution_clock::now();
    kernel_matrix_mul_3x3 << < 1, 1 >> > (d_transition_single_world_body, d_transition_body_lidar, d_transition_single_world_lidar);
    cudaDeviceSynchronize();

    threadsPerBlock = 1;
    blocksPerGrid = LIDAR_COORDS_LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (d_particles_world_x, d_particles_world_y, SEP,
        d_transition_single_world_lidar, d_lidar_coords, LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();
    auto stop_world_to_image_transform_1 = std::chrono::high_resolution_clock::now();

    auto start_check_extend = std::chrono::high_resolution_clock::now();
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
    auto stop_check_extend = std::chrono::high_resolution_clock::now();

    printf("xmin=%d, xmax=%d, ymin=%d, ymax=%d\n", xmin, xmax, ymin, ymax);
    for (int i = 0; i < 4; i++)
        std::cout << "Should Extend: " << res_should_extend[i] << std::endl;

    assert(EXTEND == ST_EXTEND);

    if (EXTEND == true) {

        auto start_extend = std::chrono::high_resolution_clock::now();

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
        printf("GRID_WIDTH=%d, AF_GRID_WIDTH=%d, GRID_HEIGHT=%d, AF_GRID_HEIGHT=%d\n", GRID_WIDTH, GRID_HEIGHT, AF_GRID_WIDTH, AF_GRID_HEIGHT);
        assert(GRID_WIDTH == AF_GRID_WIDTH);
        assert(GRID_HEIGHT == AF_GRID_HEIGHT);

        const int PRE_GRID_SIZE = PRE_GRID_WIDTH * PRE_GRID_HEIGHT;
        const int NEW_GRID_SIZE = GRID_WIDTH * GRID_HEIGHT;
        //gpuErrchk(cudaFree(d_grid_map));

        sz_grid_map = GRID_WIDTH * GRID_HEIGHT * sizeof(int);
        gpuErrchk(cudaMalloc((void**)&d_grid_map, sz_grid_map));
        gpuErrchk(cudaMemset(d_grid_map, 0, sz_grid_map));

        sz_log_odds = GRID_WIDTH * GRID_HEIGHT * sizeof(float);
        gpuErrchk(cudaMalloc((void**)&d_log_odds, sz_log_odds));
        //gpuErrchk(cudaMemset(d_log_odds, LOG_ODD_PRIOR, sz_log_odds));

        res_grid_map = (int*)malloc(sz_grid_map);
        res_log_odds = (float*)malloc(sz_log_odds);

        threadsPerBlock = 256;
        blocksPerGrid = (NEW_GRID_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        kernel_index_arr_const << <blocksPerGrid, threadsPerBlock >> > (d_log_odds, LOG_ODD_PRIOR, NEW_GRID_SIZE);
        cudaDeviceSynchronize();

        threadsPerBlock = 256;
        blocksPerGrid = (PRE_GRID_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        kernel_2d_copy_with_offset << <blocksPerGrid, threadsPerBlock >> > (d_grid_map, d_log_odds, SEP,
            dc_grid_map, dc_log_odds, res_coord[0], res_coord[1],
            PRE_GRID_HEIGHT, GRID_HEIGHT, PRE_GRID_SIZE);
        cudaDeviceSynchronize();

        auto stop_extend = std::chrono::high_resolution_clock::now();
        auto duration_extend = std::chrono::duration_cast<std::chrono::microseconds>(stop_extend - start_extend);

        std::cout << std::endl;
        std::cout << "Time taken by function (Extend): " << duration_extend.count() << " microseconds" << std::endl;

        gpuErrchk(cudaMemcpy(res_grid_map, d_grid_map, sz_grid_map, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(res_log_odds, d_log_odds, sz_log_odds, cudaMemcpyDeviceToHost));

        int error_map = 0;
        int error_log = 0;
        for (int i = 0; i < (GRID_WIDTH * GRID_HEIGHT); i++) {
            if (res_grid_map[i] != h_bg_grid_map[i]) {
                error_map += 1;
                //printf("Grid Map: %d <> %d\n", res_grid_map[i], bg_grid_map[i]);
            }
            if ( abs(res_log_odds[i] - h_bg_log_odds[i]) > 1e-4) {
                error_log += 1;
                printf("Log Odds: (%d) %f <> %f\n", i, res_log_odds[i], h_bg_log_odds[i]);
            }
        }
        printf("Map Erros: %d\n", error_map);
        printf("Log Erros: %d\n", error_log);
    }


    auto start_world_to_image_transform_2 = std::chrono::high_resolution_clock::now();
    threadsPerBlock = 1;
    blocksPerGrid = LIDAR_COORDS_LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (d_processed_single_measure_x, d_processed_single_measure_y, SEP,
        d_transition_single_world_lidar, d_lidar_coords, res, xmin, ymax, LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    kernel_position_to_image << < 1, 1 >> > (d_position_image_body, SEP, d_transition_single_world_lidar, res, xmin, ymax);
    cudaDeviceSynchronize();
    auto stop_world_to_image_transform_2 = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_transition_world_lidar, d_transition_single_world_lidar, sz_transition_single_frame, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_processed_single_measure_x, d_processed_single_measure_x, sz_processed_single_measure_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_processed_single_measure_y, d_processed_single_measure_y, sz_processed_single_measure_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_world_x, d_particles_world_x, sz_particles_world_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_world_y, d_particles_world_y, sz_particles_world_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_position_image_body, d_position_image_body, sz_position_image_body, cudaMemcpyDeviceToHost));

    ASSERT_transition_world_lidar(res_transition_world_lidar, h_transition_world_lidar, 9, false);
    ASSERT_particles_world_frame(res_particles_world_x, res_particles_world_y, h_particles_world_x, h_particles_world_y, LIDAR_COORDS_LEN, false);
    ASSERT_processed_measurements(res_processed_single_measure_x, res_processed_single_measure_y, h_particles_occupied_x, h_particles_occupied_y, LIDAR_COORDS_LEN);
    ASSERT_position_image_body(res_position_image_body, h_position_image_body, true, true);

    auto duration_world_to_image_transform_1 = std::chrono::duration_cast<std::chrono::microseconds>(stop_world_to_image_transform_1 - start_world_to_image_transform_1);
    auto duration_world_to_image_transform_2 = std::chrono::duration_cast<std::chrono::microseconds>(stop_world_to_image_transform_2 - start_world_to_image_transform_2);
    std::cout << "Time taken by function (World To Image Transform 1): " << duration_world_to_image_transform_1.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (World To Image Transform 2): " << duration_world_to_image_transform_2.count() << " microseconds" << std::endl;
    std::cout << std::endl;
}

void host_bresenham() {

    printf("/***************************** BRESENHAM ****************************/\n");

    LIDAR_COORDS_LEN = ST_LIDAR_COORDS_LEN;
    GRID_WIDTH = ST_GRID_WIDTH;
    GRID_HEIGHT = ST_GRID_HEIGHT;

    printf("~~$ GRID_WIDTH = \t%d\n", GRID_WIDTH);
    printf("~~$ GRID_HEIGHT = \t%d\n", GRID_HEIGHT);
    MAX_DIST_IN_MAP = sqrt(pow(GRID_WIDTH, 2) + pow(GRID_HEIGHT, 2));
    printf("~~$ MAX_DIST_IN_MAP = \t%d\n", MAX_DIST_IN_MAP);

    /************************ BRESENHAM VARIABLES ***********************/
    PARTICLES_OCCUPIED_LEN = ST_PARTICLES_OCCUPIED_LEN;
    PARTICLES_FREE_LEN = ST_PARTICLES_FREE_LEN;
    PARTICLE_UNIQUE_COUNTER = PARTICLES_OCCUPIED_LEN + 1;

    sz_particles_occupied_pos = PARTICLES_OCCUPIED_LEN * sizeof(int);
    sz_particles_free_pos = 0;
    sz_particles_free_pos_max = PARTICLES_OCCUPIED_LEN * MAX_DIST_IN_MAP * sizeof(int);
    sz_particles_free_counter = PARTICLE_UNIQUE_COUNTER * sizeof(int);
    sz_position_image_body = 2 * sizeof(int);

    res_particles_free_counter = (int*)malloc(sz_particles_free_counter);

    gpuErrchk(cudaMalloc((void**)&d_particles_occupied_x, sz_particles_occupied_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_occupied_y, sz_particles_occupied_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_x_max, sz_particles_free_pos_max));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_y_max, sz_particles_free_pos_max));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_counter, sz_particles_free_counter));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_idx, sz_particles_occupied_pos));
    gpuErrchk(cudaMalloc((void**)&d_position_image_body, sz_position_image_body));

    gpuErrchk(cudaMemset(d_particles_free_x_max, 0, sz_particles_free_pos_max));
    gpuErrchk(cudaMemset(d_particles_free_y_max, 0, sz_particles_free_pos_max));
    gpuErrchk(cudaMemset(d_particles_free_counter, 0, sz_particles_free_counter));

    gpuErrchk(cudaMemcpy(d_particles_occupied_x, h_particles_occupied_x, sz_particles_occupied_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particles_occupied_y, h_particles_occupied_y, sz_particles_occupied_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particles_free_idx, h_particles_free_idx, sz_particles_occupied_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_position_image_body, h_position_image_body, sz_position_image_body, cudaMemcpyHostToDevice));

    /************************* BRESENHAM KERNEL *************************/
    auto start_bresenham = std::chrono::high_resolution_clock::now();

    threadsPerBlock = 256;
    blocksPerGrid = (PARTICLES_OCCUPIED_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_bresenham << <blocksPerGrid, threadsPerBlock >> > (d_particles_free_x_max, d_particles_free_y_max, d_particles_free_counter, SEP,
        d_particles_occupied_x, d_particles_occupied_y, d_position_image_body,
        PARTICLES_OCCUPIED_LEN, MAX_DIST_IN_MAP);
    cudaDeviceSynchronize();
    
    thrust::exclusive_scan(thrust::device, d_particles_free_counter, d_particles_free_counter + PARTICLE_UNIQUE_COUNTER, d_particles_free_counter, 0);
    auto stop_bresenham = std::chrono::high_resolution_clock::now();

    auto start_bresenham_rearrange = std::chrono::high_resolution_clock::now();
    gpuErrchk(cudaMemcpy(res_particles_free_counter, d_particles_free_counter, sz_particles_free_counter, cudaMemcpyDeviceToHost));

    PARTICLES_FREE_LEN = res_particles_free_counter[PARTICLE_UNIQUE_COUNTER - 1];
    sz_particles_free_pos = PARTICLES_FREE_LEN * sizeof(int);
    gpuErrchk(cudaMalloc((void**)&d_particles_free_x, sz_particles_free_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_y, sz_particles_free_pos));

    kernel_bresenham_rearrange << <blocksPerGrid, threadsPerBlock >> > (d_particles_free_x, d_particles_free_y, SEP,
        d_particles_free_x_max, d_particles_free_y_max,
        d_particles_free_counter, MAX_DIST_IN_MAP, PARTICLES_OCCUPIED_LEN);
    cudaDeviceSynchronize();
    auto stop_bresenham_rearrange = std::chrono::high_resolution_clock::now();

    res_particles_free_x = (int*)malloc(sz_particles_free_pos);
    res_particles_free_y = (int*)malloc(sz_particles_free_pos);

    gpuErrchk(cudaMemcpy(res_particles_free_x, d_particles_free_x, sz_particles_free_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_free_y, d_particles_free_y, sz_particles_free_pos, cudaMemcpyDeviceToHost));

    ASSERT_particles_free_index(res_particles_free_counter, h_particles_free_idx, PARTICLES_OCCUPIED_LEN);
    ASSERT_particles_free_new_len(PARTICLES_FREE_LEN, ST_PARTICLES_FREE_LEN);
    ASSERT_particles_free(res_particles_free_x, res_particles_free_y, h_particles_free_x, h_particles_free_y, PARTICLES_FREE_LEN, true, true);

    auto duration_bresenham = std::chrono::duration_cast<std::chrono::microseconds>(stop_bresenham - start_bresenham);
    auto duration_bresenham_rearrange = std::chrono::duration_cast<std::chrono::microseconds>(stop_bresenham_rearrange - start_bresenham_rearrange);
    std::cout << "Time taken by function (Bresenham): " << duration_bresenham.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Bresenham Rearrange): " << duration_bresenham_rearrange.count() << " microseconds" << std::endl;
    std::cout << std::endl;
}

void host_update_map() {

    printf("/**************************** UPDATE MAP ****************************/\n");

    xmin = ST_xmin;
    xmax = ST_xmax;
    ymin = ST_ymin;
    ymax = ST_ymax;

    LIDAR_COORDS_LEN = ST_LIDAR_COORDS_LEN;
    GRID_WIDTH = AF_GRID_WIDTH;
    GRID_HEIGHT = AF_GRID_HEIGHT;

    PARTICLES_OCCUPIED_LEN = ST_PARTICLES_OCCUPIED_LEN;
    PARTICLES_FREE_LEN = ST_PARTICLES_FREE_LEN;
    PARTICLE_UNIQUE_COUNTER = PARTICLES_OCCUPIED_LEN + 1;

    printf("~~$ GRID_WIDTH = \t%d\n", GRID_WIDTH);
    printf("~~$ GRID_HEIGHT = \t%d\n", GRID_HEIGHT);

    /**************************** MAP VARIABLES *************************/
    sz_particles_occupied_pos = PARTICLES_OCCUPIED_LEN * sizeof(int);
    sz_particles_free_pos = PARTICLES_FREE_LEN * sizeof(int);
    sz_grid_map = (GRID_WIDTH * GRID_HEIGHT) * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_grid_map, sz_grid_map));
    gpuErrchk(cudaMalloc((void**)&d_particles_occupied_x, sz_particles_occupied_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_occupied_y, sz_particles_occupied_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_x, sz_particles_free_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_y, sz_particles_free_pos));

    gpuErrchk(cudaMemcpy(d_grid_map, h_bg_grid_map, sz_grid_map, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particles_occupied_x, h_particles_occupied_x, sz_particles_occupied_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particles_occupied_y, h_particles_occupied_y, sz_particles_occupied_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particles_free_x, h_particles_free_x, sz_particles_free_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particles_free_y, h_particles_free_y, sz_particles_free_pos, cudaMemcpyHostToDevice));


    /************************* LOG-ODDS VARIABLES ***********************/
    sz_free_map_2d = GRID_WIDTH * GRID_HEIGHT * sizeof(uint8_t);
    sz_free_unique_counter = 1 * sizeof(int);
    sz_free_unique_counter_col = (GRID_WIDTH + 1) * sizeof(int);
    sz_free_map_idx = 2 * sizeof(int);

    sz_occupied_map_2d = GRID_WIDTH * GRID_HEIGHT * sizeof(uint8_t);
    sz_occupied_unique_counter = 1 * sizeof(int);
    sz_occupied_unique_counter_col = (GRID_WIDTH + 1) * sizeof(int);
    sz_occupied_map_idx = 2 * sizeof(int);


    d_occupied_map_2d = NULL;
    d_free_map_2d = NULL;
    d_occupied_unique_counter = NULL;
    d_free_unique_counter = NULL;
    d_occupied_unique_counter_col = NULL;
    d_free_unique_counter_col = NULL;

    res_occupied_unique_counter = (int*)malloc(sz_occupied_unique_counter);
    res_free_unique_counter = (int*)malloc(sz_free_unique_counter);
    res_occupied_unique_counter_col = (int*)malloc(sz_occupied_unique_counter_col);
    res_free_unique_counter_col = (int*)malloc(sz_free_unique_counter_col);

    h_occupied_map_idx[1] = PARTICLES_OCCUPIED_LEN;
    h_free_map_idx[1] = PARTICLES_FREE_LEN;

    gpuErrchk(cudaMalloc((void**)&d_occupied_map_2d, sz_occupied_map_2d));
    gpuErrchk(cudaMalloc((void**)&d_free_map_2d, sz_free_map_2d));
    gpuErrchk(cudaMalloc((void**)&d_occupied_unique_counter, sz_occupied_unique_counter));
    gpuErrchk(cudaMalloc((void**)&d_occupied_unique_counter_col, sz_occupied_unique_counter_col));
    gpuErrchk(cudaMalloc((void**)&d_free_unique_counter, sz_free_unique_counter));
    gpuErrchk(cudaMalloc((void**)&d_free_unique_counter_col, sz_free_unique_counter_col));

    gpuErrchk(cudaMalloc((void**)&d_occupied_map_idx, sz_occupied_map_idx));
    gpuErrchk(cudaMalloc((void**)&d_free_map_idx, sz_free_map_idx));

    gpuErrchk(cudaMemset(d_occupied_map_2d, 0, sz_occupied_map_2d));
    gpuErrchk(cudaMemset(d_free_map_2d, 0, sz_free_map_2d));

    gpuErrchk(cudaMemcpy(d_occupied_map_idx, h_occupied_map_idx, sz_occupied_map_idx, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_free_map_idx, h_free_map_idx, sz_free_map_idx, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemset(d_occupied_unique_counter_col, 0, sz_occupied_unique_counter_col));
    gpuErrchk(cudaMemset(d_free_unique_counter_col, 0, sz_free_unique_counter_col));

    /**************************** CREATE MAP ****************************/
    auto start_create_map = std::chrono::high_resolution_clock::now();
    threadsPerBlock = 256;
    blocksPerGrid = 1;
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (d_occupied_map_2d, SEP, 
        d_particles_occupied_x, d_particles_occupied_y, d_occupied_map_idx,
        PARTICLES_OCCUPIED_LEN, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (d_free_map_2d, SEP, 
        d_particles_free_x, d_particles_free_y, d_free_map_idx,
        PARTICLES_FREE_LEN, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();
    auto stop_create_map = std::chrono::high_resolution_clock::now();


    auto start_unique_counter = std::chrono::high_resolution_clock::now();
    threadsPerBlock = GRID_WIDTH;
    blocksPerGrid = 1;
    kernel_2d_map_counter << <blocksPerGrid, threadsPerBlock >> > (d_occupied_unique_counter, d_occupied_unique_counter_col, SEP, 
        d_occupied_map_2d, GRID_WIDTH, GRID_HEIGHT);
    kernel_2d_map_counter << <blocksPerGrid, threadsPerBlock >> > (d_free_unique_counter, d_free_unique_counter_col, SEP, 
        d_free_map_2d, GRID_WIDTH, GRID_HEIGHT);
    cudaDeviceSynchronize();
    auto stop_unique_counter = std::chrono::high_resolution_clock::now();

    auto start_unique_sum = std::chrono::high_resolution_clock::now();
    threadsPerBlock = 1;
    blocksPerGrid = 1;
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (d_occupied_unique_counter_col, GRID_WIDTH);
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (d_free_unique_counter_col, GRID_WIDTH);
    cudaDeviceSynchronize();
    auto stop_unique_sum = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(h_occupied_map_idx, d_occupied_map_idx, sz_occupied_map_idx, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_occupied_unique_counter, d_occupied_unique_counter, sz_occupied_unique_counter, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_free_unique_counter, d_free_unique_counter, sz_free_unique_counter, cudaMemcpyDeviceToHost));

    //gpuErrchk(cudaMemcpy(res_unique_occupied_counter_col, d_unique_occupied_counter_col, sz_unique_counter_col, cudaMemcpyDeviceToHost));
    //gpuErrchk(cudaMemcpy(res_unique_free_counter_col, d_unique_free_counter_col, sz_unique_counter_col, cudaMemcpyDeviceToHost));

    /*---------------------------------------------------------------------*/
    /*-------------------- REINITIALIZE MAP VARIABLES ---------------------*/
    PARTICLES_OCCUPIED_UNIQUE_LEN = res_occupied_unique_counter[0];
    PARTICLES_FREE_UNIQUE_LEN = res_free_unique_counter[0];
    
    printf("--> Occupied Unique: %d, %d\n", PARTICLES_OCCUPIED_UNIQUE_LEN, ST_PARTICLES_OCCUPIED_UNIQUE_LEN);
    assert(PARTICLES_OCCUPIED_UNIQUE_LEN == ST_PARTICLES_OCCUPIED_UNIQUE_LEN);
    printf("--> Free Unique: %d, %d\n", PARTICLES_FREE_UNIQUE_LEN, ST_PARTICLES_FREE_UNIQUE_LEN);
    assert(PARTICLES_FREE_UNIQUE_LEN == ST_PARTICLES_FREE_UNIQUE_LEN);
    

    sz_particles_occupied_pos = PARTICLES_OCCUPIED_UNIQUE_LEN * sizeof(int);
    sz_particles_free_pos = PARTICLES_FREE_UNIQUE_LEN * sizeof(int);

    res_particles_occupied_x = (int*)malloc(sz_particles_occupied_pos);
    res_particles_occupied_y = (int*)malloc(sz_particles_occupied_pos);
    res_particles_free_x = (int*)malloc(sz_particles_free_pos);
    res_particles_free_y = (int*)malloc(sz_particles_free_pos);


    gpuErrchk(cudaMalloc((void**)&d_particles_occupied_x, sz_particles_occupied_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_occupied_y, sz_particles_occupied_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_x, sz_particles_free_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_y, sz_particles_free_pos));

    auto start_restructure_map = std::chrono::high_resolution_clock::now();
    threadsPerBlock = GRID_WIDTH;
    blocksPerGrid = 1;
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (d_particles_occupied_x, d_particles_occupied_y, SEP,
        d_occupied_map_2d, d_occupied_unique_counter_col, GRID_WIDTH, GRID_HEIGHT);
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (d_particles_free_x, d_particles_free_y, SEP,
        d_free_map_2d, d_free_unique_counter_col, GRID_WIDTH, GRID_HEIGHT);
    cudaDeviceSynchronize();
    auto stop_restructure_map = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_particles_occupied_x, d_particles_occupied_x, sz_particles_occupied_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_occupied_y, d_particles_occupied_y, sz_particles_occupied_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_free_x, d_particles_free_x, sz_particles_free_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_free_y, d_particles_free_y, sz_particles_free_pos, cudaMemcpyDeviceToHost));

    ASSERT_particles_occupied(res_particles_occupied_x, res_particles_occupied_y, h_particles_occupied_unique_x, h_particles_occupied_unique_y,
        "Occupied", PARTICLES_OCCUPIED_UNIQUE_LEN);
    ASSERT_particles_occupied(res_particles_free_x, res_particles_free_y, h_particles_free_unique_x, h_particles_free_unique_y,
        "Free", PARTICLES_FREE_UNIQUE_LEN);

    /************************* LOG-ODDS VARIABLES ***********************/
    sz_log_odds = (GRID_WIDTH * GRID_HEIGHT) * sizeof(float);

    res_grid_map = (int*)malloc(sz_grid_map);
    res_log_odds = (float*)malloc(sz_log_odds);

    memset(res_log_odds, 0, sz_log_odds);
    gpuErrchk(cudaMalloc((void**)&d_log_odds, sz_log_odds));
    gpuErrchk(cudaMemcpy(d_log_odds, h_bg_log_odds, sz_log_odds, cudaMemcpyHostToDevice));


    /************************** LOG-ODDS KERNEL *************************/
    auto start_update_map = std::chrono::high_resolution_clock::now();

    threadsPerBlock = 256;
    blocksPerGrid = (PARTICLES_OCCUPIED_UNIQUE_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_update_log_odds << <blocksPerGrid, threadsPerBlock >> > (d_log_odds, SEP,
        d_particles_occupied_x, d_particles_occupied_y, 2 * log_t, GRID_WIDTH, GRID_HEIGHT, PARTICLES_OCCUPIED_UNIQUE_LEN);
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

    auto stop_update_map = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_log_odds, d_log_odds, sz_log_odds, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_grid_map, d_grid_map, sz_grid_map, cudaMemcpyDeviceToHost));

    ASSERT_log_odds(res_log_odds, h_log_odds, h_post_log_odds, (GRID_WIDTH * GRID_HEIGHT));
    ASSERT_log_odds_maps(res_grid_map, h_grid_map, h_post_grid_map, (GRID_WIDTH * GRID_HEIGHT));
    printf("\n");

    auto duration_create_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_create_map - start_create_map);
    auto duration_unique_counter = std::chrono::duration_cast<std::chrono::microseconds>(stop_unique_counter - start_unique_counter);
    auto duration_unique_sum = std::chrono::duration_cast<std::chrono::microseconds>(stop_unique_sum - start_unique_sum);
    auto duration_restructure_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_restructure_map - start_restructure_map);
    auto duration_update_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_update_map - start_update_map);

    std::cout << "Time taken by function (Create Map): " << duration_create_map.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Unique Counter): " << duration_unique_counter.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Unique Sum): " << duration_unique_sum.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Restructure Map): " << duration_restructure_map.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Update Map): " << duration_update_map.count() << " microseconds" << std::endl;
    std::cout << std::endl;
}

void host_map() {

    printf("/******************************** MAP *******************************/\n");

    xmin = ST_xmin;
    xmax = ST_xmax;
    ymin = ST_ymin;
    ymax = ST_ymax;

    int xmin_pre = xmin;
    int ymax_pre = ymax;

    GRID_WIDTH = ST_GRID_WIDTH;
    GRID_HEIGHT = ST_GRID_HEIGHT;
    LIDAR_COORDS_LEN = ST_LIDAR_COORDS_LEN;

    PARTICLES_OCCUPIED_LEN = ST_PARTICLES_OCCUPIED_LEN;
    PARTICLES_FREE_LEN = ST_PARTICLES_FREE_LEN;

    sz_should_extend = 4 * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_should_extend, sz_should_extend));
    res_should_extend = (int*)malloc(sz_should_extend);
    memset(res_should_extend, 0, sz_should_extend);
    gpuErrchk(cudaMemset(d_should_extend, 0, sz_should_extend));

    /********************* IMAGE TRANSFORM VARIABLES ********************/
    sz_transition_single_frame = 9 * sizeof(float);
    sz_transition_body_lidar = 9 * sizeof(float);
    sz_lidar_coords = 2 * LIDAR_COORDS_LEN * sizeof(float);
    sz_particles_occupied_pos = LIDAR_COORDS_LEN * sizeof(int);
    sz_particles_world_pos = LIDAR_COORDS_LEN * sizeof(float);
    sz_position_image_body = 2 * sizeof(int);

    res_transition_world_lidar = (float*)malloc(sz_transition_single_frame);
    res_particles_occupied_x = (int*)malloc(sz_particles_occupied_pos);
    res_particles_occupied_y = (int*)malloc(sz_particles_occupied_pos);
    res_particles_world_x = (float*)malloc(sz_particles_world_pos);
    res_particles_world_y = (float*)malloc(sz_particles_world_pos);
    res_position_image_body = (int*)malloc(sz_position_image_body);

    gpuErrchk(cudaMalloc((void**)&d_lidar_coords, sz_lidar_coords));
    gpuErrchk(cudaMalloc((void**)&d_transition_body_lidar, sz_transition_body_lidar));
    gpuErrchk(cudaMalloc((void**)&d_transition_single_world_body, sz_transition_single_frame));
    gpuErrchk(cudaMalloc((void**)&d_transition_single_world_lidar, sz_transition_single_frame));
    gpuErrchk(cudaMalloc((void**)&d_particles_occupied_x, sz_particles_occupied_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_occupied_y, sz_particles_occupied_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_world_x, sz_particles_world_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_world_y, sz_particles_world_pos));

    gpuErrchk(cudaMemcpy(d_lidar_coords, h_lidar_coords, sz_lidar_coords, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_transition_body_lidar, h_transition_body_lidar, sz_transition_body_lidar, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_transition_single_world_body, h_transition_world_body, sz_transition_single_frame, cudaMemcpyHostToDevice));


    /************************ BRESENHAM VARIABLES ***********************/
    MAX_DIST_IN_MAP = sqrt(pow(GRID_WIDTH, 2) + pow(GRID_HEIGHT, 2));
    PARTICLE_UNIQUE_COUNTER = PARTICLES_OCCUPIED_LEN + 1;

    sz_particles_free_pos = 0;
    sz_particles_free_pos_max = PARTICLES_OCCUPIED_LEN * MAX_DIST_IN_MAP * sizeof(int);
    sz_particles_free_counter = PARTICLE_UNIQUE_COUNTER * sizeof(int);
    sz_position_image_body = 2 * sizeof(int);
    sz_particles_free_idx = PARTICLES_OCCUPIED_LEN * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_particles_free_x_max, sz_particles_free_pos_max));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_y_max, sz_particles_free_pos_max));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_counter, sz_particles_free_counter));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_idx, sz_particles_free_idx));
    gpuErrchk(cudaMalloc((void**)&d_position_image_body, sz_position_image_body));

    res_particles_free_counter = (int*)malloc(sz_particles_free_counter);

    gpuErrchk(cudaMemset(d_particles_free_x_max, 0, sz_particles_free_pos_max));
    gpuErrchk(cudaMemset(d_particles_free_y_max, 0, sz_particles_free_pos_max));
    gpuErrchk(cudaMemset(d_particles_free_counter, 0, sz_particles_free_counter));


    /**************************** MAP VARIABLES *************************/
    sz_grid_map = (GRID_WIDTH * GRID_HEIGHT) * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_grid_map, sz_grid_map));
    gpuErrchk(cudaMemcpy(d_grid_map, h_grid_map, sz_grid_map, cudaMemcpyHostToDevice));

    /************************* LOG-ODDS VARIABLES ***********************/
    sz_log_odds = (GRID_WIDTH * GRID_HEIGHT) * sizeof(float);

    sz_free_map_2d = GRID_WIDTH * GRID_HEIGHT * sizeof(uint8_t);
    sz_free_unique_counter = 1 * sizeof(int);
    sz_free_unique_counter_col = (GRID_WIDTH + 1) * sizeof(int);
    sz_free_map_idx = 2 * sizeof(int);

    res_free_unique_counter = (int*)malloc(sz_free_unique_counter);
    res_free_unique_counter_col = (int*)malloc(sz_free_unique_counter_col);
    
    gpuErrchk(cudaMalloc((void**)&d_free_map_2d, sz_free_map_2d));
    gpuErrchk(cudaMalloc((void**)&d_free_unique_counter, sz_free_unique_counter));
    gpuErrchk(cudaMalloc((void**)&d_free_unique_counter_col, sz_free_unique_counter_col));


    sz_occupied_map_2d = GRID_WIDTH * GRID_HEIGHT * sizeof(uint8_t);
    sz_occupied_unique_counter = 1 * sizeof(int);
    sz_occupied_unique_counter_col = (GRID_WIDTH + 1) * sizeof(int);
    sz_occupied_map_idx = 2 * sizeof(int);

    res_occupied_unique_counter = (int*)malloc(sz_occupied_unique_counter);
    res_occupied_unique_counter_col = (int*)malloc(sz_occupied_unique_counter_col);

    gpuErrchk(cudaMalloc((void**)&d_occupied_map_2d, sz_occupied_map_2d));
    gpuErrchk(cudaMalloc((void**)&d_occupied_unique_counter, sz_occupied_unique_counter));
    gpuErrchk(cudaMalloc((void**)&d_occupied_unique_counter_col, sz_occupied_unique_counter_col));

    res_grid_map = (int*)malloc(sz_grid_map);
    res_log_odds = (float*)malloc(sz_log_odds);

    gpuErrchk(cudaMalloc((void**)&d_log_odds, sz_log_odds));

    h_occupied_map_idx[1] = PARTICLES_OCCUPIED_LEN;
    h_free_map_idx[1] = 0;

    memset(res_log_odds, 0, sz_log_odds);

    gpuErrchk(cudaMalloc((void**)&d_occupied_map_idx, sz_occupied_map_idx));
    gpuErrchk(cudaMalloc((void**)&d_free_map_idx, sz_free_map_idx));
    gpuErrchk(cudaMemcpy(d_occupied_map_idx, h_occupied_map_idx, sz_occupied_map_idx, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_free_map_idx, h_free_map_idx, sz_free_map_idx, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_log_odds, h_log_odds, sz_log_odds, cudaMemcpyHostToDevice));


    /************************************************************* KERNEL EXECUTION SCOPE *************************************************************/

    /***************** World to IMAGE TRANSFORM KERNEL ******************/
    auto start_world_to_image_transform_1 = std::chrono::high_resolution_clock::now();
    kernel_matrix_mul_3x3 << < 1, 1 >> > (d_transition_single_world_body, d_transition_body_lidar, d_transition_single_world_lidar);
    cudaDeviceSynchronize();

    threadsPerBlock = 1;
    blocksPerGrid = LIDAR_COORDS_LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (d_particles_world_x, d_particles_world_y, SEP,
        d_transition_single_world_lidar, d_lidar_coords, LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();
    auto stop_world_to_image_transform_1 = std::chrono::high_resolution_clock::now();


    auto start_check_extend = std::chrono::high_resolution_clock::now();
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
    auto stop_check_extend = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 4; i++)
        std::cout << "Should Extend: " << res_should_extend[i] << std::endl;

    printf("xmin=%d, xmax=%d, ymin=%d, ymax=%d\n", xmin, xmax, ymin, ymax);
    printf("xmin=%d, xmax=%d, ymin=%d, ymax=%d\n", AF_xmin, AF_xmax, AF_ymin, AF_ymax);
    assert(EXTEND == ST_EXTEND);

    if (EXTEND == true) {

        auto start_extend = std::chrono::high_resolution_clock::now();

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
        printf("GRID_WIDTH=%d, AF_GRID_WIDTH=%d, GRID_HEIGHT=%d, AF_GRID_HEIGHT=%d\n", GRID_WIDTH, GRID_HEIGHT, AF_GRID_WIDTH, AF_GRID_HEIGHT);
        assert(GRID_WIDTH == AF_GRID_WIDTH);
        assert(GRID_HEIGHT == AF_GRID_HEIGHT);

        const int PRE_GRID_SIZE = PRE_GRID_WIDTH * PRE_GRID_HEIGHT;
        const int NEW_GRID_SIZE = GRID_WIDTH * GRID_HEIGHT;

        //gpuErrchk(cudaFree(d_grid_map));

        sz_grid_map = GRID_WIDTH * GRID_HEIGHT * sizeof(int);
        gpuErrchk(cudaMalloc((void**)&d_grid_map, sz_grid_map));
        gpuErrchk(cudaMemset(d_grid_map, 0, sz_grid_map));

        sz_log_odds = GRID_WIDTH * GRID_HEIGHT * sizeof(float);
        gpuErrchk(cudaMalloc((void**)&d_log_odds, sz_log_odds));
        //gpuErrchk(cudaMemset(d_log_odds, LOG_ODD_PRIOR, sz_log_odds));

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
        
        sz_occupied_unique_counter_col = (GRID_WIDTH + 1) * sizeof(int);
        gpuErrchk(cudaMalloc((void**)&d_occupied_unique_counter_col, sz_occupied_unique_counter_col));
        res_occupied_unique_counter_col = (int*)malloc(sz_occupied_unique_counter_col);

        sz_free_map_2d = GRID_WIDTH * GRID_HEIGHT * sizeof(uint8_t);
        gpuErrchk(cudaMalloc((void**)&d_free_map_2d, sz_free_map_2d));
        
        sz_occupied_map_2d = GRID_WIDTH * GRID_HEIGHT * sizeof(uint8_t);
        gpuErrchk(cudaMalloc((void**)&d_occupied_map_2d, sz_occupied_map_2d));


        auto stop_extend = std::chrono::high_resolution_clock::now();
        auto duration_extend = std::chrono::duration_cast<std::chrono::microseconds>(stop_extend - start_extend);

        std::cout << std::endl;
        std::cout << "Time taken by function (Extend): " << duration_extend.count() << " microseconds" << std::endl;

        gpuErrchk(cudaMemcpy(res_grid_map, d_grid_map, sz_grid_map, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(res_log_odds, d_log_odds, sz_log_odds, cudaMemcpyDeviceToHost));

        int error_map = 0;
        int error_log = 0;
        for (int i = 0; i < (GRID_WIDTH * GRID_HEIGHT); i++) {
            if (res_grid_map[i] != h_bg_grid_map[i]) {
                error_map += 1;
                //printf("Grid Map: %d <> %d\n", res_grid_map[i], bg_grid_map[i]);
            }
            if ( abs(res_log_odds[i] - h_bg_log_odds[i]) > 1e-4) {
                error_log += 1;
                //printf("Log Odds: (%d) %f <> %f\n", i, res_log_odds[i], bg_log_odds[i]);
            }
            if (error_log > 200)
                break;
        }
        printf("Map Erros: %d\n", error_map);
        printf("Log Erros: %d\n", error_log);
    }

    for (int i = 0; i < 4; i++)
        std::cout << "Should Extend: " << res_should_extend[i] << std::endl;

    auto start_world_to_image_transform_2 = std::chrono::high_resolution_clock::now();
    threadsPerBlock = 1;
    blocksPerGrid = LIDAR_COORDS_LEN;    
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (d_particles_occupied_x, d_particles_occupied_y, SEP,
        d_transition_single_world_lidar, d_lidar_coords, res, xmin, ymax, LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    kernel_position_to_image << < 1, 1 >> > (d_position_image_body, SEP, d_transition_single_world_lidar, res, xmin, ymax);
    cudaDeviceSynchronize();
    auto stop_world_to_image_transform_2 = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_transition_world_lidar, d_transition_single_world_lidar, sz_transition_single_frame, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_occupied_x, d_particles_occupied_x, sz_particles_occupied_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_occupied_y, d_particles_occupied_y, sz_particles_occupied_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_world_x, d_particles_world_x, sz_particles_world_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_world_y, d_particles_world_y, sz_particles_world_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_position_image_body, d_position_image_body, sz_position_image_body, cudaMemcpyDeviceToHost));

    ASSERT_transition_world_lidar(res_transition_world_lidar, h_transition_world_lidar, 9, false);
    ASSERT_particles_world_frame(res_particles_world_x, res_particles_world_y, h_particles_world_x, h_particles_world_y, LIDAR_COORDS_LEN, false);
    ASSERT_processed_measurements(res_particles_occupied_x, res_particles_occupied_y, h_particles_occupied_x, h_particles_occupied_y, LIDAR_COORDS_LEN);
    ASSERT_position_image_body(res_position_image_body, h_position_image_body, true, true);

    /************************* BRESENHAM KERNEL *************************/
    auto start_bresenham = std::chrono::high_resolution_clock::now();

    threadsPerBlock = 256;
    blocksPerGrid = (PARTICLES_OCCUPIED_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_bresenham << <blocksPerGrid, threadsPerBlock >> > (d_particles_free_x_max, d_particles_free_y_max, d_particles_free_counter, SEP,
        d_particles_occupied_x, d_particles_occupied_y, d_position_image_body,
        PARTICLES_OCCUPIED_LEN, MAX_DIST_IN_MAP);
    cudaDeviceSynchronize();
    thrust::exclusive_scan(thrust::device, d_particles_free_counter, d_particles_free_counter + PARTICLE_UNIQUE_COUNTER, d_particles_free_counter, 0); // in-place scan
    auto stop_bresenham = std::chrono::high_resolution_clock::now();

    auto start_bresenham_rearrange = std::chrono::high_resolution_clock::now();
    gpuErrchk(cudaMemcpy(res_particles_free_counter, d_particles_free_counter, sz_particles_free_counter, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(d_particles_free_idx, d_particles_free_counter, sz_particles_free_idx, cudaMemcpyDeviceToDevice));

    PARTICLES_FREE_LEN = res_particles_free_counter[PARTICLE_UNIQUE_COUNTER - 1];
    printf("^^^ PARTICLES_FREE_LEN = %d\n", PARTICLES_FREE_LEN);
    sz_particles_free_pos = PARTICLES_FREE_LEN * sizeof(int);
    gpuErrchk(cudaMalloc((void**)&d_particles_free_x, sz_particles_free_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_y, sz_particles_free_pos));

    kernel_bresenham_rearrange << <blocksPerGrid, threadsPerBlock >> > (d_particles_free_x, d_particles_free_y, SEP, 
        d_particles_free_x_max, d_particles_free_y_max,
        d_particles_free_counter, MAX_DIST_IN_MAP, PARTICLES_OCCUPIED_LEN);
    cudaDeviceSynchronize();
    auto stop_bresenham_rearrange = std::chrono::high_resolution_clock::now();

    res_particles_free_x = (int*)malloc(sz_particles_free_pos);
    res_particles_free_y = (int*)malloc(sz_particles_free_pos);

    gpuErrchk(cudaMemcpy(res_particles_free_x, d_particles_free_x, sz_particles_free_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_free_y, d_particles_free_y, sz_particles_free_pos, cudaMemcpyDeviceToHost));

    printf("~~$ PARTICLES_FREE_LEN = %d\n", PARTICLES_FREE_LEN);

    ASSERT_particles_free_index(res_particles_free_counter, h_particles_free_idx, PARTICLES_OCCUPIED_LEN, false);
    ASSERT_particles_free_new_len(PARTICLES_FREE_LEN, ST_PARTICLES_FREE_LEN);
    ASSERT_particles_free(res_particles_free_x, res_particles_free_y, h_particles_free_x, h_particles_free_y, PARTICLES_FREE_LEN);

    /*---------------------------------------------------------------------*/
    /*-------------------- REINITIALIZE MAP VARIABLES ---------------------*/
    h_free_map_idx[1] = PARTICLES_FREE_LEN;
    gpuErrchk(cudaMemcpy(d_free_map_idx, h_free_map_idx, sz_free_map_idx, cudaMemcpyHostToDevice));


    /************************** CREATE 2D MAP ***************************/
    auto start_create_map = std::chrono::high_resolution_clock::now();
    threadsPerBlock = 256;
    blocksPerGrid = 1;
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (d_occupied_map_2d, SEP, 
        d_particles_occupied_x, d_particles_occupied_y, d_occupied_map_idx,
        PARTICLES_OCCUPIED_LEN, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (d_free_map_2d, SEP, 
        d_particles_free_x, d_particles_free_y, d_free_map_idx,
        PARTICLES_FREE_LEN, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();
    auto stop_create_map = std::chrono::high_resolution_clock::now();

    auto start_unique_counter = std::chrono::high_resolution_clock::now();
    threadsPerBlock = GRID_WIDTH;
    blocksPerGrid = 1;
    kernel_2d_map_counter << <blocksPerGrid, threadsPerBlock >> > (d_occupied_unique_counter, d_occupied_unique_counter_col, SEP, 
        d_occupied_map_2d, GRID_WIDTH, GRID_HEIGHT);
    kernel_2d_map_counter << <blocksPerGrid, threadsPerBlock >> > (d_free_unique_counter, d_free_unique_counter_col, SEP, 
        d_free_map_2d, GRID_WIDTH, GRID_HEIGHT);
    cudaDeviceSynchronize();
    auto stop_unique_counter = std::chrono::high_resolution_clock::now();

    auto start_unique_sum = std::chrono::high_resolution_clock::now();
    threadsPerBlock = 1;
    blocksPerGrid = 1;
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (d_occupied_unique_counter_col, GRID_WIDTH);
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (d_free_unique_counter_col, GRID_WIDTH);
    cudaDeviceSynchronize();
    auto stop_unique_sum = std::chrono::high_resolution_clock::now();

    //gpuErrchk(cudaMemcpy(h_particles_occupied_idx, d_particles_occupied_idx, sz_particles_idx, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_occupied_unique_counter, d_occupied_unique_counter, sz_occupied_unique_counter, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_free_unique_counter, d_free_unique_counter, sz_free_unique_counter, cudaMemcpyDeviceToHost));

    //gpuErrchk(cudaMemcpy(res_unique_occupied_counter_col, d_unique_occupied_counter_col, sz_unique_counter_col, cudaMemcpyDeviceToHost));
    //gpuErrchk(cudaMemcpy(res_unique_free_counter_col, d_unique_free_counter_col, sz_unique_counter_col, cudaMemcpyDeviceToHost));

    /*---------------------------------------------------------------------*/
    /*-------------------- REINITIALIZE MAP VARIABLES ---------------------*/
    PARTICLES_OCCUPIED_UNIQUE_LEN = res_occupied_unique_counter[0];
    PARTICLES_FREE_UNIQUE_LEN = res_free_unique_counter[0];

    printf("\n--> Occupied Unique: %d, %d\n", PARTICLES_OCCUPIED_UNIQUE_LEN, ST_PARTICLES_OCCUPIED_UNIQUE_LEN);
    assert(PARTICLES_OCCUPIED_UNIQUE_LEN == ST_PARTICLES_OCCUPIED_UNIQUE_LEN);
    printf("\n--> Free Unique: %d, %d\n", PARTICLES_FREE_UNIQUE_LEN, ST_PARTICLES_FREE_UNIQUE_LEN);
    assert(PARTICLES_FREE_UNIQUE_LEN == ST_PARTICLES_FREE_UNIQUE_LEN);
    
    //gpuErrchk(cudaFree(d_particles_occupied_x));
    //gpuErrchk(cudaFree(d_particles_occupied_y));
    //gpuErrchk(cudaFree(d_particles_free_x));
    //gpuErrchk(cudaFree(d_particles_free_y));

    free(res_particles_occupied_x);
    free(res_particles_occupied_y);
    free(res_particles_free_x);
    free(res_particles_free_y);

    sz_particles_occupied_pos = PARTICLES_OCCUPIED_UNIQUE_LEN * sizeof(int);
    sz_particles_free_pos = PARTICLES_FREE_UNIQUE_LEN * sizeof(int);

    res_particles_occupied_x = (int*)malloc(sz_particles_occupied_pos);
    res_particles_occupied_y = (int*)malloc(sz_particles_occupied_pos);
    res_particles_free_x = (int*)malloc(sz_particles_free_pos);
    res_particles_free_y = (int*)malloc(sz_particles_free_pos);

    gpuErrchk(cudaMalloc((void**)&d_particles_occupied_x, sz_particles_occupied_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_occupied_y, sz_particles_occupied_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_x, sz_particles_free_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_y, sz_particles_free_pos));

    auto start_restructure_map = std::chrono::high_resolution_clock::now();
    threadsPerBlock = GRID_WIDTH;
    blocksPerGrid = 1;
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (d_particles_occupied_x, d_particles_occupied_y, SEP,
        d_occupied_map_2d, d_occupied_unique_counter_col, GRID_WIDTH, GRID_HEIGHT);
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (d_particles_free_x, d_particles_free_y, SEP,
        d_free_map_2d, d_free_unique_counter_col, GRID_WIDTH, GRID_HEIGHT);
    cudaDeviceSynchronize();
    auto stop_restructure_map = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_particles_occupied_x, d_particles_occupied_x, sz_particles_occupied_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_occupied_y, d_particles_occupied_y, sz_particles_occupied_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_free_x, d_particles_free_x, sz_particles_free_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_free_y, d_particles_free_y, sz_particles_free_pos, cudaMemcpyDeviceToHost));

    ASSERT_particles_occupied(res_particles_occupied_x, res_particles_occupied_y, h_particles_occupied_unique_x, h_particles_occupied_unique_y,
        "Occupied", PARTICLES_OCCUPIED_UNIQUE_LEN, false);
    ASSERT_particles_occupied(res_particles_free_x, res_particles_free_y, h_particles_free_unique_x, h_particles_free_unique_y,
        "Free", PARTICLES_FREE_UNIQUE_LEN, false);

    /************************** LOG-ODDS KERNEL *************************/
    auto start_update_map = std::chrono::high_resolution_clock::now();
    
    threadsPerBlock = 256;
    blocksPerGrid = (PARTICLES_OCCUPIED_UNIQUE_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_update_log_odds << <blocksPerGrid, threadsPerBlock >> > (d_log_odds, SEP, d_particles_occupied_x, d_particles_occupied_y, 2 * log_t, GRID_WIDTH, GRID_HEIGHT, PARTICLES_OCCUPIED_UNIQUE_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = 256;
    blocksPerGrid = (PARTICLES_FREE_UNIQUE_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_update_log_odds << <blocksPerGrid, threadsPerBlock >> > (d_log_odds, SEP, d_particles_free_x, d_particles_free_y, (-1) * log_t, GRID_WIDTH, GRID_HEIGHT, PARTICLES_FREE_UNIQUE_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = 256;
    blocksPerGrid = ((GRID_WIDTH * GRID_HEIGHT) + threadsPerBlock - 1) / threadsPerBlock;
    kernel_update_map << <blocksPerGrid, threadsPerBlock >> > (d_grid_map, SEP, d_log_odds, LOG_ODD_PRIOR, WALL, FREE, GRID_WIDTH * GRID_HEIGHT);
    cudaDeviceSynchronize();
    auto stop_update_map = std::chrono::high_resolution_clock::now();

    memset(res_log_odds, 0, sz_log_odds);
    gpuErrchk(cudaMemcpy(res_log_odds, d_log_odds, sz_log_odds, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_grid_map, d_grid_map, sz_grid_map, cudaMemcpyDeviceToHost));

    ASSERT_log_odds(res_log_odds, h_log_odds, h_post_log_odds, (GRID_WIDTH * GRID_HEIGHT));

    ASSERT_log_odds_maps(res_grid_map, h_grid_map, h_post_grid_map, (GRID_WIDTH * GRID_HEIGHT));
    printf("\n");


    auto duration_world_to_image_transform_1 = std::chrono::duration_cast<std::chrono::microseconds>(stop_world_to_image_transform_1 - start_world_to_image_transform_1);
    auto duration_world_to_image_transform_2 = std::chrono::duration_cast<std::chrono::microseconds>(stop_world_to_image_transform_2 - start_world_to_image_transform_2);
    auto duration_bresenham = std::chrono::duration_cast<std::chrono::microseconds>(stop_bresenham - start_bresenham);
    auto duration_bresenham_rearrange = std::chrono::duration_cast<std::chrono::microseconds>(stop_bresenham_rearrange - start_bresenham_rearrange);
    auto duration_create_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_create_map - start_create_map);
    auto duration_unique_counter = std::chrono::duration_cast<std::chrono::microseconds>(stop_unique_counter - start_unique_counter);
    auto duration_unique_sum = std::chrono::duration_cast<std::chrono::microseconds>(stop_unique_sum - start_unique_sum);
    auto duration_restructure_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_restructure_map - start_restructure_map);
    auto duration_update_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_update_map - start_update_map);

    auto duration_total = duration_world_to_image_transform_1 + duration_world_to_image_transform_2 + duration_bresenham + duration_bresenham_rearrange + duration_create_map +
        duration_unique_counter + duration_unique_sum + duration_restructure_map + duration_update_map;

    std::cout << "Time taken by function (World To Image Transform 1): " << duration_world_to_image_transform_1.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (World To Image Transform 2): " << duration_world_to_image_transform_2.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Bresenham): " << duration_bresenham.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Bresenham Rearrange): " << duration_bresenham_rearrange.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Create Map): " << duration_create_map.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Unique Counter): " << duration_unique_counter.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Unique Sum): " << duration_unique_sum.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Restructure Map): " << duration_restructure_map.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Update Map): " << duration_update_map.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Total): " << duration_total.count() << " microseconds" << std::endl;
    std::cout << std::endl;
}



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
        assert(GRID_WIDTH == AF_GRID_WIDTH);
        assert(GRID_HEIGHT == AF_GRID_HEIGHT);

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

void assertResults() {

    printf("\n--> Occupied Unique: %d, %d\n", PARTICLES_OCCUPIED_UNIQUE_LEN, ST_PARTICLES_OCCUPIED_UNIQUE_LEN);
    assert(PARTICLES_OCCUPIED_UNIQUE_LEN == ST_PARTICLES_OCCUPIED_UNIQUE_LEN);
    printf("\n--> Free Unique: %d, %d\n", PARTICLES_FREE_UNIQUE_LEN, ST_PARTICLES_FREE_UNIQUE_LEN);
    assert(PARTICLES_FREE_UNIQUE_LEN == ST_PARTICLES_FREE_UNIQUE_LEN);

    printf("~~$ PARTICLES_FREE_LEN=%d\n", PARTICLES_FREE_LEN);
    ASSERT_particles_free_index(res_particles_free_counter, h_particles_free_idx, PARTICLES_OCCUPIED_LEN, false);
    ASSERT_particles_free_new_len(PARTICLES_FREE_LEN, ST_PARTICLES_FREE_LEN);

    gpuErrchk(cudaMemcpy(res_log_odds, d_log_odds, sz_log_odds, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_grid_map, d_grid_map, sz_grid_map, cudaMemcpyDeviceToHost));

    ASSERT_log_odds(res_log_odds, h_log_odds, h_post_log_odds, (GRID_WIDTH * GRID_HEIGHT));
    ASSERT_log_odds_maps(res_grid_map, h_grid_map, h_post_grid_map, (GRID_WIDTH * GRID_HEIGHT));

    printf("\n~~$ Verification All Passed\n");
}


void test_map_func() {

    printf("/********************************************************************/\n");
    printf("/****************************** MAP MAIN ****************************/\n");
    printf("/********************************************************************/\n");

    const int data_len = 6;
    int data[data_len] = { 1, 0, 2, 2, 1, 3 };
    int* d_data = NULL;
    gpuErrchk(cudaMalloc((void**)&d_data, data_len * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_data, data, data_len * sizeof(int), cudaMemcpyHostToDevice));

    thrust::exclusive_scan(thrust::host, data, data + 6, data, 0);
    thrust::exclusive_scan(thrust::device, d_data, d_data + 6, d_data, 0);

    GRID_WIDTH = ST_GRID_WIDTH;
    GRID_HEIGHT = ST_GRID_HEIGHT;
    log_t = ST_log_t;
    LIDAR_COORDS_LEN = ST_LIDAR_COORDS_LEN;
    PARTICLES_OCCUPIED_LEN = ST_PARTICLES_OCCUPIED_LEN;
    MAX_DIST_IN_MAP = sqrt(pow(GRID_WIDTH, 2) + pow(GRID_HEIGHT, 2));
    PARTICLE_UNIQUE_COUNTER = PARTICLES_OCCUPIED_LEN + 1;

    xmin = ST_xmin;
    xmax = ST_xmax;;
    ymin = ST_ymin;
    ymax = ST_ymax;

    h_occupied_map_idx[1] = PARTICLES_OCCUPIED_LEN;
    h_free_map_idx[1] = PARTICLES_FREE_LEN;

    printf("~~$ LIDAR_COORDS_LEN = \t%d\n", LIDAR_COORDS_LEN);
    printf("~~$ PARTICLES_OCCUPIED_LEN = \t%d\n", PARTICLES_OCCUPIED_LEN);
    printf("~~$ PARTICLES_OCCUPIED_UNIQUE_LEN = \t%d\n", PARTICLES_OCCUPIED_UNIQUE_LEN);
    printf("~~$ PARTICLES_FREE_LEN = \t%d\n", PARTICLES_FREE_LEN);
    printf("~~$ PARTICLES_FREE_UNIQUE_LEN = \t%d\n", PARTICLES_FREE_UNIQUE_LEN);
    printf("~~$ PARTICLE_UNIQUE_COUNTER = \t%d\n", PARTICLE_UNIQUE_COUNTER);
    printf("~~$ MAX_DIST_IN_MAP = \t\t%d\n", MAX_DIST_IN_MAP);


    auto start_mapping_alloc = std::chrono::high_resolution_clock::now();
    alloc_init_transition_vars(h_transition_body_lidar, h_transition_world_body);
    alloc_init_lidar_coords_var(h_lidar_coords, LIDAR_COORDS_LEN);
    alloc_particles_world_vars(LIDAR_COORDS_LEN);
    alloc_particles_free_vars();
    alloc_particles_occupied_vars();
    alloc_bresenham_vars();
    alloc_init_map_vars(h_grid_map, GRID_WIDTH, GRID_HEIGHT);
    alloc_log_odds_vars();
    alloc_init_log_odds_free_vars();
    alloc_init_log_odds_occupied_vars();
    init_log_odds_vars(h_log_odds);
    auto stop_mapping_alloc = std::chrono::high_resolution_clock::now();


    auto start_mapping_kernel = std::chrono::high_resolution_clock::now();
    exec_world_to_image_transform_step_1();
    exec_map_extend();
    exec_world_to_image_transform_step_2();
    exec_bresenham();
    reinit_map_idx_vars();

    exec_create_map();
    reinit_map_vars();

    exec_log_odds(ST_log_t);
    auto stop_mapping_kernel = std::chrono::high_resolution_clock::now();

    assertResults();

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
