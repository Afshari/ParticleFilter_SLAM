#ifndef _TEST_MAP_PARTIALS_H_
#define _TEST_MAP_PARTIALS_H_

#include "headers.h"
#include "host_asserts.h"
#include "host_utils.h"
#include "kernels.cuh"

// #define BRESENHAM_EXEC
// #define UPDATE_MAP_EXEC
// #define UPDATE_MAP_INIT_EXEC
#define MAP_FUNC_EXEC


#ifdef BRESENHAM_EXEC
#include "data/bresenham/1400.h"
#endif

#ifdef UPDATE_MAP_EXEC
#include "data/log_odds/2000.h"
#endif

#ifdef UPDATE_MAP_INIT_EXEC
#include "data/update_map_init/900.h"
#endif

#ifdef MAP_FUNC_EXEC
#include "data/map/200.h"
#endif

void host_bresenham();
void host_update_map();
void host_update_map_init();
void host_map();

int test_map_partials_main() {

#ifdef BRESENHAM_EXEC
    host_bresenham();
#endif

#ifdef UPDATE_MAP_EXEC
    host_update_map();
#endif

#ifdef UPDATE_MAP_INIT_EXEC
    host_update_map_init();
#endif

#ifdef MAP_FUNC_EXEC
    host_map();
#endif

    return 0;
}


#ifdef BRESENHAM_EXEC
void host_bresenham() {

    const int data_len = 6;
    int data[data_len] = { 1, 0, 2, 2, 1, 3 };
    int* d_data = NULL;
    gpuErrchk(cudaMalloc((void**)&d_data, data_len * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_data, data, data_len * sizeof(int), cudaMemcpyHostToDevice));

    thrust::exclusive_scan(thrust::host, data, data + 6, data, 0); // in-place scan
    thrust::exclusive_scan(thrust::device, d_data, d_data + 6, d_data, 0); // in-place scan


    printf("width=%d, height=%d\n", GRID_WIDTH, GRID_HEIGHT);
    int MAX_DIST_IN_MAP = sqrt(pow(GRID_WIDTH, 2) + pow(GRID_HEIGHT, 2));
    printf("MAX_DIST_IN_MAP=%d\n", MAX_DIST_IN_MAP);

    /********************************************************************/
    /************************ BRESENHAM VARIABLES ***********************/
    /********************************************************************/
    int PARTICLES_OCCUPIED_LEN = ST_PARTICLES_OCCUPIED_LEN;
    int PARTICLES_FREE_LEN = ST_PARTICLES_FREE_LEN;

    int PARTICLE_UNIQUE_COUNTER = PARTICLES_OCCUPIED_LEN + 1;

    size_t sz_particles_occupied_pos = PARTICLES_OCCUPIED_LEN * sizeof(int);
    size_t sz_particles_free_pos = 0;
    size_t sz_particles_free_pos_max = PARTICLES_OCCUPIED_LEN * MAX_DIST_IN_MAP * sizeof(int);
    size_t sz_particles_free_counter = PARTICLE_UNIQUE_COUNTER * sizeof(int);
    size_t sz_position_image_body = 2 * sizeof(int);

    int* d_particles_occupied_x = NULL;
    int* d_particles_occupied_y = NULL;
    int* d_particles_occupied_idx = NULL;
    int* d_particles_free_x = NULL;
    int* d_particles_free_y = NULL;
    int* d_particles_free_x_max = NULL;
    int* d_particles_free_y_max = NULL;
    int* d_particles_free_counter = NULL;
    int* d_particles_free_idx = NULL;
    int* d_position_image_body = NULL;

    gpuErrchk(cudaMalloc((void**)&d_particles_occupied_x, sz_particles_occupied_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_occupied_y, sz_particles_occupied_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_x_max, sz_particles_free_pos_max));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_y_max, sz_particles_free_pos_max));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_counter, sz_particles_free_counter));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_idx, sz_particles_occupied_pos));
    gpuErrchk(cudaMalloc((void**)&d_position_image_body, sz_position_image_body));

    int* res_particles_free_x = (int*)malloc(sz_particles_free_pos);
    int* res_particles_free_y = (int*)malloc(sz_particles_free_pos);
    int* res_particles_free_counter = (int*)malloc(sz_particles_free_counter);

    gpuErrchk(cudaMemcpy(d_particles_occupied_x, h_particles_occupied_x, sz_particles_occupied_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particles_occupied_y, h_particles_occupied_y, sz_particles_occupied_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particles_free_idx, h_particles_free_idx, sz_particles_occupied_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_position_image_body, h_position_image_body, sz_position_image_body, cudaMemcpyHostToDevice));


    memset(res_particles_free_x, 0, sz_particles_free_pos);
    memset(res_particles_free_y, 0, sz_particles_free_pos);
    gpuErrchk(cudaMemset(d_particles_free_x, 0, sz_particles_free_pos));
    gpuErrchk(cudaMemset(d_particles_free_y, 0, sz_particles_free_pos));
    gpuErrchk(cudaMemset(d_particles_free_x_max, 0, sz_particles_free_pos_max));
    gpuErrchk(cudaMemset(d_particles_free_y_max, 0, sz_particles_free_pos_max));
    gpuErrchk(cudaMemset(d_particles_free_counter, 0, sz_particles_free_counter));



    /********************************************************************/
    /************************* BRESENHAM KERNEL *************************/
    /********************************************************************/
    auto start_bresenham = std::chrono::high_resolution_clock::now();

    int threadsPerBlock = 256;
    int blocksPerGrid = (PARTICLES_OCCUPIED_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_bresenham << <blocksPerGrid, threadsPerBlock >> > (d_particles_occupied_x, d_particles_occupied_y, d_position_image_body,
        d_particles_free_x_max, d_particles_free_y_max, d_particles_free_counter, PARTICLES_OCCUPIED_LEN, MAX_DIST_IN_MAP);
    cudaDeviceSynchronize();
    //kernel_update_unique_sum << <1, 1 >> > (d_particles_free_counter, PARTICLE_UNIQUE_COUNTER);
    //cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, d_particles_free_counter, d_particles_free_counter + PARTICLE_UNIQUE_COUNTER, d_particles_free_counter, 0); // in-place scan
    auto stop_bresenham = std::chrono::high_resolution_clock::now();


    auto start_bresenham_rearrange = std::chrono::high_resolution_clock::now();
    gpuErrchk(cudaMemcpy(res_particles_free_counter, d_particles_free_counter, sz_particles_free_counter, cudaMemcpyDeviceToHost));

    PARTICLES_FREE_LEN = res_particles_free_counter[PARTICLE_UNIQUE_COUNTER - 1];
    sz_particles_free_pos = PARTICLES_FREE_LEN * sizeof(int);
    gpuErrchk(cudaMalloc((void**)&d_particles_free_x, sz_particles_free_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_y, sz_particles_free_pos));

    kernel_bresenham_rearrange << <blocksPerGrid, threadsPerBlock >> > (d_particles_free_x, d_particles_free_y, d_particles_free_x_max, d_particles_free_y_max,
        d_particles_free_counter, MAX_DIST_IN_MAP, PARTICLES_OCCUPIED_LEN);
    cudaDeviceSynchronize();
    auto stop_bresenham_rearrange = std::chrono::high_resolution_clock::now();


    auto duration_bresenham = std::chrono::duration_cast<std::chrono::microseconds>(stop_bresenham - start_bresenham);
    auto duration_bresenham_rearrange = std::chrono::duration_cast<std::chrono::microseconds>(stop_bresenham_rearrange - start_bresenham_rearrange);
    std::cout << "Time taken by function (Bresenham): " << duration_bresenham.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Bresenham Rearrange): " << duration_bresenham_rearrange.count() << " microseconds" << std::endl;

    res_particles_free_x = (int*)malloc(sz_particles_free_pos);
    res_particles_free_y = (int*)malloc(sz_particles_free_pos);

    gpuErrchk(cudaMemcpy(res_particles_free_x, d_particles_free_x, sz_particles_free_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_free_y, d_particles_free_y, sz_particles_free_pos, cudaMemcpyDeviceToHost));

    ASSERT_particles_free_index(res_particles_free_counter, h_particles_free_idx, PARTICLES_OCCUPIED_LEN, false);

    ASSERT_particles_free_new_len(PARTICLES_FREE_LEN, ST_PARTICLES_FREE_LEN);

    ASSERT_particles_free(res_particles_free_x, res_particles_free_y, h_particles_free_x, h_particles_free_y, PARTICLES_FREE_LEN);

    printf("Program Finished\n");

    gpuErrchk(cudaFree(d_particles_occupied_x));
    gpuErrchk(cudaFree(d_particles_occupied_y));
    gpuErrchk(cudaFree(d_particles_free_idx));
    gpuErrchk(cudaFree(d_particles_free_x));
    gpuErrchk(cudaFree(d_particles_free_y));
}
#endif

#ifdef UPDATE_MAP_INIT_EXEC
void host_update_map_init() {

    /********************************************************************/
    /********************* IMAGE TRANSFORM VARIABLES ********************/
    /********************************************************************/
    //size_t sz_lidar_coords = LIDAR_COORDS_LEN * sizeof(float);
    size_t sz_transition_frames = 9 * sizeof(float);
    size_t sz_lidar_coords = 2 * LIDAR_COORDS_LEN * sizeof(float);
    size_t sz_processed_measure_pos = LIDAR_COORDS_LEN * sizeof(int);
    size_t sz_particles_wframe_pos = LIDAR_COORDS_LEN * sizeof(float);
    size_t sz_position_image = 2 * sizeof(int);

    float* d_lidar_coords = NULL;
    float* d_transition_body_lidar = NULL;
    float* d_transition_world_body = NULL;
    float* d_transition_world_lidar = NULL;
    int* d_processed_measure_x = NULL;
    int* d_processed_measure_y = NULL;
    float* d_particles_wframe_x = NULL;
    float* d_particles_wframe_y = NULL;
    int* d_position_image_body = NULL;

    float* res_transition_world_lidar = (float*)malloc(sz_transition_frames);
    int* res_processed_measure_x = (int*)malloc(sz_processed_measure_pos);
    int* res_processed_measure_y = (int*)malloc(sz_processed_measure_pos);
    float* res_particles_wframe_x = (float*)malloc(sz_particles_wframe_pos);
    float* res_particles_wframe_y = (float*)malloc(sz_particles_wframe_pos);
    int* res_position_image_body = (int*)malloc(sz_position_image);


    gpuErrchk(cudaMalloc((void**)&d_lidar_coords, sz_lidar_coords));
    gpuErrchk(cudaMalloc((void**)&d_transition_body_lidar, sz_transition_frames));
    gpuErrchk(cudaMalloc((void**)&d_transition_world_body, sz_transition_frames));
    gpuErrchk(cudaMalloc((void**)&d_transition_world_lidar, sz_transition_frames));
    gpuErrchk(cudaMalloc((void**)&d_processed_measure_x, sz_processed_measure_pos));
    gpuErrchk(cudaMalloc((void**)&d_processed_measure_y, sz_processed_measure_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_wframe_x, sz_particles_wframe_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_wframe_y, sz_particles_wframe_pos));
    gpuErrchk(cudaMalloc((void**)&d_position_image_body, sz_position_image));


    gpuErrchk(cudaMemcpy(d_lidar_coords, lidar_coords, sz_lidar_coords, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_transition_body_lidar, h_transition_body_lidar, sz_transition_frames, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_transition_world_body, h_transition_world_body, sz_transition_frames, cudaMemcpyHostToDevice));


    /********************************************************************/
    /********************* IMAGE TRANSFORM KERNEL ********************/
    /********************************************************************/
    auto start_world_to_image_transform = std::chrono::high_resolution_clock::now();
    kernel_matrix_mul_3x3 << < 1, 1 >> > (d_transition_world_body, d_transition_body_lidar, d_transition_world_lidar);
    cudaDeviceSynchronize();

    int threadsPerBlock = 1;
    int blocksPerGrid = LIDAR_COORDS_LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (d_transition_world_lidar, d_processed_measure_x, d_processed_measure_y,
        d_particles_wframe_x, d_particles_wframe_y, d_lidar_coords, res, xmin, ymax, LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    kernel_position_to_image << < 1, 1 >> > (d_position_image_body, d_transition_world_lidar, res, xmin, ymax);
    cudaDeviceSynchronize();

    auto stop_world_to_image_transform = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_transition_world_lidar, d_transition_world_lidar, sz_transition_frames, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_processed_measure_x, d_processed_measure_x, sz_processed_measure_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_processed_measure_y, d_processed_measure_y, sz_processed_measure_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_wframe_x, d_particles_wframe_x, sz_particles_wframe_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_wframe_y, d_particles_wframe_y, sz_particles_wframe_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_position_image_body, d_position_image_body, sz_position_image, cudaMemcpyDeviceToHost));

    ASSERT_transition_world_lidar(res_transition_world_lidar, h_transition_world_lidar, 9, false);

    ASSERT_particles_wframe(res_particles_wframe_x, res_particles_wframe_y, h_particles_wframe_x, h_particles_wframe_y, LIDAR_COORDS_LEN, false);

    ASSERT_processed_measurements(res_processed_measure_x, res_processed_measure_y, h_particles_x, h_particles_y, LIDAR_COORDS_LEN);

    ASSERT_position_image_body(res_position_image_body, h_position_image_body);

    auto duration_world_to_image_transform = std::chrono::duration_cast<std::chrono::microseconds>(stop_world_to_image_transform - start_world_to_image_transform);
    std::cout << "Time taken by function (Kernel): " << duration_world_to_image_transform.count() << " microseconds" << std::endl;

}
#endif

#ifdef UPDATE_MAP_EXEC
void host_update_map() {

    // [✓]

    int PARTICLES_OCCUPIED_LEN = ST_PARTICLES_OCCUPIED_LEN;
    int PARTICLES_OCCUPIED_UNIQUE_LEN = 0;
    int PARTICLES_FREE_LEN = ST_PARTICLES_FREE_LEN;
    int PARTICLES_FREE_UNIQUE_LEN = 0;

    /********************************************************************/
    /**************************** MAP VARIABLES *************************/
    /********************************************************************/
    size_t sz_particles_occupied_pos = PARTICLES_OCCUPIED_LEN * sizeof(int);
    size_t sz_particles_free_pos = PARTICLES_FREE_LEN * sizeof(int);
    size_t sz_map = (GRID_WIDTH * GRID_HEIGHT) * sizeof(int);

    int* d_grid_map = NULL;
    int* d_particles_occupied_x = NULL;
    int* d_particles_occupied_y = NULL;
    int* d_particles_free_x = NULL;
    int* d_particles_free_y = NULL;

    gpuErrchk(cudaMalloc((void**)&d_grid_map, sz_map));
    gpuErrchk(cudaMalloc((void**)&d_particles_occupied_x, sz_particles_occupied_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_occupied_y, sz_particles_occupied_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_x, sz_particles_free_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_y, sz_particles_free_pos));

    gpuErrchk(cudaMemcpy(d_grid_map, pre_grid_map, sz_map, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particles_occupied_x, h_particles_occupied_x, sz_particles_occupied_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particles_occupied_y, h_particles_occupied_y, sz_particles_occupied_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particles_free_x, h_particles_free_x, sz_particles_free_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particles_free_y, h_particles_free_y, sz_particles_free_pos, cudaMemcpyHostToDevice));

    /********************************************************************/
    /************************* LOG-ODDS VARIABLES ***********************/
    /********************************************************************/
    size_t   sz_map_2d = GRID_WIDTH * GRID_HEIGHT * sizeof(uint8_t);
    size_t   sz_unique_counter = 1 * sizeof(int);
    size_t   sz_unique_counter_col = (GRID_WIDTH + 1) * sizeof(int);

    uint8_t* d_map_occupied_2d = NULL;
    uint8_t* d_map_free_2d = NULL;
    int* d_unique_occupied_counter = NULL;
    int* d_unique_free_counter = NULL;
    int* d_unique_occupied_counter_col = NULL;
    int* d_unique_free_counter_col = NULL;

    int* res_unique_occupied_counter = (int*)malloc(sz_unique_counter);
    int* res_unique_free_counter = (int*)malloc(sz_unique_counter);
    int* res_unique_occupied_counter_col = (int*)malloc(sz_unique_counter_col);
    int* res_unique_free_counter_col = (int*)malloc(sz_unique_counter_col);


    gpuErrchk(cudaMalloc((void**)&d_map_occupied_2d, sz_map_2d));
    gpuErrchk(cudaMalloc((void**)&d_map_free_2d, sz_map_2d));
    gpuErrchk(cudaMalloc((void**)&d_unique_occupied_counter, sz_unique_counter));
    gpuErrchk(cudaMalloc((void**)&d_unique_occupied_counter_col, sz_unique_counter_col));
    gpuErrchk(cudaMalloc((void**)&d_unique_free_counter, sz_unique_counter));
    gpuErrchk(cudaMalloc((void**)&d_unique_free_counter_col, sz_unique_counter_col));

    size_t sz_particles_idx = 2 * sizeof(int);

    int h_particles_occupied_idx[] = { 0, PARTICLES_OCCUPIED_LEN };
    int h_particles_free_idx[] = { 0, PARTICLES_FREE_LEN };
    int* d_particles_occupied_idx = NULL;
    int* d_particles_free_idx = NULL;

    gpuErrchk(cudaMalloc((void**)&d_particles_occupied_idx, sz_particles_idx));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_idx, sz_particles_idx));

    gpuErrchk(cudaMemset(d_map_occupied_2d, 0, sz_map_2d));
    gpuErrchk(cudaMemset(d_map_free_2d, 0, sz_map_2d));

    gpuErrchk(cudaMemcpy(d_particles_occupied_idx, h_particles_occupied_idx, sz_particles_idx, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particles_free_idx, h_particles_free_idx, sz_particles_idx, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemset(d_unique_occupied_counter_col, 0, sz_unique_counter_col));
    gpuErrchk(cudaMemset(d_unique_free_counter_col, 0, sz_unique_counter_col));

    /********************************************************************/
    /**************************** CREATE MAP ****************************/
    /********************************************************************/
    auto start_create_map = std::chrono::high_resolution_clock::now();
    int threadsPerBlock = 256;
    int blocksPerGrid = 1;
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (d_particles_occupied_x, d_particles_occupied_y, d_particles_occupied_idx,
        PARTICLES_OCCUPIED_LEN, d_map_occupied_2d, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (d_particles_free_x, d_particles_free_y, d_particles_free_idx,
        PARTICLES_FREE_LEN, d_map_free_2d, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();
    auto stop_create_map = std::chrono::high_resolution_clock::now();


    auto start_unique_counter = std::chrono::high_resolution_clock::now();
    threadsPerBlock = GRID_WIDTH;
    blocksPerGrid = 1;
    kernel_2d_map_counter << <blocksPerGrid, threadsPerBlock >> > (d_map_occupied_2d, d_unique_occupied_counter, d_unique_occupied_counter_col, GRID_WIDTH, GRID_HEIGHT);
    kernel_2d_map_counter << <blocksPerGrid, threadsPerBlock >> > (d_map_free_2d, d_unique_free_counter, d_unique_free_counter_col, GRID_WIDTH, GRID_HEIGHT);
    cudaDeviceSynchronize();
    auto stop_unique_counter = std::chrono::high_resolution_clock::now();

    auto start_unique_sum = std::chrono::high_resolution_clock::now();
    threadsPerBlock = 1;
    blocksPerGrid = 1;
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (d_unique_occupied_counter_col, GRID_WIDTH);
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (d_unique_free_counter_col, GRID_WIDTH);
    cudaDeviceSynchronize();
    auto stop_unique_sum = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(h_particles_occupied_idx, d_particles_occupied_idx, sz_particles_idx, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_unique_occupied_counter, d_unique_occupied_counter, sz_unique_counter, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_unique_free_counter, d_unique_free_counter, sz_unique_counter, cudaMemcpyDeviceToHost));

    //gpuErrchk(cudaMemcpy(res_unique_occupied_counter_col, d_unique_occupied_counter_col, sz_unique_counter_col, cudaMemcpyDeviceToHost));
    //gpuErrchk(cudaMemcpy(res_unique_free_counter_col, d_unique_free_counter_col, sz_unique_counter_col, cudaMemcpyDeviceToHost));

    /*---------------------------------------------------------------------*/
    /*-------------------- REINITIALIZE MAP VARIABLES ---------------------*/
    /*---------------------------------------------------------------------*/
    /*---------------------------------------------------------------------*/
    PARTICLES_OCCUPIED_UNIQUE_LEN = res_unique_occupied_counter[0];
    PARTICLES_FREE_UNIQUE_LEN = res_unique_free_counter[0];

    printf("\n--> Occupied Unique: %d, %d\n", PARTICLES_OCCUPIED_UNIQUE_LEN, ST_PARTICLES_OCCUPIED_UNIQUE_LEN);
    assert(PARTICLES_OCCUPIED_UNIQUE_LEN == ST_PARTICLES_OCCUPIED_UNIQUE_LEN);
    printf("\n--> Free Unique: %d, %d\n", PARTICLES_FREE_UNIQUE_LEN, ST_PARTICLES_FREE_UNIQUE_LEN);
    assert(PARTICLES_FREE_UNIQUE_LEN == ST_PARTICLES_FREE_UNIQUE_LEN);


    gpuErrchk(cudaFree(d_particles_occupied_x));
    gpuErrchk(cudaFree(d_particles_occupied_y));
    gpuErrchk(cudaFree(d_particles_free_x));
    gpuErrchk(cudaFree(d_particles_free_y));

    sz_particles_occupied_pos = PARTICLES_OCCUPIED_UNIQUE_LEN * sizeof(int);
    sz_particles_free_pos = PARTICLES_FREE_UNIQUE_LEN * sizeof(int);

    int* res_particles_occupied_x = (int*)malloc(sz_particles_occupied_pos);
    int* res_particles_occupied_y = (int*)malloc(sz_particles_occupied_pos);
    int* res_particles_free_x = (int*)malloc(sz_particles_free_pos);
    int* res_particles_free_y = (int*)malloc(sz_particles_free_pos);


    gpuErrchk(cudaMalloc((void**)&d_particles_occupied_x, sz_particles_occupied_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_occupied_y, sz_particles_occupied_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_x, sz_particles_free_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_y, sz_particles_free_pos));

    auto start_restructure_map = std::chrono::high_resolution_clock::now();
    threadsPerBlock = GRID_WIDTH;
    blocksPerGrid = 1;
    kernel_update_unique_restructure2 << <blocksPerGrid, threadsPerBlock >> > (d_map_occupied_2d, d_particles_occupied_x, d_particles_occupied_y, d_particles_occupied_idx,
        d_unique_occupied_counter_col, GRID_WIDTH, GRID_HEIGHT);
    kernel_update_unique_restructure2 << <blocksPerGrid, threadsPerBlock >> > (d_map_free_2d, d_particles_free_x, d_particles_free_y, d_particles_free_idx,
        d_unique_free_counter_col, GRID_WIDTH, GRID_HEIGHT);
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

    /********************************************************************/
    /************************* LOG-ODDS VARIABLES ***********************/
    /********************************************************************/
    size_t sz_log_odds = (GRID_WIDTH * GRID_HEIGHT) * sizeof(float);

    float* d_log_odds = NULL;

    int* res_grid_map = (int*)malloc(sz_map);
    float* res_log_odds = (float*)malloc(sz_log_odds);
    memset(res_log_odds, 0, sz_log_odds);

    gpuErrchk(cudaMalloc((void**)&d_log_odds, sz_log_odds));
    gpuErrchk(cudaMemcpy(d_log_odds, pre_log_odds, sz_log_odds, cudaMemcpyHostToDevice));


    /********************************************************************/
    /************************** LOG-ODDS KERNEL *************************/
    /********************************************************************/
    auto start_update_map = std::chrono::high_resolution_clock::now();

    threadsPerBlock = 256;
    blocksPerGrid = (PARTICLES_OCCUPIED_UNIQUE_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_update_log_odds << <blocksPerGrid, threadsPerBlock >> > (d_log_odds, d_particles_occupied_x, d_particles_occupied_y, 2 * log_t, GRID_WIDTH, GRID_HEIGHT, PARTICLES_OCCUPIED_UNIQUE_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = 256;
    blocksPerGrid = (PARTICLES_FREE_UNIQUE_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_update_log_odds << <blocksPerGrid, threadsPerBlock >> > (d_log_odds, d_particles_free_x, d_particles_free_y, (-1) * log_t, GRID_WIDTH, GRID_HEIGHT, PARTICLES_FREE_UNIQUE_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = 256;
    blocksPerGrid = ((GRID_WIDTH * GRID_HEIGHT) + threadsPerBlock - 1) / threadsPerBlock;
    kernel_update_map << <blocksPerGrid, threadsPerBlock >> > (d_grid_map, d_log_odds, LOG_ODD_PRIOR, WALL, FREE, GRID_WIDTH * GRID_HEIGHT);
    cudaDeviceSynchronize();

    auto stop_update_map = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_log_odds, d_log_odds, sz_log_odds, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_grid_map, d_grid_map, sz_map, cudaMemcpyDeviceToHost));

    ASSERT_log_odds(res_log_odds, pre_log_odds, post_log_odds, (GRID_WIDTH * GRID_HEIGHT));
    ASSERT_log_odds_maps(res_grid_map, pre_grid_map, post_grid_map, (GRID_WIDTH * GRID_HEIGHT));
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

}

#endif

#ifdef MAP_FUNC_EXEC
void host_map() {

    const int data_len = 6;
    int data[data_len] = { 1, 0, 2, 2, 1, 3 };
    int* d_data = NULL;
    gpuErrchk(cudaMalloc((void**)&d_data, data_len * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_data, data, data_len * sizeof(int), cudaMemcpyHostToDevice));

    thrust::exclusive_scan(thrust::host, data, data + 6, data, 0);
    thrust::exclusive_scan(thrust::device, d_data, d_data + 6, d_data, 0);

    int PARTICLES_OCCUPIED_LEN = ST_PARTICLES_OCCUPIED_LEN;
    int PARTICLES_FREE_LEN = 0;
    int PARTICLES_OCCUPIED_UNIQUE_LEN = 0;
    int PARTICLES_FREE_UNIQUE_LEN = 0;
    int LIDAR_COORDS_LEN = ST_LIDAR_COORDS_LEN;
    int GRID_WIDTH = ST_GRID_WIDTH;
    int GRID_HEIGHT = ST_GRID_HEIGHT;

    /**************************************************************************************************************************************************/
    /**************************************************************** VARIABLES SCOPE *****************************************************************/
    /**************************************************************************************************************************************************/

    /********************************************************************/
    /********************* IMAGE TRANSFORM VARIABLES ********************/
    /********************************************************************/
    size_t sz_transition_frames = 9 * sizeof(float);
    size_t sz_lidar_coords = 2 * LIDAR_COORDS_LEN * sizeof(float);
    size_t sz_particles_occupied_pos = LIDAR_COORDS_LEN * sizeof(int);
    size_t sz_particles_wframe_pos = LIDAR_COORDS_LEN * sizeof(float);
    size_t sz_position_image = 2 * sizeof(int);

    float* d_lidar_coords = NULL;
    float* d_transition_body_lidar = NULL;
    float* d_transition_world_body = NULL;
    float* d_transition_world_lidar = NULL;
    int* d_particles_occupied_x = NULL;
    int* d_particles_occupied_y = NULL;
    float* d_particles_wframe_x = NULL;
    float* d_particles_wframe_y = NULL;


    float* res_transition_world_lidar = (float*)malloc(sz_transition_frames);
    int* res_particles_occupied_x = (int*)malloc(sz_particles_occupied_pos);
    int* res_particles_occupied_y = (int*)malloc(sz_particles_occupied_pos);
    float* res_particles_wframe_x = (float*)malloc(sz_particles_wframe_pos);
    float* res_particles_wframe_y = (float*)malloc(sz_particles_wframe_pos);
    int* res_position_image_body = (int*)malloc(sz_position_image);

    gpuErrchk(cudaMalloc((void**)&d_lidar_coords, sz_lidar_coords));
    gpuErrchk(cudaMalloc((void**)&d_transition_body_lidar, sz_transition_frames));
    gpuErrchk(cudaMalloc((void**)&d_transition_world_body, sz_transition_frames));
    gpuErrchk(cudaMalloc((void**)&d_transition_world_lidar, sz_transition_frames));
    gpuErrchk(cudaMalloc((void**)&d_particles_occupied_x, sz_particles_occupied_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_occupied_y, sz_particles_occupied_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_wframe_x, sz_particles_wframe_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_wframe_y, sz_particles_wframe_pos));

    gpuErrchk(cudaMemcpy(d_lidar_coords, lidar_coords, sz_lidar_coords, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_transition_body_lidar, h_transition_body_lidar, sz_transition_frames, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_transition_world_body, h_transition_world_body, sz_transition_frames, cudaMemcpyHostToDevice));


    /********************************************************************/
    /************************ BRESENHAM VARIABLES ***********************/
    /********************************************************************/
    int MAX_DIST_IN_MAP = sqrt(pow(GRID_WIDTH, 2) + pow(GRID_HEIGHT, 2));
    int PARTICLE_UNIQUE_COUNTER = PARTICLES_OCCUPIED_LEN + 1;

    //size_t sz_particles_occupied_pos = PARTICLES_OCCUPIED_LEN * sizeof(int);
    size_t sz_particles_free_pos = 0;
    size_t sz_particles_free_pos_max = PARTICLES_OCCUPIED_LEN * MAX_DIST_IN_MAP * sizeof(int);
    size_t sz_particles_free_counter = PARTICLE_UNIQUE_COUNTER * sizeof(int);
    size_t sz_particles_free_idx = PARTICLES_OCCUPIED_LEN * sizeof(int);
    size_t sz_position_image_body = 2 * sizeof(int);

    int* d_particles_occupied_idx = NULL;
    int* d_particles_free_x = NULL;
    int* d_particles_free_y = NULL;
    int* d_particles_free_x_max = NULL;
    int* d_particles_free_y_max = NULL;
    int* d_particles_free_counter = NULL;
    int* d_particles_free_idx = NULL;
    int* d_position_image_body = NULL;

    gpuErrchk(cudaMalloc((void**)&d_particles_free_x_max, sz_particles_free_pos_max));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_y_max, sz_particles_free_pos_max));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_counter, sz_particles_free_counter));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_idx, sz_particles_free_idx));
    gpuErrchk(cudaMalloc((void**)&d_position_image_body, sz_position_image_body));

    int* res_particles_free_x = (int*)malloc(sz_particles_free_pos);
    int* res_particles_free_y = (int*)malloc(sz_particles_free_pos);
    int* res_particles_free_counter = (int*)malloc(sz_particles_free_counter);

    memset(res_particles_free_x, 0, sz_particles_free_pos);
    memset(res_particles_free_y, 0, sz_particles_free_pos);
    gpuErrchk(cudaMemset(d_particles_free_x, 0, sz_particles_free_pos));
    gpuErrchk(cudaMemset(d_particles_free_y, 0, sz_particles_free_pos));
    gpuErrchk(cudaMemset(d_particles_free_x_max, 0, sz_particles_free_pos_max));
    gpuErrchk(cudaMemset(d_particles_free_y_max, 0, sz_particles_free_pos_max));
    gpuErrchk(cudaMemset(d_particles_free_counter, 0, sz_particles_free_counter));


    /********************************************************************/
    /**************************** MAP VARIABLES *************************/
    /********************************************************************/
    size_t sz_map = (GRID_WIDTH * GRID_HEIGHT) * sizeof(int);

    int* d_grid_map = NULL;
    gpuErrchk(cudaMalloc((void**)&d_grid_map, sz_map));
    gpuErrchk(cudaMemcpy(d_grid_map, pre_grid_map, sz_map, cudaMemcpyHostToDevice));

    /********************************************************************/
    /************************* LOG-ODDS VARIABLES ***********************/
    /********************************************************************/
    size_t  sz_map_2d = GRID_WIDTH * GRID_HEIGHT * sizeof(uint8_t);
    size_t  sz_unique_counter = 1 * sizeof(int);
    size_t  sz_unique_counter_col = (GRID_WIDTH + 1) * sizeof(int);
    size_t  sz_log_odds = (GRID_WIDTH * GRID_HEIGHT) * sizeof(float);
    size_t  sz_map_idx = 2 * sizeof(int);


    uint8_t* d_map_occupied_2d = NULL;
    uint8_t* d_map_free_2d = NULL;
    int* d_unique_occupied_counter = NULL;
    int* d_unique_free_counter = NULL;
    int* d_unique_occupied_counter_col = NULL;
    int* d_unique_free_counter_col = NULL;
    float* d_log_odds = NULL;

    int* res_unique_occupied_counter = (int*)malloc(sz_unique_counter);
    int* res_unique_free_counter = (int*)malloc(sz_unique_counter);
    int* res_unique_occupied_counter_col = (int*)malloc(sz_unique_counter_col);
    int* res_unique_free_counter_col = (int*)malloc(sz_unique_counter_col);
    int* res_grid_map = (int*)malloc(sz_map);
    float* res_log_odds = (float*)malloc(sz_log_odds);


    gpuErrchk(cudaMalloc((void**)&d_map_occupied_2d, sz_map_2d));
    gpuErrchk(cudaMalloc((void**)&d_map_free_2d, sz_map_2d));
    gpuErrchk(cudaMalloc((void**)&d_unique_occupied_counter, sz_unique_counter));
    gpuErrchk(cudaMalloc((void**)&d_unique_occupied_counter_col, sz_unique_counter_col));
    gpuErrchk(cudaMalloc((void**)&d_unique_free_counter, sz_unique_counter));
    gpuErrchk(cudaMalloc((void**)&d_unique_free_counter_col, sz_unique_counter_col));
    gpuErrchk(cudaMalloc((void**)&d_log_odds, sz_log_odds));

    printf("PARTICLES_OCCUPIED_LEN=%d\n", PARTICLES_OCCUPIED_LEN);

    int h_map_occupied_idx[] = { 0, PARTICLES_OCCUPIED_LEN };
    int h_map_free_idx[] = { 0, 0 };
    int* d_map_occupied_idx = NULL;
    int* d_map_free_idx = NULL;

    memset(res_log_odds, 0, sz_log_odds);

    gpuErrchk(cudaMalloc((void**)&d_map_occupied_idx, sz_map_idx));
    gpuErrchk(cudaMalloc((void**)&d_map_free_idx, sz_map_idx));
    gpuErrchk(cudaMemcpy(d_map_occupied_idx, h_map_occupied_idx, sz_map_idx, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_map_free_idx, h_map_free_idx, sz_map_idx, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_log_odds, pre_log_odds, sz_log_odds, cudaMemcpyHostToDevice));

    //gpuErrchk(cudaMemset(d_map_occupied_2d, 0, sz_map_2d));
    //gpuErrchk(cudaMemset(d_map_free_2d, 0, sz_map_2d));

    //gpuErrchk(cudaMemset(d_unique_occupied_counter_col, 0, sz_unique_counter_col));
    //gpuErrchk(cudaMemset(d_unique_free_counter_col, 0, sz_unique_counter_col));


    /**************************************************************************************************************************************************/
    /************************************************************* KERNEL EXECUTION SCOPE *************************************************************/
    /**************************************************************************************************************************************************/

    /********************************************************************/
    /***************** World to IMAGE TRANSFORM KERNEL ******************/
    /********************************************************************/
    auto start_world_to_image_transform = std::chrono::high_resolution_clock::now();
    kernel_matrix_mul_3x3 << < 1, 1 >> > (d_transition_world_body, d_transition_body_lidar, d_transition_world_lidar);
    cudaDeviceSynchronize();

    int threadsPerBlock = 1;
    int blocksPerGrid = LIDAR_COORDS_LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (d_transition_world_lidar, d_particles_occupied_x, d_particles_occupied_y,
        d_particles_wframe_x, d_particles_wframe_y, d_lidar_coords, res, xmin, ymax, LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    kernel_position_to_image << < 1, 1 >> > (d_position_image_body, d_transition_world_lidar, res, xmin, ymax);
    cudaDeviceSynchronize();
    auto stop_world_to_image_transform = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_transition_world_lidar, d_transition_world_lidar, sz_transition_frames, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_occupied_x, d_particles_occupied_x, sz_particles_occupied_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_occupied_y, d_particles_occupied_y, sz_particles_occupied_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_wframe_x, d_particles_wframe_x, sz_particles_wframe_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_wframe_y, d_particles_wframe_y, sz_particles_wframe_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_position_image_body, d_position_image_body, sz_position_image, cudaMemcpyDeviceToHost));

    //ASSERT_transition_world_lidar(res_transition_world_lidar, h_transition_world_lidar, 9, false);
    //ASSERT_particles_wframe(res_particles_wframe_x, res_particles_wframe_y, h_particles_wframe_x, h_particles_wframe_y, LIDAR_COORDS_LEN, false);
    ASSERT_processed_measurements(res_particles_occupied_x, res_particles_occupied_y, h_particles_occupied_x, h_particles_occupied_y, LIDAR_COORDS_LEN);
    ASSERT_position_image_body(res_position_image_body, h_position_image_body);


    /********************************************************************/
    /************************* BRESENHAM KERNEL *************************/
    /********************************************************************/
    auto start_bresenham = std::chrono::high_resolution_clock::now();

    threadsPerBlock = 256;
    blocksPerGrid = (PARTICLES_OCCUPIED_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_bresenham << <blocksPerGrid, threadsPerBlock >> > (d_particles_occupied_x, d_particles_occupied_y, d_position_image_body,
        d_particles_free_x_max, d_particles_free_y_max, d_particles_free_counter, PARTICLES_OCCUPIED_LEN, MAX_DIST_IN_MAP);
    cudaDeviceSynchronize();
    thrust::exclusive_scan(thrust::device, d_particles_free_counter, d_particles_free_counter + PARTICLE_UNIQUE_COUNTER, d_particles_free_counter, 0); // in-place scan
    auto stop_bresenham = std::chrono::high_resolution_clock::now();

    auto start_bresenham_rearrange = std::chrono::high_resolution_clock::now();
    gpuErrchk(cudaMemcpy(res_particles_free_counter, d_particles_free_counter, sz_particles_free_counter, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(d_particles_free_idx, d_particles_free_counter, sz_particles_free_idx, cudaMemcpyDeviceToDevice));

    PARTICLES_FREE_LEN = res_particles_free_counter[PARTICLE_UNIQUE_COUNTER - 1];
    sz_particles_free_pos = PARTICLES_FREE_LEN * sizeof(int);
    gpuErrchk(cudaMalloc((void**)&d_particles_free_x, sz_particles_free_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_free_y, sz_particles_free_pos));

    kernel_bresenham_rearrange << <blocksPerGrid, threadsPerBlock >> > (d_particles_free_x, d_particles_free_y, d_particles_free_x_max, d_particles_free_y_max,
        d_particles_free_counter, MAX_DIST_IN_MAP, PARTICLES_OCCUPIED_LEN);
    cudaDeviceSynchronize();
    auto stop_bresenham_rearrange = std::chrono::high_resolution_clock::now();

    res_particles_free_x = (int*)malloc(sz_particles_free_pos);
    res_particles_free_y = (int*)malloc(sz_particles_free_pos);

    gpuErrchk(cudaMemcpy(res_particles_free_x, d_particles_free_x, sz_particles_free_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_free_y, d_particles_free_y, sz_particles_free_pos, cudaMemcpyDeviceToHost));

    printf("PARTICLES_FREE_LEN=%d\n", PARTICLES_FREE_LEN);
    ASSERT_particles_free_index(res_particles_free_counter, h_particles_free_idx, PARTICLES_OCCUPIED_LEN, false);
    ASSERT_particles_free_new_len(PARTICLES_FREE_LEN, ST_PARTICLES_FREE_LEN);
    ASSERT_particles_free(res_particles_free_x, res_particles_free_y, h_particles_free_x, h_particles_free_y, PARTICLES_FREE_LEN);


    /*---------------------------------------------------------------------*/
    /*-------------------- REINITIALIZE MAP VARIABLES ---------------------*/
    /*---------------------------------------------------------------------*/
    /*---------------------------------------------------------------------*/
    h_map_free_idx[1] = PARTICLES_FREE_LEN;
    gpuErrchk(cudaMemcpy(d_map_free_idx, h_map_free_idx, sz_map_idx, cudaMemcpyHostToDevice));


    /********************************************************************/
    /**************************** CREATE MAP ****************************/
    /********************************************************************/
    auto start_create_map = std::chrono::high_resolution_clock::now();
    threadsPerBlock = 256;
    blocksPerGrid = 1;
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (d_particles_occupied_x, d_particles_occupied_y, d_map_occupied_idx,
        PARTICLES_OCCUPIED_LEN, d_map_occupied_2d, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (d_particles_free_x, d_particles_free_y, d_map_free_idx,
        PARTICLES_FREE_LEN, d_map_free_2d, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();
    auto stop_create_map = std::chrono::high_resolution_clock::now();


    auto start_unique_counter = std::chrono::high_resolution_clock::now();
    threadsPerBlock = GRID_WIDTH;
    blocksPerGrid = 1;
    kernel_2d_map_counter << <blocksPerGrid, threadsPerBlock >> > (d_map_occupied_2d, d_unique_occupied_counter, d_unique_occupied_counter_col, GRID_WIDTH, GRID_HEIGHT);
    kernel_2d_map_counter << <blocksPerGrid, threadsPerBlock >> > (d_map_free_2d, d_unique_free_counter, d_unique_free_counter_col, GRID_WIDTH, GRID_HEIGHT);
    cudaDeviceSynchronize();
    auto stop_unique_counter = std::chrono::high_resolution_clock::now();

    auto start_unique_sum = std::chrono::high_resolution_clock::now();
    threadsPerBlock = 1;
    blocksPerGrid = 1;
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (d_unique_occupied_counter_col, GRID_WIDTH);
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (d_unique_free_counter_col, GRID_WIDTH);
    cudaDeviceSynchronize();
    auto stop_unique_sum = std::chrono::high_resolution_clock::now();

    //gpuErrchk(cudaMemcpy(h_particles_occupied_idx, d_particles_occupied_idx, sz_particles_idx, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_unique_occupied_counter, d_unique_occupied_counter, sz_unique_counter, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_unique_free_counter, d_unique_free_counter, sz_unique_counter, cudaMemcpyDeviceToHost));

    //gpuErrchk(cudaMemcpy(res_unique_occupied_counter_col, d_unique_occupied_counter_col, sz_unique_counter_col, cudaMemcpyDeviceToHost));
    //gpuErrchk(cudaMemcpy(res_unique_free_counter_col, d_unique_free_counter_col, sz_unique_counter_col, cudaMemcpyDeviceToHost));

    /*---------------------------------------------------------------------*/
    /*-------------------- REINITIALIZE MAP VARIABLES ---------------------*/
    /*---------------------------------------------------------------------*/
    /*---------------------------------------------------------------------*/
    PARTICLES_OCCUPIED_UNIQUE_LEN = res_unique_occupied_counter[0];
    PARTICLES_FREE_UNIQUE_LEN = res_unique_free_counter[0];

    printf("\n--> Occupied Unique: %d, %d\n", PARTICLES_OCCUPIED_UNIQUE_LEN, ST_PARTICLES_OCCUPIED_UNIQUE_LEN);
    assert(PARTICLES_OCCUPIED_UNIQUE_LEN == ST_PARTICLES_OCCUPIED_UNIQUE_LEN);
    printf("\n--> Free Unique: %d, %d\n", PARTICLES_FREE_UNIQUE_LEN, ST_PARTICLES_FREE_UNIQUE_LEN);
    assert(PARTICLES_FREE_UNIQUE_LEN == ST_PARTICLES_FREE_UNIQUE_LEN);

    gpuErrchk(cudaFree(d_particles_occupied_x));
    gpuErrchk(cudaFree(d_particles_occupied_y));
    gpuErrchk(cudaFree(d_particles_free_x));
    gpuErrchk(cudaFree(d_particles_free_y));

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
    kernel_update_unique_restructure2 << <blocksPerGrid, threadsPerBlock >> > (d_map_occupied_2d, d_particles_occupied_x, d_particles_occupied_y, d_particles_occupied_idx,
        d_unique_occupied_counter_col, GRID_WIDTH, GRID_HEIGHT);
    kernel_update_unique_restructure2 << <blocksPerGrid, threadsPerBlock >> > (d_map_free_2d, d_particles_free_x, d_particles_free_y, d_particles_free_idx,
        d_unique_free_counter_col, GRID_WIDTH, GRID_HEIGHT);
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


    /********************************************************************/
    /************************** LOG-ODDS KERNEL *************************/
    /********************************************************************/
    auto start_update_map = std::chrono::high_resolution_clock::now();

    threadsPerBlock = 256;
    blocksPerGrid = (PARTICLES_OCCUPIED_UNIQUE_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_update_log_odds << <blocksPerGrid, threadsPerBlock >> > (d_log_odds, d_particles_occupied_x, d_particles_occupied_y, 2 * log_t, GRID_WIDTH, GRID_HEIGHT, PARTICLES_OCCUPIED_UNIQUE_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = 256;
    blocksPerGrid = (PARTICLES_FREE_UNIQUE_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_update_log_odds << <blocksPerGrid, threadsPerBlock >> > (d_log_odds, d_particles_free_x, d_particles_free_y, (-1) * log_t, GRID_WIDTH, GRID_HEIGHT, PARTICLES_FREE_UNIQUE_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = 256;
    blocksPerGrid = ((GRID_WIDTH * GRID_HEIGHT) + threadsPerBlock - 1) / threadsPerBlock;
    kernel_update_map << <blocksPerGrid, threadsPerBlock >> > (d_grid_map, d_log_odds, LOG_ODD_PRIOR, WALL, FREE, GRID_WIDTH * GRID_HEIGHT);
    cudaDeviceSynchronize();

    auto stop_update_map = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_log_odds, d_log_odds, sz_log_odds, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_grid_map, d_grid_map, sz_map, cudaMemcpyDeviceToHost));

    ASSERT_log_odds(res_log_odds, pre_log_odds, post_log_odds, (GRID_WIDTH * GRID_HEIGHT));
    ASSERT_log_odds_maps(res_grid_map, pre_grid_map, post_grid_map, (GRID_WIDTH * GRID_HEIGHT));
    printf("\n");


    auto duration_world_to_image_transform = std::chrono::duration_cast<std::chrono::microseconds>(stop_world_to_image_transform - start_world_to_image_transform);
    auto duration_bresenham = std::chrono::duration_cast<std::chrono::microseconds>(stop_bresenham - start_bresenham);
    auto duration_bresenham_rearrange = std::chrono::duration_cast<std::chrono::microseconds>(stop_bresenham_rearrange - start_bresenham_rearrange);
    auto duration_create_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_create_map - start_create_map);
    auto duration_unique_counter = std::chrono::duration_cast<std::chrono::microseconds>(stop_unique_counter - start_unique_counter);
    auto duration_unique_sum = std::chrono::duration_cast<std::chrono::microseconds>(stop_unique_sum - start_unique_sum);
    auto duration_restructure_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_restructure_map - start_restructure_map);
    auto duration_update_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_update_map - start_update_map);

    auto duration_total = duration_world_to_image_transform + duration_bresenham + duration_bresenham_rearrange + duration_create_map +
        duration_unique_counter + duration_unique_sum + duration_restructure_map + duration_update_map;

    std::cout << "Time taken by function (World to Image Transform Kernel): " << duration_world_to_image_transform.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Bresenham): " << duration_bresenham.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Bresenham Rearrange): " << duration_bresenham_rearrange.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Create Map): " << duration_create_map.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Unique Counter): " << duration_unique_counter.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Unique Sum): " << duration_unique_sum.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Restructure Map): " << duration_restructure_map.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Update Map): " << duration_update_map.count() << " microseconds" << std::endl;

    std::cout << "Time taken by function (Total): " << duration_total.count() << " microseconds" << std::endl;
}
#endif



#endif
