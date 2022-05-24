
#ifndef _TEST_ROBOT_H_
#define _TEST_ROBOT_H_

#include "headers.h"
#include "host_asserts.h"
#include "host_utils.h"
#include "kernels.cuh"
#include "kernels_robot.cuh"
#include "kernels_utils.cuh"

/**************************************************/
//#include "data/robot/1900.h"
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



int threadsPerBlock = 1;
int blocksPerGrid = 1;

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


HostMapData h_map;
HostMeasurements h_measurements;
HostParticlesData h_particles;
HostRobotParticles h_robot_particles;
HostRobotParticles h_robot_particles_before_resampling;
HostRobotParticles h_robot_particles_after_resampling;
HostRobotParticles h_robot_particles_unique;
HostProcessedMeasure h_processed_measure;
HostState h_state;
HostState h_state_updated;
HostPositionTransition h_position_transition;
HostResampling h_resampling;
HostRobotState h_robot_state;
HostParticlesTransition h_particles_transition;
host_vector<float> weights_pre;
host_vector<float> weights_new;
host_vector<float> weights_updated;
GeneralInfo general_info;


int test_robot_particles_partials_main() {

    const int data_len = 6;
    int data[data_len] = { 1, 0, 2, 2, 1, 3 };
    int* d_data = NULL;
    gpuErrchk(cudaMalloc((void**)&d_data, data_len * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_data, data, data_len * sizeof(int), cudaMemcpyHostToDevice));

    thrust::exclusive_scan(thrust::host, data, data + 6, data, 0);
    thrust::exclusive_scan(thrust::device, d_data, d_data + 6, d_data, 0);
    
    read_update_robot(500, h_map, h_measurements, h_particles, h_robot_particles, h_robot_particles_before_resampling, 
        h_robot_particles_after_resampling, h_robot_particles_unique, h_processed_measure, h_state,
        h_state_updated, h_position_transition,h_resampling, h_robot_state, h_particles_transition,
        weights_pre, weights_new, weights_updated, general_info);

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

    printf("/************************** UPDATE PARTICLES ************************/\n");
    
    DeviceMeasurements d_measurements;
    DeviceState d_state;
    DevicePositionTransition d_position_transition;
    DeviceParticlesTransition d_particles_transition;
    DeviceProcessedMeasure d_processed_measure;

    HostParticlesTransition res_particles_transition;
    HostProcessedMeasure res_processed_measure;

    /************************* STATES VARIABLES *************************/

    d_state.x.resize(NUM_PARTICLES, 0);
    d_state.y.resize(NUM_PARTICLES, 0);
    d_state.theta.resize(NUM_PARTICLES, 0);

    d_state.x.assign(h_state.x.begin(), h_state.x.end());
    d_state.y.assign(h_state.y.begin(), h_state.y.end());
    d_state.theta.assign(h_state.theta.begin(), h_state.theta.end());

    d_measurements.lidar_coords.resize(2 * h_measurements.LIDAR_COORDS_LEN, 0);
    d_measurements.lidar_coords.assign(h_measurements.lidar_coords.begin(), h_measurements.lidar_coords.end());


    /************************ TRANSFORM VARIABLES ***********************/
    d_particles_transition.transition_multi_world_body.resize(9 * NUM_PARTICLES, 0);
    d_particles_transition.transition_multi_world_lidar.resize(9 * NUM_PARTICLES, 0);

    /*------------------------ RESULT VARIABLES -----------------------*/
    res_particles_transition.transition_multi_world_body.resize(9 * NUM_PARTICLES, 0);
    res_particles_transition.transition_multi_world_lidar.resize(9 * NUM_PARTICLES, 0);

    /********************* PROCESSED MEASURE VARIABLES ******************/
    d_processed_measure.x.resize(NUM_PARTICLES * h_measurements.LIDAR_COORDS_LEN, 0);
    d_processed_measure.y.resize(NUM_PARTICLES * h_measurements.LIDAR_COORDS_LEN, 0);
    d_processed_measure.idx.resize(NUM_PARTICLES * h_measurements.LIDAR_COORDS_LEN, 0);

    /*------------------------ RESULT VARIABLES -----------------------*/
    res_processed_measure.x.resize(NUM_PARTICLES * h_measurements.LIDAR_COORDS_LEN, 0);
    res_processed_measure.y.resize(NUM_PARTICLES * h_measurements.LIDAR_COORDS_LEN, 0);
    res_processed_measure.idx.resize(NUM_PARTICLES * h_measurements.LIDAR_COORDS_LEN, 0);

    d_position_transition.transition_body_lidar.resize(9, 0);
    d_position_transition.transition_body_lidar.assign(hvec_transition_body_lidar.begin(), hvec_transition_body_lidar.end());

    /*************************** KERNEL EXEC ****************************/
    auto start_kernel = std::chrono::high_resolution_clock::now();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;

    kernel_update_particles_states << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles_transition.transition_multi_world_body), 
        THRUST_RAW_CAST(d_particles_transition.transition_multi_world_lidar), SEP,
        THRUST_RAW_CAST(d_state.x), THRUST_RAW_CAST(d_state.y), THRUST_RAW_CAST(d_state.theta),
        THRUST_RAW_CAST(d_position_transition.transition_body_lidar), NUM_PARTICLES);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = h_measurements.LIDAR_COORDS_LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_processed_measure.x), THRUST_RAW_CAST(d_processed_measure.y), SEP,
        THRUST_RAW_CAST(d_particles_transition.transition_multi_world_lidar), THRUST_RAW_CAST(d_measurements.lidar_coords), 
        general_info.res, h_map.xmin, h_map.ymax, h_measurements.LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();
    
    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_index_init_const << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_processed_measure.idx), h_measurements.LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = 1;
    blocksPerGrid = 1;
    thrust::exclusive_scan(thrust::device, d_processed_measure.idx.begin(), d_processed_measure.idx.end(), 
        d_processed_measure.idx.begin(), 0);

    auto stop_kernel = std::chrono::high_resolution_clock::now();

    res_particles_transition.transition_multi_world_body.assign(
        d_particles_transition.transition_multi_world_body.begin(), d_particles_transition.transition_multi_world_body.end());
    res_particles_transition.transition_multi_world_lidar.assign(
        d_particles_transition.transition_multi_world_lidar.begin(), d_particles_transition.transition_multi_world_lidar.end());

    res_processed_measure.x.assign(d_processed_measure.x.begin(), d_processed_measure.x.end());
    res_processed_measure.y.assign(d_processed_measure.y.begin(), d_processed_measure.y.end());
    res_processed_measure.idx.assign(d_processed_measure.idx.begin(), d_processed_measure.idx.end());


    ASSERT_transition_frames(res_particles_transition.transition_multi_world_body.data(), 
        res_particles_transition.transition_multi_world_lidar.data(),
        h_particles_transition.transition_multi_world_body.data(), h_particles_transition.transition_multi_world_lidar.data(), 
        NUM_PARTICLES, false, true, false);

    ASSERT_processed_measurements(res_processed_measure.x.data(), res_processed_measure.y.data(),
        res_processed_measure.idx.data(), h_processed_measure.x.data(), h_processed_measure.y.data(),
        (NUM_PARTICLES * h_measurements.LIDAR_COORDS_LEN), h_measurements.LIDAR_COORDS_LEN, false, true, true);

    
    auto duration_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_kernel - start_kernel);
    std::cout << "Time taken by function (Kernel): " << duration_kernel.count() << " microseconds" << std::endl;
    std::cout << std::endl;
}

// Step 1.2
void host_update_unique() {

    printf("/************************** UPDATE UNIQUE ***************************/\n");

    int negative_before_counter = getNegativeCounter(h_robot_particles.x.data(), h_robot_particles.y.data(), h_robot_particles.LEN);
    int count_bigger_than_height = getGreaterThanCounter(h_robot_particles.y.data(), h_map.GRID_HEIGHT, h_robot_particles.LEN);
    int negative_after_unique_counter = getNegativeCounter(h_robot_particles_unique.x.data(), h_robot_particles_unique.y.data(), h_robot_particles_unique.LEN);
    int negative_after_counter = getNegativeCounter(h_robot_particles_after_resampling.x.data(), h_robot_particles_after_resampling.y.data(),
        h_robot_particles_after_resampling.LEN);

    printf("~~$ GRID_WIDTH: \t\t%d\n", h_map.GRID_WIDTH);
    printf("~~$ GRID_HEIGHT: \t\t%d\n", h_map.GRID_HEIGHT);
    printf("~~$ MEASURE_LEN: \t\t%d\n", h_measurements.LIDAR_COORDS_LEN);
    printf("~~$ negative_before_counter: \t%d\n", negative_before_counter);
    printf("~~$ negative_after_counter: \t%d\n", negative_after_counter);
    printf("~~$ count_bigger_than_height: \t%d\n", count_bigger_than_height);

    const int MEASURE_LEN = NUM_PARTICLES * h_measurements.LIDAR_COORDS_LEN;
    const int UNIQUE_COUNTER_LEN = NUM_PARTICLES + 1;

    DeviceRobotParticles d_robot_particles;
    Device2DUniqueFinder d_2d_unique;
    DeviceProcessedMeasure d_processed_measure;

    HostRobotParticles res_robot_particles;
    Host2DUniqueFinder res_2d_unique;
    HostProcessedMeasure res_processed_measure;

    /************************** PRIOR VARIABLES *************************/
    d_robot_particles.x.resize(h_robot_particles.LEN, 0);
    d_robot_particles.y.resize(h_robot_particles.LEN, 0);
    d_robot_particles.idx.resize(NUM_PARTICLES, 0);

    d_robot_particles.x.assign(h_robot_particles.x.begin(), h_robot_particles.x.end());
    d_robot_particles.y.assign(h_robot_particles.y.begin(), h_robot_particles.y.end());
    d_robot_particles.idx.assign(h_robot_particles.idx.begin(), h_robot_particles.idx.end());

    /**************************** MAP VARIABLES *************************/

    d_2d_unique.map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT * NUM_PARTICLES, 0);
    d_2d_unique.in_map.resize(UNIQUE_COUNTER_LEN, 0);
    d_2d_unique.in_col.resize(UNIQUE_COUNTER_LEN * h_map.GRID_WIDTH, 0);

    res_2d_unique.map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT * NUM_PARTICLES, 0);
    res_2d_unique.in_map.resize(UNIQUE_COUNTER_LEN, 0);
    res_2d_unique.in_col.resize(UNIQUE_COUNTER_LEN * h_map.GRID_WIDTH, 0);

    /*********************** MEASUREMENT VARIABLES **********************/
    d_processed_measure.x.resize(NUM_PARTICLES * h_measurements.LIDAR_COORDS_LEN, 0);
    d_processed_measure.y.resize(NUM_PARTICLES * h_measurements.LIDAR_COORDS_LEN, 0);
    d_processed_measure.idx.resize(NUM_PARTICLES, 0);

    d_processed_measure.x.assign(h_processed_measure.x.begin(), h_processed_measure.x.end());
    d_processed_measure.y.assign(h_processed_measure.y.begin(), h_processed_measure.y.end());
    d_processed_measure.idx.assign(h_processed_measure.idx.begin(), h_processed_measure.idx.end());

    /**************************** CREATE MAP ****************************/
    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;
    auto start_create_map = std::chrono::high_resolution_clock::now();
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_2d_unique.map), THRUST_RAW_CAST(d_2d_unique.in_map), THRUST_RAW_CAST(d_2d_unique.in_col), SEP,
        THRUST_RAW_CAST(d_robot_particles.x), THRUST_RAW_CAST(d_robot_particles.y), THRUST_RAW_CAST(d_robot_particles.idx), h_robot_particles.LEN,
        h_map.GRID_WIDTH, h_map.GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();
    auto stop_create_map = std::chrono::high_resolution_clock::now();
    res_2d_unique.map.assign(d_2d_unique.map.begin(), d_2d_unique.map.end());

    ASSERT_create_2d_map_elements(res_2d_unique.map.data(), negative_before_counter, 
        h_map.GRID_WIDTH, h_map.GRID_HEIGHT, NUM_PARTICLES, h_robot_particles.LEN, true, true);

    /**************************** UPDATE MAP ****************************/
    auto start_update_map = std::chrono::high_resolution_clock::now();
    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;

    kernel_update_2d_map_with_measure << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_2d_unique.map), THRUST_RAW_CAST(d_2d_unique.in_map), THRUST_RAW_CAST(d_2d_unique.in_col), SEP,
        THRUST_RAW_CAST(d_processed_measure.x), THRUST_RAW_CAST(d_processed_measure.y), THRUST_RAW_CAST(d_processed_measure.idx), 
        MEASURE_LEN, h_map.GRID_WIDTH, h_map.GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();
    auto stop_update_map = std::chrono::high_resolution_clock::now();
    res_2d_unique.map.assign(d_2d_unique.map.begin(), d_2d_unique.map.end());

    /************************* CUMULATIVE SUM ***************************/
    threadsPerBlock = UNIQUE_COUNTER_LEN;
    blocksPerGrid = 1;

    auto start_cumulative_sum = std::chrono::high_resolution_clock::now();
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_2d_unique.in_col), h_map.GRID_WIDTH);
    thrust::exclusive_scan(thrust::device, 
        d_2d_unique.in_map.begin(), d_2d_unique.in_map.end(), d_2d_unique.in_map.begin(), 0);
    cudaDeviceSynchronize();
    auto stop_cumulative_sum = std::chrono::high_resolution_clock::now();

    res_2d_unique.map.assign(d_2d_unique.map.begin(), d_2d_unique.map.end());
    res_2d_unique.in_map.assign(d_2d_unique.in_map.begin(), d_2d_unique.in_map.end());
    res_2d_unique.in_col.assign(d_2d_unique.in_col.begin(), d_2d_unique.in_col.end());

    res_robot_particles.LEN = res_2d_unique.in_map[NUM_PARTICLES];
    printf("\n~~$ PARTICLES_ITEMS_LEN=%d, AF_PARTICLES_ITEMS_LEN=%d\n", res_robot_particles.LEN, h_robot_particles_unique.LEN);
    ASSERT_new_len_calculation(res_robot_particles.LEN, h_robot_particles_unique.LEN, negative_after_unique_counter);


    d_robot_particles.x.clear();
    d_robot_particles.x.resize(res_robot_particles.LEN, 0);
    d_robot_particles.y.clear();
    d_robot_particles.y.resize(res_robot_particles.LEN, 0);
    d_robot_particles.extended_idx.clear();
    d_robot_particles.extended_idx.resize(res_robot_particles.LEN, 0);

    res_robot_particles.x.resize(res_robot_particles.LEN, 0);
    res_robot_particles.y.resize(res_robot_particles.LEN, 0);
    res_robot_particles.idx.resize(NUM_PARTICLES, 0);

    /************************ MAP RESTRUCTURE ***************************/
    threadsPerBlock = h_map.GRID_WIDTH;
    blocksPerGrid = NUM_PARTICLES;

    auto start_map_restructure = std::chrono::high_resolution_clock::now();
    thrust::fill(d_robot_particles.idx.begin(), d_robot_particles.idx.end(), 0);
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_robot_particles.x), THRUST_RAW_CAST(d_robot_particles.y), THRUST_RAW_CAST(d_robot_particles.idx), SEP,
        THRUST_RAW_CAST(d_2d_unique.map), THRUST_RAW_CAST(d_2d_unique.in_map), THRUST_RAW_CAST(d_2d_unique.in_col), 
        h_map.GRID_WIDTH, h_map.GRID_HEIGHT);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, d_robot_particles.idx.begin(), d_robot_particles.idx.end(), d_robot_particles.idx.begin(), 0);
    auto stop_map_restructure = std::chrono::high_resolution_clock::now();

    res_robot_particles.x.assign(d_robot_particles.x.begin(), d_robot_particles.x.end());
    res_robot_particles.y.assign(d_robot_particles.y.begin(), d_robot_particles.y.end());
    res_robot_particles.idx.assign(d_robot_particles.idx.begin(), d_robot_particles.idx.end());

    //printf("~~$ PARTICLES_ITEMS_LEN: \t%d\n", PARTICLES_ITEMS_LEN);
    //printf("~~$ AF_PARTICLES_ITEMS_LEN: \t%d\n", AF_PARTICLES_ITEMS_LEN_UNIQUE);
    //printf("~~$ Measurement Length: \t%d\n", MEASURE_LEN);

    ASSERT_particles_pos_unique(res_robot_particles.x.data(), res_robot_particles.y.data(), 
        h_robot_particles_unique.x.data(), h_robot_particles_unique.y.data(), res_robot_particles.LEN, false, true, true);
    ASSERT_particles_idx_unique(res_robot_particles.idx.data(), h_robot_particles_unique.idx.data(), negative_after_counter, NUM_PARTICLES, false, true);

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

    printf("/**************************** CORRELATION ***************************/\n");

    DeviceMapData d_map;
    DeviceRobotParticles d_robot_particles;
    DeviceCorrelation d_correlation;

    HostCorrelation res_correlation;

    /************************** PRIOR VARIABLES *************************/

    d_map.grid_map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT);
    d_robot_particles.x.resize(h_robot_particles_unique.LEN, 0);
    d_robot_particles.y.resize(h_robot_particles_unique.LEN, 0);
    d_robot_particles.idx.resize(NUM_PARTICLES, 0);
    d_robot_particles.extended_idx.resize(h_robot_particles_unique.LEN, 0);

    auto start_memory_copy = std::chrono::high_resolution_clock::now();

    d_map.grid_map.assign(h_map.grid_map.begin(), h_map.grid_map.end());
    d_robot_particles.x.assign(h_robot_particles_unique.x.begin(), h_robot_particles_unique.x.end());
    d_robot_particles.y.assign(h_robot_particles_unique.y.begin(), h_robot_particles_unique.y.end());
    d_robot_particles.idx.assign(h_robot_particles_unique.idx.begin(), h_robot_particles_unique.idx.end());

    d_correlation.weight.resize(NUM_PARTICLES, 0);
    d_correlation.raw.resize(25 * NUM_PARTICLES, 0);

    res_correlation.weight.resize(NUM_PARTICLES, 0);

    auto stop_memory_copy = std::chrono::high_resolution_clock::now();

    /*************************** PRINT SUMMARY **************************/
    //printf("Elements of particles_x: \t%d  \tSize of particles_x: \t%d\n", (int)PARTICLES_ITEMS_LEN, (int)sz_particles_pos);
    //printf("Elements of particles_y: \t%d  \tSize of particles_y: \t%d\n", (int)PARTICLES_ITEMS_LEN, (int)sz_particles_pos);
    //printf("Elements of particles_idx: \t%d  \tSize of particles_idx: \t%d\n", (int)PARTICLES_ITEMS_LEN, (int)sz_extended_idx);
    //printf("\n");
    //printf("Elements of Grid_Map: \t\t%d  \tSize of Grid_Map: \t%d\n", (int)GRID_MAP_ITEMS_LEN, (int)sz_grid_map);

    /************************* INDEX EXPANSION **************************/
    auto start_index_expansion = std::chrono::high_resolution_clock::now();
    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;
    kernel_index_expansion << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_robot_particles.extended_idx), THRUST_RAW_CAST(d_robot_particles.idx), h_robot_particles_unique.LEN);
    cudaDeviceSynchronize();
    auto stop_index_expansion = std::chrono::high_resolution_clock::now();

    /************************ KERNEL CORRELATION ************************/

    auto start_kernel = std::chrono::high_resolution_clock::now();
    threadsPerBlock = 256;
    blocksPerGrid = (h_robot_particles_unique.LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_correlation << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.raw), SEP,
        THRUST_RAW_CAST(d_map.grid_map), THRUST_RAW_CAST(d_robot_particles.x), THRUST_RAW_CAST(d_robot_particles.y), 
        THRUST_RAW_CAST(d_robot_particles.extended_idx), h_map.GRID_WIDTH, h_map.GRID_HEIGHT, h_robot_particles_unique.LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_correlation_max << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.weight), THRUST_RAW_CAST(d_correlation.raw), NUM_PARTICLES);

    auto stop_kernel = std::chrono::high_resolution_clock::now();

    res_correlation.weight.assign(d_correlation.weight.begin(), d_correlation.weight.end());

    ASSERT_correlation_Equality(res_correlation.weight.data(), weights_pre.data(), NUM_PARTICLES, true, true);

    auto duration_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_kernel - start_kernel);
    auto duration_memory_copy = std::chrono::duration_cast<std::chrono::microseconds>(stop_memory_copy - start_memory_copy);
    auto duration_index_expansion = std::chrono::duration_cast<std::chrono::microseconds>(stop_index_expansion - start_index_expansion);
    std::cout << "Time taken by function (Correlation): " << duration_kernel.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Memory Copy): " << duration_memory_copy.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Index Expansion): " << duration_index_expansion.count() << " microseconds" << std::endl;
    std::cout << std::endl;
}

// Step 1
void host_update_loop() {

    printf("/**************************** UPDATE LOOP ***************************/\n");

    const int MEASURE_LEN = NUM_PARTICLES * h_measurements.LIDAR_COORDS_LEN;
    const int UNIQUE_COUNTER_LEN = NUM_PARTICLES + 1;

    int negative_before_counter = getNegativeCounter(
        h_robot_particles.x.data(), h_robot_particles.y.data(), h_robot_particles.LEN);
    int count_bigger_than_height = getGreaterThanCounter(h_robot_particles.y.data(), h_map.GRID_HEIGHT, h_robot_particles.LEN);
    int negative_after_unique_counter = getNegativeCounter(h_robot_particles_unique.x.data(), h_robot_particles_unique.y.data(), h_robot_particles_unique.LEN);
    int negative_after_resampling_counter = getNegativeCounter(h_robot_particles_after_resampling.x.data(), h_robot_particles_after_resampling.y.data(), h_robot_particles_after_resampling.LEN);

    printf("~~$ GRID_WIDTH: \t\t\t%d\n", h_map.GRID_WIDTH);
    printf("~~$ GRID_HEIGHT: \t\t\t%d\n", h_map.GRID_HEIGHT);
    printf("~~$ LIDAR_COORDS_LEN: \t\t\t%d\n", h_measurements.LIDAR_COORDS_LEN);
    printf("~~$ negative_before_counter: \t\t%d\n", negative_before_counter);
    printf("~~$ negative_after_unique_counter: \t%d\n", negative_after_unique_counter);
    printf("~~$ negative_after_resampling_counter: \t%d\n", negative_after_resampling_counter);
    printf("~~$ count_bigger_than_height: \t\t%d\n", count_bigger_than_height);
    printf("~~$ MEASURE_LEN: \t\t\t%d \n", MEASURE_LEN);

    HostState res_state;
    HostRobotParticles res_robot_particles;
    HostCorrelation res_correlation;
    HostParticlesTransition res_particles_transition;
    HostProcessedMeasure res_processed_measure;
    Host2DUniqueFinder res_2d_unique;

    DeviceMapData d_map;
    DeviceState d_state;
    DeviceMeasurements d_measurements;
    DeviceRobotParticles d_robot_particles;
    DeviceCorrelation d_correlation;
    DeviceParticlesTransition d_particles_transition;
    DevicePositionTransition d_transition;
    DeviceProcessedMeasure d_processed_measure;
    Device2DUniqueFinder d_2d_unique;

    /************************** PRIOR VARIABLES *************************/
    d_state.x.resize(NUM_PARTICLES, 0);
    d_state.y.resize(NUM_PARTICLES, 0);
    d_state.theta.resize(NUM_PARTICLES, 0);
    d_measurements.lidar_coords.resize(2 * h_measurements.LIDAR_COORDS_LEN, 0);

    d_state.x.assign(h_state.x.begin(), h_state.x.end());
    d_state.y.assign(h_state.y.begin(), h_state.y.end());
    d_state.theta.assign(h_state.theta.begin(), h_state.theta.end());
    d_measurements.lidar_coords.assign(h_measurements.lidar_coords.begin(), h_measurements.lidar_coords.end());

    /**************************** MAP VARIABLES *************************/
    d_map.grid_map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);
    d_robot_particles.x.resize(h_robot_particles.LEN, 0);
    d_robot_particles.y.resize(h_robot_particles.LEN, 0);
    d_robot_particles.idx.resize(NUM_PARTICLES, 0);
    d_robot_particles.extended_idx.resize(h_robot_particles.LEN, 0);

    res_robot_particles.x.resize(h_robot_particles.LEN, 0);
    res_robot_particles.y.resize(h_robot_particles.LEN, 0);
    res_robot_particles.idx.resize(NUM_PARTICLES, 0);
    res_robot_particles.extended_idx.resize(h_robot_particles.LEN, 0);

    d_map.grid_map.assign(h_map.grid_map.begin(), h_map.grid_map.end());
    d_robot_particles.x.assign(h_robot_particles.x.begin(), h_robot_particles.x.end());
    d_robot_particles.y.assign(h_robot_particles.y.begin(), h_robot_particles.y.end());
    d_robot_particles.idx.assign(h_robot_particles.idx.begin(), h_robot_particles.idx.end());
    /********************** CORRELATION VARIABLES ***********************/
    res_correlation.weight.resize(NUM_PARTICLES, 0);
    res_correlation.raw.resize(25 * NUM_PARTICLES, 0);

    d_correlation.weight.resize(NUM_PARTICLES, 0);
    d_correlation.raw.resize(25 * NUM_PARTICLES, 0);

    /*********************** TRANSITION VARIABLES ***********************/
    res_particles_transition.transition_multi_world_body.resize(9 * NUM_PARTICLES, 0);
    res_particles_transition.transition_multi_world_lidar.resize(9 * NUM_PARTICLES, 0);
    res_processed_measure.x.resize(NUM_PARTICLES* h_measurements.LIDAR_COORDS_LEN, 0);
    res_processed_measure.y.resize(NUM_PARTICLES* h_measurements.LIDAR_COORDS_LEN, 0);
    res_processed_measure.idx.resize(NUM_PARTICLES, 0);

    d_particles_transition.transition_multi_world_body.resize(9 * NUM_PARTICLES, 0);
    d_particles_transition.transition_multi_world_lidar.resize(9 * NUM_PARTICLES, 0);
    d_transition.transition_body_lidar.resize(9, 0);
    d_processed_measure.x.resize(NUM_PARTICLES* h_measurements.LIDAR_COORDS_LEN, 0);
    d_processed_measure.y.resize(NUM_PARTICLES* h_measurements.LIDAR_COORDS_LEN, 0);
    d_processed_measure.idx.resize(NUM_PARTICLES, 0);

    d_transition.transition_body_lidar.assign(hvec_transition_body_lidar.begin(), hvec_transition_body_lidar.end());

    /**************************** MAP VARIABLES *************************/
    d_2d_unique.map.resize(h_map.GRID_WIDTH* h_map.GRID_HEIGHT* NUM_PARTICLES, 0);
    d_2d_unique.in_map.resize(UNIQUE_COUNTER_LEN, 0);
    d_2d_unique.in_col.resize(UNIQUE_COUNTER_LEN* h_map.GRID_WIDTH, 0);

    res_2d_unique.map.resize(h_map.GRID_WIDTH* h_map.GRID_HEIGHT* NUM_PARTICLES, 0);
    res_2d_unique.in_map.resize(UNIQUE_COUNTER_LEN, 0);

    /************************ TRANSITION KERNEL *************************/
    auto start_transition_kernel = std::chrono::high_resolution_clock::now();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;

    kernel_update_particles_states << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles_transition.transition_multi_world_body), 
        THRUST_RAW_CAST(d_particles_transition.transition_multi_world_lidar), SEP,
        THRUST_RAW_CAST(d_state.x), THRUST_RAW_CAST(d_state.y), THRUST_RAW_CAST(d_state.theta),
        THRUST_RAW_CAST(d_transition.transition_body_lidar), NUM_PARTICLES);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = h_measurements.LIDAR_COORDS_LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_processed_measure.x), THRUST_RAW_CAST(d_processed_measure.y), SEP,
        THRUST_RAW_CAST(d_particles_transition.transition_multi_world_lidar), THRUST_RAW_CAST(d_measurements.lidar_coords), 
        general_info.res, h_map.xmin, h_map.ymax, h_measurements.LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_index_init_const << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_processed_measure.idx), h_measurements.LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, d_processed_measure.idx.begin(), d_processed_measure.idx.end(), 
        d_processed_measure.idx.begin(), 0);

    auto stop_transition_kernel = std::chrono::high_resolution_clock::now();

    res_particles_transition.transition_multi_world_body.assign(d_particles_transition.transition_multi_world_body.begin(),
        d_particles_transition.transition_multi_world_body.end());
    res_particles_transition.transition_multi_world_lidar.assign(d_particles_transition.transition_multi_world_lidar.begin(),
        d_particles_transition.transition_multi_world_lidar.end());
    res_processed_measure.x.assign(d_processed_measure.x.begin(), d_processed_measure.x.end());
    res_processed_measure.y.assign(d_processed_measure.y.begin(), d_processed_measure.y.end());
    res_processed_measure.idx.assign(d_processed_measure.idx.begin(), d_processed_measure.idx.end());

    ASSERT_transition_frames(res_particles_transition.transition_multi_world_body.data(), res_particles_transition.transition_multi_world_lidar.data(), 
        h_particles_transition.transition_multi_world_body.data(), h_particles_transition.transition_multi_world_lidar.data(),
        NUM_PARTICLES, false, true, false);

    ASSERT_processed_measurements(res_processed_measure.x.data(), res_processed_measure.y.data(), res_processed_measure.idx.data(),
        h_processed_measure.x.data(), h_processed_measure.y.data(),
        (NUM_PARTICLES* h_measurements.LIDAR_COORDS_LEN), h_measurements.LIDAR_COORDS_LEN, false, true, false);


    /************************** CREATE 2D MAP ***************************/
    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;

    auto start_create_map = std::chrono::high_resolution_clock::now();
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_2d_unique.map), THRUST_RAW_CAST(d_2d_unique.in_map), THRUST_RAW_CAST(d_2d_unique.in_col), SEP,
        THRUST_RAW_CAST(d_robot_particles.x), THRUST_RAW_CAST(d_robot_particles.y), THRUST_RAW_CAST(d_robot_particles.idx), h_robot_particles.LEN,
        h_map.GRID_WIDTH, h_map.GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_create_map = std::chrono::high_resolution_clock::now();

    //cudaError_t err = cudaPeekAtLastError();
    //printf("%s\n", cudaGetErrorString(err));

    res_2d_unique.map.assign(d_2d_unique.map.begin(), d_2d_unique.map.end());

    ASSERT_create_2d_map_elements(res_2d_unique.map.data(), negative_before_counter, h_map.GRID_WIDTH, h_map.GRID_HEIGHT, 
        NUM_PARTICLES, h_robot_particles.LEN, true, false);
    
    ///**************************** UPDATE MAP ****************************/
    auto start_update_map = std::chrono::high_resolution_clock::now();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_update_2d_map_with_measure << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_2d_unique.map), THRUST_RAW_CAST(d_2d_unique.in_map), THRUST_RAW_CAST(d_2d_unique.in_col), SEP,
        THRUST_RAW_CAST(d_processed_measure.x), THRUST_RAW_CAST(d_processed_measure.y), THRUST_RAW_CAST(d_processed_measure.idx),
        MEASURE_LEN, h_map.GRID_WIDTH, h_map.GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_update_map = std::chrono::high_resolution_clock::now();

    res_processed_measure.idx.assign(d_processed_measure.idx.begin(), d_processed_measure.idx.end());
    
    ///************************* CUMULATIVE SUM ***************************/
    threadsPerBlock = UNIQUE_COUNTER_LEN;
    blocksPerGrid = 1;

    auto start_cumulative_sum = std::chrono::high_resolution_clock::now();

    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_2d_unique.in_col), h_map.GRID_WIDTH);
    thrust::exclusive_scan(thrust::device, d_2d_unique.in_map.begin(), d_2d_unique.in_map.end(), d_2d_unique.in_map.begin(), 0);
    cudaDeviceSynchronize();

    auto stop_cumulative_sum = std::chrono::high_resolution_clock::now();

    res_2d_unique.in_map.assign(d_2d_unique.in_map.begin(), d_2d_unique.in_map.end());

    res_robot_particles.LEN = res_2d_unique.in_map.data()[UNIQUE_COUNTER_LEN - 1];
    printf("\n~~$ PARTICLES_ITEMS_LEN=%d, AF_PARTICLES_ITEMS_LEN=%d\n", res_robot_particles.LEN, h_robot_particles_unique.LEN);
    ASSERT_new_len_calculation(res_robot_particles.LEN, h_robot_particles_unique.LEN, negative_after_resampling_counter);

    ///******************* REINITIALIZE MAP VARIABLES *********************/
    d_robot_particles.x.clear();
    d_robot_particles.x.resize(res_robot_particles.LEN, 0);
    d_robot_particles.y.clear();
    d_robot_particles.y.resize(res_robot_particles.LEN, 0);
    d_robot_particles.extended_idx.clear();
    d_robot_particles.extended_idx.resize(res_robot_particles.LEN, 0);

    res_robot_particles.x.clear();
    res_robot_particles.x.resize(res_robot_particles.LEN, 0);
    res_robot_particles.y.clear();
    res_robot_particles.y.resize(res_robot_particles.LEN, 0);
    res_robot_particles.extended_idx.clear();
    res_robot_particles.extended_idx.resize(res_robot_particles.LEN, 0);

    ///************************ MAP RESTRUCTURE ***************************/
    threadsPerBlock = h_map.GRID_WIDTH;
    blocksPerGrid = NUM_PARTICLES;

    auto start_map_restructure = std::chrono::high_resolution_clock::now();

    thrust::fill(d_robot_particles.idx.begin(), d_robot_particles.idx.end(), 0);

    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_robot_particles.x), THRUST_RAW_CAST(d_robot_particles.y), THRUST_RAW_CAST(d_robot_particles.idx), SEP,
        THRUST_RAW_CAST(d_2d_unique.map), THRUST_RAW_CAST(d_2d_unique.in_map), THRUST_RAW_CAST(d_2d_unique.in_col), 
        h_map.GRID_WIDTH, h_map.GRID_HEIGHT);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, d_robot_particles.idx.begin(), d_robot_particles.idx.end(), d_robot_particles.idx.begin(), 0);

    auto stop_map_restructure = std::chrono::high_resolution_clock::now();

    auto start_copy_particles_pos = std::chrono::high_resolution_clock::now();
    res_robot_particles.x.assign(d_robot_particles.x.begin(), d_robot_particles.x.end());
    res_robot_particles.y.assign(d_robot_particles.y.begin(), d_robot_particles.y.end());
    auto stop_copy_particles_pos = std::chrono::high_resolution_clock::now();

    ASSERT_particles_pos_unique(res_robot_particles.x.data(), res_robot_particles.y.data(), 
        h_robot_particles_unique.x.data(), h_robot_particles_unique.y.data(), res_robot_particles.LEN, false, false, true);

    ///************************* INDEX EXPANSION **************************/
    auto start_index_expansion = std::chrono::high_resolution_clock::now();

    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;
    
    kernel_index_expansion << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_robot_particles.extended_idx), THRUST_RAW_CAST(d_robot_particles.idx), res_robot_particles.LEN);
    cudaDeviceSynchronize();

    auto stop_index_expansion = std::chrono::high_resolution_clock::now();

    res_robot_particles.extended_idx.assign(d_robot_particles.extended_idx.begin(), d_robot_particles.extended_idx.end());
    res_robot_particles.idx.assign(d_robot_particles.idx.begin(), d_robot_particles.idx.end());

    /************************ KERNEL CORRELATION ************************/
    threadsPerBlock = 256;
    blocksPerGrid = (res_robot_particles.LEN + threadsPerBlock - 1) / threadsPerBlock;

    auto start_correlation = std::chrono::high_resolution_clock::now();    
    kernel_correlation << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.raw), SEP,
        THRUST_RAW_CAST(d_map.grid_map), THRUST_RAW_CAST(d_robot_particles.x), THRUST_RAW_CAST(d_robot_particles.y), 
        THRUST_RAW_CAST(d_robot_particles.extended_idx), h_map.GRID_WIDTH, h_map.GRID_HEIGHT, res_robot_particles.LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    
    kernel_correlation_max << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.weight), THRUST_RAW_CAST(d_correlation.raw), NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_correlation = std::chrono::high_resolution_clock::now();

    res_correlation.weight.assign(d_correlation.weight.begin(), d_correlation.weight.end());

    ASSERT_correlation_Equality(res_correlation.weight.data(), weights_pre.data(), NUM_PARTICLES, false, true);

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

    printf("/********************** UPDATE PARTICLE WEIGHTS *********************/\n");

    DeviceCorrelation d_correlation;

    HostCorrelation res_correlation;

    /************************ WEIGHTS VARIABLES *************************/
    d_correlation.weight.resize(NUM_PARTICLES, 0);
    d_correlation.vec_max.resize(1, 0);
    d_correlation.vec_sum_exp.resize(1, 0);

    res_correlation.weight.resize(NUM_PARTICLES, 0);
    res_correlation.vec_max.resize(1, 0);
    res_correlation.vec_sum_exp.resize(1, 0);

    d_correlation.weight.assign(weights_pre.begin(), weights_pre.end());

    /********************** UPDATE WEIGHTS KERNEL ***********************/
    auto start_update_particle_weights = std::chrono::high_resolution_clock::now();

    threadsPerBlock = 1;
    blocksPerGrid = 1;

    kernel_arr_max << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.weight), THRUST_RAW_CAST(d_correlation.vec_max), NUM_PARTICLES);
    cudaDeviceSynchronize();

    res_correlation.vec_max.assign(d_correlation.vec_max.begin(), d_correlation.vec_max.end());
    printf("~~$ res_weights_max[0]=%f\n", res_correlation.vec_max[0]);

    float norm_value = -res_correlation.vec_max[0] + 50;

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_arr_increase << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.weight), norm_value, 0);
    cudaDeviceSynchronize();

    threadsPerBlock = 1;
    blocksPerGrid = 1;
    kernel_arr_sum_exp << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.vec_sum_exp), THRUST_RAW_CAST(d_correlation.weight), NUM_PARTICLES);
    cudaDeviceSynchronize();

    res_correlation.vec_sum_exp.assign(d_correlation.vec_sum_exp.begin(), d_correlation.vec_sum_exp.end());

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_arr_normalize << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.weight), res_correlation.vec_sum_exp[0]);
    cudaDeviceSynchronize();

    auto stop_update_particle_weights = std::chrono::high_resolution_clock::now();

    res_correlation.weight.assign(d_correlation.weight.begin(), d_correlation.weight.end());

    ASSERT_update_particle_weights(res_correlation.weight.data(), weights_new.data(), NUM_PARTICLES, "weights", false, false, true);

    auto duration_update_particle_weights = std::chrono::duration_cast<std::chrono::microseconds>(stop_update_particle_weights - start_update_particle_weights);
    std::cout << "Time taken by function (Update Particle Weights): " << duration_update_particle_weights.count() << " microseconds" << std::endl;
    std::cout << std::endl;
}


// Step 3
void host_resampling() {

    printf("/***************************** RESAMPLING ***************************/\n");

    int negative_before_counter = getNegativeCounter(h_robot_particles.x.data(), h_robot_particles.y.data(), h_robot_particles.LEN);
    int count_bigger_than_height = getGreaterThanCounter(h_robot_particles.y.data(), h_map.GRID_HEIGHT, h_robot_particles.LEN);
    int negative_after_counter = getNegativeCounter(h_robot_particles_after_resampling.x.data(), 
        h_robot_particles_after_resampling.y.data(), h_robot_particles_after_resampling.LEN);

    printf("~~$ negative_before_counter: \t%d\n", negative_before_counter);
    printf("~~$ negative_after_counter: \t%d\n", negative_after_counter);
    printf("~~$ count_bigger_than_height: \t%d\n", count_bigger_than_height);

    HostResampling res_resampling;

    DeviceResampling d_resampling;
    DeviceCorrelation d_correlation;

    /*********************** RESAMPLING VARIABLES ***********************/
    d_correlation.weight.resize(NUM_PARTICLES, 0);
    d_resampling.js.resize(NUM_PARTICLES, 0);
    d_resampling.rnds.resize(NUM_PARTICLES, 0);

    res_resampling.js.resize(NUM_PARTICLES, 0);

    d_correlation.weight.assign(weights_new.begin(), weights_new.end());
    d_resampling.rnds.assign(h_resampling.rnds.begin(), h_resampling.rnds.end());

    /************************ RESAMPLING kerenel ************************/
    auto start_resampling = std::chrono::high_resolution_clock::now();
    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    
    kernel_resampling << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_resampling.js), THRUST_RAW_CAST(d_correlation.weight), THRUST_RAW_CAST(d_resampling.rnds), NUM_PARTICLES);
    cudaDeviceSynchronize();
    auto stop_resampling = std::chrono::high_resolution_clock::now();

    res_resampling.js.assign(d_resampling.js.begin(), d_resampling.js.end());

    ASSERT_resampling_indices(res_resampling.js.data(), h_resampling.js.data(), NUM_PARTICLES, false, true, false);
    ASSERT_resampling_states(h_state.x.data(), h_state.y.data(), h_state.theta.data(),
        h_state_updated.x.data(), h_state_updated.y.data(), h_state_updated.theta.data(), res_resampling.js.data(), NUM_PARTICLES, false, true, true);

    auto duration_resampling = std::chrono::duration_cast<std::chrono::microseconds>(stop_resampling - start_resampling);
    std::cout << "Time taken by function (Kernel Resampling): " << duration_resampling.count() << " microseconds" << std::endl;
    std::cout << std::endl;
}


// Step 4
void host_update_state() {

    printf("/**************************** UPDATE STATE **************************/\n");

    auto start_update_states = std::chrono::high_resolution_clock::now();

    std::vector<float> std_vec_states_x(h_state_updated.x.begin(), h_state_updated.x.end());
    std::vector<float> std_vec_states_y(h_state_updated.y.begin(), h_state_updated.y.end());
    std::vector<float> std_vec_states_theta(h_state_updated.theta.begin(), h_state_updated.theta.end());

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
        printf("%f ", h_robot_state.transition_world_body[i]);
    }

    printf("\n\n");
    printf("~~$ Robot State (Result): %f, %f, %f\n", std::get<0>(key), std::get<1>(key), std::get<2>(key));
    printf("~~$ Robot State (Host)  : %f, %f, %f\n", h_robot_state.state[0], h_robot_state.state[1], h_robot_state.state[2]);

    std::cout << std::endl;
    auto duration_update_states = std::chrono::duration_cast<std::chrono::microseconds>(stop_update_states - start_update_states);
    std::cout << "Time taken by function (Update States): " << duration_update_states.count() << " microseconds" << std::endl;
    std::cout << std::endl;
}


// Step X
void host_update_func() {

    printf("/**************************** UPDATE FUNC ***************************/\n");

    int MEASURE_LEN = NUM_PARTICLES * h_measurements.LIDAR_COORDS_LEN;
    const int UNIQUE_COUNTER_LEN = NUM_PARTICLES + 1;

    int negative_before_counter = getNegativeCounter(h_robot_particles.x.data(), h_robot_particles.y.data(), h_robot_particles.LEN);
    int count_bigger_than_height = getGreaterThanCounter(h_robot_particles.y.data(), h_map.GRID_HEIGHT, h_robot_particles.LEN);
    int negative_after_counter = getNegativeCounter(h_robot_particles_after_resampling.x.data(), h_robot_particles_after_resampling.y.data(), 
        h_robot_particles_after_resampling.LEN);

    printf("~~$ GRID_WIDTH: \t\t%d\n", h_map.GRID_WIDTH);
    printf("~~$ GRID_HEIGHT: \t\t%d\n", h_map.GRID_HEIGHT);
    printf("~~$ negative_before_counter: \t%d\n", negative_before_counter);
    printf("~~$ negative_after_counter: \t%d\n", negative_after_counter);
    printf("~~$ count_bigger_than_height: \t%d\n", count_bigger_than_height);
    printf("~~$ MEASURE_LEN: \t\t%d\n", MEASURE_LEN);

    /**************************************************************** VARIABLES SCOPE *****************************************************************/

    DeviceMapData d_map;
    DeviceState d_state;
    DeviceState d_clone_state;
    DeviceMeasurements d_measurements;
    DeviceRobotParticles d_robot_particles;
    DeviceRobotParticles d_clone_robot_particles;
    DeviceCorrelation d_correlation;
    DeviceParticlesTransition d_particles_transition;
    DevicePositionTransition d_transition;
    DeviceProcessedMeasure d_processed_measure;
    Device2DUniqueFinder d_2d_unique;
    DeviceResampling d_resampling;

    HostState res_state;
    HostRobotParticles res_robot_particles;
    HostRobotParticles res_clone_robot_particles;
    HostCorrelation res_correlation;
    HostParticlesTransition res_particles_transition;
    HostProcessedMeasure res_processed_measure;
    Host2DUniqueFinder res_2d_unique;
    HostResampling res_resampling;

    /************************** STATES VARIABLES ************************/
    res_state.x.resize(NUM_PARTICLES, 0);
    res_state.y.resize(NUM_PARTICLES, 0);
    res_state.theta.resize(NUM_PARTICLES, 0);

    d_state.x.resize(NUM_PARTICLES, 0);
    d_state.y.resize(NUM_PARTICLES, 0);
    d_state.theta.resize(NUM_PARTICLES, 0);
    d_measurements.resize(2 * h_measurements.LIDAR_COORDS_LEN, 0);

    d_state.x.assign(h_state.x.begin(), h_state.x.end());
    d_state.y.assign(h_state.y.begin(), h_state.y.end());
    d_state.theta.assign(h_state.theta.begin(), h_state.theta.end());
    d_measurements.lidar_coords.assign(h_measurements.lidar_coords.begin(), h_measurements.lidar_coords.end());

    /************************* PARTICLES VARIABLES **********************/
    d_map.grid_map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);

    res_robot_particles.x.resize(h_robot_particles.LEN, 0);
    res_robot_particles.y.resize(h_robot_particles.LEN, 0);
    res_robot_particles.idx.resize(NUM_PARTICLES, 0);
    res_robot_particles.extended_idx.resize(h_robot_particles.LEN, 0);
    res_robot_particles.weight.resize(NUM_PARTICLES, 0);
    
    d_robot_particles.x.resize(h_robot_particles.LEN, 0);
    d_robot_particles.y.resize(h_robot_particles.LEN, 0);
    d_robot_particles.idx.resize(NUM_PARTICLES, 0);
    d_robot_particles.extended_idx.resize(h_robot_particles.LEN, 0);
    d_robot_particles.weight.resize(NUM_PARTICLES, 0);
    
    d_correlation.weight.resize(NUM_PARTICLES, 0);
    d_correlation.raw.resize(25 * NUM_PARTICLES, 0);

    d_map.grid_map.assign(h_map.grid_map.begin(), h_map.grid_map.end());
    d_robot_particles.x.assign(h_robot_particles.x.begin(), h_robot_particles.x.end());
    d_robot_particles.y.assign(h_robot_particles.y.begin(), h_robot_particles.y.end());
    d_robot_particles.idx.assign(h_robot_particles.idx.begin(), h_robot_particles.idx.end());
    d_correlation.weight.assign(weights_pre.begin(), weights_pre.end());

    /******************** PARTICLES COPY VARIABLES **********************/

    d_clone_state.x.resize(NUM_PARTICLES, 0);
    d_clone_state.y.resize(NUM_PARTICLES, 0);
    d_clone_state.theta.resize(NUM_PARTICLES, 0);

    /********************** CORRELATION VARIABLES ***********************/
    res_correlation.weight.resize(NUM_PARTICLES, 0);
    res_correlation.raw.resize(25 * NUM_PARTICLES, 0);

    /*********************** TRANSITION VARIABLES ***********************/
    res_particles_transition.transition_multi_world_body.resize(9 * NUM_PARTICLES, 0);
    res_particles_transition.transition_multi_world_lidar.resize(9 * NUM_PARTICLES, 0);

    d_particles_transition.transition_multi_world_body.resize(9 * NUM_PARTICLES, 0);
    d_particles_transition.transition_multi_world_lidar.resize(9 * NUM_PARTICLES, 0);

    res_processed_measure.x.resize(NUM_PARTICLES* h_measurements.LIDAR_COORDS_LEN);
    res_processed_measure.y.resize(NUM_PARTICLES* h_measurements.LIDAR_COORDS_LEN);

    d_processed_measure.x.resize(NUM_PARTICLES* h_measurements.LIDAR_COORDS_LEN, 0);
    d_processed_measure.y.resize(NUM_PARTICLES* h_measurements.LIDAR_COORDS_LEN, 0);
    d_processed_measure.idx.resize(NUM_PARTICLES* h_measurements.LIDAR_COORDS_LEN, 0);

    d_transition.transition_body_lidar.resize(9, 0);
    d_transition.transition_body_lidar.assign(hvec_transition_body_lidar.begin(), hvec_transition_body_lidar.end());

    /**************************** MAP VARIABLES *************************/
    d_2d_unique.map.resize(h_map.GRID_WIDTH* h_map.GRID_HEIGHT* NUM_PARTICLES, 0);
    d_2d_unique.in_map.resize(UNIQUE_COUNTER_LEN, 0);
    d_2d_unique.in_col.resize(UNIQUE_COUNTER_LEN * h_map.GRID_WIDTH, 0);

    res_2d_unique.map.resize(h_map.GRID_WIDTH* h_map.GRID_HEIGHT* NUM_PARTICLES, 0);
    res_2d_unique.in_map.resize(UNIQUE_COUNTER_LEN, 0);

    /************************ WEIGHTS VARIABLES *************************/
    res_correlation.vec_max.resize(1, 0);
    res_correlation.vec_sum_exp.resize(1, 0);

    d_correlation.vec_max.resize(1, 0);
    d_correlation.vec_sum_exp.resize(1, 0);

    /*********************** RESAMPLING VARIABLES ***********************/

    res_resampling.js.resize(NUM_PARTICLES, 0);

    d_resampling.js.resize(NUM_PARTICLES, 0);
    d_resampling.rnds.resize(NUM_PARTICLES, 0);

    d_resampling.rnds.assign(h_resampling.rnds.begin(), h_resampling.rnds.end());

    /************************************************************* KERNEL EXECUTION SCOPE *************************************************************/

    /************************ TRANSITION KERNEL *************************/
    auto start_transition_kernel = std::chrono::high_resolution_clock::now();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;

    kernel_update_particles_states << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles_transition.transition_multi_world_body), 
        THRUST_RAW_CAST(d_particles_transition.transition_multi_world_lidar), SEP,
        THRUST_RAW_CAST(d_state.x), THRUST_RAW_CAST(d_state.y), THRUST_RAW_CAST(d_state.theta),
        THRUST_RAW_CAST(d_transition.transition_body_lidar), NUM_PARTICLES);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = h_measurements.LIDAR_COORDS_LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_processed_measure.x), THRUST_RAW_CAST(d_processed_measure.y), SEP,
        THRUST_RAW_CAST(d_particles_transition.transition_multi_world_lidar), THRUST_RAW_CAST(d_measurements.lidar_coords), 
        general_info.res, h_map.xmin, h_map.ymax, h_measurements.LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_index_init_const << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_processed_measure.idx), h_measurements.LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, d_processed_measure.idx.begin(), 
        d_processed_measure.idx.end(), d_processed_measure.idx.begin(), 0);

    auto stop_transition_kernel = std::chrono::high_resolution_clock::now();

    res_particles_transition.transition_multi_world_body.assign(
        d_particles_transition.transition_multi_world_body.begin(), d_particles_transition.transition_multi_world_body.end());
    res_particles_transition.transition_multi_world_lidar.assign(
        d_particles_transition.transition_multi_world_lidar.begin(), d_particles_transition.transition_multi_world_lidar.end());
    
    res_processed_measure.x.assign(d_processed_measure.x.begin(), d_processed_measure.x.end());
    res_processed_measure.y.assign(d_processed_measure.y.begin(), d_processed_measure.y.end());
    res_processed_measure.idx.assign(d_processed_measure.idx.begin(), d_processed_measure.idx.end());

    //ASSERT_transition_frames(res_transition_world_body, res_transition_world_lidar,
    //    h_transition_world_body, h_transition_world_lidar, NUM_PARTICLES, false);
    // ASSERT_processed_measurements(res_processed_measure_x, res_processed_measure_y, processed_measure, NUM_PARTICLES, LIDAR_COORDS_LEN);

    ASSERT_processed_measurements(res_processed_measure.x.data(), res_processed_measure.y.data(),
        res_processed_measure.idx.data(), h_processed_measure.x.data(), h_processed_measure.y.data(),
        (NUM_PARTICLES* h_measurements.LIDAR_COORDS_LEN), h_measurements.LIDAR_COORDS_LEN, false, true, true);


    /************************** CREATE 2D MAP ***************************/
    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;

    auto start_create_map = std::chrono::high_resolution_clock::now();
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_2d_unique.map), THRUST_RAW_CAST(d_2d_unique.in_map), THRUST_RAW_CAST(d_2d_unique.in_col), SEP,
        THRUST_RAW_CAST(d_robot_particles.x), THRUST_RAW_CAST(d_robot_particles.y), THRUST_RAW_CAST(d_robot_particles.idx),
        h_robot_particles.LEN, h_map.GRID_WIDTH, h_map.GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();
    auto stop_create_map = std::chrono::high_resolution_clock::now();

    //cudaError_t err = cudaPeekAtLastError();
    //printf("%s\n", cudaGetErrorString(err));

    res_2d_unique.map.assign(d_2d_unique.map.begin(), d_2d_unique.map.end());

    ASSERT_create_2d_map_elements(res_2d_unique.map.data(), negative_before_counter, h_map.GRID_WIDTH, h_map.GRID_HEIGHT, NUM_PARTICLES, h_robot_particles.LEN, true, true);

    /**************************** UPDATE MAP ****************************/
    auto start_update_map = std::chrono::high_resolution_clock::now();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_update_2d_map_with_measure << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_2d_unique.map), THRUST_RAW_CAST(d_2d_unique.in_map), THRUST_RAW_CAST(d_2d_unique.in_col), SEP,
        THRUST_RAW_CAST(d_processed_measure.x), THRUST_RAW_CAST(d_processed_measure.y), THRUST_RAW_CAST(d_processed_measure.idx),
        MEASURE_LEN, h_map.GRID_WIDTH, h_map.GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_update_map = std::chrono::high_resolution_clock::now();

    /************************* CUMULATIVE SUM ***************************/
    auto start_cumulative_sum = std::chrono::high_resolution_clock::now();

    threadsPerBlock = UNIQUE_COUNTER_LEN;
    blocksPerGrid = 1;
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_2d_unique.in_col), h_map.GRID_WIDTH);
    thrust::exclusive_scan(thrust::device, d_2d_unique.in_map.begin(), d_2d_unique.in_map.end(), d_2d_unique.in_map.begin(), 0);
    cudaDeviceSynchronize();

    auto stop_cumulative_sum = std::chrono::high_resolution_clock::now();

    res_2d_unique.in_map.assign(d_2d_unique.in_map.begin(), d_2d_unique.in_map.end());

    res_robot_particles.LEN = res_2d_unique.in_map[UNIQUE_COUNTER_LEN - 1];
    ASSERT_new_len_calculation(res_robot_particles.LEN, h_robot_particles_unique.LEN, negative_after_counter);


    ///*---------------------------------------------------------------------*/
    ///*-------------------- REINITIALIZE MAP VARIABLES ---------------------*/

    d_robot_particles.x.clear();
    d_robot_particles.x.resize(res_robot_particles.LEN, 0);
    d_robot_particles.y.clear();
    d_robot_particles.y.resize(res_robot_particles.LEN, 0);
    d_robot_particles.extended_idx.clear();
    d_robot_particles.extended_idx.resize(res_robot_particles.LEN, 0);

    res_robot_particles.x.clear();
    res_robot_particles.x.resize(res_robot_particles.LEN, 0);
    res_robot_particles.y.clear();
    res_robot_particles.y.resize(res_robot_particles.LEN, 0);

    /************************ MAP RESTRUCTURE ***************************/
    threadsPerBlock = h_map.GRID_WIDTH;
    blocksPerGrid = NUM_PARTICLES;

    auto start_map_restructure = std::chrono::high_resolution_clock::now();

    thrust::fill(d_robot_particles.idx.begin(), d_robot_particles.idx.end(), 0);

    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_robot_particles.x), THRUST_RAW_CAST(d_robot_particles.y), THRUST_RAW_CAST(d_robot_particles.idx), SEP,
        THRUST_RAW_CAST(d_2d_unique.map), THRUST_RAW_CAST(d_2d_unique.in_map), 
        THRUST_RAW_CAST(d_2d_unique.in_col), h_map.GRID_WIDTH, h_map.GRID_HEIGHT);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, d_robot_particles.idx.begin(), d_robot_particles.idx.end(), d_robot_particles.idx.begin(), 0);

    auto stop_map_restructure = std::chrono::high_resolution_clock::now();

    auto start_copy_particles_pos = std::chrono::high_resolution_clock::now();
    res_robot_particles.x.assign(d_robot_particles.x.begin(), d_robot_particles.x.end());
    res_robot_particles.y.assign(d_robot_particles.y.begin(), d_robot_particles.y.end());
    res_robot_particles.idx.assign(d_robot_particles.idx.begin(), d_robot_particles.idx.end());
    auto stop_copy_particles_pos = std::chrono::high_resolution_clock::now();

    ASSERT_particles_pos_unique(res_robot_particles.x.data(), res_robot_particles.y.data(),
        h_robot_particles_unique.x.data(), h_robot_particles_unique.y.data(), res_robot_particles.LEN, false, true, true);
    ASSERT_particles_idx_unique(res_robot_particles.idx.data(), h_robot_particles_unique.idx.data(), negative_after_counter, NUM_PARTICLES, false, true);

    ///************************* INDEX EXPANSION **************************/
    auto start_index_expansion = std::chrono::high_resolution_clock::now();

    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;
    kernel_index_expansion << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_robot_particles.extended_idx), THRUST_RAW_CAST(d_robot_particles.idx), res_robot_particles.LEN);
    cudaDeviceSynchronize();

    auto stop_index_expansion = std::chrono::high_resolution_clock::now();

    res_robot_particles.extended_idx.clear();
    res_robot_particles.extended_idx.resize(res_robot_particles.LEN, 0);

    res_robot_particles.extended_idx.assign(
        d_robot_particles.extended_idx.begin(), d_robot_particles.extended_idx.end());
    res_robot_particles.idx.assign(d_robot_particles.idx.begin(), d_robot_particles.idx.end());

    /************************ KERNEL CORRELATION ************************/
    threadsPerBlock = 256;
    blocksPerGrid = (res_robot_particles.LEN + threadsPerBlock - 1) / threadsPerBlock;

    auto start_correlation = std::chrono::high_resolution_clock::now();
    kernel_correlation << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.raw), SEP,
        THRUST_RAW_CAST(d_map.grid_map), THRUST_RAW_CAST(d_robot_particles.x), THRUST_RAW_CAST(d_robot_particles.y), 
        THRUST_RAW_CAST(d_robot_particles.extended_idx), h_map.GRID_WIDTH, h_map.GRID_HEIGHT, res_robot_particles.LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_correlation_max << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.weight), THRUST_RAW_CAST(d_correlation.raw), NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_correlation = std::chrono::high_resolution_clock::now();

    res_correlation.weight.assign(d_correlation.weight.begin(), d_correlation.weight.end());
    res_robot_particles.extended_idx.assign(d_robot_particles.extended_idx.begin(), d_robot_particles.extended_idx.end());

    ASSERT_correlation_Equality(res_correlation.weight.data(), weights_pre.data(), NUM_PARTICLES, false, true);

    /********************** UPDATE WEIGHTS KERNEL ***********************/
    auto start_update_particle_weights = std::chrono::high_resolution_clock::now();

    threadsPerBlock = 1;
    blocksPerGrid = 1;

    kernel_arr_max << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.weight), THRUST_RAW_CAST(d_correlation.vec_max), NUM_PARTICLES);
    cudaDeviceSynchronize();

    res_correlation.vec_max.assign(d_correlation.vec_max.begin(), d_correlation.vec_max.end());

    float norm_value = -res_correlation.vec_max[0] + 50;

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_arr_increase << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.weight), norm_value, 0);
    cudaDeviceSynchronize();

    threadsPerBlock = 1;
    blocksPerGrid = 1;
    kernel_arr_sum_exp << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.vec_sum_exp), THRUST_RAW_CAST(d_correlation.weight), NUM_PARTICLES);
    cudaDeviceSynchronize();

    res_correlation.vec_sum_exp.assign(d_correlation.vec_sum_exp.begin(), d_correlation.vec_sum_exp.end());

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_arr_normalize << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.weight), res_correlation.vec_sum_exp[0]);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_arr_mult << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_robot_particles.weight), THRUST_RAW_CAST(d_correlation.weight));
    cudaDeviceSynchronize();

    auto stop_update_particle_weights = std::chrono::high_resolution_clock::now();

    res_correlation.weight.assign(d_correlation.weight.begin(), d_correlation.weight.end());
    res_robot_particles.weight.assign(d_robot_particles.weight.begin(), d_robot_particles.weight.end());

    ASSERT_update_particle_weights(res_correlation.weight.data(), weights_new.data(), NUM_PARTICLES, "weights", false, false, true);
    //ASSERT_update_particle_weights(res_robot_particles.weight.data(), h_robot_particles_unique.weight.data(), NUM_PARTICLES, "particles weight", true, false, true);


    ///************************ RESAMPLING KERNEL *************************/
    auto start_resampling = std::chrono::high_resolution_clock::now();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    
    kernel_resampling << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_resampling.js), THRUST_RAW_CAST(d_correlation.weight), 
        THRUST_RAW_CAST(d_resampling.rnds), NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_resampling = std::chrono::high_resolution_clock::now();

    res_resampling.js.assign(d_resampling.js.begin(), d_resampling.js.end());

    ASSERT_resampling_indices(res_resampling.js.data(), h_resampling.js.data(), NUM_PARTICLES, false, false, true);
    ASSERT_resampling_states(h_state.x.data(), h_state.y.data(), h_state.theta.data(),
        h_state_updated.x.data(), h_state_updated.y.data(), h_state_updated.theta.data(), res_resampling.js.data(), NUM_PARTICLES, false, true, true);


    /*---------------------------------------------------------------------*/
    /*----------------- REINITIALIZE PARTICLES VARIABLES ------------------*/
    d_clone_robot_particles.x.resize(res_robot_particles.LEN, 0);
    d_clone_robot_particles.y.resize(res_robot_particles.LEN, 0);
    d_clone_robot_particles.idx.resize(res_robot_particles.LEN, 0);


    int* d_last_len;
    int* res_last_len = (int*)malloc(sizeof(int));
    gpuErrchk(cudaMalloc((void**)&d_last_len, sizeof(int)));

    auto start_clone_particles = std::chrono::high_resolution_clock::now();

    d_clone_robot_particles.x.assign(d_robot_particles.x.begin(), d_robot_particles.x.end());
    d_clone_robot_particles.y.assign(d_robot_particles.y.begin(), d_robot_particles.y.end());
    d_clone_robot_particles.idx.assign(d_robot_particles.idx.begin(), d_robot_particles.idx.end());

    d_clone_state.x.assign(d_state.x.begin(), d_state.x.end());
    d_clone_state.y.assign(d_state.y.begin(), d_state.y.end());
    d_clone_state.theta.assign(d_state.theta.begin(), d_state.theta.end());

    auto stop_clone_particles = std::chrono::high_resolution_clock::now();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_rearrange_indecies << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_robot_particles.idx), d_last_len, SEP,
        THRUST_RAW_CAST(d_clone_robot_particles.idx), THRUST_RAW_CAST(d_resampling.js), res_robot_particles.LEN);
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(res_last_len, d_last_len, sizeof(int), cudaMemcpyDeviceToHost));
    res_robot_particles.idx.assign(d_robot_particles.idx.begin(), d_robot_particles.idx.end());

    /********************** REARRANGEMENT KERNEL ************************/
    auto start_rearrange_index = std::chrono::high_resolution_clock::now();
    thrust::exclusive_scan(thrust::device, d_robot_particles.idx.begin(), d_robot_particles.idx.end(), d_robot_particles.idx.begin(), 0);
    auto stop_rearrange_index = std::chrono::high_resolution_clock::now();

    res_robot_particles.idx.assign(d_robot_particles.idx.begin(), d_robot_particles.idx.end());
    
    res_clone_robot_particles.LEN = res_robot_particles.LEN;
    res_robot_particles.LEN = res_robot_particles.idx[NUM_PARTICLES - 1] + res_last_len[0];

    printf("--> PARTICLES_ITEMS_LEN=%d <> ELEMS_PARTICLES_AFTER=%d\n", res_robot_particles.LEN, 
        h_robot_particles_after_resampling.LEN);
    assert(res_robot_particles.LEN + negative_after_counter == h_robot_particles_after_resampling.LEN);

    res_robot_particles.x.clear();
    res_robot_particles.y.clear();
    res_robot_particles.x.resize(res_robot_particles.LEN, 0);
    res_robot_particles.y.resize(res_robot_particles.LEN, 0);

    ASSERT_resampling_particles_index(h_robot_particles_after_resampling.idx.data(), res_robot_particles.idx.data(), NUM_PARTICLES, false, negative_after_counter);

    auto start_rearrange_particles_states = std::chrono::high_resolution_clock::now();
    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;
    
    kernel_rearrange_particles << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_robot_particles.x), THRUST_RAW_CAST(d_robot_particles.y), SEP,
        THRUST_RAW_CAST(d_robot_particles.idx), THRUST_RAW_CAST(d_clone_robot_particles.x), THRUST_RAW_CAST(d_clone_robot_particles.y), 
        THRUST_RAW_CAST(d_clone_robot_particles.idx), THRUST_RAW_CAST(d_resampling.js),
        h_map.GRID_WIDTH, h_map.GRID_HEIGHT, NUM_PARTICLES, res_robot_particles.LEN, res_clone_robot_particles.LEN);
    kernel_rearrange_states << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_state.x), THRUST_RAW_CAST(d_state.y), THRUST_RAW_CAST(d_state.theta), SEP,
        THRUST_RAW_CAST(d_clone_state.x), THRUST_RAW_CAST(d_clone_state.y), THRUST_RAW_CAST(d_clone_state.theta), 
        THRUST_RAW_CAST(d_resampling.js));
    cudaDeviceSynchronize();
    auto stop_rearrange_particles_states = std::chrono::high_resolution_clock::now();

    res_robot_particles.x.assign(d_robot_particles.x.begin(), d_robot_particles.x.end());
    res_robot_particles.y.assign(d_robot_particles.y.begin(), d_robot_particles.y.end());

    res_state.x.assign(d_state.x.begin(), d_state.x.end());
    res_state.y.assign(d_state.y.begin(), d_state.y.end());
    res_state.theta.assign(d_state.theta.begin(), d_state.theta.end());

    ASSERT_rearrange_particles_states(res_robot_particles.x.data(), res_robot_particles.y.data(), 
        res_state.x.data(), res_state.y.data(), res_state.theta.data(),
        h_robot_particles_after_resampling.x.data(), h_robot_particles_after_resampling.y.data(), 
        h_state_updated.x.data(), h_state_updated.y.data(), h_state_updated.theta.data(),
        res_robot_particles.LEN, NUM_PARTICLES);


    ///********************** UPDATE STATES KERNEL ************************/
    auto start_update_states = std::chrono::high_resolution_clock::now();

    //std::vector<float> std_vec_states_x(h_state_updated.x.begin(), h_state_updated.x.end());
    //std::vector<float> std_vec_states_y(h_state_updated.y.begin(), h_state_updated.y.end());
    //std::vector<float> std_vec_states_theta(h_state_updated.theta.begin(), h_state_updated.theta.end());

    std::vector<float> std_vec_states_x(res_state.x.begin(), res_state.x.end());
    std::vector<float> std_vec_states_y(res_state.y.begin(), res_state.y.end());
    std::vector<float> std_vec_states_theta(res_state.theta.begin(), res_state.theta.end());

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
        printf("%f ", h_robot_state.transition_world_body[i]);
    }

    printf("\n\n");
    printf("~~$ Robot State (Result): %f, %f, %f\n", std::get<0>(key), std::get<1>(key), std::get<2>(key));
    printf("~~$ Robot State (Host)  : %f, %f, %f\n", h_robot_state.state[0], h_robot_state.state[1], h_robot_state.state[2]);


    /************************* EXECUTION TIMES **************************/
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




//void alloc_init_state_vars(float* h_states_x, float* h_states_y, float* h_states_theta) {
//
//    sz_states_pos = NUM_PARTICLES * sizeof(float);
//
//    gpuErrchk(cudaMalloc((void**)&d_states_x, sz_states_pos));
//    gpuErrchk(cudaMalloc((void**)&d_states_y, sz_states_pos));
//    gpuErrchk(cudaMalloc((void**)&d_states_theta, sz_states_pos));
//
//    gpuErrchk(cudaMemcpy(d_states_x, h_states_x, sz_states_pos, cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemcpy(d_states_y, h_states_y, sz_states_pos, cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemcpy(d_states_theta, h_states_theta, sz_states_pos, cudaMemcpyHostToDevice));
//
//    res_robot_state = (float*)malloc(3 * sizeof(float));
//}

void alloc_init_state_vars(DeviceState& d_state, HostState& res_state, HostRobotState& res_robot_state, HostState& h_state) {

    d_state.x.resize(NUM_PARTICLES, 0);
    d_state.y.resize(NUM_PARTICLES, 0);
    d_state.theta.resize(NUM_PARTICLES, 0);

    res_state.x.resize(NUM_PARTICLES, 0);
    res_state.y.resize(NUM_PARTICLES, 0);
    res_state.theta.resize(NUM_PARTICLES, 0);

    d_state.x.assign(h_state.x.begin(), h_state.x.end());
    d_state.y.assign(h_state.y.begin(), h_state.y.end());
    d_state.theta.assign(h_state.theta.begin(), h_state.theta.end());

    res_robot_state.state.resize(3, 0);
}

//void alloc_init_lidar_coords_var(float* h_lidar_coords, const int LIDAR_COORDS_LEN) {
//
//    sz_lidar_coords = 2 * LIDAR_COORDS_LEN * sizeof(float);
//    gpuErrchk(cudaMalloc((void**)&d_lidar_coords, sz_lidar_coords));
//
//    gpuErrchk(cudaMemcpy(d_lidar_coords, h_lidar_coords, sz_lidar_coords, cudaMemcpyHostToDevice));
//}

void alloc_init_lidar_coords_var(DeviceMeasurements& d_measurements, HostMeasurements& res_measurements, 
    HostMeasurements& h_measurements) {

    d_measurements.lidar_coords.resize(2 * h_measurements.LIDAR_COORDS_LEN, 0);
    d_measurements.lidar_coords.assign(h_measurements.lidar_coords.begin(), h_measurements.lidar_coords.end());

    res_measurements.LIDAR_COORDS_LEN = h_measurements.LIDAR_COORDS_LEN;
}

//void alloc_init_grid_map(int* h_grid_map, const int GRID_WIDTH, const int GRID_HEIGHT) {
//
//    sz_grid_map = GRID_WIDTH * GRID_HEIGHT * sizeof(int);
//    gpuErrchk(cudaMalloc((void**)&d_grid_map, sz_grid_map));
//
//    gpuErrchk(cudaMemcpy(d_grid_map, h_grid_map, sz_grid_map, cudaMemcpyHostToDevice));
//}

void alloc_init_grid_map(DeviceMapData& d_map, HostMapData& res_map, HostMapData& h_map) {

    res_map.GRID_WIDTH = h_map.GRID_WIDTH;
    res_map.GRID_HEIGHT = h_map.GRID_HEIGHT;
    res_map.xmin = h_map.xmin;
    res_map.xmax = h_map.xmax;
    res_map.ymin = h_map.ymin;
    res_map.ymax = h_map.ymax;

    d_map.grid_map.resize(res_map.GRID_WIDTH * res_map.GRID_HEIGHT, 0);
    d_map.grid_map.assign(h_map.grid_map.begin(), h_map.grid_map.end());
}

//void alloc_init_particles_vars(int* h_particles_x, int* h_particles_y, int* h_particles_idx, 
//    float* h_particles_weight, const int PARTICLES_ITEMS_LEN) {
//
//    sz_particles_pos = PARTICLES_ITEMS_LEN * sizeof(int);
//    sz_particles_idx = NUM_PARTICLES * sizeof(int);
//    sz_particles_weight = NUM_PARTICLES * sizeof(float);
//
//    gpuErrchk(cudaMalloc((void**)&d_particles_x, sz_particles_pos));
//    gpuErrchk(cudaMalloc((void**)&d_particles_y, sz_particles_pos));
//    gpuErrchk(cudaMalloc((void**)&d_particles_idx, sz_particles_idx));
//    gpuErrchk(cudaMalloc((void**)&d_particles_weight, sz_particles_weight));
//
//    gpuErrchk(cudaMemcpy(d_particles_x, h_particles_x, sz_particles_pos, cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemcpy(d_particles_y, h_particles_y, sz_particles_pos, cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemcpy(d_particles_idx, h_particles_idx, sz_particles_idx, cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemcpy(d_particles_weight, h_particles_weight, sz_particles_weight, cudaMemcpyHostToDevice));
//
//    res_particles_idx = (int*)malloc(sz_particles_idx);
//}

void alloc_init_particles_vars(DeviceRobotParticles& d_robot_particles, HostRobotParticles& res_robot_particles,
    HostRobotParticles& h_robot_particles) {

    res_robot_particles.LEN = h_robot_particles.LEN;
    res_robot_particles.x.resize(res_robot_particles.LEN, 0);
    res_robot_particles.y.resize(res_robot_particles.LEN, 0);
    res_robot_particles.idx.resize(NUM_PARTICLES, 0);
    res_robot_particles.extended_idx.resize(res_robot_particles.LEN, 0);
    res_robot_particles.weight.resize(NUM_PARTICLES, 0);


    d_robot_particles.x.resize(res_robot_particles.LEN, 0);
    d_robot_particles.y.resize(res_robot_particles.LEN, 0);
    d_robot_particles.idx.resize(NUM_PARTICLES, 0);
    d_robot_particles.weight.resize(NUM_PARTICLES, 0);

    d_robot_particles.x.assign(h_robot_particles.x.begin(), h_robot_particles.x.end());
    d_robot_particles.y.assign(h_robot_particles.y.begin(), h_robot_particles.y.end());
    d_robot_particles.idx.assign(h_robot_particles.idx.begin(), h_robot_particles.idx.end());
    d_robot_particles.weight.assign(h_robot_particles.weight.begin(), h_robot_particles.weight.end());
}

//void alloc_extended_idx() {
//
//    sz_extended_idx = PARTICLES_ITEMS_LEN * sizeof(int);
//    gpuErrchk(cudaMalloc((void**)&d_extended_idx, sz_extended_idx));
//
//    res_extended_idx = (int*)malloc(sz_extended_idx);
//}

void alloc_extended_idx(DeviceRobotParticles& d_robot_particles, HostRobotParticles& res_robot_particles) {

    d_robot_particles.extended_idx.resize(res_robot_particles.LEN, 0);
    res_robot_particles.extended_idx.resize(res_robot_particles.LEN, 0);
}

//void alloc_states_copy_vars() {
//
//    gpuErrchk(cudaMalloc((void**)&dc_states_x, sz_states_pos));
//    gpuErrchk(cudaMalloc((void**)&dc_states_y, sz_states_pos));
//    gpuErrchk(cudaMalloc((void**)&dc_states_theta, sz_states_pos));
//}

void alloc_states_copy_vars(DeviceState& d_clone_state) {

    d_clone_state.x.resize(NUM_PARTICLES, 0);
    d_clone_state.y.resize(NUM_PARTICLES, 0);
    d_clone_state.theta.resize(NUM_PARTICLES, 0);
}

//void alloc_correlation_vars() {
//
//    sz_correlation_weights = NUM_PARTICLES * sizeof(float);
//    sz_correlation_weights_raw = 25 * sz_correlation_weights;
//
//    gpuErrchk(cudaMalloc((void**)&d_correlation_weights, sz_correlation_weights));
//    gpuErrchk(cudaMalloc((void**)&d_correlation_weights_raw, sz_correlation_weights_raw));
//    gpuErrchk(cudaMemset(d_correlation_weights_raw, 0, sz_correlation_weights_raw));
//
//    res_correlation_weights = (float*)malloc(sz_correlation_weights);
//    memset(res_correlation_weights, 0, sz_correlation_weights);
//
//    //res_extended_idx = (int*)malloc(sz_extended_idx);
//}

void alloc_correlation_vars(DeviceCorrelation& d_correlation, HostCorrelation& res_correlation) {

    d_correlation.weight.resize(NUM_PARTICLES, 0);
    d_correlation.raw.resize(25 * NUM_PARTICLES, 0);

    res_correlation.weight.resize(NUM_PARTICLES, 0);
    res_correlation.raw.resize(25 * NUM_PARTICLES, 0);
}

//void alloc_init_transition_vars(float* h_transition_body_lidar) {
//
//    sz_transition_multi_world_frame = 9 * NUM_PARTICLES * sizeof(float);
//    sz_transition_body_lidar = 9 * sizeof(float);
//
//    gpuErrchk(cudaMalloc((void**)&d_transition_multi_world_body, sz_transition_multi_world_frame));
//    gpuErrchk(cudaMalloc((void**)&d_transition_multi_world_lidar, sz_transition_multi_world_frame));
//    gpuErrchk(cudaMalloc((void**)&d_transition_body_lidar, sz_transition_body_lidar));
//
//    gpuErrchk(cudaMemcpy(d_transition_body_lidar, h_transition_body_lidar, sz_transition_body_lidar, cudaMemcpyHostToDevice));
//
//    gpuErrchk(cudaMemset(d_transition_multi_world_body, 0, sz_transition_multi_world_frame));
//    gpuErrchk(cudaMemset(d_transition_multi_world_lidar, 0, sz_transition_multi_world_frame));
//
//    res_transition_world_body = (float*)malloc(sz_transition_multi_world_frame);
//    //res_transition_world_lidar = (float*)malloc(sz_transition_world_frame);
//    res_robot_world_body = (float*)malloc(sz_transition_multi_world_frame);
//}

void alloc_init_transition_vars(DevicePositionTransition& d_transition, DeviceParticlesTransition& d_particles_transition,
    HostParticlesTransition& res_particles_transition) {

    d_particles_transition.transition_multi_world_body.resize(9 * NUM_PARTICLES, 0);
    d_particles_transition.transition_multi_world_lidar.resize(9 * NUM_PARTICLES, 0);

    res_particles_transition.transition_multi_world_body.resize(9 * NUM_PARTICLES, 0);
    res_particles_transition.transition_multi_world_lidar.resize(9 * NUM_PARTICLES, 0);

    d_transition.transition_body_lidar.resize(9, 0);
    d_transition.transition_body_lidar.assign(hvec_transition_body_lidar.begin(), hvec_transition_body_lidar.end());
}

//void alloc_init_processed_measurement_vars(const int LIDAR_COORDS_LEN) {
//
//    sz_processed_measure_pos = NUM_PARTICLES * LIDAR_COORDS_LEN * sizeof(int);
//    sz_processed_measure_idx = NUM_PARTICLES * LIDAR_COORDS_LEN * sizeof(int);
//
//    gpuErrchk(cudaMalloc((void**)&d_processed_measure_x, sz_processed_measure_pos));
//    gpuErrchk(cudaMalloc((void**)&d_processed_measure_y, sz_processed_measure_pos));
//    gpuErrchk(cudaMalloc((void**)&d_processed_measure_idx, sz_processed_measure_idx));
//
//    gpuErrchk(cudaMemset(d_processed_measure_x, 0, sz_processed_measure_pos));
//    gpuErrchk(cudaMemset(d_processed_measure_y, 0, sz_processed_measure_pos));
//    gpuErrchk(cudaMemset(d_processed_measure_idx, 0, sz_processed_measure_idx));
//}

void alloc_init_processed_measurement_vars(DeviceProcessedMeasure& d_processed_measure, HostProcessedMeasure& res_processed_measure,
    HostMeasurements& res_measurements) {

    d_processed_measure.x.resize(NUM_PARTICLES * res_measurements.LIDAR_COORDS_LEN, 0);
    d_processed_measure.y.resize(NUM_PARTICLES * res_measurements.LIDAR_COORDS_LEN, 0);
    d_processed_measure.idx.resize(NUM_PARTICLES * res_measurements.LIDAR_COORDS_LEN, 0);

    res_processed_measure.x.resize(NUM_PARTICLES * h_measurements.LIDAR_COORDS_LEN);
    res_processed_measure.y.resize(NUM_PARTICLES * h_measurements.LIDAR_COORDS_LEN);
}

//void alloc_map_2d_var(const int GRID_WIDTH, const int GRID_HEIGHT) {
//
//    sz_map_2d = GRID_WIDTH * GRID_HEIGHT * NUM_PARTICLES * sizeof(uint8_t);
//
//    gpuErrchk(cudaMalloc((void**)&d_map_2d, sz_map_2d));
//    gpuErrchk(cudaMemset(d_map_2d, 0, sz_map_2d));
//}

void alloc_map_2d_var(Device2DUniqueFinder& d_2d_unique, Host2DUniqueFinder& res_2d_unique, HostMapData& res_map) {

    d_2d_unique.map.resize(res_map.GRID_WIDTH * res_map.GRID_HEIGHT * NUM_PARTICLES, 0);
    res_2d_unique.map.resize(res_map.GRID_WIDTH * res_map.GRID_HEIGHT * NUM_PARTICLES, 0);
}

//void alloc_map_2d_unique_counter_vars(const int UNIQUE_COUNTER_LEN, const int GRID_WIDTH) {
//
//    sz_unique_in_particle = UNIQUE_COUNTER_LEN * sizeof(int);
//    sz_unique_in_particle_col = UNIQUE_COUNTER_LEN * GRID_WIDTH * sizeof(int);
//
//    gpuErrchk(cudaMalloc((void**)&d_unique_in_particle, sz_unique_in_particle));
//    gpuErrchk(cudaMalloc((void**)&d_unique_in_particle_col, sz_unique_in_particle_col));
//
//    gpuErrchk(cudaMemset(d_unique_in_particle, 0, sz_unique_in_particle));
//    gpuErrchk(cudaMemset(d_unique_in_particle_col, 0, sz_unique_in_particle_col));
//
//    res_unique_in_particle = (int*)malloc(sz_unique_in_particle);
//}

void alloc_map_2d_unique_counter_vars(Device2DUniqueFinder& d_2d_unique, Host2DUniqueFinder& res_2d_unique, HostMapData& res_map) {

    const int UNIQUE_COUNTER_LEN = NUM_PARTICLES + 1;

    d_2d_unique.in_map.resize(UNIQUE_COUNTER_LEN, 0);
    d_2d_unique.in_col.resize(UNIQUE_COUNTER_LEN * res_map.GRID_WIDTH, 0);

    res_2d_unique.in_map.resize(UNIQUE_COUNTER_LEN, 0);
}

//void alloc_correlation_weights_vars() {
//
//    sz_correlation_sum_exp = sizeof(double);
//    sz_correlation_weights_max = sizeof(float);
//
//    gpuErrchk(cudaMalloc((void**)&d_correlation_sum_exp, sz_correlation_sum_exp));
//    gpuErrchk(cudaMalloc((void**)&d_correlation_weights_max, sz_correlation_weights_max));
//
//    gpuErrchk(cudaMemset(d_correlation_sum_exp, 0, sz_correlation_sum_exp));
//    gpuErrchk(cudaMemset(d_correlation_weights_max, 0, sz_correlation_weights_max));
//
//    res_correlation_sum_exp = (double*)malloc(sz_correlation_sum_exp);
//    res_correlation_weights_max = (float*)malloc(sz_correlation_weights_max);
//}

void alloc_correlation_weights_vars(DeviceCorrelation& d_correlation, HostCorrelation& res_correlation) {

    d_correlation.vec_sum_exp.resize(1, 0);
    d_correlation.vec_max.resize(1, 0);

    res_correlation.vec_sum_exp.resize(1, 0);
    res_correlation.vec_max.resize(1, 0);
}

//void alloc_resampling_vars(float* h_resampling_rnds) {
//
//    sz_resampling_js = NUM_PARTICLES * sizeof(int);
//    sz_resampling_rnd = NUM_PARTICLES * sizeof(float);
//
//    gpuErrchk(cudaMalloc((void**)&d_resampling_js, sz_resampling_js));
//    gpuErrchk(cudaMalloc((void**)&d_resampling_rnd, sz_resampling_rnd));
//
//    gpuErrchk(cudaMemcpy(d_resampling_rnd, h_resampling_rnds, sz_resampling_rnd, cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemset(d_resampling_js, 0, sz_resampling_js));
//}

void alloc_resampling_vars(DeviceResampling& d_resampling, HostResampling& res_resampling, HostResampling& h_resampling) {

    res_resampling.js.resize(NUM_PARTICLES, 0);

    d_resampling.js.resize(NUM_PARTICLES, 0);
    d_resampling.rnds.resize(NUM_PARTICLES, 0);

    d_resampling.rnds.assign(h_resampling.rnds.begin(), h_resampling.rnds.end());
}

//void exec_calc_transition() {
//
//    threadsPerBlock = NUM_PARTICLES;
//    blocksPerGrid = 1;
//    kernel_update_particles_states << <blocksPerGrid, threadsPerBlock >> > (d_transition_multi_world_body, d_transition_multi_world_lidar, SEP,
//        d_states_x, d_states_y, d_states_theta,
//        d_transition_body_lidar, NUM_PARTICLES);
//    cudaDeviceSynchronize();
//
//    gpuErrchk(cudaMemcpy(res_transition_world_body, d_transition_multi_world_body, sz_transition_multi_world_frame, cudaMemcpyDeviceToHost));
//}

void exec_calc_transition(DeviceParticlesTransition& d_particles_transition, DeviceState& d_state, 
    DevicePositionTransition& d_transition, HostParticlesTransition& res_particles_transition) {

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_update_particles_states << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles_transition.transition_multi_world_body),
        THRUST_RAW_CAST(d_particles_transition.transition_multi_world_lidar), SEP,
        THRUST_RAW_CAST(d_state.x), THRUST_RAW_CAST(d_state.y), THRUST_RAW_CAST(d_state.theta),
        THRUST_RAW_CAST(d_transition.transition_body_lidar), NUM_PARTICLES);
    cudaDeviceSynchronize();
}

//void exec_process_measurements() {
//
//    threadsPerBlock = NUM_PARTICLES;
//    blocksPerGrid = LIDAR_COORDS_LEN;
//    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (d_processed_measure_x, d_processed_measure_y, SEP,
//        d_transition_multi_world_lidar, d_lidar_coords, res, xmin, ymax, LIDAR_COORDS_LEN);
//    cudaDeviceSynchronize();
//
//    threadsPerBlock = NUM_PARTICLES;
//    blocksPerGrid = 1;
//    kernel_index_init_const << < blocksPerGrid, threadsPerBlock >> > (d_processed_measure_idx, LIDAR_COORDS_LEN);
//    cudaDeviceSynchronize();
//
//    thrust::exclusive_scan(thrust::device, d_processed_measure_idx, d_processed_measure_idx + NUM_PARTICLES, d_processed_measure_idx, 0);
//}

void exec_process_measurements(DeviceProcessedMeasure& d_processed_measure, DeviceParticlesTransition& d_particles_transition,
    DeviceMeasurements& d_measurements, HostMapData& res_map, HostMeasurements& res_measurements, GeneralInfo& general_info) {

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = res_measurements.LIDAR_COORDS_LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_processed_measure.x), THRUST_RAW_CAST(d_processed_measure.y), SEP,
        THRUST_RAW_CAST(d_particles_transition.transition_multi_world_lidar), THRUST_RAW_CAST(d_measurements.lidar_coords), 
        general_info.res, res_map.xmin, res_map.ymax, res_measurements.LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_index_init_const << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_processed_measure.idx), res_measurements.LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, d_processed_measure.idx.begin(), d_processed_measure.idx.end(), d_processed_measure.idx.begin(), 0);
}

void assert_processed_measures(DeviceParticlesTransition& d_particles_transition, DeviceProcessedMeasure& d_processed_measure,
    HostParticlesTransition& res_particles_transition, HostProcessedMeasure& res_processed_measure) {

    res_particles_transition.transition_multi_world_body.assign(
        d_particles_transition.transition_multi_world_body.begin(), d_particles_transition.transition_multi_world_body.end());
    res_particles_transition.transition_multi_world_lidar.assign(
        d_particles_transition.transition_multi_world_lidar.begin(), d_particles_transition.transition_multi_world_lidar.end());

    res_processed_measure.x.assign(d_processed_measure.x.begin(), d_processed_measure.x.end());
    res_processed_measure.y.assign(d_processed_measure.y.begin(), d_processed_measure.y.end());
    res_processed_measure.idx.assign(d_processed_measure.idx.begin(), d_processed_measure.idx.end());

    //ASSERT_transition_frames(res_transition_world_body, res_transition_world_lidar,
    //    h_transition_world_body, h_transition_world_lidar, NUM_PARTICLES, false);
    // ASSERT_processed_measurements(res_processed_measure_x, res_processed_measure_y, processed_measure, NUM_PARTICLES, LIDAR_COORDS_LEN);

    ASSERT_processed_measurements(res_processed_measure.x.data(), res_processed_measure.y.data(),
        res_processed_measure.idx.data(), h_processed_measure.x.data(), h_processed_measure.y.data(),
        (NUM_PARTICLES * h_measurements.LIDAR_COORDS_LEN), h_measurements.LIDAR_COORDS_LEN, false, true, true);
}

//void exec_create_2d_map() {
//
//    threadsPerBlock = 100;
//    blocksPerGrid = NUM_PARTICLES;
//    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (d_map_2d, d_unique_in_particle, d_unique_in_particle_col, SEP,
//        d_particles_x, d_particles_y, d_particles_idx, PARTICLES_ITEMS_LEN, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
//    cudaDeviceSynchronize();
//}

void exec_create_2d_map(Device2DUniqueFinder& d_2d_unique, DeviceRobotParticles& d_robot_particles, HostMapData& res_map,
    HostRobotParticles& res_robot_particles) {

    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_2d_unique.map), THRUST_RAW_CAST(d_2d_unique.in_map), THRUST_RAW_CAST(d_2d_unique.in_col), SEP,
        THRUST_RAW_CAST(d_robot_particles.x), THRUST_RAW_CAST(d_robot_particles.y), THRUST_RAW_CAST(d_robot_particles.idx),
        res_robot_particles.LEN, res_map.GRID_WIDTH, res_map.GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();
}

void assert_create_2d_map(Device2DUniqueFinder& d_2d_unique, Host2DUniqueFinder& res_2d_unique, HostMapData& res_map, HostRobotParticles& res_robot_particles,
    const int negative_before_counter) {

    res_2d_unique.map.assign(d_2d_unique.map.begin(), d_2d_unique.map.end());
    ASSERT_create_2d_map_elements(res_2d_unique.map.data(), negative_before_counter, res_map.GRID_WIDTH, res_map.GRID_HEIGHT, NUM_PARTICLES, res_robot_particles.LEN, true, true);
}

//void exec_update_map() {
//
//    threadsPerBlock = NUM_PARTICLES;
//    blocksPerGrid = 1;
//
//    kernel_update_2d_map_with_measure << <blocksPerGrid, threadsPerBlock >> > (d_map_2d, d_unique_in_particle, d_unique_in_particle_col, SEP,
//        d_processed_measure_x, d_processed_measure_y, d_processed_measure_idx,
//        MEASURE_LEN, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
//    cudaDeviceSynchronize();
//}

void exec_update_map(Device2DUniqueFinder& d_2d_unique, DeviceProcessedMeasure& d_processed_measure, HostMapData& res_map,
    const int MEASURE_LEN) {

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;

    kernel_update_2d_map_with_measure << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_2d_unique.map), THRUST_RAW_CAST(d_2d_unique.in_map), THRUST_RAW_CAST(d_2d_unique.in_col), SEP,
        THRUST_RAW_CAST(d_processed_measure.x), THRUST_RAW_CAST(d_processed_measure.y), THRUST_RAW_CAST(d_processed_measure.idx),
        MEASURE_LEN, res_map.GRID_WIDTH, res_map.GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();
}

//void exec_particle_unique_cum_sum() {
//
//    threadsPerBlock = UNIQUE_COUNTER_LEN;
//    blocksPerGrid = 1;
//    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (d_unique_in_particle_col, GRID_WIDTH);
//    thrust::exclusive_scan(thrust::device, d_unique_in_particle, d_unique_in_particle + UNIQUE_COUNTER_LEN, d_unique_in_particle, 0);
//    cudaDeviceSynchronize();
//
//    gpuErrchk(cudaMemcpy(res_unique_in_particle, d_unique_in_particle, sz_unique_in_particle, cudaMemcpyDeviceToHost));
//
//    PARTICLES_ITEMS_LEN = res_unique_in_particle[UNIQUE_COUNTER_LEN - 1];
//    C_PARTICLES_ITEMS_LEN = 0;
//}

void exec_particle_unique_cum_sum(Device2DUniqueFinder& d_2d_unique, HostMapData& res_map, Host2DUniqueFinder& res_2d_unique, 
    HostRobotParticles& res_robot_particles) {

    const int UNIQUE_COUNTER_LEN = NUM_PARTICLES + 1;

    threadsPerBlock = UNIQUE_COUNTER_LEN;
    blocksPerGrid = 1;
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_2d_unique.in_col), res_map.GRID_WIDTH);
    thrust::exclusive_scan(thrust::device, d_2d_unique.in_map.begin(), d_2d_unique.in_map.end(), d_2d_unique.in_map.begin(), 0);
    cudaDeviceSynchronize();

    res_2d_unique.in_map.assign(d_2d_unique.in_map.begin(), d_2d_unique.in_map.end());

    res_robot_particles.LEN = res_2d_unique.in_map[UNIQUE_COUNTER_LEN - 1];
}

void assert_particles_unique(HostRobotParticles& res_robot_particles, HostRobotParticles& h_robot_particles_unique,
    const int negative_after_counter) {

    ASSERT_new_len_calculation(res_robot_particles.LEN, h_robot_particles_unique.LEN, negative_after_counter);
}

//void reinit_map_vars() {
//
//    //gpuErrchk(cudaFree(d_particles_x));
//    //gpuErrchk(cudaFree(d_particles_y));
//    //gpuErrchk(cudaFree(d_extended_idx));
//
//    sz_particles_pos = PARTICLES_ITEMS_LEN * sizeof(int);
//    sz_extended_idx = PARTICLES_ITEMS_LEN * sizeof(int);
//
//    gpuErrchk(cudaMalloc((void**)&d_particles_x, sz_particles_pos));
//    gpuErrchk(cudaMalloc((void**)&d_particles_y, sz_particles_pos));
//    gpuErrchk(cudaMalloc((void**)&d_extended_idx, sz_extended_idx));
//}

void reinit_map_vars(DeviceRobotParticles& d_robot_particles, HostRobotParticles& res_robot_particles) {

    d_robot_particles.x.clear();
    d_robot_particles.x.resize(res_robot_particles.LEN, 0);
    d_robot_particles.y.clear();
    d_robot_particles.y.resize(res_robot_particles.LEN, 0);
    d_robot_particles.extended_idx.clear();
    d_robot_particles.extended_idx.resize(res_robot_particles.LEN, 0);

    res_robot_particles.x.clear();
    res_robot_particles.x.resize(res_robot_particles.LEN, 0);
    res_robot_particles.y.clear();
    res_robot_particles.y.resize(res_robot_particles.LEN, 0);
}

//void exec_map_restructure() {
//
//    threadsPerBlock = GRID_WIDTH;
//    blocksPerGrid = NUM_PARTICLES;
//
//    cudaMemset(d_particles_idx, 0, sz_particles_idx);
//    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (d_particles_x, d_particles_y, d_particles_idx, SEP,
//        d_map_2d, d_unique_in_particle, d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT);
//    cudaDeviceSynchronize();
//
//    thrust::exclusive_scan(thrust::device, d_particles_idx, d_particles_idx + NUM_PARTICLES, d_particles_idx, 0);
//}

void exec_map_restructure(DeviceRobotParticles& d_robot_particles, Device2DUniqueFinder& d_2d_unique,
    HostMapData& res_map) {

    threadsPerBlock = res_map.GRID_WIDTH;
    blocksPerGrid = NUM_PARTICLES;

    thrust::fill(d_robot_particles.idx.begin(), d_robot_particles.idx.end(), 0);
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_robot_particles.x), THRUST_RAW_CAST(d_robot_particles.y), THRUST_RAW_CAST(d_robot_particles.idx), SEP,
        THRUST_RAW_CAST(d_2d_unique.map), THRUST_RAW_CAST(d_2d_unique.in_map), 
        THRUST_RAW_CAST(d_2d_unique.in_col), res_map.GRID_WIDTH, res_map.GRID_HEIGHT);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, d_robot_particles.idx.begin(), d_robot_particles.idx.end(), d_robot_particles.idx.begin(), 0);
}

void assert_particles_unique(DeviceRobotParticles& d_robot_particles, HostRobotParticles& res_robot_particles,
    const int negative_after_counter) {

    res_robot_particles.x.assign(d_robot_particles.x.begin(), d_robot_particles.x.end());
    res_robot_particles.y.assign(d_robot_particles.y.begin(), d_robot_particles.y.end());
    res_robot_particles.idx.assign(d_robot_particles.idx.begin(), d_robot_particles.idx.end());

    ASSERT_particles_pos_unique(res_robot_particles.x.data(), res_robot_particles.y.data(),
        h_robot_particles_unique.x.data(), h_robot_particles_unique.y.data(), res_robot_particles.LEN, false, true, true);
    ASSERT_particles_idx_unique(res_robot_particles.idx.data(), h_robot_particles_unique.idx.data(), negative_after_counter, NUM_PARTICLES, false, true);
}

//void exec_index_expansion() {
//
//    threadsPerBlock = 100;
//    blocksPerGrid = NUM_PARTICLES;
//    kernel_index_expansion << <blocksPerGrid, threadsPerBlock >> > (d_extended_idx, d_particles_idx, PARTICLES_ITEMS_LEN);
//    cudaDeviceSynchronize();
//
//    res_extended_idx = (int*)malloc(sz_extended_idx);
//    gpuErrchk(cudaMemcpy(res_extended_idx, d_extended_idx, sz_extended_idx, cudaMemcpyDeviceToHost));
//    gpuErrchk(cudaMemcpy(res_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToHost));
//}

void exec_index_expansion(DeviceRobotParticles& d_robot_particles, HostRobotParticles& res_robot_particles) {

    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;
    kernel_index_expansion << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_robot_particles.extended_idx), THRUST_RAW_CAST(d_robot_particles.idx), res_robot_particles.LEN);
    cudaDeviceSynchronize();

    res_robot_particles.extended_idx.clear();
    res_robot_particles.extended_idx.resize(h_robot_particles.LEN, 0);
    res_robot_particles.extended_idx.assign(d_robot_particles.extended_idx.begin(), d_robot_particles.extended_idx.end());
    res_robot_particles.idx.assign(d_robot_particles.idx.begin(), d_robot_particles.idx.end());
}

//void exec_correlation() {
//
//    threadsPerBlock = 256;
//    blocksPerGrid = (PARTICLES_ITEMS_LEN + threadsPerBlock - 1) / threadsPerBlock;
//
//    kernel_correlation << <blocksPerGrid, threadsPerBlock >> > (d_correlation_weights_raw, SEP, 
//        d_grid_map, d_particles_x, d_particles_y,
//        d_extended_idx, GRID_WIDTH, GRID_HEIGHT, PARTICLES_ITEMS_LEN);
//    cudaDeviceSynchronize();
//
//    threadsPerBlock = NUM_PARTICLES;
//    blocksPerGrid = 1;
//    kernel_correlation_max << <blocksPerGrid, threadsPerBlock >> > (d_correlation_weights, d_correlation_weights_raw, NUM_PARTICLES);
//    cudaDeviceSynchronize();
//
//    gpuErrchk(cudaMemcpy(res_extended_idx, d_extended_idx, sz_extended_idx, cudaMemcpyDeviceToHost));
//}

void exec_correlation(DeviceMapData& d_map, DeviceRobotParticles& d_robot_particles, DeviceCorrelation& d_correlation, 
    HostMapData& res_map, HostRobotParticles& res_robot_particles) {

    threadsPerBlock = 256;
    blocksPerGrid = (res_robot_particles.LEN + threadsPerBlock - 1) / threadsPerBlock;

    kernel_correlation << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.raw), SEP,
        THRUST_RAW_CAST(d_map.grid_map), THRUST_RAW_CAST(d_robot_particles.x), THRUST_RAW_CAST(d_robot_particles.y),
        THRUST_RAW_CAST(d_robot_particles.extended_idx), res_map.GRID_WIDTH, res_map.GRID_HEIGHT, res_robot_particles.LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_correlation_max << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.weight), THRUST_RAW_CAST(d_correlation.raw), NUM_PARTICLES);
    cudaDeviceSynchronize();

    res_robot_particles.extended_idx.assign(d_robot_particles.extended_idx.begin(), d_robot_particles.extended_idx.end());
}

void assert_correlation(DeviceCorrelation& d_correlation, DeviceRobotParticles& d_robot_particles,  
    HostCorrelation& res_correlation, HostRobotParticles& res_robot_particles) {

    res_correlation.weight.assign(d_correlation.weight.begin(), d_correlation.weight.end());
    res_robot_particles.extended_idx.assign(d_robot_particles.extended_idx.begin(), d_robot_particles.extended_idx.end());

    ASSERT_correlation_Equality(res_correlation.weight.data(), weights_pre.data(), NUM_PARTICLES, false, true);
}

//void exec_update_weights() {
//
//    threadsPerBlock = 1;
//    blocksPerGrid = 1;
//
//    kernel_arr_max << < blocksPerGrid, threadsPerBlock >> > (d_correlation_weights, d_correlation_weights_max, NUM_PARTICLES);
//    cudaDeviceSynchronize();
//
//    gpuErrchk(cudaMemcpy(res_correlation_weights_max, d_correlation_weights_max, sz_correlation_weights_max, cudaMemcpyDeviceToHost));
//
//    float norm_value = -res_correlation_weights_max[0] + 50;
//
//    threadsPerBlock = NUM_PARTICLES;
//    blocksPerGrid = 1;
//    kernel_arr_increase << < blocksPerGrid, threadsPerBlock >> > (d_correlation_weights, norm_value, 0);
//    cudaDeviceSynchronize();
//
//    threadsPerBlock = 1;
//    blocksPerGrid = 1;
//    kernel_arr_sum_exp << < blocksPerGrid, threadsPerBlock >> > (d_correlation_sum_exp, d_correlation_weights, NUM_PARTICLES);
//    cudaDeviceSynchronize();
//
//    gpuErrchk(cudaMemcpy(res_correlation_sum_exp, d_correlation_sum_exp, sz_correlation_sum_exp, cudaMemcpyDeviceToHost));
//
//    threadsPerBlock = NUM_PARTICLES;
//    blocksPerGrid = 1;
//    kernel_arr_normalize << < blocksPerGrid, threadsPerBlock >> > (d_correlation_weights, res_correlation_sum_exp[0]);
//    cudaDeviceSynchronize();
//
//    threadsPerBlock = NUM_PARTICLES;
//    blocksPerGrid = 1;
//    kernel_arr_mult << < blocksPerGrid, threadsPerBlock >> > (d_particles_weight, d_correlation_weights);
//    cudaDeviceSynchronize();
//}

void exec_update_weights(DeviceRobotParticles& d_robot_particles, DeviceCorrelation& d_correlation, 
    HostRobotParticles& res_robot_particles, HostCorrelation& res_correlation) {

    threadsPerBlock = 1;
    blocksPerGrid = 1;

    kernel_arr_max << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.weight), THRUST_RAW_CAST(d_correlation.vec_max), NUM_PARTICLES);
    cudaDeviceSynchronize();

    res_correlation.vec_max.assign(d_correlation.vec_max.begin(), d_correlation.vec_max.end());

    float norm_value = -res_correlation.vec_max[0] + 50;

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_arr_increase << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.weight), norm_value, 0);
    cudaDeviceSynchronize();

    threadsPerBlock = 1;
    blocksPerGrid = 1;
    kernel_arr_sum_exp << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.vec_sum_exp), THRUST_RAW_CAST(d_correlation.weight), NUM_PARTICLES);
    cudaDeviceSynchronize();

    res_correlation.vec_sum_exp.assign(d_correlation.vec_sum_exp.begin(), d_correlation.vec_sum_exp.end());

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_arr_normalize << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.weight), res_correlation.vec_sum_exp[0]);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_arr_mult << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_robot_particles.weight), THRUST_RAW_CAST(d_correlation.weight));
    cudaDeviceSynchronize();

    res_correlation.weight.assign(d_correlation.weight.begin(), d_correlation.weight.end());
    res_robot_particles.weight.assign(d_robot_particles.weight.begin(), d_robot_particles.weight.end());
}

void assert_update_weights(DeviceCorrelation& d_correlation, DeviceRobotParticles& d_robot_particles,
    HostCorrelation& res_correlation, HostRobotParticles& res_robot_particles) {

    res_correlation.weight.assign(d_correlation.weight.begin(), d_correlation.weight.end());
    res_robot_particles.weight.assign(d_robot_particles.weight.begin(), d_robot_particles.weight.end());

    ASSERT_update_particle_weights(res_correlation.weight.data(), weights_new.data(), NUM_PARTICLES, "weights", false, false, true);
    //ASSERT_update_particle_weights(res_robot_particles.weight.data(), h_robot_particles_unique.weight.data(), NUM_PARTICLES, "particles weight", true, false, true);
}

//void exec_resampling() {
//
//    threadsPerBlock = NUM_PARTICLES;
//    blocksPerGrid = 1;
//    
//    kernel_resampling << <blocksPerGrid, threadsPerBlock >> > (d_resampling_js, d_correlation_weights, d_resampling_rnd, NUM_PARTICLES);
//    cudaDeviceSynchronize();
//}

void exec_resampling(DeviceCorrelation& d_correlation, DeviceResampling& d_resampling) {

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    
    kernel_resampling << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_resampling.js), THRUST_RAW_CAST(d_correlation.weight), 
        THRUST_RAW_CAST(d_resampling.rnds), NUM_PARTICLES);
    cudaDeviceSynchronize();
}

void assert_resampling(DeviceResampling& d_resampling, HostResampling& res_resampling) {

    res_resampling.js.assign(d_resampling.js.begin(), d_resampling.js.end());

    ASSERT_resampling_indices(res_resampling.js.data(), h_resampling.js.data(), NUM_PARTICLES, false, false, true);
    ASSERT_resampling_states(h_state.x.data(), h_state.y.data(), h_state.theta.data(),
        h_state_updated.x.data(), h_state_updated.y.data(), h_state_updated.theta.data(), res_resampling.js.data(), NUM_PARTICLES, false, true, true);
}

//void reinit_particles_vars() {
//
//    sz_last_len = sizeof(int);
//    d_last_len = NULL;
//    res_last_len = (int*)malloc(sizeof(int));
//
//    gpuErrchk(cudaMalloc((void**)&d_last_len, sizeof(int)));
//    gpuErrchk(cudaMalloc((void**)&dc_particles_x, sz_particles_pos));
//    gpuErrchk(cudaMalloc((void**)&dc_particles_y, sz_particles_pos));
//    gpuErrchk(cudaMalloc((void**)&dc_particles_idx, sz_particles_idx));
//
//    gpuErrchk(cudaMemcpy(dc_particles_x, d_particles_x, sz_particles_pos, cudaMemcpyDeviceToDevice));
//    gpuErrchk(cudaMemcpy(dc_particles_y, d_particles_y, sz_particles_pos, cudaMemcpyDeviceToDevice));
//    gpuErrchk(cudaMemcpy(dc_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToDevice));
//
//    gpuErrchk(cudaMemcpy(dc_states_x, d_states_x, sz_states_pos, cudaMemcpyDeviceToDevice));
//    gpuErrchk(cudaMemcpy(dc_states_y, d_states_y, sz_states_pos, cudaMemcpyDeviceToDevice));
//    gpuErrchk(cudaMemcpy(dc_states_theta, d_states_theta, sz_states_pos, cudaMemcpyDeviceToDevice));
//
//    threadsPerBlock = NUM_PARTICLES;
//    blocksPerGrid = 1;
//    kernel_rearrange_indecies << <blocksPerGrid, threadsPerBlock >> > (d_particles_idx, d_last_len, SEP, 
//        dc_particles_idx, d_resampling_js, PARTICLES_ITEMS_LEN);
//    cudaDeviceSynchronize();
//
//    gpuErrchk(cudaMemcpy(res_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToHost));
//    gpuErrchk(cudaMemcpy(res_last_len, d_last_len, sz_last_len, cudaMemcpyDeviceToHost));
//}

void reinit_particles_vars(DeviceState& d_state, DeviceRobotParticles& d_robot_particles, DeviceResampling& d_resampling,
    DeviceRobotParticles& d_clone_robot_particles, DeviceState& d_clone_state, HostRobotParticles& res_robot_particles, 
    HostState& res_state, int* res_last_len) {

    int* d_last_len = NULL;
    gpuErrchk(cudaMalloc((void**)&d_last_len, sizeof(int)));
    
    d_clone_robot_particles.x.resize(res_robot_particles.LEN, 0);
    d_clone_robot_particles.y.resize(res_robot_particles.LEN, 0);
    d_clone_robot_particles.idx.resize(res_robot_particles.LEN, 0);

    d_clone_robot_particles.x.assign(d_robot_particles.x.begin(), d_robot_particles.x.end());
    d_clone_robot_particles.y.assign(d_robot_particles.y.begin(), d_robot_particles.y.end());
    d_clone_robot_particles.idx.assign(d_robot_particles.idx.begin(), d_robot_particles.idx.end());

    d_clone_state.x.assign(d_state.x.begin(), d_state.x.end());
    d_clone_state.y.assign(d_state.y.begin(), d_state.y.end());
    d_clone_state.theta.assign(d_state.theta.begin(), d_state.theta.end());

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_rearrange_indecies << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_robot_particles.idx), d_last_len, SEP,
        THRUST_RAW_CAST(d_clone_robot_particles.idx), THRUST_RAW_CAST(d_resampling.js), res_robot_particles.LEN);
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(res_last_len, d_last_len, sizeof(int), cudaMemcpyDeviceToHost));
    res_robot_particles.idx.assign(d_robot_particles.idx.begin(), d_robot_particles.idx.end());
}

//void exec_rearrangement() {
//
//    thrust::exclusive_scan(thrust::device, d_particles_idx, d_particles_idx + NUM_PARTICLES, d_particles_idx, 0);
//
//    gpuErrchk(cudaMemcpy(res_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToHost));
//    C_PARTICLES_ITEMS_LEN = PARTICLES_ITEMS_LEN;
//    PARTICLES_ITEMS_LEN = res_particles_idx[NUM_PARTICLES - 1] + res_last_len[0];
//
//    sz_particles_pos = PARTICLES_ITEMS_LEN * sizeof(int);
//
//    threadsPerBlock = 100;
//    blocksPerGrid = NUM_PARTICLES;
//    kernel_rearrange_particles << <blocksPerGrid, threadsPerBlock >> > (d_particles_x, d_particles_y, SEP,
//        d_particles_idx, dc_particles_x, dc_particles_y, dc_particles_idx, d_resampling_js,
//        GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES, PARTICLES_ITEMS_LEN, C_PARTICLES_ITEMS_LEN);
//
//    kernel_rearrange_states << <blocksPerGrid, threadsPerBlock >> > (d_states_x, d_states_y, d_states_theta, SEP,
//        dc_states_x, dc_states_y, dc_states_theta, d_resampling_js);
//    cudaDeviceSynchronize();
//}

void exec_rearrangement(DeviceRobotParticles& d_robot_particles, DeviceState& d_state, DeviceResampling& d_resampling, 
    DeviceRobotParticles& d_clone_robot_particles, DeviceState& d_clone_state, HostMapData& res_map, 
    HostRobotParticles& res_robot_particles, HostRobotParticles& res_clone_robot_particles, int* res_last_len) {

    thrust::exclusive_scan(thrust::device, d_robot_particles.idx.begin(), d_robot_particles.idx.end(), d_robot_particles.idx.begin(), 0);

    res_robot_particles.idx.assign(d_robot_particles.idx.begin(), d_robot_particles.idx.end());
    
    res_clone_robot_particles.LEN = res_robot_particles.LEN;
    res_robot_particles.LEN = res_robot_particles.idx[NUM_PARTICLES - 1] + res_last_len[0];

    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;
    kernel_rearrange_particles << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_robot_particles.x), THRUST_RAW_CAST(d_robot_particles.y), SEP,
        THRUST_RAW_CAST(d_robot_particles.idx), THRUST_RAW_CAST(d_clone_robot_particles.x), THRUST_RAW_CAST(d_clone_robot_particles.y), 
        THRUST_RAW_CAST(d_clone_robot_particles.idx), THRUST_RAW_CAST(d_resampling.js),
        res_map.GRID_WIDTH, res_map.GRID_HEIGHT, NUM_PARTICLES, res_robot_particles.LEN, res_clone_robot_particles.LEN);

    kernel_rearrange_states << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_state.x), THRUST_RAW_CAST(d_state.y), THRUST_RAW_CAST(d_state.theta), SEP,
        THRUST_RAW_CAST(d_clone_state.x), THRUST_RAW_CAST(d_clone_state.y), THRUST_RAW_CAST(d_clone_state.theta), 
        THRUST_RAW_CAST(d_resampling.js));
    cudaDeviceSynchronize();
}


//void exec_update_states() {
//
//    thrust::device_vector<float> d_vec_states_x(d_states_x, d_states_x + NUM_PARTICLES);
//    thrust::device_vector<float> d_vec_states_y(d_states_y, d_states_y + NUM_PARTICLES);
//    thrust::device_vector<float> d_vec_states_theta(d_states_theta, d_states_theta + NUM_PARTICLES);
//
//    thrust::host_vector<float> h_vec_states_x(d_vec_states_x.begin(), d_vec_states_x.end());
//    thrust::host_vector<float> h_vec_states_y(d_vec_states_y.begin(), d_vec_states_y.end());
//    thrust::host_vector<float> h_vec_states_theta(d_vec_states_theta.begin(), d_vec_states_theta.end());
//
//    std_vec_states_x.clear();
//    std_vec_states_y.clear();
//    std_vec_states_theta.clear();
//    std_vec_states_x.resize(h_vec_states_x.size());
//    std_vec_states_y.resize(h_vec_states_y.size());
//    std_vec_states_theta.resize(h_vec_states_theta.size());
//
//    std::copy(h_vec_states_x.begin(), h_vec_states_x.end(), std_vec_states_x.begin());
//    std::copy(h_vec_states_y.begin(), h_vec_states_y.end(), std_vec_states_y.begin());
//    std::copy(h_vec_states_theta.begin(), h_vec_states_theta.end(), std_vec_states_theta.begin());
//
//    std::map<std::tuple<float, float, float>, int> states;
//
//    for (int i = 0; i < NUM_PARTICLES; i++) {
//        if (states.find(std::make_tuple(std_vec_states_x[i], std_vec_states_y[i], std_vec_states_theta[i])) == states.end())
//            states.insert({ std::make_tuple(std_vec_states_x[i], std_vec_states_y[i], std_vec_states_theta[i]), 1 });
//        else
//            states[std::make_tuple(std_vec_states_x[i], std_vec_states_y[i], std_vec_states_theta[i])] += 1;
//    }
//
//    std::map<std::tuple<float, float, float>, int>::iterator best
//        = std::max_element(states.begin(), states.end(), [](const std::pair<std::tuple<float, float, float>, int>& a,
//            const std::pair<std::tuple<float, float, float>, int>& b)->bool { return a.second < b.second; });
//
//    auto key = best->first;
//
//    float theta = std::get<2>(key);
//
//    res_robot_world_body[0] = cos(theta);	res_robot_world_body[1] = -sin(theta);	res_robot_world_body[2] = std::get<0>(key);
//    res_robot_world_body[3] = sin(theta);   res_robot_world_body[4] = cos(theta);	res_robot_world_body[5] = std::get<1>(key);
//    res_robot_world_body[6] = 0;			res_robot_world_body[7] = 0;			res_robot_world_body[8] = 1;
//
//    res_robot_state[0] = std::get<0>(key); res_robot_state[1] = std::get<1>(key); res_robot_state[2] = std::get<2>(key);
//}


void exec_update_states(DeviceState& d_state, HostState& res_state, HostRobotState& res_robot_state) {

    res_state.x.assign(d_state.x.begin(), d_state.x.end());
    res_state.y.assign(d_state.y.begin(), d_state.y.end());
    res_state.theta.assign(d_state.theta.begin(), d_state.theta.end());

    std::vector<float> std_vec_states_x(res_state.x.begin(), res_state.x.end());
    std::vector<float> std_vec_states_y(res_state.y.begin(), res_state.y.end());
    std::vector<float> std_vec_states_theta(res_state.theta.begin(), res_state.theta.end());


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

    res_robot_state.transition_world_body.resize(9, 0);
    res_robot_state.state.resize(3, 0);

    res_robot_state.transition_world_body[0] = cos(theta);	res_robot_state.transition_world_body[1] = -sin(theta);	res_robot_state.transition_world_body[2] = std::get<0>(key);
    res_robot_state.transition_world_body[3] = sin(theta);   res_robot_state.transition_world_body[4] = cos(theta);	res_robot_state.transition_world_body[5] = std::get<1>(key);
    res_robot_state.transition_world_body[6] = 0;			res_robot_state.transition_world_body[7] = 0;			res_robot_state.transition_world_body[8] = 1;

    res_robot_state.state[0] = std::get<0>(key); res_robot_state.state[1] = std::get<1>(key); res_robot_state.state[2] = std::get<2>(key);
}


void assert_results(DeviceRobotParticles& d_robot_particles, DeviceCorrelation& d_correlation, 
    HostRobotParticles& res_robot_particles, HostCorrelation& res_correlation, HostRobotState& res_robot_state,
    HostRobotParticles& h_robot_particles_after_resampling, int negative_after_counter) {

    res_robot_particles.idx.assign(d_robot_particles.idx.begin(), d_robot_particles.idx.end());

    ASSERT_resampling_particles_index(h_robot_particles_after_resampling.idx.data(), res_robot_particles.idx.data(), NUM_PARTICLES, false, negative_after_counter);

    res_correlation.weight.resize(NUM_PARTICLES, 0);
    res_robot_particles.weight.resize(NUM_PARTICLES, 0);

    ASSERT_update_particle_weights(res_correlation.weight.data(), weights_new.data(), NUM_PARTICLES, "weights", false, false, true);
    ASSERT_update_particle_weights(res_robot_particles.weight.data(), h_robot_particles_unique.weight.data(), NUM_PARTICLES, "particles weight", false, false, true);

    printf("\n");
    printf("~~$ Transition World to Body (Result): ");
    for (int i = 0; i < 9; i++) {
        printf("%f ", res_robot_state.transition_world_body[i]);
    }
    printf("\n");
    printf("~~$ Transition World to Body (Host)  : ");
    for (int i = 0; i < 9; i++) {
        printf("%f ", h_robot_state.transition_world_body[i]);
    }

    printf("\n\n");
    printf("~~$ Robot State (Result): %f, %f, %f\n", res_robot_state.state[0], res_robot_state.state[1], res_robot_state.state[2]);
    printf("~~$ Robot State (Host)  : %f, %f, %f\n", h_robot_state.state[0], h_robot_state.state[1], h_robot_state.state[2]);
}


void test_robot_particles_main() {

    printf("/****************************** ROBOT  ******************************/\n");

    int MEASURE_LEN = NUM_PARTICLES * h_measurements.LIDAR_COORDS_LEN;

    int negative_before_counter = getNegativeCounter(h_robot_particles.x.data(), h_robot_particles.y.data(), h_robot_particles.LEN);
    int count_bigger_than_height = getGreaterThanCounter(h_robot_particles.y.data(), h_map.GRID_HEIGHT, h_robot_particles.LEN);
    int negative_after_counter = getNegativeCounter(h_robot_particles_after_resampling.x.data(), h_robot_particles_after_resampling.y.data(), h_robot_particles_after_resampling.LEN);

    printf("~~$ GRID_WIDTH: \t\t%d\n", h_map.GRID_WIDTH);
    printf("~~$ GRID_HEIGHT: \t\t%d\n", h_map.GRID_HEIGHT);
    printf("~~$ negative_before_counter: \t%d\n", negative_before_counter);
    printf("~~$ negative_after_counter: \t%d\n", negative_after_counter);
    printf("~~$ count_bigger_than_height: \t%d\n", count_bigger_than_height);

    DeviceState d_state;
    DeviceState d_clone_state;
    DeviceMeasurements d_measurements;
    DeviceMapData d_map;
    DeviceRobotParticles d_robot_particles;
    DeviceRobotParticles d_clone_robot_particles;
    DeviceCorrelation d_correlation;
    DevicePositionTransition d_transition;
    DeviceParticlesTransition d_particles_transition;
    DeviceProcessedMeasure d_processed_measure;
    Device2DUniqueFinder d_2d_unique;
    DeviceResampling d_resampling;

    HostState res_state;
    HostMeasurements res_measurements;
    HostMapData res_map;
    HostRobotParticles res_robot_particles;
    HostRobotParticles res_clone_robot_particles;
    HostCorrelation res_correlation;
    HostRobotState res_robot_state;
    HostParticlesTransition res_particles_transition;
    Host2DUniqueFinder res_2d_unique;
    HostProcessedMeasure res_processed_measure;
    HostResampling res_resampling;

    auto start_robot_particles_alloc = std::chrono::high_resolution_clock::now();
    alloc_init_state_vars(d_state, res_state, res_robot_state, h_state);
    alloc_init_lidar_coords_var(d_measurements, res_measurements, h_measurements);
    alloc_init_grid_map(d_map, res_map, h_map);
    alloc_init_particles_vars(d_robot_particles, res_robot_particles, h_robot_particles);
    alloc_extended_idx(d_robot_particles, res_robot_particles);
    alloc_states_copy_vars(d_clone_state);
    alloc_correlation_vars(d_correlation, res_correlation);
    alloc_init_transition_vars(d_transition, d_particles_transition, res_particles_transition);
    alloc_init_processed_measurement_vars(d_processed_measure, res_processed_measure, res_measurements);
    alloc_map_2d_var(d_2d_unique, res_2d_unique, res_map);
    alloc_map_2d_unique_counter_vars(d_2d_unique, res_2d_unique, res_map);
    alloc_correlation_weights_vars(d_correlation, res_correlation);
    alloc_resampling_vars(d_resampling, res_resampling, h_resampling);
    auto stop_robot_particles_alloc = std::chrono::high_resolution_clock::now();

    int* res_last_len = (int*)malloc(sizeof(int));
    bool should_assert = false;

    auto start_robot_particles_kernel = std::chrono::high_resolution_clock::now();
    exec_calc_transition(d_particles_transition, d_state, d_transition, res_particles_transition);
    exec_process_measurements(d_processed_measure, d_particles_transition, d_measurements, res_map, res_measurements, general_info);
    if(should_assert == true) assert_processed_measures(d_particles_transition, d_processed_measure, res_particles_transition, res_processed_measure);
    exec_create_2d_map(d_2d_unique, d_robot_particles, res_map, res_robot_particles);
    if (should_assert == true) assert_create_2d_map(d_2d_unique, res_2d_unique, res_map, res_robot_particles, negative_before_counter);
    exec_update_map(d_2d_unique, d_processed_measure, res_map, MEASURE_LEN);
    exec_particle_unique_cum_sum(d_2d_unique, res_map, res_2d_unique, res_robot_particles);
    if (should_assert == true) assert_particles_unique(res_robot_particles, h_robot_particles_unique, negative_after_counter);
    reinit_map_vars(d_robot_particles, res_robot_particles);
    exec_map_restructure(d_robot_particles, d_2d_unique, res_map);
    if (should_assert == true) assert_particles_unique(d_robot_particles, res_robot_particles, negative_after_counter);
    exec_index_expansion(d_robot_particles, res_robot_particles);
    exec_correlation(d_map, d_robot_particles, d_correlation, res_map, res_robot_particles);
    if (should_assert == true) assert_correlation(d_correlation, d_robot_particles, res_correlation, res_robot_particles);
    exec_update_weights(d_robot_particles, d_correlation, res_robot_particles, res_correlation);
    if (should_assert == true) assert_update_weights(d_correlation, d_robot_particles, res_correlation, res_robot_particles);
    exec_resampling(d_correlation, d_resampling);
    reinit_particles_vars(d_state, d_robot_particles, d_resampling, d_clone_robot_particles, d_clone_state, res_robot_particles,
        res_state, res_last_len);
    if (should_assert == true) assert_resampling(d_resampling, res_resampling);
    exec_rearrangement(d_robot_particles, d_state, d_resampling, d_clone_robot_particles, d_clone_state, res_map,
        res_robot_particles, res_clone_robot_particles, res_last_len);
    exec_update_states(d_state, res_state, res_robot_state);
    auto stop_robot_particles_kernel = std::chrono::high_resolution_clock::now();


    assert_results(d_robot_particles, d_correlation, res_robot_particles, res_correlation, res_robot_state,
        h_robot_particles_after_resampling ,negative_after_counter);


    auto duration_robot_particles_alloc = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_particles_alloc - start_robot_particles_alloc);
    auto duration_robot_particles_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_particles_kernel - start_robot_particles_kernel);
    auto duration_robot_particles_total = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_particles_kernel - start_robot_particles_alloc);

    std::cout << std::endl;
    std::cout << "Time taken by function (Allocation): " << duration_robot_particles_alloc.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Kernel): " << duration_robot_particles_kernel.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Total): " << duration_robot_particles_total.count() << " microseconds" << std::endl;
}

#endif
