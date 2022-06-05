#ifndef _TEST_ROBOT_H_
#define _TEST_ROBOT_H_

#include "headers.h"
#include "host_asserts.h"
#include "host_utils.h"
#include "kernels.cuh"
#include "kernels_robot.cuh"
#include "kernels_utils.cuh"
#include "device_init_robot.h"
#include "device_init_common.h"
#include "device_exec_robot.h"
#include "device_assert_robot.h"


void host_update_loop(HostMap&, HostState&, HostMeasurements&, HostProcessedMeasure&,
    HostParticlesTransition&, HostRobotParticles&,
    HostRobotParticles&, HostRobotParticles&, GeneralInfo&, host_vector<float>&);   // Step 1
void host_update_particles(HostMap&, HostState&, HostMeasurements&, HostParticlesTransition&, HostProcessedMeasure&,
    GeneralInfo&);  // Step 1.1
void host_update_unique(HostMap&, HostMeasurements&, HostProcessedMeasure&,HostRobotParticles&, HostRobotParticles&,
        HostRobotParticles&);                          // Step 1.2
void host_correlation(HostMap&, HostRobotParticles&, host_vector<float>&);                            // Step 1.3
void host_update_particle_weights(host_vector<float>&, host_vector<float>&);                // Step 2
void host_resampling(HostMap&, HostState&, HostState&, HostResampling&,
    HostRobotParticles&, HostRobotParticles&, host_vector<float>&);                             // Step 3
void host_update_state(HostState&, HostRobotState&);                           // Step 4
void host_update_func(HostMap&, HostState&, HostState&, HostMeasurements&, HostResampling&,
    HostRobotState&, HostProcessedMeasure&, HostRobotParticles&,
    HostRobotParticles&, HostRobotParticles&, GeneralInfo&, host_vector<float>&, host_vector<float>&);  // Step X

void test_robot_particles_main(HostMap&, HostState&, HostState&, HostMeasurements&,
    HostParticlesTransition&, HostResampling&, HostRobotState&,
    HostRobotParticles&, HostRobotParticles&, HostRobotParticles&,
    HostProcessedMeasure&, GeneralInfo&, host_vector<float>&, host_vector<float>&);

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


int test_robot() {

    const int data_len = 6;
    int data[data_len] = { 1, 0, 2, 2, 1, 3 };
    int* d_data = NULL;
    gpuErrchk(cudaMalloc((void**)&d_data, data_len * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_data, data, data_len * sizeof(int), cudaMemcpyHostToDevice));

    thrust::exclusive_scan(thrust::host, data, data + 6, data, 0);
    thrust::exclusive_scan(thrust::device, d_data, d_data + 6, d_data, 0);
    
    HostMap pre_map;
    HostMeasurements pre_measurements;
    HostParticles pre_particles;
    HostRobotParticles pre_robot_particles;
    HostRobotParticles pre_resampling_robot_particles;
    HostRobotParticles post_resampling_robot_particles;
    HostRobotParticles post_unique_robot_particles;
    HostProcessedMeasure post_processed_measure;
    HostState pre_state;
    HostState post_state;
    //HostParticlesPosition pre_particles_position;
    //HostParticlesRotation pre_particles_rotation;
    //HostTransition pre_transition;
    HostResampling pre_resampling;
    HostRobotState post_robot_state;
    HostParticlesTransition post_particles_transition;
    host_vector<float> pre_weights;
    host_vector<float> post_loop_weights;
    host_vector<float> post_weights;
    GeneralInfo general_info;


    read_update_robot(500, pre_map, pre_measurements, pre_robot_particles, pre_resampling_robot_particles, 
        post_resampling_robot_particles, post_unique_robot_particles, post_processed_measure, pre_state,
        post_state, pre_resampling, post_robot_state, post_particles_transition,
        pre_weights, post_loop_weights, post_weights, general_info);

    host_update_particles(pre_map, pre_state, pre_measurements, post_particles_transition, post_processed_measure,
        general_info);                                                                                              // Step 1.1
    host_update_unique(pre_map, pre_measurements, post_processed_measure, pre_robot_particles, post_unique_robot_particles,
        post_resampling_robot_particles);                                                                           // Step 1.2
    host_correlation(pre_map, post_unique_robot_particles, pre_weights);                                            // Step 1.3
    host_update_loop(pre_map, pre_state, pre_measurements, post_processed_measure,
        post_particles_transition, pre_robot_particles,
        post_unique_robot_particles, post_resampling_robot_particles,
        general_info, pre_weights);                                                                                 // Step 1
    host_update_particle_weights(pre_weights, post_loop_weights);                                                   // Step 2
    host_resampling(pre_map, pre_state, post_state, pre_resampling,
        pre_robot_particles, post_resampling_robot_particles, post_loop_weights);                                   // Step 3
    host_update_state(post_state, post_robot_state);                                                                // Step 4
    host_update_func(pre_map, pre_state, post_state, pre_measurements, pre_resampling,
        post_robot_state, post_processed_measure, pre_robot_particles,
        post_unique_robot_particles, post_resampling_robot_particles,
        general_info, pre_weights, post_loop_weights);                                                              // Step X

    test_robot_particles_main(pre_map, pre_state, post_state, pre_measurements,
        post_particles_transition, pre_resampling, post_robot_state,
        pre_robot_particles, post_unique_robot_particles, post_resampling_robot_particles,
        post_processed_measure, general_info, pre_weights, post_loop_weights);

    return 0;
}


// Step 1.1
void host_update_particles(HostMap& pre_map, HostState& pre_state, HostMeasurements& pre_measurements, 
    HostParticlesTransition& post_particles_transition, HostProcessedMeasure& post_processed_measure,
    GeneralInfo& general_info) {

    printf("/************************** UPDATE PARTICLES ************************/\n");
    
    DeviceMeasurements d_measurements;
    DeviceState d_state;
    DevicePosition d_position;
    DeviceTransition d_transition;
    DeviceParticlesTransition d_particles_transition;
    DeviceProcessedMeasure d_processed_measure;

    HostParticlesTransition h_particles_transition;
    HostProcessedMeasure h_processed_measure;
    HostMeasurements h_measurements;

    /************************* STATES VARIABLES *************************/
    h_measurements.LEN = pre_measurements.LEN;

    d_state.c_x.resize(NUM_PARTICLES, 0);
    d_state.c_y.resize(NUM_PARTICLES, 0);
    d_state.c_theta.resize(NUM_PARTICLES, 0);

    d_state.c_x.assign(pre_state.c_x.begin(), pre_state.c_x.end());
    d_state.c_y.assign(pre_state.c_y.begin(), pre_state.c_y.end());
    d_state.c_theta.assign(pre_state.c_theta.begin(), pre_state.c_theta.end());

    d_measurements.v_lidar_coords.resize(2 * h_measurements.MAX_LEN, 0);
    thrust::copy(pre_measurements.v_lidar_coords.begin(), pre_measurements.v_lidar_coords.end(), d_measurements.v_lidar_coords.begin());


    /************************ TRANSFORM VARIABLES ***********************/
    d_particles_transition.c_world_body.resize(9 * NUM_PARTICLES, 0);
    d_particles_transition.c_world_lidar.resize(9 * NUM_PARTICLES, 0);

    /*------------------------ RESULT VARIABLES -----------------------*/
    h_particles_transition.c_world_body.resize(9 * NUM_PARTICLES, 0);
    h_particles_transition.c_world_lidar.resize(9 * NUM_PARTICLES, 0);

    /********************* PROCESSED MEASURE VARIABLES ******************/
    d_processed_measure.v_x.resize(NUM_PARTICLES * h_measurements.MAX_LEN, 0);
    d_processed_measure.v_y.resize(NUM_PARTICLES * h_measurements.MAX_LEN, 0);
    //d_processed_measure.c_idx.resize(NUM_PARTICLES * h_measurements.MAX_LEN, 0);
    d_processed_measure.c_idx.resize(NUM_PARTICLES, 0);

    /*------------------------ RESULT VARIABLES -----------------------*/
    h_processed_measure.v_x.resize(NUM_PARTICLES * h_measurements.MAX_LEN, 0);
    h_processed_measure.v_y.resize(NUM_PARTICLES * h_measurements.MAX_LEN, 0);
    h_processed_measure.c_idx.resize(NUM_PARTICLES, 0);

    d_transition.c_body_lidar.resize(9, 0);
    d_transition.c_body_lidar.assign(hvec_transition_body_lidar.begin(), hvec_transition_body_lidar.end());

    /*************************** KERNEL EXEC ****************************/
    auto start_kernel = std::chrono::high_resolution_clock::now();

    int threadsPerBlock = NUM_PARTICLES;
    int blocksPerGrid = 1;

    kernel_update_particles_states << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles_transition.c_world_body), 
        THRUST_RAW_CAST(d_particles_transition.c_world_lidar), SEP,
        THRUST_RAW_CAST(d_state.c_x), THRUST_RAW_CAST(d_state.c_y), THRUST_RAW_CAST(d_state.c_theta),
        THRUST_RAW_CAST(d_transition.c_body_lidar), NUM_PARTICLES);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = h_measurements.LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_processed_measure.v_x), THRUST_RAW_CAST(d_processed_measure.v_y), SEP,
        THRUST_RAW_CAST(d_particles_transition.c_world_lidar), THRUST_RAW_CAST(d_measurements.v_lidar_coords), 
        general_info.res, pre_map.xmin, pre_map.ymax, h_measurements.LEN);
    cudaDeviceSynchronize();
    
    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_index_init_const << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_processed_measure.c_idx), h_measurements.LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = 1;
    blocksPerGrid = 1;
    thrust::exclusive_scan(thrust::device, d_processed_measure.c_idx.begin(), d_processed_measure.c_idx.end(), 
        d_processed_measure.c_idx.begin(), 0);

    auto stop_kernel = std::chrono::high_resolution_clock::now();

    h_particles_transition.c_world_body.assign(
        d_particles_transition.c_world_body.begin(), d_particles_transition.c_world_body.end());
    h_particles_transition.c_world_lidar.assign(
        d_particles_transition.c_world_lidar.begin(), d_particles_transition.c_world_lidar.end());

    int PROCESSED_MEASURE_ACTUAL_LEN = h_measurements.LEN * NUM_PARTICLES;
    thrust::copy(d_processed_measure.v_x.begin(), d_processed_measure.v_x.begin() + PROCESSED_MEASURE_ACTUAL_LEN, h_processed_measure.v_x.begin());
    thrust::copy(d_processed_measure.v_y.begin(), d_processed_measure.v_y.begin() + PROCESSED_MEASURE_ACTUAL_LEN, h_processed_measure.v_y.begin());
    //thrust::copy(d_processed_measure.c_idx.begin(), d_processed_measure.c_idx.begin() + PROCESSED_MEASURE_ACTUAL_LEN, h_processed_measure.c_idx.begin());
    h_processed_measure.c_idx.assign(d_processed_measure.c_idx.begin(), d_processed_measure.c_idx.end());

    ASSERT_transition_frames(h_particles_transition.c_world_body.data(), 
        h_particles_transition.c_world_lidar.data(),
        post_particles_transition.c_world_body.data(), post_particles_transition.c_world_lidar.data(), 
        NUM_PARTICLES, false, true, false);

    ASSERT_processed_measurements(h_processed_measure.v_x.data(), h_processed_measure.v_y.data(),
        h_processed_measure.c_idx.data(), post_processed_measure.v_x.data(), post_processed_measure.v_y.data(),
        (NUM_PARTICLES * h_measurements.LEN), h_measurements.LEN, false, true, true);

    
    auto duration_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_kernel - start_kernel);
    std::cout << "Time taken by function (Kernel): " << duration_kernel.count() << " microseconds" << std::endl;
    std::cout << std::endl;
}

// Step 1.2
void host_update_unique(HostMap& pre_map, HostMeasurements& pre_measurements, HostProcessedMeasure& post_processed_measure,
    HostRobotParticles& pre_robot_particles, HostRobotParticles& post_unique_robot_particles, 
    HostRobotParticles& post_resampling_robot_particles) {

    printf("/************************** UPDATE UNIQUE ***************************/\n");

    int negative_before_counter = getNegativeCounter(pre_robot_particles.f_x.data(), pre_robot_particles.f_y.data(), pre_robot_particles.LEN);
    int count_bigger_than_height = getGreaterThanCounter(pre_robot_particles.f_y.data(), pre_map.GRID_HEIGHT, pre_robot_particles.LEN);
    int negative_after_unique_counter = getNegativeCounter(post_unique_robot_particles.f_x.data(), post_unique_robot_particles.f_y.data(), post_unique_robot_particles.LEN);
    int negative_after_counter = getNegativeCounter(post_resampling_robot_particles.f_x.data(), post_resampling_robot_particles.f_y.data(),
        post_resampling_robot_particles.LEN);

    printf("~~$ GRID_WIDTH: \t\t%d\n", pre_map.GRID_WIDTH);
    printf("~~$ GRID_HEIGHT: \t\t%d\n", pre_map.GRID_HEIGHT);
    printf("~~$ MEASURE_LEN: \t\t%d\n", pre_measurements.LEN);
    printf("~~$ negative_before_counter: \t%d\n", negative_before_counter);
    printf("~~$ negative_after_counter: \t%d\n", negative_after_counter);
    printf("~~$ count_bigger_than_height: \t%d\n", count_bigger_than_height);

    const int MEASURE_LEN = NUM_PARTICLES * pre_measurements.LEN;
    const int UNIQUE_COUNTER_LEN = NUM_PARTICLES + 1;

    DeviceRobotParticles d_robot_particles;
    Device2DUniqueFinder d_2d_unique;
    DeviceProcessedMeasure d_processed_measure;

    HostRobotParticles h_robot_particles;
    h_robot_particles.LEN = pre_robot_particles.LEN;

    Host2DUniqueFinder h_2d_unique;
    HostProcessedMeasure h_processed_measure;

    HostMeasurements h_measurements;
    h_measurements.LEN = pre_measurements.LEN;

    HostMap h_map;
    h_map.GRID_WIDTH = pre_map.GRID_WIDTH;
    h_map.GRID_HEIGHT = pre_map.GRID_HEIGHT;

    /************************** PRIOR VARIABLES *************************/
    d_robot_particles.f_x.resize(h_robot_particles.MAX_LEN, 0);
    d_robot_particles.f_y.resize(h_robot_particles.MAX_LEN, 0);
    d_robot_particles.f_extended_idx.resize(h_robot_particles.MAX_LEN, 0);
    d_robot_particles.c_idx.resize(NUM_PARTICLES, 0);

    h_robot_particles.f_x.resize(h_robot_particles.MAX_LEN, 0);
    h_robot_particles.f_y.resize(h_robot_particles.MAX_LEN, 0);
    h_robot_particles.f_extended_idx.resize(h_robot_particles.MAX_LEN, 0);
    h_robot_particles.c_idx.resize(NUM_PARTICLES, 0);

    //d_robot_particles.f_x.assign(pre_robot_particles.f_x.begin(), pre_robot_particles.f_x.end());
    //d_robot_particles.f_y.assign(pre_robot_particles.f_y.begin(), pre_robot_particles.f_y.end());
    thrust::copy(pre_robot_particles.f_x.begin(), pre_robot_particles.f_x.begin() + h_robot_particles.LEN, d_robot_particles.f_x.begin());
    thrust::copy(pre_robot_particles.f_y.begin(), pre_robot_particles.f_y.begin() + h_robot_particles.LEN, d_robot_particles.f_y.begin());
    d_robot_particles.c_idx.assign(pre_robot_particles.c_idx.begin(), pre_robot_particles.c_idx.end());

    /**************************** MAP VARIABLES *************************/

    d_2d_unique.s_map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT * NUM_PARTICLES, 0);
    d_2d_unique.c_in_map.resize(UNIQUE_COUNTER_LEN, 0);
    d_2d_unique.s_in_col.resize(UNIQUE_COUNTER_LEN * h_map.GRID_WIDTH, 0);

    h_2d_unique.s_map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT * NUM_PARTICLES, 0);
    h_2d_unique.c_in_map.resize(UNIQUE_COUNTER_LEN, 0);
    h_2d_unique.s_in_col.resize(UNIQUE_COUNTER_LEN * h_map.GRID_WIDTH, 0);

    /*********************** MEASUREMENT VARIABLES **********************/
    d_processed_measure.v_y.resize(NUM_PARTICLES * h_measurements.MAX_LEN, 0);
    d_processed_measure.v_x.resize(NUM_PARTICLES * h_measurements.MAX_LEN, 0);
    d_processed_measure.c_idx.resize(NUM_PARTICLES, 0);

    //d_processed_measure.v_x.assign(post_processed_measure.v_x.begin(), post_processed_measure.v_x.end());
    //d_processed_measure.v_y.assign(post_processed_measure.v_y.begin(), post_processed_measure.v_y.end());
    thrust::copy(post_processed_measure.v_x.begin(), post_processed_measure.v_x.begin() + MEASURE_LEN, d_processed_measure.v_x.begin());
    thrust::copy(post_processed_measure.v_y.begin(), post_processed_measure.v_y.begin() + MEASURE_LEN, d_processed_measure.v_y.begin());
    d_processed_measure.c_idx.assign(post_processed_measure.c_idx.begin(), post_processed_measure.c_idx.end());

    /**************************** CREATE MAP ****************************/
    int threadsPerBlock = 100;
    int blocksPerGrid = NUM_PARTICLES;
    auto start_create_map = std::chrono::high_resolution_clock::now();
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_2d_unique.s_map), THRUST_RAW_CAST(d_2d_unique.c_in_map), THRUST_RAW_CAST(d_2d_unique.s_in_col), SEP,
        THRUST_RAW_CAST(d_robot_particles.f_x), THRUST_RAW_CAST(d_robot_particles.f_y), THRUST_RAW_CAST(d_robot_particles.c_idx), 
        h_robot_particles.LEN, h_map.GRID_WIDTH, h_map.GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();
    auto stop_create_map = std::chrono::high_resolution_clock::now();
    h_2d_unique.s_map.assign(d_2d_unique.s_map.begin(), d_2d_unique.s_map.end());

    ASSERT_create_2d_map_elements(h_2d_unique.s_map.data(), negative_before_counter, 
        pre_map.GRID_WIDTH, pre_map.GRID_HEIGHT, NUM_PARTICLES, pre_robot_particles.LEN, true, true);

    /**************************** UPDATE MAP ****************************/
    auto start_update_map = std::chrono::high_resolution_clock::now();
    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;

    kernel_update_2d_map_with_measure << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_2d_unique.s_map), THRUST_RAW_CAST(d_2d_unique.c_in_map), THRUST_RAW_CAST(d_2d_unique.s_in_col), SEP,
        THRUST_RAW_CAST(d_processed_measure.v_x), THRUST_RAW_CAST(d_processed_measure.v_y), THRUST_RAW_CAST(d_processed_measure.c_idx), 
        MEASURE_LEN, h_map.GRID_WIDTH, h_map.GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();
    auto stop_update_map = std::chrono::high_resolution_clock::now();
    h_2d_unique.s_map.assign(d_2d_unique.s_map.begin(), d_2d_unique.s_map.end());

    /************************* CUMULATIVE SUM ***************************/
    threadsPerBlock = UNIQUE_COUNTER_LEN;
    blocksPerGrid = 1;

    auto start_cumulative_sum = std::chrono::high_resolution_clock::now();
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_2d_unique.s_in_col), h_map.GRID_WIDTH);
    thrust::exclusive_scan(thrust::device, 
        d_2d_unique.c_in_map.begin(), d_2d_unique.c_in_map.end(), d_2d_unique.c_in_map.begin(), 0);
    cudaDeviceSynchronize();
    auto stop_cumulative_sum = std::chrono::high_resolution_clock::now();

    h_2d_unique.s_map.assign(d_2d_unique.s_map.begin(), d_2d_unique.s_map.end());
    h_2d_unique.c_in_map.assign(d_2d_unique.c_in_map.begin(), d_2d_unique.c_in_map.end());
    h_2d_unique.s_in_col.assign(d_2d_unique.s_in_col.begin(), d_2d_unique.s_in_col.end());

    int pre_robot_particles_LEN = h_robot_particles.LEN;
    h_robot_particles.LEN = h_2d_unique.c_in_map[NUM_PARTICLES];
    printf("\n~~$ PARTICLES_ITEMS_LEN=%d, AF_PARTICLES_ITEMS_LEN=%d\n", h_robot_particles.LEN, post_unique_robot_particles.LEN);
    ASSERT_new_len_calculation(h_robot_particles.LEN, post_unique_robot_particles.LEN, negative_after_unique_counter);


    //d_robot_particles.f_x.clear();
    //d_robot_particles.f_x.resize(h_robot_particles.LEN, 0);
    //d_robot_particles.f_y.clear();
    //d_robot_particles.f_y.resize(h_robot_particles.LEN, 0);
    //d_robot_particles.f_extended_idx.clear();
    //d_robot_particles.f_extended_idx.resize(h_robot_particles.LEN, 0);
    thrust::fill(d_robot_particles.f_x.begin(), d_robot_particles.f_x.begin() + pre_robot_particles_LEN, 0);
    thrust::fill(d_robot_particles.f_y.begin(), d_robot_particles.f_y.begin() + pre_robot_particles_LEN, 0);
    thrust::fill(d_robot_particles.f_extended_idx.begin(), d_robot_particles.f_extended_idx.begin() + pre_robot_particles_LEN, 0);

    //h_robot_particles.f_x.resize(h_robot_particles.LEN, 0);
    //h_robot_particles.f_y.resize(h_robot_particles.LEN, 0);
    //h_robot_particles.c_idx.resize(NUM_PARTICLES, 0);
    thrust::fill(h_robot_particles.f_x.begin(), h_robot_particles.f_x.begin() + pre_robot_particles_LEN, 0);
    thrust::fill(h_robot_particles.f_y.begin(), h_robot_particles.f_y.begin() + pre_robot_particles_LEN, 0);
    thrust::fill(h_robot_particles.c_idx.begin(), h_robot_particles.c_idx.end(), 0);

    /************************ MAP RESTRUCTURE ***************************/
    threadsPerBlock = pre_map.GRID_WIDTH;
    blocksPerGrid = NUM_PARTICLES;

    auto start_map_restructure = std::chrono::high_resolution_clock::now();
    thrust::fill(d_robot_particles.c_idx.begin(), d_robot_particles.c_idx.end(), 0);
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_robot_particles.f_x), THRUST_RAW_CAST(d_robot_particles.f_y), THRUST_RAW_CAST(d_robot_particles.c_idx), SEP,
        THRUST_RAW_CAST(d_2d_unique.s_map), THRUST_RAW_CAST(d_2d_unique.c_in_map), THRUST_RAW_CAST(d_2d_unique.s_in_col), 
        h_map.GRID_WIDTH, h_map.GRID_HEIGHT);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, d_robot_particles.c_idx.begin(), d_robot_particles.c_idx.end(), d_robot_particles.c_idx.begin(), 0);
    auto stop_map_restructure = std::chrono::high_resolution_clock::now();

    //h_robot_particles.f_x.assign(d_robot_particles.f_x.begin(), d_robot_particles.f_x.end());
    //h_robot_particles.f_y.assign(d_robot_particles.f_y.begin(), d_robot_particles.f_y.end());
    thrust::copy(d_robot_particles.f_x.begin(), d_robot_particles.f_x.begin() + h_robot_particles.LEN, h_robot_particles.f_x.begin());
    thrust::copy(d_robot_particles.f_y.begin(), d_robot_particles.f_y.begin() + h_robot_particles.LEN, h_robot_particles.f_y.begin());
    h_robot_particles.c_idx.assign(d_robot_particles.c_idx.begin(), d_robot_particles.c_idx.end());

    //printf("~~$ PARTICLES_ITEMS_LEN: \t%d\n", PARTICLES_ITEMS_LEN);
    //printf("~~$ AF_PARTICLES_ITEMS_LEN: \t%d\n", AF_PARTICLES_ITEMS_LEN_UNIQUE);
    //printf("~~$ Measurement Length: \t%d\n", MEASURE_LEN);

    ASSERT_particles_pos_unique(h_robot_particles.f_x.data(), h_robot_particles.f_y.data(), 
        post_unique_robot_particles.f_x.data(), post_unique_robot_particles.f_y.data(), h_robot_particles.LEN, false, true, true);
    ASSERT_particles_idx_unique(h_robot_particles.c_idx.data(), post_unique_robot_particles.c_idx.data(), negative_after_counter, NUM_PARTICLES, false, true);

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
void host_correlation(HostMap& pre_map, HostRobotParticles& post_unique_robot_particles, host_vector<float>& pre_weights) {

    printf("/**************************** CORRELATION ***************************/\n");

    DeviceMap d_map;
    DeviceRobotParticles d_robot_particles;
    DeviceCorrelation d_correlation;

    HostCorrelation h_correlation;

    HostRobotParticles h_robot_particles;
    h_robot_particles.LEN = post_unique_robot_particles.LEN;

    HostMap h_map;
    h_map.GRID_WIDTH = pre_map.GRID_WIDTH;
    h_map.GRID_HEIGHT = pre_map.GRID_HEIGHT;

    /************************** PRIOR VARIABLES *************************/

    d_map.s_grid_map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT);
    //d_robot_particles.f_x.resize(post_unique_robot_particles.LEN, 0);
    //d_robot_particles.f_y.resize(post_unique_robot_particles.LEN, 0);
    d_robot_particles.f_x.resize(h_robot_particles.MAX_LEN, 0);
    d_robot_particles.f_y.resize(h_robot_particles.MAX_LEN, 0);
    d_robot_particles.c_idx.resize(NUM_PARTICLES, 0);
    d_robot_particles.f_extended_idx.resize(h_robot_particles.MAX_LEN, 0);

    auto start_memory_copy = std::chrono::high_resolution_clock::now();

    d_map.s_grid_map.assign(pre_map.s_grid_map.begin(), pre_map.s_grid_map.end());
    //d_robot_particles.f_x.assign(post_unique_robot_particles.f_x.begin(), post_unique_robot_particles.f_x.end());
    //d_robot_particles.f_y.assign(post_unique_robot_particles.f_y.begin(), post_unique_robot_particles.f_y.end());
    thrust::copy(post_unique_robot_particles.f_x.begin(), post_unique_robot_particles.f_x.begin() + h_robot_particles.LEN, d_robot_particles.f_x.begin());
    thrust::copy(post_unique_robot_particles.f_y.begin(), post_unique_robot_particles.f_y.begin() + h_robot_particles.LEN, d_robot_particles.f_y.begin());
    d_robot_particles.c_idx.assign(post_unique_robot_particles.c_idx.begin(), post_unique_robot_particles.c_idx.end());

    d_correlation.c_weight.resize(NUM_PARTICLES, 0);
    d_correlation.c_raw.resize(25 * NUM_PARTICLES, 0);

    h_correlation.c_weight.resize(NUM_PARTICLES, 0);

    auto stop_memory_copy = std::chrono::high_resolution_clock::now();

    /*************************** PRINT SUMMARY **************************/
    //printf("Elements of particles_x: \t%d  \tSize of particles_x: \t%d\n", (int)PARTICLES_ITEMS_LEN, (int)sz_particles_pos);
    //printf("Elements of particles_y: \t%d  \tSize of particles_y: \t%d\n", (int)PARTICLES_ITEMS_LEN, (int)sz_particles_pos);
    //printf("Elements of particles_idx: \t%d  \tSize of particles_idx: \t%d\n", (int)PARTICLES_ITEMS_LEN, (int)sz_extended_idx);
    //printf("\n");
    //printf("Elements of Grid_Map: \t\t%d  \tSize of Grid_Map: \t%d\n", (int)GRID_MAP_ITEMS_LEN, (int)sz_grid_map);

    /************************* INDEX EXPANSION **************************/
    auto start_index_expansion = std::chrono::high_resolution_clock::now();
    int threadsPerBlock = 100;
    int blocksPerGrid = NUM_PARTICLES;
    kernel_index_expansion << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_robot_particles.f_extended_idx), THRUST_RAW_CAST(d_robot_particles.c_idx), h_robot_particles.LEN);
    cudaDeviceSynchronize();
    auto stop_index_expansion = std::chrono::high_resolution_clock::now();

    /************************ KERNEL CORRELATION ************************/

    auto start_kernel = std::chrono::high_resolution_clock::now();
    threadsPerBlock = 256;
    blocksPerGrid = (h_robot_particles.LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_correlation << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.c_raw), SEP,
        THRUST_RAW_CAST(d_map.s_grid_map), THRUST_RAW_CAST(d_robot_particles.f_x), THRUST_RAW_CAST(d_robot_particles.f_y), 
        THRUST_RAW_CAST(d_robot_particles.f_extended_idx), h_map.GRID_WIDTH, h_map.GRID_HEIGHT, h_robot_particles.LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_correlation_max << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.c_weight), THRUST_RAW_CAST(d_correlation.c_raw), NUM_PARTICLES);

    auto stop_kernel = std::chrono::high_resolution_clock::now();

    h_correlation.c_weight.assign(d_correlation.c_weight.begin(), d_correlation.c_weight.end());

    ASSERT_correlation_Equality(h_correlation.c_weight.data(), pre_weights.data(), NUM_PARTICLES, true, true);

    auto duration_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_kernel - start_kernel);
    auto duration_memory_copy = std::chrono::duration_cast<std::chrono::microseconds>(stop_memory_copy - start_memory_copy);
    auto duration_index_expansion = std::chrono::duration_cast<std::chrono::microseconds>(stop_index_expansion - start_index_expansion);
    std::cout << "Time taken by function (Correlation): " << duration_kernel.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Memory Copy): " << duration_memory_copy.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Index Expansion): " << duration_index_expansion.count() << " microseconds" << std::endl;
    std::cout << std::endl;
}

// Step 1
void host_update_loop(HostMap& pre_map, HostState& pre_state, HostMeasurements& pre_measurements, HostProcessedMeasure& post_processed_measure,
    HostParticlesTransition& post_particles_transition, HostRobotParticles& pre_robot_particles, 
    HostRobotParticles& post_unique_robot_particles, HostRobotParticles& post_resampling_robot_particles,
    GeneralInfo& general_info, host_vector<float>& pre_weights) {

    printf("/**************************** UPDATE LOOP ***************************/\n");

    const int MEASURE_LEN = NUM_PARTICLES * pre_measurements.LEN;
    const int UNIQUE_COUNTER_LEN = NUM_PARTICLES + 1;

    int negative_before_counter = getNegativeCounter(
        pre_robot_particles.f_x.data(), pre_robot_particles.f_y.data(), pre_robot_particles.LEN);
    int count_bigger_than_height = getGreaterThanCounter(pre_robot_particles.f_y.data(), pre_map.GRID_HEIGHT, pre_robot_particles.LEN);
    int negative_after_unique_counter = getNegativeCounter(post_unique_robot_particles.f_x.data(), post_unique_robot_particles.f_y.data(), post_unique_robot_particles.LEN);
    int negative_after_resampling_counter = getNegativeCounter(post_resampling_robot_particles.f_x.data(), post_resampling_robot_particles.f_y.data(), post_resampling_robot_particles.LEN);

    printf("~~$ GRID_WIDTH: \t\t\t%d\n", pre_map.GRID_WIDTH);
    printf("~~$ GRID_HEIGHT: \t\t\t%d\n", pre_map.GRID_HEIGHT);
    printf("~~$ LEN: \t\t\t%d\n", pre_measurements.LEN);
    printf("~~$ negative_before_counter: \t\t%d\n", negative_before_counter);
    printf("~~$ negative_after_unique_counter: \t%d\n", negative_after_unique_counter);
    printf("~~$ negative_after_resampling_counter: \t%d\n", negative_after_resampling_counter);
    printf("~~$ count_bigger_than_height: \t\t%d\n", count_bigger_than_height);
    printf("~~$ MEASURE_LEN: \t\t\t%d \n", MEASURE_LEN);

    HostState h_state;
    HostRobotParticles h_robot_particles;
    HostCorrelation h_correlation;
    HostParticlesTransition h_particles_transition;
    HostProcessedMeasure h_processed_measure;
    Host2DUniqueFinder h_2d_unique;

    DeviceMap d_map;
    DeviceState d_state;
    DeviceMeasurements d_measurements;
    DeviceRobotParticles d_robot_particles;
    DeviceCorrelation d_correlation;
    DeviceParticlesTransition d_particles_transition;
    DeviceTransition d_transition;
    DeviceProcessedMeasure d_processed_measure;
    Device2DUniqueFinder d_2d_unique;

    /************************** PRIOR VARIABLES *************************/
    d_state.c_x.resize(NUM_PARTICLES, 0);
    d_state.c_y.resize(NUM_PARTICLES, 0);
    d_state.c_theta.resize(NUM_PARTICLES, 0);
    d_measurements.v_lidar_coords.resize(2 * pre_measurements.MAX_LEN, 0);

    d_state.c_x.assign(pre_state.c_x.begin(), pre_state.c_x.end());
    d_state.c_y.assign(pre_state.c_y.begin(), pre_state.c_y.end());
    d_state.c_theta.assign(pre_state.c_theta.begin(), pre_state.c_theta.end());
    //d_measurements.v_lidar_coords.assign(pre_measurements.v_lidar_coords.begin(), pre_measurements.v_lidar_coords.end());
    thrust::copy(pre_measurements.v_lidar_coords.begin(), pre_measurements.v_lidar_coords.end(), d_measurements.v_lidar_coords.begin());

    /**************************** MAP VARIABLES *************************/
    d_map.s_grid_map.resize(pre_map.GRID_WIDTH * pre_map.GRID_HEIGHT, 0);
    d_robot_particles.f_x.resize(pre_robot_particles.MAX_LEN, 0);
    d_robot_particles.f_y.resize(pre_robot_particles.MAX_LEN, 0);
    d_robot_particles.c_idx.resize(NUM_PARTICLES, 0);
    d_robot_particles.f_extended_idx.resize(pre_robot_particles.MAX_LEN, 0);

    h_robot_particles.f_x.resize(pre_robot_particles.MAX_LEN, 0);
    h_robot_particles.f_y.resize(pre_robot_particles.MAX_LEN, 0);
    h_robot_particles.c_idx.resize(NUM_PARTICLES, 0);
    h_robot_particles.f_extended_idx.resize(pre_robot_particles.MAX_LEN, 0);

    d_map.s_grid_map.assign(pre_map.s_grid_map.begin(), pre_map.s_grid_map.end());
    //d_robot_particles.f_x.assign(pre_robot_particles.f_x.begin(), pre_robot_particles.f_x.end());
    //d_robot_particles.f_y.assign(pre_robot_particles.f_y.begin(), pre_robot_particles.f_y.end());
    thrust::copy(pre_robot_particles.f_x.begin(), pre_robot_particles.f_x.end(), d_robot_particles.f_x.begin());
    thrust::copy(pre_robot_particles.f_y.begin(), pre_robot_particles.f_y.end(), d_robot_particles.f_y.begin());
    d_robot_particles.c_idx.assign(pre_robot_particles.c_idx.begin(), pre_robot_particles.c_idx.end());
    /********************** CORRELATION VARIABLES ***********************/
    h_correlation.c_weight.resize(NUM_PARTICLES, 0);
    h_correlation.c_raw.resize(25 * NUM_PARTICLES, 0);

    d_correlation.c_weight.resize(NUM_PARTICLES, 0);
    d_correlation.c_raw.resize(25 * NUM_PARTICLES, 0);

    /*********************** TRANSITION VARIABLES ***********************/
    h_particles_transition.c_world_body.resize(9 * NUM_PARTICLES, 0);
    h_particles_transition.c_world_lidar.resize(9 * NUM_PARTICLES, 0);
    h_processed_measure.v_x.resize(NUM_PARTICLES* pre_measurements.MAX_LEN, 0);
    h_processed_measure.v_y.resize(NUM_PARTICLES* pre_measurements.MAX_LEN, 0);
    h_processed_measure.c_idx.resize(NUM_PARTICLES, 0);

    d_particles_transition.c_world_body.resize(9 * NUM_PARTICLES, 0);
    d_particles_transition.c_world_lidar.resize(9 * NUM_PARTICLES, 0);
    d_transition.c_body_lidar.resize(9, 0);
    d_processed_measure.v_x.resize(NUM_PARTICLES* pre_measurements.MAX_LEN, 0);
    d_processed_measure.v_y.resize(NUM_PARTICLES* pre_measurements.MAX_LEN, 0);
    d_processed_measure.c_idx.resize(NUM_PARTICLES, 0);

    d_transition.c_body_lidar.assign(hvec_transition_body_lidar.begin(), hvec_transition_body_lidar.end());

    /**************************** MAP VARIABLES *************************/
    d_2d_unique.s_map.resize(pre_map.GRID_WIDTH* pre_map.GRID_HEIGHT* NUM_PARTICLES, 0);
    d_2d_unique.c_in_map.resize(UNIQUE_COUNTER_LEN, 0);
    d_2d_unique.s_in_col.resize(UNIQUE_COUNTER_LEN* pre_map.GRID_WIDTH, 0);

    h_2d_unique.s_map.resize(pre_map.GRID_WIDTH* pre_map.GRID_HEIGHT* NUM_PARTICLES, 0);
    h_2d_unique.c_in_map.resize(UNIQUE_COUNTER_LEN, 0);

    /************************ TRANSITION KERNEL *************************/
    auto start_transition_kernel = std::chrono::high_resolution_clock::now();

    int threadsPerBlock = NUM_PARTICLES;
    int blocksPerGrid = 1;

    kernel_update_particles_states << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles_transition.c_world_body), 
        THRUST_RAW_CAST(d_particles_transition.c_world_lidar), SEP,
        THRUST_RAW_CAST(d_state.c_x), THRUST_RAW_CAST(d_state.c_y), THRUST_RAW_CAST(d_state.c_theta),
        THRUST_RAW_CAST(d_transition.c_body_lidar), NUM_PARTICLES);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = pre_measurements.LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_processed_measure.v_x), THRUST_RAW_CAST(d_processed_measure.v_y), SEP,
        THRUST_RAW_CAST(d_particles_transition.c_world_lidar), THRUST_RAW_CAST(d_measurements.v_lidar_coords), 
        general_info.res, pre_map.xmin, pre_map.ymax, pre_measurements.LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_index_init_const << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_processed_measure.c_idx), pre_measurements.LEN);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, d_processed_measure.c_idx.begin(), d_processed_measure.c_idx.end(), 
        d_processed_measure.c_idx.begin(), 0);

    auto stop_transition_kernel = std::chrono::high_resolution_clock::now();

    h_particles_transition.c_world_body.assign(d_particles_transition.c_world_body.begin(),
        d_particles_transition.c_world_body.end());
    h_particles_transition.c_world_lidar.assign(d_particles_transition.c_world_lidar.begin(),
        d_particles_transition.c_world_lidar.end());
    //h_processed_measure.v_x.assign(d_processed_measure.v_x.begin(), d_processed_measure.v_x.end());
    //h_processed_measure.v_y.assign(d_processed_measure.v_y.begin(), d_processed_measure.v_y.end());
    thrust::copy(d_processed_measure.v_x.begin(), d_processed_measure.v_x.begin() + MEASURE_LEN, h_processed_measure.v_x.begin());
    thrust::copy(d_processed_measure.v_y.begin(), d_processed_measure.v_y.begin() + MEASURE_LEN, h_processed_measure.v_y.begin());
    h_processed_measure.c_idx.assign(d_processed_measure.c_idx.begin(), d_processed_measure.c_idx.end());

    ASSERT_transition_frames(h_particles_transition.c_world_body.data(), h_particles_transition.c_world_lidar.data(), 
        post_particles_transition.c_world_body.data(), post_particles_transition.c_world_lidar.data(),
        NUM_PARTICLES, false, true, false);

    ASSERT_processed_measurements(h_processed_measure.v_x.data(), h_processed_measure.v_y.data(), h_processed_measure.c_idx.data(),
        post_processed_measure.v_x.data(), post_processed_measure.v_y.data(),
        (NUM_PARTICLES* pre_measurements.LEN), pre_measurements.LEN, false, true, false);


    /************************** CREATE 2D MAP ***************************/
    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;

    auto start_create_map = std::chrono::high_resolution_clock::now();
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_2d_unique.s_map), THRUST_RAW_CAST(d_2d_unique.c_in_map), THRUST_RAW_CAST(d_2d_unique.s_in_col), SEP,
        THRUST_RAW_CAST(d_robot_particles.f_x), THRUST_RAW_CAST(d_robot_particles.f_y), THRUST_RAW_CAST(d_robot_particles.c_idx), pre_robot_particles.LEN,
        pre_map.GRID_WIDTH, pre_map.GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_create_map = std::chrono::high_resolution_clock::now();

    //cudaError_t err = cudaPeekAtLastError();
    //printf("%s\n", cudaGetErrorString(err));

    h_2d_unique.s_map.assign(d_2d_unique.s_map.begin(), d_2d_unique.s_map.end());

    ASSERT_create_2d_map_elements(h_2d_unique.s_map.data(), negative_before_counter, pre_map.GRID_WIDTH, pre_map.GRID_HEIGHT, 
        NUM_PARTICLES, pre_robot_particles.LEN, true, false);
    
    ///**************************** UPDATE MAP ****************************/
    auto start_update_map = std::chrono::high_resolution_clock::now();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_update_2d_map_with_measure << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_2d_unique.s_map), THRUST_RAW_CAST(d_2d_unique.c_in_map), THRUST_RAW_CAST(d_2d_unique.s_in_col), SEP,
        THRUST_RAW_CAST(d_processed_measure.v_x), THRUST_RAW_CAST(d_processed_measure.v_y), THRUST_RAW_CAST(d_processed_measure.c_idx),
        MEASURE_LEN, pre_map.GRID_WIDTH, pre_map.GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_update_map = std::chrono::high_resolution_clock::now();

    h_processed_measure.c_idx.assign(d_processed_measure.c_idx.begin(), d_processed_measure.c_idx.end());
    
    ///************************* CUMULATIVE SUM ***************************/
    threadsPerBlock = UNIQUE_COUNTER_LEN;
    blocksPerGrid = 1;

    auto start_cumulative_sum = std::chrono::high_resolution_clock::now();

    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_2d_unique.s_in_col), pre_map.GRID_WIDTH);
    thrust::exclusive_scan(thrust::device, d_2d_unique.c_in_map.begin(), d_2d_unique.c_in_map.end(), d_2d_unique.c_in_map.begin(), 0);
    cudaDeviceSynchronize();

    auto stop_cumulative_sum = std::chrono::high_resolution_clock::now();

    h_2d_unique.c_in_map.assign(d_2d_unique.c_in_map.begin(), d_2d_unique.c_in_map.end());

    h_robot_particles.LEN = h_2d_unique.c_in_map.data()[UNIQUE_COUNTER_LEN - 1];
    printf("\n~~$ PARTICLES_ITEMS_LEN=%d, AF_PARTICLES_ITEMS_LEN=%d\n", h_robot_particles.LEN, post_unique_robot_particles.LEN);
    ASSERT_new_len_calculation(h_robot_particles.LEN, post_unique_robot_particles.LEN, negative_after_resampling_counter);

    ///******************* REINITIALIZE MAP VARIABLES *********************/
    //d_robot_particles.f_x.clear();
    //d_robot_particles.f_x.resize(h_robot_particles.LEN, 0);
    //d_robot_particles.f_y.clear();
    //d_robot_particles.f_y.resize(h_robot_particles.LEN, 0);
    //d_robot_particles.f_extended_idx.clear();
    //d_robot_particles.f_extended_idx.resize(h_robot_particles.LEN, 0);
    thrust::fill(d_robot_particles.f_x.begin(), d_robot_particles.f_x.begin() + h_robot_particles.LEN, 0);
    thrust::fill(d_robot_particles.f_y.begin(), d_robot_particles.f_y.begin() + h_robot_particles.LEN, 0);
    thrust::fill(d_robot_particles.f_extended_idx.begin(), d_robot_particles.f_extended_idx.begin() + h_robot_particles.LEN, 0);

    //h_robot_particles.f_x.clear();
    //h_robot_particles.f_x.resize(h_robot_particles.LEN, 0);
    //h_robot_particles.f_y.clear();
    //h_robot_particles.f_y.resize(h_robot_particles.LEN, 0);
    //h_robot_particles.f_extended_idx.clear();
    //h_robot_particles.f_extended_idx.resize(h_robot_particles.LEN, 0);
    thrust::fill(h_robot_particles.f_x.begin(), h_robot_particles.f_x.begin() + h_robot_particles.LEN, 0);
    thrust::fill(h_robot_particles.f_y.begin(), h_robot_particles.f_y.begin() + h_robot_particles.LEN, 0);
    thrust::fill(h_robot_particles.f_extended_idx.begin(), h_robot_particles.f_extended_idx.begin() + h_robot_particles.LEN, 0);


    ///************************ MAP RESTRUCTURE ***************************/
    threadsPerBlock = pre_map.GRID_WIDTH;
    blocksPerGrid = NUM_PARTICLES;

    auto start_map_restructure = std::chrono::high_resolution_clock::now();

    thrust::fill(d_robot_particles.c_idx.begin(), d_robot_particles.c_idx.end(), 0);

    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_robot_particles.f_x), THRUST_RAW_CAST(d_robot_particles.f_y), THRUST_RAW_CAST(d_robot_particles.c_idx), SEP,
        THRUST_RAW_CAST(d_2d_unique.s_map), THRUST_RAW_CAST(d_2d_unique.c_in_map), THRUST_RAW_CAST(d_2d_unique.s_in_col), 
        pre_map.GRID_WIDTH, pre_map.GRID_HEIGHT);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, d_robot_particles.c_idx.begin(), d_robot_particles.c_idx.end(), d_robot_particles.c_idx.begin(), 0);

    auto stop_map_restructure = std::chrono::high_resolution_clock::now();

    auto start_copy_particles_pos = std::chrono::high_resolution_clock::now();
    //h_robot_particles.f_x.assign(d_robot_particles.f_x.begin(), d_robot_particles.f_x.end());
    //h_robot_particles.f_y.assign(d_robot_particles.f_y.begin(), d_robot_particles.f_y.end());
    thrust::copy(d_robot_particles.f_x.begin(), d_robot_particles.f_x.begin() + h_robot_particles.LEN, h_robot_particles.f_x.begin());
    thrust::copy(d_robot_particles.f_y.begin(), d_robot_particles.f_y.begin() + h_robot_particles.LEN, h_robot_particles.f_y.begin());
    auto stop_copy_particles_pos = std::chrono::high_resolution_clock::now();

    ASSERT_particles_pos_unique(h_robot_particles.f_x.data(), h_robot_particles.f_y.data(), 
        post_unique_robot_particles.f_x.data(), post_unique_robot_particles.f_y.data(), h_robot_particles.LEN, false, false, true);

    ///************************* INDEX EXPANSION **************************/
    auto start_index_expansion = std::chrono::high_resolution_clock::now();

    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;
    
    kernel_index_expansion << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_robot_particles.f_extended_idx), THRUST_RAW_CAST(d_robot_particles.c_idx), h_robot_particles.LEN);
    cudaDeviceSynchronize();

    auto stop_index_expansion = std::chrono::high_resolution_clock::now();

    //h_robot_particles.f_extended_idx.assign(d_robot_particles.f_extended_idx.begin(), d_robot_particles.f_extended_idx.end());
    thrust::copy(d_robot_particles.f_extended_idx.begin(), d_robot_particles.f_extended_idx.begin() + h_robot_particles.LEN, h_robot_particles.f_extended_idx.begin());
    h_robot_particles.c_idx.assign(d_robot_particles.c_idx.begin(), d_robot_particles.c_idx.end());

    /************************ KERNEL CORRELATION ************************/
    threadsPerBlock = 256;
    blocksPerGrid = (h_robot_particles.LEN + threadsPerBlock - 1) / threadsPerBlock;

    auto start_correlation = std::chrono::high_resolution_clock::now();    
    kernel_correlation << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.c_raw), SEP,
        THRUST_RAW_CAST(d_map.s_grid_map), THRUST_RAW_CAST(d_robot_particles.f_x), THRUST_RAW_CAST(d_robot_particles.f_y), 
        THRUST_RAW_CAST(d_robot_particles.f_extended_idx), pre_map.GRID_WIDTH, pre_map.GRID_HEIGHT, h_robot_particles.LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    
    kernel_correlation_max << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.c_weight), THRUST_RAW_CAST(d_correlation.c_raw), NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_correlation = std::chrono::high_resolution_clock::now();

    h_correlation.c_weight.assign(d_correlation.c_weight.begin(), d_correlation.c_weight.end());

    ASSERT_correlation_Equality(h_correlation.c_weight.data(), pre_weights.data(), NUM_PARTICLES, false, true);

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
void host_update_particle_weights(host_vector<float>& pre_weights, host_vector<float>& post_loop_weights) {

    printf("/********************** UPDATE PARTICLE WEIGHTS *********************/\n");

    DeviceCorrelation d_correlation;

    HostCorrelation h_correlation;

    /************************ WEIGHTS VARIABLES *************************/
    d_correlation.c_weight.resize(NUM_PARTICLES, 0);
    d_correlation.c_max.resize(1, 0);
    d_correlation.c_sum_exp.resize(1, 0);

    h_correlation.c_weight.resize(NUM_PARTICLES, 0);
    h_correlation.c_max.resize(1, 0);
    h_correlation.c_sum_exp.resize(1, 0);

    d_correlation.c_weight.assign(pre_weights.begin(), pre_weights.end());

    /********************** UPDATE WEIGHTS KERNEL ***********************/
    auto start_update_particle_weights = std::chrono::high_resolution_clock::now();

    int threadsPerBlock = 1;
    int blocksPerGrid = 1;

    kernel_arr_max << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.c_weight), THRUST_RAW_CAST(d_correlation.c_max), NUM_PARTICLES);
    cudaDeviceSynchronize();

    h_correlation.c_max.assign(d_correlation.c_max.begin(), d_correlation.c_max.end());
    printf("~~$ h_weights_max[0]=%f\n", h_correlation.c_max[0]);

    float norm_value = -h_correlation.c_max[0] + 50;

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_arr_increase << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.c_weight), norm_value, 0);
    cudaDeviceSynchronize();

    threadsPerBlock = 1;
    blocksPerGrid = 1;
    kernel_arr_sum_exp << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.c_sum_exp), THRUST_RAW_CAST(d_correlation.c_weight), NUM_PARTICLES);
    cudaDeviceSynchronize();

    h_correlation.c_sum_exp.assign(d_correlation.c_sum_exp.begin(), d_correlation.c_sum_exp.end());

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_arr_normalize << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.c_weight), h_correlation.c_sum_exp[0]);
    cudaDeviceSynchronize();

    auto stop_update_particle_weights = std::chrono::high_resolution_clock::now();

    h_correlation.c_weight.assign(d_correlation.c_weight.begin(), d_correlation.c_weight.end());

    ASSERT_update_particle_weights(h_correlation.c_weight.data(), post_loop_weights.data(), NUM_PARTICLES, "weights", false, false, true);

    auto duration_update_particle_weights = std::chrono::duration_cast<std::chrono::microseconds>(stop_update_particle_weights - start_update_particle_weights);
    std::cout << "Time taken by function (Update Particle Weights): " << duration_update_particle_weights.count() << " microseconds" << std::endl;
    std::cout << std::endl;
}


// Step 3
void host_resampling(HostMap& pre_map, HostState& pre_state, HostState& post_state, HostResampling& pre_resampling,
    HostRobotParticles& pre_robot_particles, HostRobotParticles& post_resampling_robot_particles,
    host_vector<float>& post_loop_weights) {

    printf("/***************************** RESAMPLING ***************************/\n");

    int negative_before_counter = getNegativeCounter(pre_robot_particles.f_x.data(), pre_robot_particles.f_y.data(), pre_robot_particles.LEN);
    int count_bigger_than_height = getGreaterThanCounter(pre_robot_particles.f_y.data(), pre_map.GRID_HEIGHT, pre_robot_particles.LEN);
    int negative_after_counter = getNegativeCounter(post_resampling_robot_particles.f_x.data(), 
        post_resampling_robot_particles.f_y.data(), post_resampling_robot_particles.LEN);

    printf("~~$ negative_before_counter: \t%d\n", negative_before_counter);
    printf("~~$ negative_after_counter: \t%d\n", negative_after_counter);
    printf("~~$ count_bigger_than_height: \t%d\n", count_bigger_than_height);

    HostResampling h_resampling;

    DeviceResampling d_resampling;
    DeviceCorrelation d_correlation;

    /*********************** RESAMPLING VARIABLES ***********************/
    d_correlation.c_weight.resize(NUM_PARTICLES, 0);
    d_resampling.c_js.resize(NUM_PARTICLES, 0);
    d_resampling.c_rnds.resize(NUM_PARTICLES, 0);

    h_resampling.c_js.resize(NUM_PARTICLES, 0);

    d_correlation.c_weight.assign(post_loop_weights.begin(), post_loop_weights.end());
    d_resampling.c_rnds.assign(pre_resampling.c_rnds.begin(), pre_resampling.c_rnds.end());

    /************************ RESAMPLING kerenel ************************/
    auto start_resampling = std::chrono::high_resolution_clock::now();
    int threadsPerBlock = NUM_PARTICLES;
    int blocksPerGrid = 1;
    
    kernel_resampling << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_resampling.c_js), THRUST_RAW_CAST(d_correlation.c_weight), THRUST_RAW_CAST(d_resampling.c_rnds), NUM_PARTICLES);
    cudaDeviceSynchronize();
    auto stop_resampling = std::chrono::high_resolution_clock::now();

    h_resampling.c_js.assign(d_resampling.c_js.begin(), d_resampling.c_js.end());

    ASSERT_resampling_indices(h_resampling.c_js.data(), pre_resampling.c_js.data(), NUM_PARTICLES, false, true, false);
    ASSERT_resampling_states(pre_state.c_x.data(), pre_state.c_y.data(), pre_state.c_theta.data(),
        post_state.c_x.data(), post_state.c_y.data(), post_state.c_theta.data(), h_resampling.c_js.data(), NUM_PARTICLES, false, true, true);

    auto duration_resampling = std::chrono::duration_cast<std::chrono::microseconds>(stop_resampling - start_resampling);
    std::cout << "Time taken by function (Kernel Resampling): " << duration_resampling.count() << " microseconds" << std::endl;
    std::cout << std::endl;
}


// Step 4
void host_update_state(HostState& post_state, HostRobotState& post_robot_state) {

    printf("/**************************** UPDATE STATE **************************/\n");

    auto start_update_states = std::chrono::high_resolution_clock::now();

    std::vector<float> std_vec_states_x(post_state.c_x.begin(), post_state.c_x.end());
    std::vector<float> std_vec_states_y(post_state.c_y.begin(), post_state.c_y.end());
    std::vector<float> std_vec_states_theta(post_state.c_theta.begin(), post_state.c_theta.end());

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
    float h_transition_world_body[] = { cos(theta), -sin(theta), std::get<0>(key),
                        sin(theta),  cos(theta), std::get<1>(key),
                        0, 0, 1 };

    auto stop_update_states = std::chrono::high_resolution_clock::now();

    printf("\n");
    printf("~~$ Transition World to Body (Result): ");
    for (int i = 0; i < 9; i++) {
        printf("%f ", h_transition_world_body[i]);
    }
    printf("\n");
    printf("~~$ Transition World to Body (Host)  : ");
    for (int i = 0; i < 9; i++) {
        printf("%f ", post_robot_state.transition_world_body[i]);
    }

    printf("\n\n");
    printf("~~$ Robot State (Result): %f, %f, %f\n", std::get<0>(key), std::get<1>(key), std::get<2>(key));
    printf("~~$ Robot State (Host)  : %f, %f, %f\n", post_robot_state.state[0], post_robot_state.state[1], post_robot_state.state[2]);

    std::cout << std::endl;
    auto duration_update_states = std::chrono::duration_cast<std::chrono::microseconds>(stop_update_states - start_update_states);
    std::cout << "Time taken by function (Update States): " << duration_update_states.count() << " microseconds" << std::endl;
    std::cout << std::endl;
}


// Step X
void host_update_func(HostMap& pre_map, HostState& pre_state, HostState& post_state, HostMeasurements& pre_measurements, HostResampling& pre_resampling,
    HostRobotState& post_robot_state, HostProcessedMeasure& post_processed_measure, HostRobotParticles& pre_robot_particles,
    HostRobotParticles& post_unique_robot_particles, HostRobotParticles& post_resampling_robot_particles,
    GeneralInfo& general_info, host_vector<float>& pre_weights, host_vector<float>& post_loop_weights) {

    printf("/**************************** UPDATE FUNC ***************************/\n");

    int MEASURE_LEN = NUM_PARTICLES * pre_measurements.LEN;
    const int UNIQUE_COUNTER_LEN = NUM_PARTICLES + 1;

    int negative_before_counter = getNegativeCounter(pre_robot_particles.f_x.data(), pre_robot_particles.f_y.data(), pre_robot_particles.LEN);
    int count_bigger_than_height = getGreaterThanCounter(pre_robot_particles.f_y.data(), pre_map.GRID_HEIGHT, pre_robot_particles.LEN);
    int negative_after_counter = getNegativeCounter(post_resampling_robot_particles.f_x.data(), post_resampling_robot_particles.f_y.data(), 
        post_resampling_robot_particles.LEN);

    printf("~~$ GRID_WIDTH: \t\t%d\n", pre_map.GRID_WIDTH);
    printf("~~$ GRID_HEIGHT: \t\t%d\n", pre_map.GRID_HEIGHT);
    printf("~~$ negative_before_counter: \t%d\n", negative_before_counter);
    printf("~~$ negative_after_counter: \t%d\n", negative_after_counter);
    printf("~~$ count_bigger_than_height: \t%d\n", count_bigger_than_height);
    printf("~~$ MEASURE_LEN: \t\t%d\n", MEASURE_LEN);

    /**************************************************************** VARIABLES SCOPE *****************************************************************/

    DeviceMap d_map;
    DeviceState d_state;
    DeviceState d_clone_state;
    DeviceMeasurements d_measurements;
    DeviceRobotParticles d_robot_particles;
    DeviceRobotParticles d_clone_robot_particles;
    DeviceCorrelation d_correlation;
    DeviceParticlesTransition d_particles_transition;
    DeviceTransition d_transition;
    DeviceProcessedMeasure d_processed_measure;
    Device2DUniqueFinder d_2d_unique;
    DeviceResampling d_resampling;

    HostState h_state;
    HostRobotParticles h_robot_particles;
    HostRobotParticles h_clone_robot_particles;
    HostCorrelation h_correlation;
    HostParticlesTransition h_particles_transition;
    HostProcessedMeasure h_processed_measure;
    Host2DUniqueFinder h_2d_unique;
    HostResampling h_resampling;
    HostMeasurements h_measurements;


    /************************** STATES VARIABLES ************************/
    h_state.c_x.resize(NUM_PARTICLES, 0);
    h_state.c_y.resize(NUM_PARTICLES, 0);
    h_state.c_theta.resize(NUM_PARTICLES, 0);

    d_state.c_x.resize(NUM_PARTICLES, 0);
    d_state.c_y.resize(NUM_PARTICLES, 0);
    d_state.c_theta.resize(NUM_PARTICLES, 0);

    d_state.c_x.assign(pre_state.c_x.begin(), pre_state.c_x.end());
    d_state.c_y.assign(pre_state.c_y.begin(), pre_state.c_y.end());
    d_state.c_theta.assign(pre_state.c_theta.begin(), pre_state.c_theta.end());

    d_clone_state.c_x.resize(NUM_PARTICLES, 0);
    d_clone_state.c_y.resize(NUM_PARTICLES, 0);
    d_clone_state.c_theta.resize(NUM_PARTICLES, 0);


    d_measurements.v_lidar_coords.resize(2 * pre_measurements.MAX_LEN, 0);
    //d_measurements.v_lidar_coords.assign(pre_measurements.v_lidar_coords.begin(), pre_measurements.v_lidar_coords.end());
    thrust::copy(pre_measurements.v_lidar_coords.begin(), pre_measurements.v_lidar_coords.end(), d_measurements.v_lidar_coords.begin());

    /************************* PARTICLES VARIABLES **********************/
    d_map.s_grid_map.resize(pre_map.GRID_WIDTH * pre_map.GRID_HEIGHT, 0);
    d_map.s_grid_map.assign(pre_map.s_grid_map.begin(), pre_map.s_grid_map.end());


    h_robot_particles.f_x.resize(pre_robot_particles.MAX_LEN, 0);
    h_robot_particles.f_y.resize(pre_robot_particles.MAX_LEN, 0);
    h_robot_particles.c_idx.resize(NUM_PARTICLES, 0);
    h_robot_particles.f_extended_idx.resize(pre_robot_particles.MAX_LEN, 0);
    h_robot_particles.c_weight.resize(NUM_PARTICLES, 0);
    
    d_robot_particles.f_x.resize(pre_robot_particles.MAX_LEN, 0);
    d_robot_particles.f_y.resize(pre_robot_particles.MAX_LEN, 0);
    d_robot_particles.c_idx.resize(NUM_PARTICLES, 0);
    d_robot_particles.f_extended_idx.resize(pre_robot_particles.MAX_LEN, 0);
    d_robot_particles.c_weight.resize(NUM_PARTICLES, 0);
    
    //d_robot_particles.f_x.assign(pre_robot_particles.f_x.begin(), pre_robot_particles.f_x.end());
    //d_robot_particles.f_y.assign(pre_robot_particles.f_y.begin(), pre_robot_particles.f_y.end());
    thrust::copy(pre_robot_particles.f_x.begin(), pre_robot_particles.f_x.end(), d_robot_particles.f_x.begin());
    thrust::copy(pre_robot_particles.f_y.begin(), pre_robot_particles.f_y.end(), d_robot_particles.f_y.begin());
    d_robot_particles.c_idx.assign(pre_robot_particles.c_idx.begin(), pre_robot_particles.c_idx.end());
    
    d_clone_robot_particles.f_x.resize(h_robot_particles.MAX_LEN, 0);
    d_clone_robot_particles.f_y.resize(h_robot_particles.MAX_LEN, 0);
    d_clone_robot_particles.c_idx.resize(NUM_PARTICLES, 0);

    /********************** CORRELATION VARIABLES ***********************/
    //h_correlation.c_weight.resize(NUM_PARTICLES, 0);
    //h_correlation.c_raw.resize(25 * NUM_PARTICLES, 0);

    /*********************** TRANSITION VARIABLES ***********************/
    h_particles_transition.c_world_body.resize(9 * NUM_PARTICLES, 0);
    h_particles_transition.c_world_lidar.resize(9 * NUM_PARTICLES, 0);

    d_particles_transition.c_world_body.resize(9 * NUM_PARTICLES, 0);
    d_particles_transition.c_world_lidar.resize(9 * NUM_PARTICLES, 0);


    h_processed_measure.v_x.resize(NUM_PARTICLES* pre_measurements.MAX_LEN);
    h_processed_measure.v_y.resize(NUM_PARTICLES* pre_measurements.MAX_LEN);
    h_processed_measure.c_idx.resize(NUM_PARTICLES, 0);

    d_processed_measure.v_x.resize(NUM_PARTICLES* pre_measurements.MAX_LEN, 0);
    d_processed_measure.v_y.resize(NUM_PARTICLES* pre_measurements.MAX_LEN, 0);
    d_processed_measure.c_idx.resize(NUM_PARTICLES, 0);

    d_transition.c_body_lidar.resize(9, 0);
    d_transition.c_body_lidar.assign(hvec_transition_body_lidar.begin(), hvec_transition_body_lidar.end());

    /**************************** MAP VARIABLES *************************/
    d_2d_unique.s_map.resize(pre_map.GRID_WIDTH* pre_map.GRID_HEIGHT* NUM_PARTICLES, 0);
    d_2d_unique.c_in_map.resize(UNIQUE_COUNTER_LEN, 0);
    d_2d_unique.s_in_col.resize(UNIQUE_COUNTER_LEN * pre_map.GRID_WIDTH, 0);

    h_2d_unique.s_map.resize(pre_map.GRID_WIDTH* pre_map.GRID_HEIGHT* NUM_PARTICLES, 0);
    h_2d_unique.c_in_map.resize(UNIQUE_COUNTER_LEN, 0);

    /************************ WEIGHTS VARIABLES *************************/
    d_correlation.c_weight.resize(NUM_PARTICLES, 0);
    d_correlation.c_raw.resize(25 * NUM_PARTICLES, 0);
    d_correlation.c_weight.assign(pre_weights.begin(), pre_weights.end());

    h_correlation.c_max.resize(1, 0);
    h_correlation.c_sum_exp.resize(1, 0);

    d_correlation.c_max.resize(1, 0);
    d_correlation.c_sum_exp.resize(1, 0);

    /*********************** RESAMPLING VARIABLES ***********************/
    h_resampling.c_js.resize(NUM_PARTICLES, 0);

    d_resampling.c_js.resize(NUM_PARTICLES, 0);
    d_resampling.c_rnds.resize(NUM_PARTICLES, 0);

    d_resampling.c_rnds.assign(pre_resampling.c_rnds.begin(), pre_resampling.c_rnds.end());

    /************************************************************* KERNEL EXECUTION SCOPE *************************************************************/

    /************************ TRANSITION KERNEL *************************/
    auto start_transition_kernel = std::chrono::high_resolution_clock::now();

    int threadsPerBlock = NUM_PARTICLES;
    int blocksPerGrid = 1;

    kernel_update_particles_states << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles_transition.c_world_body), 
        THRUST_RAW_CAST(d_particles_transition.c_world_lidar), SEP,
        THRUST_RAW_CAST(d_state.c_x), THRUST_RAW_CAST(d_state.c_y), THRUST_RAW_CAST(d_state.c_theta),
        THRUST_RAW_CAST(d_transition.c_body_lidar), NUM_PARTICLES);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = pre_measurements.LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_processed_measure.v_x), THRUST_RAW_CAST(d_processed_measure.v_y), SEP,
        THRUST_RAW_CAST(d_particles_transition.c_world_lidar), THRUST_RAW_CAST(d_measurements.v_lidar_coords), 
        general_info.res, pre_map.xmin, pre_map.ymax, pre_measurements.LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_index_init_const << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_processed_measure.c_idx), pre_measurements.LEN);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, d_processed_measure.c_idx.begin(), 
        d_processed_measure.c_idx.end(), d_processed_measure.c_idx.begin(), 0);

    auto stop_transition_kernel = std::chrono::high_resolution_clock::now();

    h_particles_transition.c_world_body.assign(
        d_particles_transition.c_world_body.begin(), d_particles_transition.c_world_body.end());
    h_particles_transition.c_world_lidar.assign(
        d_particles_transition.c_world_lidar.begin(), d_particles_transition.c_world_lidar.end());
    
    //h_processed_measure.v_x.assign(d_processed_measure.v_x.begin(), d_processed_measure.v_x.end());
    //h_processed_measure.v_y.assign(d_processed_measure.v_y.begin(), d_processed_measure.v_y.end());
    thrust::copy(d_processed_measure.v_x.begin(), d_processed_measure.v_x.begin() + MEASURE_LEN, h_processed_measure.v_x.begin());
    thrust::copy(d_processed_measure.v_y.begin(), d_processed_measure.v_y.begin() + MEASURE_LEN, h_processed_measure.v_y.begin());
    h_processed_measure.c_idx.assign(d_processed_measure.c_idx.begin(), d_processed_measure.c_idx.end());

    //ASSERT_transition_frames(h_transition_world_body, h_transition_world_lidar,
    //    pre_transition_world_body, pre_transition_world_lidar, NUM_PARTICLES, false);
    // ASSERT_processed_measurements(h_processed_measure_x, h_processed_measure_y, processed_measure, NUM_PARTICLES, LEN);

    ASSERT_processed_measurements(h_processed_measure.v_x.data(), h_processed_measure.v_y.data(),
        h_processed_measure.c_idx.data(), post_processed_measure.v_x.data(), post_processed_measure.v_y.data(),
        (NUM_PARTICLES* pre_measurements.LEN), pre_measurements.LEN, false, true, true);


    /************************** CREATE 2D MAP ***************************/
    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;

    auto start_create_map = std::chrono::high_resolution_clock::now();
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_2d_unique.s_map), THRUST_RAW_CAST(d_2d_unique.c_in_map), THRUST_RAW_CAST(d_2d_unique.s_in_col), SEP,
        THRUST_RAW_CAST(d_robot_particles.f_x), THRUST_RAW_CAST(d_robot_particles.f_y), THRUST_RAW_CAST(d_robot_particles.c_idx),
        pre_robot_particles.LEN, pre_map.GRID_WIDTH, pre_map.GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();
    auto stop_create_map = std::chrono::high_resolution_clock::now();

    //cudaError_t err = cudaPeekAtLastError();
    //printf("%s\n", cudaGetErrorString(err));

    h_2d_unique.s_map.assign(d_2d_unique.s_map.begin(), d_2d_unique.s_map.end());

    ASSERT_create_2d_map_elements(h_2d_unique.s_map.data(), negative_before_counter, pre_map.GRID_WIDTH, pre_map.GRID_HEIGHT, NUM_PARTICLES, pre_robot_particles.LEN, true, true);

    /**************************** UPDATE MAP ****************************/
    auto start_update_map = std::chrono::high_resolution_clock::now();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_update_2d_map_with_measure << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_2d_unique.s_map), THRUST_RAW_CAST(d_2d_unique.c_in_map), THRUST_RAW_CAST(d_2d_unique.s_in_col), SEP,
        THRUST_RAW_CAST(d_processed_measure.v_x), THRUST_RAW_CAST(d_processed_measure.v_y), THRUST_RAW_CAST(d_processed_measure.c_idx),
        MEASURE_LEN, pre_map.GRID_WIDTH, pre_map.GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_update_map = std::chrono::high_resolution_clock::now();

    /************************* CUMULATIVE SUM ***************************/
    auto start_cumulative_sum = std::chrono::high_resolution_clock::now();

    threadsPerBlock = UNIQUE_COUNTER_LEN;
    blocksPerGrid = 1;
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_2d_unique.s_in_col), pre_map.GRID_WIDTH);
    thrust::exclusive_scan(thrust::device, d_2d_unique.c_in_map.begin(), d_2d_unique.c_in_map.end(), d_2d_unique.c_in_map.begin(), 0);
    cudaDeviceSynchronize();

    auto stop_cumulative_sum = std::chrono::high_resolution_clock::now();

    h_2d_unique.c_in_map.assign(d_2d_unique.c_in_map.begin(), d_2d_unique.c_in_map.end());

    h_robot_particles.LEN = h_2d_unique.c_in_map[UNIQUE_COUNTER_LEN - 1];
    ASSERT_new_len_calculation(h_robot_particles.LEN, post_unique_robot_particles.LEN, negative_after_counter);


    ///*---------------------------------------------------------------------*/
    ///*-------------------- REINITIALIZE MAP VARIABLES ---------------------*/

    //d_robot_particles.f_x.clear();
    //d_robot_particles.f_x.resize(h_robot_particles.LEN, 0);
    //d_robot_particles.f_y.clear();
    //d_robot_particles.f_y.resize(h_robot_particles.LEN, 0);
    //d_robot_particles.f_extended_idx.clear();
    //d_robot_particles.f_extended_idx.resize(h_robot_particles.LEN, 0);
    thrust::fill(d_robot_particles.f_x.begin(), d_robot_particles.f_x.begin() + h_robot_particles.LEN, 0);
    thrust::fill(d_robot_particles.f_y.begin(), d_robot_particles.f_y.begin() + h_robot_particles.LEN, 0);
    thrust::fill(d_robot_particles.f_extended_idx.begin(), d_robot_particles.f_extended_idx.begin() + h_robot_particles.LEN, 0);

    //h_robot_particles.f_x.clear();
    //h_robot_particles.f_x.resize(h_robot_particles.LEN, 0);
    //h_robot_particles.f_y.clear();
    //h_robot_particles.f_y.resize(h_robot_particles.LEN, 0);
    thrust::fill(h_robot_particles.f_x.begin(), h_robot_particles.f_x.begin() + h_robot_particles.LEN, 0);
    thrust::fill(h_robot_particles.f_y.begin(), h_robot_particles.f_y.begin() + h_robot_particles.LEN, 0);
    thrust::fill(h_robot_particles.f_extended_idx.begin(), h_robot_particles.f_extended_idx.begin() + h_robot_particles.LEN, 0);


    /************************ MAP RESTRUCTURE ***************************/
    threadsPerBlock = pre_map.GRID_WIDTH;
    blocksPerGrid = NUM_PARTICLES;

    auto start_map_restructure = std::chrono::high_resolution_clock::now();

    thrust::fill(d_robot_particles.c_idx.begin(), d_robot_particles.c_idx.end(), 0);

    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_robot_particles.f_x), THRUST_RAW_CAST(d_robot_particles.f_y), THRUST_RAW_CAST(d_robot_particles.c_idx), SEP,
        THRUST_RAW_CAST(d_2d_unique.s_map), THRUST_RAW_CAST(d_2d_unique.c_in_map), 
        THRUST_RAW_CAST(d_2d_unique.s_in_col), pre_map.GRID_WIDTH, pre_map.GRID_HEIGHT);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, d_robot_particles.c_idx.begin(), d_robot_particles.c_idx.end(), d_robot_particles.c_idx.begin(), 0);

    auto stop_map_restructure = std::chrono::high_resolution_clock::now();

    auto start_copy_particles_pos = std::chrono::high_resolution_clock::now();
    //h_robot_particles.f_x.assign(d_robot_particles.f_x.begin(), d_robot_particles.f_x.end());
    //h_robot_particles.f_y.assign(d_robot_particles.f_y.begin(), d_robot_particles.f_y.end());
    thrust::copy(d_robot_particles.f_x.begin(), d_robot_particles.f_x.begin() + h_robot_particles.LEN, h_robot_particles.f_x.begin());
    thrust::copy(d_robot_particles.f_y.begin(), d_robot_particles.f_y.begin() + h_robot_particles.LEN, h_robot_particles.f_y.begin());
    h_robot_particles.c_idx.assign(d_robot_particles.c_idx.begin(), d_robot_particles.c_idx.end());
    auto stop_copy_particles_pos = std::chrono::high_resolution_clock::now();

    ASSERT_particles_pos_unique(h_robot_particles.f_x.data(), h_robot_particles.f_y.data(),
        post_unique_robot_particles.f_x.data(), post_unique_robot_particles.f_y.data(), h_robot_particles.LEN, false, true, true);
    ASSERT_particles_idx_unique(h_robot_particles.c_idx.data(), post_unique_robot_particles.c_idx.data(), negative_after_counter, NUM_PARTICLES, false, true);

    ///************************* INDEX EXPANSION **************************/
    auto start_index_expansion = std::chrono::high_resolution_clock::now();

    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;
    kernel_index_expansion << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_robot_particles.f_extended_idx), THRUST_RAW_CAST(d_robot_particles.c_idx), h_robot_particles.LEN);
    cudaDeviceSynchronize();

    auto stop_index_expansion = std::chrono::high_resolution_clock::now();

    //h_robot_particles.f_extended_idx.clear();
    //h_robot_particles.f_extended_idx.resize(h_robot_particles.LEN, 0);

    //h_robot_particles.f_extended_idx.assign(
    //    d_robot_particles.f_extended_idx.begin(), d_robot_particles.f_extended_idx.end());
    thrust::copy(d_robot_particles.f_extended_idx.begin(), d_robot_particles.f_extended_idx.begin() + h_robot_particles.LEN,
        h_robot_particles.f_extended_idx.begin());
    h_robot_particles.c_idx.assign(d_robot_particles.c_idx.begin(), d_robot_particles.c_idx.end());

    /************************ KERNEL CORRELATION ************************/
    threadsPerBlock = 256;
    blocksPerGrid = (h_robot_particles.LEN + threadsPerBlock - 1) / threadsPerBlock;

    auto start_correlation = std::chrono::high_resolution_clock::now();
    kernel_correlation << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.c_raw), SEP,
        THRUST_RAW_CAST(d_map.s_grid_map), THRUST_RAW_CAST(d_robot_particles.f_x), THRUST_RAW_CAST(d_robot_particles.f_y), 
        THRUST_RAW_CAST(d_robot_particles.f_extended_idx), pre_map.GRID_WIDTH, pre_map.GRID_HEIGHT, h_robot_particles.LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_correlation_max << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.c_weight), THRUST_RAW_CAST(d_correlation.c_raw), NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_correlation = std::chrono::high_resolution_clock::now();

    h_correlation.c_weight.assign(d_correlation.c_weight.begin(), d_correlation.c_weight.end());
    thrust::copy(d_robot_particles.f_extended_idx.begin(), d_robot_particles.f_extended_idx.begin() + h_robot_particles.LEN,
        h_robot_particles.f_extended_idx.begin());
    //h_robot_particles.f_extended_idx.assign(d_robot_particles.f_extended_idx.begin(), d_robot_particles.f_extended_idx.end());

    ASSERT_correlation_Equality(h_correlation.c_weight.data(), pre_weights.data(), NUM_PARTICLES, false, true);

    /********************** UPDATE WEIGHTS KERNEL ***********************/
    auto start_update_particle_weights = std::chrono::high_resolution_clock::now();

    threadsPerBlock = 1;
    blocksPerGrid = 1;

    kernel_arr_max << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.c_weight), THRUST_RAW_CAST(d_correlation.c_max), NUM_PARTICLES);
    cudaDeviceSynchronize();

    h_correlation.c_max.assign(d_correlation.c_max.begin(), d_correlation.c_max.end());

    float norm_value = -h_correlation.c_max[0] + 50;

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_arr_increase << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.c_weight), norm_value, 0);
    cudaDeviceSynchronize();

    threadsPerBlock = 1;
    blocksPerGrid = 1;
    kernel_arr_sum_exp << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.c_sum_exp), THRUST_RAW_CAST(d_correlation.c_weight), NUM_PARTICLES);
    cudaDeviceSynchronize();

    h_correlation.c_sum_exp.assign(d_correlation.c_sum_exp.begin(), d_correlation.c_sum_exp.end());

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_arr_normalize << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.c_weight), h_correlation.c_sum_exp[0]);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_arr_mult << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_robot_particles.c_weight), THRUST_RAW_CAST(d_correlation.c_weight));
    cudaDeviceSynchronize();

    auto stop_update_particle_weights = std::chrono::high_resolution_clock::now();

    h_correlation.c_weight.assign(d_correlation.c_weight.begin(), d_correlation.c_weight.end());
    h_robot_particles.c_weight.assign(d_robot_particles.c_weight.begin(), d_robot_particles.c_weight.end());

    ASSERT_update_particle_weights(h_correlation.c_weight.data(), post_loop_weights.data(), NUM_PARTICLES, "weights", false, false, true);
    //ASSERT_update_particle_weights(h_robot_particles.c_weight.data(), post_unique_robot_particles.c_weight.data(), NUM_PARTICLES, "particles weight", true, false, true);


    ///************************ RESAMPLING KERNEL *************************/
    auto start_resampling = std::chrono::high_resolution_clock::now();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    
    kernel_resampling << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_resampling.c_js), THRUST_RAW_CAST(d_correlation.c_weight), 
        THRUST_RAW_CAST(d_resampling.c_rnds), NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_resampling = std::chrono::high_resolution_clock::now();

    h_resampling.c_js.assign(d_resampling.c_js.begin(), d_resampling.c_js.end());

    ASSERT_resampling_indices(h_resampling.c_js.data(), pre_resampling.c_js.data(), NUM_PARTICLES, false, false, true);
    ASSERT_resampling_states(pre_state.c_x.data(), pre_state.c_y.data(), pre_state.c_theta.data(),
        post_state.c_x.data(), post_state.c_y.data(), post_state.c_theta.data(), h_resampling.c_js.data(), NUM_PARTICLES, false, true, true);


    /*---------------------------------------------------------------------*/
    /*----------------- REINITIALIZE PARTICLES VARIABLES ------------------*/


    int* d_last_len;
    int* h_last_len = (int*)malloc(sizeof(int));
    gpuErrchk(cudaMalloc((void**)&d_last_len, sizeof(int)));

    auto start_clone_particles = std::chrono::high_resolution_clock::now();

    //d_clone_robot_particles.f_x.assign(d_robot_particles.f_x.begin(), d_robot_particles.f_x.end());
    //d_clone_robot_particles.f_y.assign(d_robot_particles.f_y.begin(), d_robot_particles.f_y.end());
    thrust::copy(d_robot_particles.f_x.begin(), d_robot_particles.f_x.begin() + h_robot_particles.LEN, d_clone_robot_particles.f_x.begin());
    thrust::copy(d_robot_particles.f_y.begin(), d_robot_particles.f_y.begin() + h_robot_particles.LEN, d_clone_robot_particles.f_y.begin());
    d_clone_robot_particles.c_idx.assign(d_robot_particles.c_idx.begin(), d_robot_particles.c_idx.end());

    d_clone_state.c_x.assign(d_state.c_x.begin(), d_state.c_x.end());
    d_clone_state.c_y.assign(d_state.c_y.begin(), d_state.c_y.end());
    d_clone_state.c_theta.assign(d_state.c_theta.begin(), d_state.c_theta.end());

    auto stop_clone_particles = std::chrono::high_resolution_clock::now();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_rearrange_indecies << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_robot_particles.c_idx), d_last_len, SEP,
        THRUST_RAW_CAST(d_clone_robot_particles.c_idx), THRUST_RAW_CAST(d_resampling.c_js), h_robot_particles.LEN);
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(h_last_len, d_last_len, sizeof(int), cudaMemcpyDeviceToHost));
    h_robot_particles.c_idx.assign(d_robot_particles.c_idx.begin(), d_robot_particles.c_idx.end());

    /********************** REARRANGEMENT KERNEL ************************/
    auto start_rearrange_index = std::chrono::high_resolution_clock::now();
    thrust::exclusive_scan(thrust::device, d_robot_particles.c_idx.begin(), d_robot_particles.c_idx.end(), d_robot_particles.c_idx.begin(), 0);
    auto stop_rearrange_index = std::chrono::high_resolution_clock::now();

    h_robot_particles.c_idx.assign(d_robot_particles.c_idx.begin(), d_robot_particles.c_idx.end());
    
    h_clone_robot_particles.LEN = h_robot_particles.LEN;
    h_robot_particles.LEN = h_robot_particles.c_idx[NUM_PARTICLES - 1] + h_last_len[0];

    printf("--> PARTICLES_ITEMS_LEN=%d <> ELEMS_PARTICLES_AFTER=%d\n", h_robot_particles.LEN, 
        post_resampling_robot_particles.LEN);
    assert(h_robot_particles.LEN + negative_after_counter == post_resampling_robot_particles.LEN);

    //h_robot_particles.f_x.clear();
    //h_robot_particles.f_y.clear();
    //h_robot_particles.f_x.resize(h_robot_particles.LEN, 0);
    //h_robot_particles.f_y.resize(h_robot_particles.LEN, 0);
    thrust::fill(h_robot_particles.f_x.begin(), h_robot_particles.f_x.begin() + h_robot_particles.LEN, 0);
    thrust::fill(h_robot_particles.f_y.begin(), h_robot_particles.f_y.begin() + h_robot_particles.LEN, 0);

    ASSERT_resampling_particles_index(post_resampling_robot_particles.c_idx.data(), h_robot_particles.c_idx.data(), NUM_PARTICLES, false, negative_after_counter);

    auto start_rearrange_particles_states = std::chrono::high_resolution_clock::now();
    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;
    
    kernel_rearrange_particles << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_robot_particles.f_x), THRUST_RAW_CAST(d_robot_particles.f_y), SEP,
        THRUST_RAW_CAST(d_robot_particles.c_idx), THRUST_RAW_CAST(d_clone_robot_particles.f_x), THRUST_RAW_CAST(d_clone_robot_particles.f_y), 
        THRUST_RAW_CAST(d_clone_robot_particles.c_idx), THRUST_RAW_CAST(d_resampling.c_js),
        pre_map.GRID_WIDTH, pre_map.GRID_HEIGHT, NUM_PARTICLES, h_robot_particles.LEN, h_clone_robot_particles.LEN);
    kernel_rearrange_states << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_state.c_x), THRUST_RAW_CAST(d_state.c_y), THRUST_RAW_CAST(d_state.c_theta), SEP,
        THRUST_RAW_CAST(d_clone_state.c_x), THRUST_RAW_CAST(d_clone_state.c_y), THRUST_RAW_CAST(d_clone_state.c_theta), 
        THRUST_RAW_CAST(d_resampling.c_js));
    cudaDeviceSynchronize();
    auto stop_rearrange_particles_states = std::chrono::high_resolution_clock::now();

    //h_robot_particles.f_x.assign(d_robot_particles.f_x.begin(), d_robot_particles.f_x.end());
    //h_robot_particles.f_y.assign(d_robot_particles.f_y.begin(), d_robot_particles.f_y.end());
    thrust::copy(d_robot_particles.f_x.begin(), d_robot_particles.f_x.begin() + h_robot_particles.LEN, h_robot_particles.f_x.begin());
    thrust::copy(d_robot_particles.f_y.begin(), d_robot_particles.f_y.begin() + h_robot_particles.LEN, h_robot_particles.f_y.begin());

    h_state.c_x.assign(d_state.c_x.begin(), d_state.c_x.end());
    h_state.c_y.assign(d_state.c_y.begin(), d_state.c_y.end());
    h_state.c_theta.assign(d_state.c_theta.begin(), d_state.c_theta.end());

    ASSERT_rearrange_particles_states(h_robot_particles.f_x.data(), h_robot_particles.f_y.data(), 
        h_state.c_x.data(), h_state.c_y.data(), h_state.c_theta.data(),
        post_resampling_robot_particles.f_x.data(), post_resampling_robot_particles.f_y.data(), 
        post_state.c_x.data(), post_state.c_y.data(), post_state.c_theta.data(),
        h_robot_particles.LEN, NUM_PARTICLES);


    ///********************** UPDATE STATES KERNEL ************************/
    auto start_update_states = std::chrono::high_resolution_clock::now();

    //std::vector<float> std_vec_states_x(post_state.c_x.begin(), post_state.c_x.end());
    //std::vector<float> std_vec_states_y(post_state.c_y.begin(), post_state.c_y.end());
    //std::vector<float> std_vec_states_theta(post_state.c_theta.begin(), post_state.c_theta.end());

    std::vector<float> std_vec_states_x(h_state.c_x.begin(), h_state.c_x.end());
    std::vector<float> std_vec_states_y(h_state.c_y.begin(), h_state.c_y.end());
    std::vector<float> std_vec_states_theta(h_state.c_theta.begin(), h_state.c_theta.end());

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
    float h_transition_world_body[] = { cos(theta), -sin(theta), std::get<0>(key),
                        sin(theta),  cos(theta), std::get<1>(key),
                        0, 0, 1 };

    auto stop_update_states = std::chrono::high_resolution_clock::now();

    printf("\n");
    printf("~~$ Transition World to Body (Result): ");
    for (int i = 0; i < 9; i++) {
        printf("%f ", h_transition_world_body[i]);
    }
    printf("\n");
    printf("~~$ Transition World to Body (Host)  : ");
    for (int i = 0; i < 9; i++) {
        printf("%f ", post_robot_state.transition_world_body[i]);
    }

    printf("\n\n");
    printf("~~$ Robot State (Result): %f, %f, %f\n", std::get<0>(key), std::get<1>(key), std::get<2>(key));
    printf("~~$ Robot State (Host)  : %f, %f, %f\n", post_robot_state.state[0], post_robot_state.state[1], post_robot_state.state[2]);


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


void test_robot_particles_main(HostMap& pre_map, HostState& pre_state, HostState& post_state, HostMeasurements& pre_measurements,
    HostParticlesTransition& post_particles_transition, HostResampling& pre_resampling, HostRobotState& post_robot_state,
    HostRobotParticles& pre_robot_particles, HostRobotParticles& post_unique_robot_particles, HostRobotParticles& post_resampling_robot_particles,
    HostProcessedMeasure& post_processed_measure, GeneralInfo& general_info, host_vector<float>& pre_weights, 
    host_vector<float>& post_loop_weights) {

    printf("/****************************** ROBOT  ******************************/\n");

    int MEASURE_LEN = NUM_PARTICLES * pre_measurements.LEN;

    int negative_before_counter = getNegativeCounter(pre_robot_particles.f_x.data(), pre_robot_particles.f_y.data(), pre_robot_particles.LEN);
    int count_bigger_than_height = getGreaterThanCounter(pre_robot_particles.f_y.data(), pre_map.GRID_HEIGHT, pre_robot_particles.LEN);
    int negative_after_counter = getNegativeCounter(post_resampling_robot_particles.f_x.data(), post_resampling_robot_particles.f_y.data(), post_resampling_robot_particles.LEN);

    printf("~~$ GRID_WIDTH: \t\t%d\n", pre_map.GRID_WIDTH);
    printf("~~$ GRID_HEIGHT: \t\t%d\n", pre_map.GRID_HEIGHT);
    printf("~~$ negative_before_counter: \t%d\n", negative_before_counter);
    printf("~~$ negative_after_counter: \t%d\n", negative_after_counter);
    printf("~~$ count_bigger_than_height: \t%d\n", count_bigger_than_height);

    DeviceState d_state;
    DeviceState d_clone_state;
    DeviceMeasurements d_measurements;
    DeviceMap d_map;
    DeviceRobotParticles d_robot_particles;
    DeviceRobotParticles d_clone_robot_particles;
    DeviceCorrelation d_correlation;
    DevicePosition d_position;
    DeviceTransition d_transition;
    DeviceParticlesTransition d_particles_transition;
    DeviceParticlesPosition d_particles_position;
    DeviceParticlesRotation d_particles_rotation;
    DeviceProcessedMeasure d_processed_measure;
    Device2DUniqueFinder d_2d_unique;
    DeviceResampling d_resampling;

    HostState h_state;
    HostMeasurements h_measurements;
    HostMap h_map;
    HostRobotParticles h_robot_particles;
    HostRobotParticles h_clone_robot_particles;
    HostCorrelation h_correlation;
    HostPosition h_position;
    HostTransition h_transition;
    HostRobotState h_robot_state;
    HostParticlesTransition h_particles_transition;
    HostParticlesPosition h_particles_position;
    HostParticlesRotation h_particles_rotation;
    Host2DUniqueFinder h_2d_unique;
    HostProcessedMeasure h_processed_measure;
    HostResampling h_resampling;

    auto start_robot_particles_alloc = std::chrono::high_resolution_clock::now();

    auto start_init_state = std::chrono::high_resolution_clock::now();
    alloc_init_state_vars(d_state, d_clone_state, h_state, h_robot_state, pre_state);
    auto stop_init_state = std::chrono::high_resolution_clock::now();

    auto start_init_measurements = std::chrono::high_resolution_clock::now();
    alloc_init_measurement_vars(d_measurements, h_measurements, pre_measurements);
    auto stop_init_measurements = std::chrono::high_resolution_clock::now();

    auto start_init_map = std::chrono::high_resolution_clock::now();
    alloc_init_map_vars(d_map, h_map, pre_map);
    auto stop_init_map = std::chrono::high_resolution_clock::now();

    auto start_init_particles = std::chrono::high_resolution_clock::now();
    alloc_init_robot_particles_vars(d_robot_particles, d_clone_robot_particles, h_robot_particles, pre_robot_particles);
    auto stop_init_particles = std::chrono::high_resolution_clock::now();

    auto start_init_correlation = std::chrono::high_resolution_clock::now();
    alloc_correlation_vars(d_correlation, h_correlation);
    auto stop_init_correlation = std::chrono::high_resolution_clock::now();

    auto start_init_particles_transition = std::chrono::high_resolution_clock::now();
    alloc_particles_transition_vars(d_particles_transition, d_particles_position, d_particles_rotation, 
        h_particles_transition, h_particles_position, h_particles_rotation);
    auto stop_init_particles_transition = std::chrono::high_resolution_clock::now();

    auto start_init_transition = std::chrono::high_resolution_clock::now();
    //alloc_init_transition_vars(d_position, d_transition, h_position, h_transition, pre_position, pre_transition);
    alloc_init_body_lidar(d_transition);
    auto stop_init_transition = std::chrono::high_resolution_clock::now();

    auto start_init_processed_measurement = std::chrono::high_resolution_clock::now();
    alloc_init_processed_measurement_vars(d_processed_measure, h_processed_measure, h_measurements);
    auto stop_init_processed_measurement = std::chrono::high_resolution_clock::now();

    auto start_init_map_2d = std::chrono::high_resolution_clock::now();
    alloc_map_2d_var(d_2d_unique, h_2d_unique, h_map, true);
    auto stop_init_map_2d = std::chrono::high_resolution_clock::now();

    auto start_init_resampling = std::chrono::high_resolution_clock::now();
    alloc_resampling_vars(d_resampling, h_resampling, pre_resampling);
    auto stop_init_resampling = std::chrono::high_resolution_clock::now();

    auto stop_robot_particles_alloc = std::chrono::high_resolution_clock::now();

    int* h_last_len = (int*)malloc(sizeof(int));
    bool should_assert = true;

    auto start_robot_particles_kernel = std::chrono::high_resolution_clock::now();
    exec_calc_transition(d_particles_transition, d_state, d_transition, h_particles_transition);
    exec_process_measurements(d_processed_measure, d_particles_transition, d_measurements, h_map, h_measurements, general_info);
    if(should_assert == true) 
        assert_processed_measures(d_particles_transition, d_processed_measure, h_particles_transition, 
            h_measurements, h_processed_measure, post_processed_measure);
    
    exec_create_2d_map(d_2d_unique, d_robot_particles, h_map, h_robot_particles);
    if (should_assert == true) assert_create_2d_map(d_2d_unique, h_2d_unique, h_map, h_robot_particles, negative_before_counter);
    
    exec_update_map(d_2d_unique, d_processed_measure, h_map, MEASURE_LEN);
    exec_particle_unique_cum_sum(d_2d_unique, h_map, h_2d_unique, h_robot_particles);
    if (should_assert == true) assert_particles_unique(h_robot_particles, post_unique_robot_particles, negative_after_counter);
    
    reinit_map_vars(d_robot_particles, h_robot_particles);
    exec_map_restructure(d_robot_particles, d_2d_unique, h_map);
    if (should_assert == true) assert_particles_unique(d_robot_particles, h_robot_particles, post_unique_robot_particles, negative_after_counter);
    
    exec_index_expansion(d_robot_particles, h_robot_particles);
    exec_correlation(d_map, d_robot_particles, d_correlation, h_map, h_robot_particles);
    if (should_assert == true)
        assert_correlation(d_correlation, d_robot_particles, h_correlation, h_robot_particles, pre_weights);
    
    exec_update_weights(d_robot_particles, d_correlation, h_robot_particles, h_correlation);
    if (should_assert == true) 
        assert_update_weights(d_correlation, d_robot_particles,h_correlation, h_robot_particles, post_loop_weights);
    
    exec_resampling(d_correlation, d_resampling);
    reinit_particles_vars(d_state, d_robot_particles, d_resampling, d_clone_robot_particles, d_clone_state, h_robot_particles,
        h_state, h_last_len);
    if (should_assert == true) assert_resampling(d_resampling, h_resampling, pre_resampling, pre_state, post_state);
    
    exec_rearrangement(d_robot_particles, d_state, d_resampling, d_clone_robot_particles, d_clone_state, h_map,
        h_robot_particles, h_clone_robot_particles, h_last_len);

    if (should_assert == true) 
        assert_rearrangment(d_robot_particles, d_state, h_robot_particles, post_resampling_robot_particles, h_state, post_state);

    exec_update_states(d_state, h_state, h_robot_state);
    auto stop_robot_particles_kernel = std::chrono::high_resolution_clock::now();

    assert_robot_final_results(d_robot_particles, d_correlation, h_robot_particles, h_correlation, h_robot_state,
        post_resampling_robot_particles, post_robot_state, post_unique_robot_particles, post_loop_weights, negative_after_counter);



    auto duration_init_state = std::chrono::duration_cast<std::chrono::microseconds>(stop_init_state - start_init_state);
    auto duration_init_measurements = std::chrono::duration_cast<std::chrono::microseconds>(stop_init_measurements - start_init_measurements);
    auto duration_init_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_init_map - start_init_map);
    auto duration_init_particles = std::chrono::duration_cast<std::chrono::microseconds>(stop_init_particles - start_init_particles);
    auto duration_init_correlation = std::chrono::duration_cast<std::chrono::microseconds>(stop_init_correlation - start_init_correlation);
    auto duration_init_particles_transition = std::chrono::duration_cast<std::chrono::microseconds>(stop_init_particles_transition - start_init_particles_transition);
    auto duration_init_transition = std::chrono::duration_cast<std::chrono::microseconds>(stop_init_transition - start_init_transition);
    auto duration_init_processed_measurement = std::chrono::duration_cast<std::chrono::microseconds>(stop_init_processed_measurement - start_init_processed_measurement);
    auto duration_init_map_2d = std::chrono::duration_cast<std::chrono::microseconds>(stop_init_map_2d - start_init_map_2d);
    auto duration_init_resampling = std::chrono::duration_cast<std::chrono::microseconds>(stop_init_resampling - start_init_resampling);

    std::cout << "Time taken by function (Alloc State): " << duration_init_state.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Alloc Measurements): " << duration_init_measurements.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Alloc Map): " << duration_init_map.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Alloc Particles): " << duration_init_particles.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Alloc Correlation): " << duration_init_correlation.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Alloc Particles Transition): " << duration_init_particles_transition.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Alloc Transition): " << duration_init_transition.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Alloc Processed Measurement): " << duration_init_processed_measurement.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Alloc Map 2D): " << duration_init_map_2d.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Alloc Resampling): " << duration_init_resampling.count() << " microseconds" << std::endl;

    auto duration_robot_particles_alloc = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_particles_alloc - start_robot_particles_alloc);
    auto duration_robot_particles_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_particles_kernel - start_robot_particles_kernel);
    auto duration_robot_particles_total = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_particles_kernel - start_robot_particles_alloc);

    std::cout << std::endl;
    std::cout << "Time taken by function (Allocation): " << duration_robot_particles_alloc.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Kernel): " << duration_robot_particles_kernel.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Total): " << duration_robot_particles_total.count() << " microseconds" << std::endl;
}

#endif
