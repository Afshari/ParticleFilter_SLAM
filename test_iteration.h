#ifndef _TEST_ITERATION_H_
#define _TEST_ITERATION_H_

#include "headers.h"
#include "host_asserts.h"
#include "host_utils.h"
#include "kernels.cuh"
#include "kernels_robot.cuh"
#include "kernels_map.cuh"
#include "kernels_utils.cuh"
#include "device_init_common.h"
#include "device_init_map.h"
#include "device_init_robot.h"
#include "device_exec_robot.h"
#include "device_exec_map.h"
#include "device_assert_robot.h"
#include "device_assert_map.h"

const int UNIQUE_COUNTER_LEN = NUM_PARTICLES + 1;

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

void test_setup() {

    const int data_len = 6;
    int data[data_len] = { 1, 0, 2, 2, 1, 3 };
    int* d_data = NULL;
    gpuErrchk(cudaMalloc((void**)&d_data, data_len * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_data, data, data_len * sizeof(int), cudaMemcpyHostToDevice));

    thrust::exclusive_scan(thrust::host, data, data + 6, data, 0);
    thrust::exclusive_scan(thrust::device, d_data, d_data + 6, d_data, 0);
}


host_vector<int> hvec_occupied_map_idx(2, 0);
host_vector<int> hvec_free_map_idx(2, 0);

DeviceMap d_map;
DeviceState d_state;
DeviceState d_clone_state;
DeviceMeasurements d_measurements;
DeviceParticles d_particles;
DeviceRobotParticles d_robot_particles;
DeviceRobotParticles d_clone_robot_particles;
DeviceParticlesTransition d_particles_transition;
DeviceParticlesPosition d_particles_position;
DeviceParticlesRotation d_particles_rotation;
DeviceCorrelation d_correlation;
DevicePosition d_position;
DeviceTransition d_transition;
DeviceProcessedMeasure d_processed_measure;
DeviceResampling d_resampling;
Device2DUniqueFinder d_2d_unique;
Device2DUniqueFinder d_unique_occupied;
Device2DUniqueFinder d_unique_free;

HostMap res_map;
HostState res_state;
HostRobotState res_robot_state;
HostMeasurements res_measurements;
HostParticles res_particles;
HostRobotParticles res_robot_particles;
HostRobotParticles res_clone_robot_particles;
HostParticlesTransition res_particles_transition;
HostParticlesPosition res_particles_position;
HostParticlesRotation res_particles_rotation;
HostCorrelation res_correlation;
HostPosition res_position;
HostTransition res_transition;
HostProcessedMeasure res_processed_measure;
HostResampling res_resampling;
Host2DUniqueFinder res_2d_unique;
Host2DUniqueFinder res_unique_occupied;
Host2DUniqueFinder res_unique_free;

host_vector<float> weights_pre; 
host_vector<float> weights_new; 
host_vector<float> weights_updated;

HostMap h_map;
HostMap h_map_bg;
HostMap h_map_post;
HostState h_state;
HostState post_state;
HostState h_state_updated;
HostRobotState h_robot_state;
HostMeasurements h_measurements;
HostProcessedMeasure h_processed_measure;
HostParticles h_particles;
HostRobotParticles h_robot_particles;
HostRobotParticles h_robot_particles_unique;
HostRobotParticles h_robot_particles_before_resampling;
HostRobotParticles h_robot_particles_after_resampling;
HostPosition h_position;
HostTransition h_transition;
HostParticlesPosition h_particles_position;
HostParticlesTransition h_particles_transition;
HostResampling h_resampling;
GeneralInfo general_info;

bool should_assert = true;

void test_allocation_initialization() {

    printf("/********************************************************************/\n");
    printf("/****************** ALLOCATIONS & INITIALIZATIONS  ******************/\n");
    printf("/********************************************************************/\n");

    //printf("~~$ GRID_WIDTH: \t\t%d\n", GRID_WIDTH);
    //printf("~~$ GRID_HEIGHT: \t\t%d\n", GRID_HEIGHT);

    //printf("~~$ PARTICLES_OCCUPIED_LEN = \t%d\n", PARTICLES_OCCUPIED_LEN);
    //printf("~~$ PARTICLE_UNIQUE_COUNTER = \t%d\n", PARTICLE_UNIQUE_COUNTER);
    //printf("~~$ MAX_DIST_IN_MAP = \t\t%d\n", MAX_DIST_IN_MAP);

    read_iteration(400, h_state, post_state, h_state_updated, h_particles,
        h_robot_particles, h_robot_particles_unique,
        h_robot_particles_before_resampling, h_robot_particles_after_resampling,
        h_resampling, h_robot_state,
        h_map, h_map_bg, h_map_post,
        h_position, h_transition,
        h_particles_position, h_particles_transition,
        h_measurements, h_processed_measure, general_info,
        weights_pre, weights_new, weights_updated);

    alloc_init_state_vars(d_state, d_clone_state, res_state, res_robot_state, h_state);

    auto start_robot_particles_alloc = std::chrono::high_resolution_clock::now();

    alloc_init_measurement_vars(d_measurements, res_measurements, h_measurements);
    alloc_init_map_vars(d_map, res_map, h_map);
    alloc_init_robot_particles_vars(d_robot_particles, res_robot_particles, h_robot_particles);
    alloc_correlation_vars(d_correlation, res_correlation);
    alloc_particles_transition_vars(d_particles_transition, d_particles_position, d_particles_rotation,
        res_particles_transition, res_particles_position, res_particles_rotation);
    alloc_init_transition_vars(d_position, d_transition, res_position, res_transition, h_position, h_transition);
    alloc_init_body_lidar(d_transition);
    alloc_init_processed_measurement_vars(d_processed_measure, res_processed_measure, res_measurements);
    alloc_map_2d_var(d_2d_unique, res_2d_unique, res_map);
    alloc_resampling_vars(d_resampling, res_resampling, h_resampling);

    auto stop_robot_particles_alloc = std::chrono::high_resolution_clock::now();

    auto start_mapping_alloc = std::chrono::high_resolution_clock::now();
    //alloc_init_transition_vars(d_position, d_transition, res_position, res_transition, h_position, h_transition);
    //alloc_init_measurement_vars(d_measurements, res_measurements, h_measurements);

    int MAX_DIST_IN_MAP = sqrt(pow(h_map.GRID_WIDTH, 2) + pow(h_map.GRID_HEIGHT, 2));

    alloc_init_particles_vars(d_particles, res_particles, h_measurements, h_particles, MAX_DIST_IN_MAP);
    alloc_init_map_vars(d_map, res_map, h_map);

    hvec_occupied_map_idx[1] = res_particles.PARTICLES_OCCUPIED_LEN;
    hvec_free_map_idx[1] = 0;
    alloc_init_unique_map_vars(d_unique_occupied, res_unique_occupied, res_map, hvec_occupied_map_idx);
    alloc_init_unique_map_vars(d_unique_free, res_unique_free, res_map, hvec_free_map_idx);

    auto stop_mapping_alloc = std::chrono::high_resolution_clock::now();

    auto duration_robot_particles_alloc = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_particles_alloc - start_robot_particles_alloc);
    auto duration_mapping_alloc = std::chrono::duration_cast<std::chrono::microseconds>(stop_mapping_alloc - start_mapping_alloc);

    std::cout << "Time taken by function (Particles Allocation): " << duration_robot_particles_alloc.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Mapping Allocation): " << duration_mapping_alloc.count() << " microseconds" << std::endl;
}

void test_robot_advance() {

    printf("/********************************************************************/\n");
    printf("/************************** ROBOT ADVANCE ***************************/\n");
    printf("/********************************************************************/\n");

    std::cout << "Start Robot Advance" << std::endl;
        
    auto start_robot_advance_kernel = std::chrono::high_resolution_clock::now();
    exec_robot_move(d_state, res_state);
    auto stop_robot_advance_kernel = std::chrono::high_resolution_clock::now();
    
    assert_robot_move_results(d_state, res_state, post_state);
    
    auto duration_robot_advance_total = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_advance_kernel - start_robot_advance_kernel);
    std::cout << std::endl;
    std::cout << "Time taken by function (Robot Advance Kernel): " << duration_robot_advance_total.count() << " microseconds" << std::endl;
}

void test_robot() {

    printf("/********************************************************************/\n");
    printf("/****************************** ROBOT  ******************************/\n");
    printf("/********************************************************************/\n");


    int negative_before_counter = getNegativeCounter(h_robot_particles.x.data(), h_robot_particles.y.data(), h_robot_particles.LEN);
    int count_bigger_than_height = getGreaterThanCounter(h_robot_particles.y.data(), h_map.GRID_HEIGHT, h_robot_particles.LEN);
    int negative_after_counter = getNegativeCounter(h_robot_particles_after_resampling.x.data(), h_robot_particles_after_resampling.y.data(), h_robot_particles_after_resampling.LEN);

    printf("~~$ negative_before_counter: \t%d\n", negative_before_counter);
    printf("~~$ negative_after_counter: \t%d\n", negative_after_counter);
    printf("~~$ count_bigger_than_height: \t%d\n", count_bigger_than_height);

    const int MEASURE_LEN = NUM_PARTICLES * res_measurements.LIDAR_COORDS_LEN;

    int* res_last_len = (int*)malloc(sizeof(int));

    auto start_robot_particles_kernel = std::chrono::high_resolution_clock::now();
    exec_calc_transition(d_particles_transition, d_state, d_transition, res_particles_transition);
    exec_process_measurements(d_processed_measure, d_particles_transition, d_measurements, res_map, res_measurements, general_info);
    if (should_assert == true)
        assert_processed_measures(d_particles_transition, d_processed_measure, res_particles_transition,
            res_measurements, res_processed_measure, h_processed_measure);

    exec_create_2d_map(d_2d_unique, d_robot_particles, res_map, res_robot_particles);
    if (should_assert == true) assert_create_2d_map(d_2d_unique, res_2d_unique, res_map, res_robot_particles, negative_before_counter);

    exec_update_map(d_2d_unique, d_processed_measure, res_map, MEASURE_LEN);
    exec_particle_unique_cum_sum(d_2d_unique, res_map, res_2d_unique, res_robot_particles);
    if (should_assert == true) assert_particles_unique(res_robot_particles, h_robot_particles_unique, negative_after_counter);

    reinit_map_vars(d_robot_particles, res_robot_particles);
    exec_map_restructure(d_robot_particles, d_2d_unique, res_map);
    if (should_assert == true) assert_particles_unique(d_robot_particles, res_robot_particles, h_robot_particles_unique, negative_after_counter);

    exec_index_expansion(d_robot_particles, res_robot_particles);
    exec_correlation(d_map, d_robot_particles, d_correlation, res_map, res_robot_particles);
    if (should_assert == true) assert_correlation(d_correlation, d_robot_particles, res_correlation, res_robot_particles, weights_pre);

    exec_update_weights(d_robot_particles, d_correlation, res_robot_particles, res_correlation);
    if (should_assert == true) assert_update_weights(d_correlation, d_robot_particles, res_correlation,
        res_robot_particles, weights_new);

    exec_resampling(d_correlation, d_resampling);
    reinit_particles_vars(d_state, d_robot_particles, d_resampling, d_clone_robot_particles, d_clone_state, res_robot_particles,
        res_state, res_last_len);
    if (should_assert == true) assert_resampling(d_resampling, res_resampling, h_resampling, post_state, h_state_updated);

    exec_rearrangement(d_robot_particles, d_state, d_resampling, d_clone_robot_particles, d_clone_state, res_map,
        res_robot_particles, res_clone_robot_particles, res_last_len);
    exec_update_states(d_state, res_state, res_robot_state);

    auto stop_robot_particles_kernel = std::chrono::high_resolution_clock::now();

    assert_robot_final_results(d_robot_particles, d_correlation, res_robot_particles, res_correlation, res_robot_state,
        h_robot_particles_after_resampling, h_robot_state, h_robot_particles_unique, weights_new, negative_after_counter);
    
    auto duration_robot_particles_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_particles_kernel - start_robot_particles_kernel);

    std::cout << std::endl;
    std::cout << "Time taken by function (Kernel): " << duration_robot_particles_kernel.count() << " microseconds" << std::endl;
}

void test_map() {

    printf("/********************************************************************/\n");
    printf("/****************************** MAP MAIN ****************************/\n");
    printf("/********************************************************************/\n");

    const int MEASURE_LEN = NUM_PARTICLES * res_measurements.LIDAR_COORDS_LEN;
    int MAX_DIST_IN_MAP = sqrt(pow(h_map.GRID_WIDTH, 2) + pow(h_map.GRID_HEIGHT, 2));

    auto start_mapping_kernel = std::chrono::high_resolution_clock::now();

    auto start_world_to_image_transform_1 = std::chrono::high_resolution_clock::now();
    exec_world_to_image_transform_step_1(d_position, d_transition, d_particles, d_measurements, res_measurements);
    auto stop_world_to_image_transform_1 = std::chrono::high_resolution_clock::now();

    bool EXTEND = false;
    auto start_check_extend = std::chrono::high_resolution_clock::now();
    exec_map_extend(d_map, d_measurements, d_particles, d_unique_occupied, d_unique_free,
        res_map, res_measurements, res_unique_occupied, res_unique_free, general_info, EXTEND);
    auto stop_check_extend = std::chrono::high_resolution_clock::now();

    if (should_assert == true) assert_map_extend(res_map, h_map, h_map_bg, h_map_post, EXTEND);

    auto start_world_to_image_transform_2 = std::chrono::high_resolution_clock::now();
    exec_world_to_image_transform_step_2(d_measurements, d_particles, d_position, d_transition,
        res_map, res_measurements, general_info);
    auto stop_world_to_image_transform_2 = std::chrono::high_resolution_clock::now();

    if (should_assert == true)
        assert_world_to_image_transform(d_particles, d_position, d_transition,
            res_measurements, res_particles, res_position, res_transition, h_particles, h_position, h_transition);

    auto start_bresenham = std::chrono::high_resolution_clock::now();
    exec_bresenham(d_particles, d_position, d_transition, res_particles, MAX_DIST_IN_MAP);
    auto stop_bresenham = std::chrono::high_resolution_clock::now();
    
    if (should_assert == true)
        assert_bresenham(d_particles, res_particles, res_measurements, d_measurements, h_particles);

    auto start_create_map = std::chrono::high_resolution_clock::now();
    reinit_map_idx_vars(d_unique_free, res_particles, res_unique_free);
    exec_create_map(d_particles, d_unique_occupied, d_unique_free, res_map, res_particles);
    auto stop_create_map = std::chrono::high_resolution_clock::now();

    auto start_restructure_map = std::chrono::high_resolution_clock::now();
    reinit_map_vars(d_particles, d_unique_occupied, d_unique_free, res_particles, res_unique_occupied, res_unique_free);
    exec_map_restructure(d_particles, d_unique_occupied, d_unique_free, res_map);
    auto stop_restructure_map = std::chrono::high_resolution_clock::now();

    if (should_assert == true) assert_map_restructure(d_particles, res_particles, h_particles);

    auto start_update_map = std::chrono::high_resolution_clock::now();
    exec_log_odds(d_map, d_particles, res_map, res_particles, general_info);
    auto stop_update_map = std::chrono::high_resolution_clock::now();

    if (should_assert == true) assert_log_odds(d_map, res_map, h_map, h_map_post);

    auto stop_mapping_kernel = std::chrono::high_resolution_clock::now();


    auto duration_mapping_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_mapping_kernel - start_mapping_kernel);

    std::cout << std::endl;
    std::cout << "Time taken by function (Mapping Kernel): " << duration_mapping_kernel.count() << " microseconds" << std::endl;
    std::cout << std::endl;
}


void test_iteration_single() {

    test_setup();
    test_allocation_initialization();
    test_robot_advance();
    test_robot();
    test_map();

}

#endif
