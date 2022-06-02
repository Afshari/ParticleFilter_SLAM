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

HostMap h_map;
HostState h_state;
HostRobotState h_robot_state;
HostMeasurements h_measurements;
HostParticles h_particles;
HostRobotParticles h_robot_particles;
HostRobotParticles h_clone_robot_particles;
HostParticlesTransition h_particles_transition;
HostParticlesPosition h_particles_position;
HostParticlesRotation h_particles_rotation;
HostCorrelation h_correlation;
HostPosition h_position;
HostTransition h_transition;
HostProcessedMeasure h_processed_measure;
HostResampling h_resampling;
Host2DUniqueFinder h_2d_unique;
Host2DUniqueFinder h_unique_occupied;
Host2DUniqueFinder h_unique_free;

host_vector<float> pre_weights; 
host_vector<float> post_loop_weights;
host_vector<float> post_weights;

HostMap pre_map;
HostMap post_bg_map;
HostMap post_map;
HostState pre_state;
HostState post_robot_move_state;
HostState post_state;
HostRobotState post_robot_state;
HostMeasurements pre_measurements;
HostProcessedMeasure post_processed_measure;
HostParticles pre_particles;
HostParticles post_particles;
HostRobotParticles pre_robot_particles;
HostRobotParticles post_unique_robot_particles;
HostRobotParticles pre_resampling_robot_particles;
HostRobotParticles post_resampling_robot_particles;
HostPosition post_position;
HostTransition pre_transition;
HostTransition post_transition;
HostParticlesTransition post_particles_transition;
HostResampling pre_resampling;
GeneralInfo general_info;

bool should_assert = true;

void test_iteration_single() {

    printf("/********************************************************************/\n");
    printf("/****************** ALLOCATIONS & INITIALIZATIONS  ******************/\n");
    printf("/********************************************************************/\n");

    read_iteration(500, pre_state, post_robot_move_state, post_state,
        pre_robot_particles, post_unique_robot_particles,
        pre_resampling_robot_particles, post_resampling_robot_particles,
        post_processed_measure, post_particles_transition,
        pre_resampling, post_robot_state,
        pre_map, post_bg_map, post_map, pre_measurements,
        post_position, pre_transition, post_transition,
        pre_particles, post_particles, general_info,
        pre_weights, post_loop_weights, post_weights);

    alloc_init_state_vars(d_state, d_clone_state, h_state, h_robot_state, pre_state);

    auto start_robot_particles_alloc = std::chrono::high_resolution_clock::now();

    alloc_init_measurement_vars(d_measurements, h_measurements, pre_measurements);
    alloc_init_map_vars(d_map, h_map, pre_map);
    alloc_init_robot_particles_vars(d_robot_particles, h_robot_particles, pre_robot_particles);
    alloc_correlation_vars(d_correlation, h_correlation);
    alloc_particles_transition_vars(d_particles_transition, d_particles_position, d_particles_rotation,
        h_particles_transition, h_particles_position, h_particles_rotation);
    alloc_init_body_lidar(d_transition);
    alloc_init_processed_measurement_vars(d_processed_measure, h_processed_measure, h_measurements);
    alloc_map_2d_var(d_2d_unique, h_2d_unique, h_map, true);
    alloc_resampling_vars(d_resampling, h_resampling, pre_resampling);

    auto stop_robot_particles_alloc = std::chrono::high_resolution_clock::now();

    auto start_mapping_alloc = std::chrono::high_resolution_clock::now();
    alloc_init_transition_vars(d_position, d_transition, h_position, h_transition, pre_transition);

    int MAX_DIST_IN_MAP = sqrt(pow(pre_map.GRID_WIDTH, 2) + pow(pre_map.GRID_HEIGHT, 2));

    alloc_init_particles_vars(d_particles, h_particles, h_measurements, pre_particles, MAX_DIST_IN_MAP);

    hvec_occupied_map_idx[1] = h_particles.OCCUPIED_LEN;
    hvec_free_map_idx[1] = 0;
    alloc_init_unique_map_vars(d_unique_occupied, h_unique_occupied, h_map, hvec_occupied_map_idx);
    alloc_init_unique_map_vars(d_unique_free, h_unique_free, h_map, hvec_free_map_idx);

    auto stop_mapping_alloc = std::chrono::high_resolution_clock::now();

    auto duration_robot_particles_alloc = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_particles_alloc - start_robot_particles_alloc);
    auto duration_mapping_alloc = std::chrono::duration_cast<std::chrono::microseconds>(stop_mapping_alloc - start_mapping_alloc);

    std::cout << "Time taken by function (Particles Allocation): " << duration_robot_particles_alloc.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Mapping Allocation): " << duration_mapping_alloc.count() << " microseconds" << std::endl;


    printf("/********************************************************************/\n");
    printf("/************************** ROBOT ADVANCE ***************************/\n");
    printf("/********************************************************************/\n");

    std::cout << "Start Robot Advance" << std::endl;
        
    auto start_robot_advance_kernel = std::chrono::high_resolution_clock::now();
    exec_robot_move(d_state, h_state);
    auto stop_robot_advance_kernel = std::chrono::high_resolution_clock::now();
    
    if (should_assert == true) assert_robot_move_results(d_state, h_state, post_robot_move_state);
    
    auto duration_robot_advance_total = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_advance_kernel - start_robot_advance_kernel);
    std::cout << std::endl;
    std::cout << "Time taken by function (Robot Advance Kernel): " << duration_robot_advance_total.count() << " microseconds" << std::endl;


    printf("/********************************************************************/\n");
    printf("/****************************** ROBOT  ******************************/\n");
    printf("/********************************************************************/\n");


    int negative_before_counter = getNegativeCounter(pre_robot_particles.x.data(), pre_robot_particles.y.data(), pre_robot_particles.LEN);
    int count_bigger_than_height = getGreaterThanCounter(pre_robot_particles.y.data(), h_map.GRID_HEIGHT, pre_robot_particles.LEN);
    int negative_after_counter = getNegativeCounter(post_resampling_robot_particles.x.data(), post_resampling_robot_particles.y.data(), post_resampling_robot_particles.LEN);

    printf("~~$ negative_before_counter: \t%d\n", negative_before_counter);
    printf("~~$ negative_after_counter: \t%d\n", negative_after_counter);
    printf("~~$ count_bigger_than_height: \t%d\n", count_bigger_than_height);

    const int MEASURE_LEN = NUM_PARTICLES * h_measurements.LIDAR_COORDS_LEN;

    int* h_last_len = (int*)malloc(sizeof(int));

    auto start_robot_particles_kernel = std::chrono::high_resolution_clock::now();
    exec_calc_transition(d_particles_transition, d_state, d_transition, h_particles_transition);
    exec_process_measurements(d_processed_measure, d_particles_transition, d_measurements, h_map, h_measurements, general_info);
    if (should_assert == true)
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
    if (should_assert == true) assert_correlation(d_correlation, d_robot_particles, h_correlation, h_robot_particles, pre_weights);

    exec_update_weights(d_robot_particles, d_correlation, h_robot_particles, h_correlation);
    if (should_assert == true) assert_update_weights(d_correlation, d_robot_particles, h_correlation, h_robot_particles, post_loop_weights);

    exec_resampling(d_correlation, d_resampling);
    reinit_particles_vars(d_state, d_robot_particles, d_resampling, d_clone_robot_particles, d_clone_state, h_robot_particles, h_state, h_last_len);
    if (should_assert == true) assert_resampling(d_resampling, h_resampling, pre_resampling, post_robot_move_state, post_state);

    exec_rearrangement(d_robot_particles, d_state, d_resampling, d_clone_robot_particles, d_clone_state, h_map,
        h_robot_particles, h_clone_robot_particles, h_last_len);
    exec_update_states(d_state, h_state, h_robot_state);

    auto stop_robot_particles_kernel = std::chrono::high_resolution_clock::now();

    if (should_assert == true)
        assert_robot_final_results(d_robot_particles, d_correlation, h_robot_particles, h_correlation, h_robot_state,
            post_resampling_robot_particles, post_robot_state, post_unique_robot_particles, post_loop_weights, negative_after_counter);
    
    auto duration_robot_particles_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_particles_kernel - start_robot_particles_kernel);

    std::cout << std::endl;
    std::cout << "Time taken by function (Kernel): " << duration_robot_particles_kernel.count() << " microseconds" << std::endl;


    printf("/********************************************************************/\n");
    printf("/****************************** MAP MAIN ****************************/\n");
    printf("/********************************************************************/\n");

    //const int MEASURE_LEN = NUM_PARTICLES * h_measurements.LIDAR_COORDS_LEN;
    //int MAX_DIST_IN_MAP = sqrt(pow(h_map.GRID_WIDTH, 2) + pow(h_map.GRID_HEIGHT, 2));

    auto start_mapping_kernel = std::chrono::high_resolution_clock::now();

    auto start_world_to_image_transform_1 = std::chrono::high_resolution_clock::now();
    exec_world_to_image_transform_step_1(d_position, d_transition, d_particles, d_measurements, h_measurements);
    auto stop_world_to_image_transform_1 = std::chrono::high_resolution_clock::now();

    bool EXTEND = false;
    auto start_check_extend = std::chrono::high_resolution_clock::now();
    exec_map_extend(d_map, d_measurements, d_particles, d_unique_occupied, d_unique_free,
        h_map, h_measurements, h_unique_occupied, h_unique_free, general_info, EXTEND);
    auto stop_check_extend = std::chrono::high_resolution_clock::now();

    if (should_assert == true) assert_map_extend(h_map, pre_map, post_bg_map, post_map, EXTEND);

    auto start_world_to_image_transform_2 = std::chrono::high_resolution_clock::now();
    exec_world_to_image_transform_step_2(d_measurements, d_particles, d_position, d_transition,
        h_map, h_measurements, general_info);
    auto stop_world_to_image_transform_2 = std::chrono::high_resolution_clock::now();

    if (should_assert == true)
        assert_world_to_image_transform(d_particles, d_position, d_transition,
            h_measurements, h_particles, h_position, h_transition, post_particles, post_position, post_transition);

    auto start_bresenham = std::chrono::high_resolution_clock::now();
    exec_bresenham(d_particles, d_position, d_transition, h_particles, MAX_DIST_IN_MAP);
    auto stop_bresenham = std::chrono::high_resolution_clock::now();
    
    if (should_assert == true) assert_bresenham(d_particles, h_particles, h_measurements, d_measurements, post_particles);

    auto start_create_map = std::chrono::high_resolution_clock::now();
    reinit_map_idx_vars(d_unique_free, h_particles, h_unique_free);
    exec_create_map(d_particles, d_unique_occupied, d_unique_free, h_map, h_particles);
    auto stop_create_map = std::chrono::high_resolution_clock::now();

    auto start_restructure_map = std::chrono::high_resolution_clock::now();
    reinit_map_vars(d_particles, d_unique_occupied, d_unique_free, h_particles, h_unique_occupied, h_unique_free);
    exec_map_restructure(d_particles, d_unique_occupied, d_unique_free, h_map);
    auto stop_restructure_map = std::chrono::high_resolution_clock::now();

    if (should_assert == true) assert_map_restructure(d_particles, h_particles, post_particles);

    auto start_update_map = std::chrono::high_resolution_clock::now();
    exec_log_odds(d_map, d_particles, h_map, h_particles, general_info);
    auto stop_update_map = std::chrono::high_resolution_clock::now();

    if (should_assert == true) assert_log_odds(d_map, h_map, pre_map, post_map);

    auto stop_mapping_kernel = std::chrono::high_resolution_clock::now();

    auto duration_mapping_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_mapping_kernel - start_mapping_kernel);

    std::cout << std::endl;
    std::cout << "Time taken by function (Mapping Kernel): " << duration_mapping_kernel.count() << " microseconds" << std::endl;
    std::cout << std::endl;


}

#endif
