#ifndef _TEST_ITERATION_MULTI_H_
#define _TEST_ITERATION_MULTI_H_

//#define ADD_HEADER_DATA

#include "headers.h"
#include "host_asserts.h"
#include "host_utils.h"
#include "kernels.cuh"
#include "kernels_robot.cuh"
#include "kernels_map.cuh"
#include "kernels_utils.cuh"
#include "device_init_common.h"
#include "device_init_robot.h"
#include "device_init_map.h"
#include "device_exec_robot.h"
#include "device_exec_map.h"
#include "device_assert_robot.h"
#include "device_assert_map.h"
#include "device_set_reset_map.h"
#include "device_set_reset_robot.h"


//#define VERBOSE_BORDER_LINE_COUNTER
//#define VERBOSE_TOTAL_INFO
//#define VERBOSE_BANNER
//#define VERBOSE_EXECUTION_TIME
#define GET_EXTRA_TRANSITION_WORLD_BODY


bool map_size_changed = false;


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


void test_alloc_init_robot(DeviceState& d_state, DeviceState& d_clone_state, DeviceMeasurements& d_measurements,
    DeviceMap& d_map, DeviceRobotParticles& d_robot_particles, DeviceRobotParticles& d_clone_robot_particles, 
    DeviceCorrelation& d_correlation, DeviceParticlesTransition& d_particles_transition,
    DeviceParticlesPosition& d_particles_position, DeviceParticlesRotation& d_particles_rotation, DeviceTransition& d_transition,
    DeviceProcessedMeasure& d_processed_measure, Device2DUniqueFinder& d_2d_unique, DeviceResampling& d_resampling,
    HostState& h_state, HostRobotState& h_robot_state, HostMeasurements& h_measurements, HostMap& h_map, HostRobotParticles& h_robot_particles,
    HostCorrelation& h_correlation, HostParticlesTransition& h_particles_transition, HostParticlesPosition& h_particles_position,
    HostParticlesRotation& h_particles_rotation, HostProcessedMeasure& h_processed_measure, Host2DUniqueFinder& h_2d_unique, HostResampling& h_resampling,
    HostState& pre_state, HostMeasurements& pre_measurements, HostMap& pre_map, HostRobotParticles& pre_robot_particles, HostResampling& pre_resampling) {

#ifdef VERBOSE_BANNER
    printf("/****************** ALLOCATIONS & INITIALIZATIONS  ******************/\n");
#endif

#ifdef VERBOSE_TOTAL_INFO
    printf("~~$ GRID_WIDTH: \t\t%d\n", GRID_WIDTH);
    printf("~~$ GRID_HEIGHT: \t\t%d\n", GRID_HEIGHT);

    printf("~~$ PARTICLES_OCCUPIED_LEN: \t%d\n", OCCUPIED_LEN);
    printf("~~$ PARTICLE_UNIQUE_COUNTER: \t%d\n", PARTICLE_UNIQUE_COUNTER);
    printf("~~$ MAX_DIST_IN_MAP: \t\t%d\n", MAX_DIST_IN_MAP);
    printf("~~$ LIDAR_COORDS_LEN: \t\t%d\n", LIDAR_COORDS_LEN);
    printf("~~$ PARTICLES_ITEMS_LEN: \t%d\n", PARTICLES_ITEMS_LEN);
#endif

    auto start_robot_particles_alloc = std::chrono::high_resolution_clock::now();
    alloc_init_state_vars(d_state, d_clone_state, h_state, h_robot_state, pre_state);
    alloc_init_measurement_vars(d_measurements, h_measurements, pre_measurements);
    alloc_init_map_vars(d_map, h_map, pre_map);
    alloc_init_robot_particles_vars(d_robot_particles, d_clone_robot_particles, h_robot_particles, pre_robot_particles);
    alloc_correlation_vars(d_correlation, h_correlation);
    alloc_particles_transition_vars(d_particles_transition, d_particles_position, d_particles_rotation,
        h_particles_transition, h_particles_position, h_particles_rotation);
    alloc_init_body_lidar(d_transition);
    alloc_init_processed_measurement_vars(d_processed_measure, h_processed_measure, h_measurements);
    alloc_map_2d_var(d_2d_unique, h_2d_unique, h_map);
    alloc_resampling_vars(d_resampling, h_resampling, pre_resampling);
    auto stop_robot_particles_alloc = std::chrono::high_resolution_clock::now();


#ifdef VERBOSE_EXECUTION_TIME
    auto duration_robot_particles_alloc = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_particles_alloc - start_robot_particles_alloc);
    auto duration_mapping_alloc = std::chrono::duration_cast<std::chrono::microseconds>(stop_mapping_alloc - start_mapping_alloc);

    std::cout << "Time taken by function (Particles Allocation): " << duration_robot_particles_alloc.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Mapping Allocation): " << duration_mapping_alloc.count() << " microseconds" << std::endl;
    std::cout << std::endl;
#endif
}

void test_alloc_init_map(DevicePosition& d_position, DeviceTransition& d_transition, DeviceParticles& d_particles,
    Device2DUniqueFinder& d_unique_occupied, Device2DUniqueFinder& d_unique_free,
    HostMap& h_map, HostPosition& h_position, HostTransition& h_transition, HostParticles& h_particles,
    HostMeasurements& h_measurements, Host2DUniqueFinder& h_unique_occupied, Host2DUniqueFinder& h_unique_free,
    HostTransition& pre_transition, HostMap& pre_map, HostParticles& pre_particles,
    host_vector<int>& hvec_occupied_map_idx, host_vector<int>& hvec_free_map_idx) {

    hvec_occupied_map_idx.resize(2, 0);
    hvec_free_map_idx.resize(2, 0);

    auto start_mapping_alloc = std::chrono::high_resolution_clock::now();
    alloc_init_transition_vars(d_position, d_transition, h_position, h_transition, pre_transition);
    int MAX_DIST_IN_MAP = sqrt(pow(pre_map.GRID_WIDTH, 2) + pow(pre_map.GRID_HEIGHT, 2));
    alloc_init_particles_vars(d_particles, h_particles, h_measurements, pre_particles, MAX_DIST_IN_MAP);
    hvec_occupied_map_idx[1] = h_particles.OCCUPIED_LEN;
    hvec_free_map_idx[1] = 0;
    alloc_init_unique_map_vars(d_unique_occupied, h_unique_occupied, h_map, hvec_occupied_map_idx);
    alloc_init_unique_map_vars(d_unique_free, h_unique_free, h_map, hvec_free_map_idx);
    auto stop_mapping_alloc = std::chrono::high_resolution_clock::now();

#ifdef VERBOSE_EXECUTION_TIME
    auto duration_mapping_alloc = std::chrono::duration_cast<std::chrono::microseconds>(stop_mapping_alloc - start_mapping_alloc);
    std::cout << "Time taken by function (Mapping Allocation): " << duration_mapping_alloc.count() << " microseconds" << std::endl;
    std::cout << std::endl;
#endif
}


void test_robot(DeviceState& d_state, DeviceState& d_clone_state, DeviceMap& d_map, DeviceTransition& d_transition, DeviceMeasurements& d_measurements, 
    DeviceProcessedMeasure& d_processed_measure, DeviceCorrelation& d_correlation, DeviceResampling& d_resampling,
    DeviceParticlesTransition& d_particles_transition, Device2DUniqueFinder& d_2d_unique, 
    DeviceRobotParticles& d_robot_particles, DeviceRobotParticles& d_clone_robot_particles,
    HostMap& h_map, HostMeasurements& h_measurements, HostParticlesTransition& h_particles_transition, HostResampling& h_resampling,
    HostRobotParticles& h_robot_particles, HostRobotParticles& h_clone_robot_particles, HostProcessedMeasure& h_processed_measure,
    Host2DUniqueFinder& h_2d_unique, HostCorrelation& h_correlation, HostState& h_state, HostRobotState& h_robot_state,
    HostRobotParticles& pre_robot_particles, HostRobotParticles& post_resampling_robot_particles, HostState& post_robot_move_state,
    HostResampling& pre_resampling, HostProcessedMeasure& post_processed_measure, HostRobotParticles& post_unique_robot_particles,
    HostState& post_state, host_vector<float>& pre_weights, host_vector<float>& post_loop_weights,
    GeneralInfo& general_info, bool should_assert) {

#ifdef VERBOSE_BANNER
    printf("/****************************** ROBOT *******************************/\n");
#endif

    const int MEASURE_LEN = NUM_PARTICLES * h_measurements.LEN;
    int* h_last_len = (int*)malloc(sizeof(int));

    int negative_before_counter = 0;
    int count_bigger_than_height = 0;
    int negative_after_unique_counter = 0;
    int negative_after_resampling_counter = 0;

    if (should_assert == true) {
        negative_before_counter = getNegativeCounter(pre_robot_particles.f_x.data(), pre_robot_particles.f_y.data(), pre_robot_particles.LEN);
        count_bigger_than_height = getGreaterThanCounter(pre_robot_particles.f_y.data(), h_map.GRID_HEIGHT, pre_robot_particles.LEN);
        negative_after_unique_counter = getNegativeCounter(post_unique_robot_particles.f_x.data(), post_unique_robot_particles.f_y.data(), post_unique_robot_particles.LEN);
        negative_after_resampling_counter = getNegativeCounter(post_resampling_robot_particles.f_x.data(), post_resampling_robot_particles.f_y.data(), post_resampling_robot_particles.LEN);
    }


#ifdef VERBOSE_BORDER_LINE_COUNTER
    printf("~~$ negative_before_counter: \t%d\n", negative_before_counter);
    printf("~~$ negative_after_unique_counter: \t%d\n", negative_after_unique_counter);
    printf("~~$ negative_after_resampling_counter: \t%d\n", negative_after_resampling_counter);
#endif

    auto start_robot_particles_kernel = std::chrono::high_resolution_clock::now();
    exec_robot_move(d_state, h_state);
    if (should_assert == true) assert_robot_move_results(d_state, h_state, post_robot_move_state);

    exec_calc_transition(d_particles_transition, d_state, d_transition, h_particles_transition);
    exec_process_measurements(d_processed_measure, d_particles_transition, d_measurements, h_map, h_measurements, general_info);
    if (should_assert == true)
        assert_processed_measures(d_particles_transition, d_processed_measure, h_particles_transition,
            h_measurements, h_processed_measure, post_processed_measure);

    exec_create_2d_map(d_2d_unique, d_robot_particles, h_map, h_robot_particles);
    //if (should_assert == true) assert_create_2d_map(d_2d_unique, h_2d_unique, h_map, h_robot_particles, negative_before_counter);
    if (should_assert == true) assert_create_2d_map(d_2d_unique, h_2d_unique, h_map, pre_robot_particles, negative_before_counter);

    exec_update_map(d_2d_unique, d_processed_measure, h_map, MEASURE_LEN);
    exec_particle_unique_cum_sum(d_2d_unique, h_map, h_2d_unique, h_robot_particles);
    if (should_assert == true) assert_particles_unique(h_robot_particles, post_unique_robot_particles, negative_after_unique_counter);

    reinit_map_vars(d_robot_particles, h_robot_particles);
    exec_map_restructure(d_robot_particles, d_2d_unique, h_map);
    if (should_assert == true) assert_particles_unique(d_robot_particles, h_robot_particles, post_unique_robot_particles, negative_after_unique_counter);

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
    exec_update_states(d_state, d_transition, h_state, h_robot_state);
    auto stop_robot_particles_kernel = std::chrono::high_resolution_clock::now();

#ifdef VERBOSE_EXECUTION_TIME
    auto duration_robot_particles_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_particles_kernel - start_robot_particles_kernel);

    std::cout << std::endl;
    std::cout << "Time taken by function (Kernel): " << duration_robot_particles_kernel.count() << " microseconds" << std::endl;
    std::cout << std::endl;
#endif
}

void test_map(DevicePosition& d_position, DeviceTransition& d_transition, DeviceParticles& d_particles,
    DeviceMeasurements& d_measurements, DeviceMap& d_map, 
    HostMeasurements& h_measurements, HostMap& h_map, HostPosition& h_position, HostTransition& h_transition,
    Device2DUniqueFinder& d_unique_occupied, Device2DUniqueFinder& d_unique_free,
    Host2DUniqueFinder& h_unique_occupied, Host2DUniqueFinder& h_unique_free,
    HostMap& pre_map, HostMap& post_bg_map, HostMap& post_map, HostParticles& post_particles, HostPosition& post_position, HostTransition& post_transition,
    HostParticles& h_particles, GeneralInfo& general_info, bool should_assert) {

#ifdef VERBOSE_BANNER
    printf("/****************************** MAP MAIN ****************************/\n");
#endif

    auto start_mapping_kernel = std::chrono::high_resolution_clock::now();

    exec_world_to_image_transform_step_1(d_position, d_transition, d_particles, d_measurements, h_measurements);

    bool EXTEND = false;
    exec_map_extend(d_map, d_measurements, d_particles, d_unique_occupied, d_unique_free,
        h_map, h_measurements, h_unique_occupied, h_unique_free, general_info, EXTEND);
    if (should_assert == true) assert_map_extend(h_map, pre_map, post_bg_map, post_map, EXTEND);

    exec_world_to_image_transform_step_2(d_measurements, d_particles, d_position, d_transition,
        h_map, h_measurements, general_info);
    if (should_assert == true)
        assert_world_to_image_transform(d_particles, d_position, d_transition,
            h_measurements, h_particles, h_position, h_transition, post_particles, post_position, post_transition);

    int MAX_DIST_IN_MAP = sqrt(pow(h_map.GRID_WIDTH, 2) + pow(h_map.GRID_HEIGHT, 2));

    exec_bresenham(d_particles, d_position, d_transition, h_particles, MAX_DIST_IN_MAP);
    if (should_assert == true) assert_bresenham(d_particles, h_particles, h_measurements, d_measurements, post_particles);

    reinit_map_idx_vars(d_unique_free, h_particles, h_unique_free);
    exec_create_map(d_particles, d_unique_occupied, d_unique_free, h_map, h_particles);
    
    reinit_map_vars(d_particles, d_unique_occupied, d_unique_free, h_particles, h_unique_occupied, h_unique_free);
    exec_map_restructure(d_particles, d_unique_occupied, d_unique_free, h_map);
    if (should_assert == true) assert_map_restructure(d_particles, h_particles, post_particles);

    exec_log_odds(d_map, d_particles, h_map, h_particles, general_info);
    if (should_assert == true) assert_log_odds(d_map, h_map, pre_map, post_map);
    
    auto stop_mapping_kernel = std::chrono::high_resolution_clock::now();

#ifdef VERBOSE_EXECUTION_TIME
    auto duration_mapping_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_mapping_kernel - start_mapping_kernel);

    std::cout << std::endl;
    std::cout << "Time taken by function (Mapping Kernel): " << duration_mapping_kernel.count() << " microseconds" << std::endl;
    std::cout << std::endl;
#endif
}



void resetMiddleVariables(DeviceCorrelation& d_correlation, DeviceProcessedMeasure& d_processed_measure, DeviceResampling& d_resampling,
    Device2DUniqueFinder& d_2d_unique, DeviceRobotParticles& d_robot_particles, 
    HostMap& h_map, HostMeasurements& h_measurements, HostProcessedMeasure& h_processed_measure) {

    int num_items = NUM_PARTICLES * h_measurements.LEN;
    reset_processed_measure(d_processed_measure, h_measurements);

    thrust::fill(d_2d_unique.s_map.begin(), d_2d_unique.s_map.end(), 0);
    thrust::fill(d_2d_unique.c_in_map.begin(), d_2d_unique.c_in_map.end(), 0);
    thrust::fill(d_2d_unique.s_in_col.begin(), d_2d_unique.s_in_col.end(), 0);

    reset_correlation(d_correlation);

    thrust::fill(d_resampling.c_js.begin(), d_resampling.c_js.end(), 0);
    thrust::fill(d_robot_particles.c_weight.begin(), d_robot_particles.c_weight.end(), 0);
}


void test_iterations() {

    vector<vector<float>> vec_arr_rnds_encoder_counts;
    vector<vector<float>> vec_arr_lidar_coords;
    vector<vector<float>> vec_arr_rnds;
    vector<vector<float>> vec_arr_transition;
    vector<vector<float>> vec_arr_rnds_yaws;

    vector<float> vec_encoder_counts;
    vector<float> vec_yaws;
    vector<float> vec_dt;

    printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
    printf("Reading Data Files\n");
    printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");

    HostState pre_state;
    HostState post_robot_move_state;
    HostState post_state;
    HostRobotParticles pre_robot_particles;
    HostRobotParticles post_unique_robot_particles;
    HostRobotParticles pre_resampling_robot_particles;
    HostRobotParticles post_resampling_robot_particles;
    HostProcessedMeasure post_processed_measure;
    HostParticlesTransition post_particles_transition;
    HostResampling pre_resampling;
    HostRobotState post_robot_state;
    HostMap pre_map;
    HostMap post_bg_map;
    HostMap post_map;
    HostMeasurements pre_measurements;
    HostPosition post_position;
    HostTransition pre_transition;
    HostTransition post_transition;
    HostParticles pre_particles;
    HostParticles post_particles;
    GeneralInfo general_info;

    DeviceState d_state;
    DeviceState d_clone_state;
    DeviceMeasurements d_measurements;
    DeviceProcessedMeasure d_processed_measure;
    DeviceMap d_map;
    DeviceRobotParticles d_robot_particles;
    DeviceRobotParticles d_clone_robot_particles;
    DeviceCorrelation d_correlation;
    DeviceParticlesTransition d_particles_transition;
    DeviceParticlesPosition d_particles_position;
    DeviceParticlesRotation d_particles_rotation;
    Device2DUniqueFinder d_2d_unique;
    DeviceResampling d_resampling;
    DeviceTransition d_transition;
    DevicePosition d_position;
    DeviceParticles d_particles;
    Device2DUniqueFinder d_unique_occupied;
    Device2DUniqueFinder d_unique_free;


    HostState h_state;
    HostRobotState h_robot_state;
    HostMeasurements h_measurements;
    HostProcessedMeasure h_processed_measure;
    HostMap h_map;
    HostRobotParticles h_robot_particles;
    HostRobotParticles h_clone_robot_particles;
    HostCorrelation h_correlation;
    HostParticlesTransition h_particles_transition;
    HostParticlesPosition h_particles_position;
    HostParticlesRotation h_particles_rotation;
    Host2DUniqueFinder h_2d_unique;
    HostResampling h_resampling;
    HostPosition h_position;
    HostTransition h_transition;
    HostParticles h_particles;
    Host2DUniqueFinder h_unique_occupied;
    Host2DUniqueFinder h_unique_free;

    host_vector<float> pre_weights;
    host_vector<float> post_loop_weights;
    host_vector<float> post_weights;

    host_vector<int> hvec_occupied_map_idx; 
    host_vector<int> hvec_free_map_idx;

    // string root = "range_700_730_step_1/";
    string root = "";
    const int INPUT_VEC_SIZE = 800;
    read_small_steps_vec("encoder_counts", vec_encoder_counts, INPUT_VEC_SIZE, root);
    read_small_steps_vec_arr("rnds_encoder_counts", vec_arr_rnds_encoder_counts, INPUT_VEC_SIZE, root);
    read_small_steps_vec_arr("lidar_coords", vec_arr_lidar_coords, INPUT_VEC_SIZE, root);
    read_small_steps_vec_arr("rnds", vec_arr_rnds, INPUT_VEC_SIZE, root);
    read_small_steps_vec_arr("transition", vec_arr_transition, INPUT_VEC_SIZE, root);
    read_small_steps_vec("yaws", vec_yaws, INPUT_VEC_SIZE, root);
    read_small_steps_vec_arr("rnds_yaws", vec_arr_rnds_yaws, INPUT_VEC_SIZE, root);
    read_small_steps_vec("dt", vec_dt, INPUT_VEC_SIZE, root);

    bool should_assert = false;

    const int LOOP_LEN = 50;
    const int ST_FILE_NUMBER = 100;
    const int CHECK_STEP = 200;
    const int DIFF_FROM_START = ST_FILE_NUMBER - 100;

    int PRE_GRID_SIZE = 0;

    for (int file_number = ST_FILE_NUMBER; file_number < ST_FILE_NUMBER + LOOP_LEN; file_number += 1) {

        should_assert = (file_number % CHECK_STEP == 0);

        if (file_number == ST_FILE_NUMBER) {

            printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
            printf("Iteration: %d\n", file_number);
            printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");

            auto start_read_file = std::chrono::high_resolution_clock::now();
            read_iteration(file_number, pre_state, post_robot_move_state, post_state,
                pre_robot_particles, post_unique_robot_particles,
                pre_resampling_robot_particles, post_resampling_robot_particles,
                post_processed_measure, post_particles_transition,
                pre_resampling, post_robot_state,
                pre_map, post_bg_map, post_map, pre_measurements,
                post_position, pre_transition, post_transition,
                pre_particles, post_particles, general_info,
                pre_weights, post_loop_weights, post_weights, root);
            auto stop_read_file = std::chrono::high_resolution_clock::now();

            test_alloc_init_robot(d_state, d_clone_state, d_measurements,
                d_map, d_robot_particles, d_clone_robot_particles, d_correlation, d_particles_transition,
                d_particles_position, d_particles_rotation, d_transition,
                d_processed_measure, d_2d_unique, d_resampling,
                h_state, h_robot_state, h_measurements, h_map, h_robot_particles,
                h_correlation, h_particles_transition, h_particles_position,
                h_particles_rotation, h_processed_measure, h_2d_unique, h_resampling,
                pre_state, pre_measurements, pre_map, pre_robot_particles, pre_resampling);

            test_alloc_init_map(d_position, d_transition, d_particles,
                d_unique_occupied, d_unique_free,
                h_map, h_position, h_transition, h_particles,
                h_measurements, h_unique_occupied, h_unique_free,
                pre_transition, pre_map, pre_particles,
                hvec_occupied_map_idx, hvec_free_map_idx);

            auto duration_read_file = std::chrono::duration_cast<std::chrono::microseconds>(stop_read_file - start_read_file);
            std::cout << std::endl;
            std::cout << "Time taken by function (Read Data Files): " << duration_read_file.count() << " microseconds" << std::endl;
            std::cout << std::endl;
        }
        else {

            //should_assert = (file_number % CHECK_STEP == 0);
            if (should_assert == true) {
                printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
                printf("Iteration: %d\n", file_number);
                printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
            }

            if (should_assert == true) {
                read_iteration(file_number, pre_state, post_robot_move_state, post_state,
                    pre_robot_particles, post_unique_robot_particles,
                    pre_resampling_robot_particles, post_resampling_robot_particles,
                    post_processed_measure, post_particles_transition,
                    pre_resampling, post_robot_state,
                    pre_map, post_bg_map, post_map, pre_measurements,
                    post_position, pre_transition, post_transition,
                    pre_particles, post_particles, general_info,
                    pre_weights, post_loop_weights, post_weights, root);
            }

            int curr_idx = file_number - ST_FILE_NUMBER + DIFF_FROM_START;

            auto start_alloc_init_step = std::chrono::high_resolution_clock::now();
            h_measurements.LEN = vec_arr_lidar_coords[curr_idx].size() / 2;
            int MEASURE_LEN = NUM_PARTICLES * h_measurements.LEN;
            h_particles.OCCUPIED_LEN = h_measurements.LEN;
            int PARTICLE_UNIQUE_COUNTER = h_particles.OCCUPIED_LEN + 1;

            set_measurement_vars(d_measurements, h_measurements, vec_arr_lidar_coords[curr_idx], vec_arr_lidar_coords[curr_idx].size() / 2);
            reset_processed_measure(d_processed_measure, h_measurements);
            reset_correlation(d_correlation);
            thrust::fill(d_robot_particles.c_weight.begin(), d_robot_particles.c_weight.end(), 0);
            set_state(d_state, h_state, pre_state, vec_arr_rnds_encoder_counts[curr_idx],
                vec_arr_rnds_yaws[curr_idx], vec_encoder_counts[curr_idx], vec_yaws[curr_idx], vec_dt[curr_idx]);
            set_resampling(d_resampling, vec_arr_rnds[curr_idx]);

            int MAX_DIST_IN_MAP = sqrt(pow(pre_map.GRID_WIDTH, 2) + pow(pre_map.GRID_HEIGHT, 2));
            //alloc_init_particles_vars(d_particles, h_particles, h_measurements, h_particles, MAX_DIST_IN_MAP);
            //resize_particles_vars(d_particles, h_measurements, MAX_DIST_IN_MAP);
            //set_robot_particles(d_robot_particles, h_robot_particles, pre_robot_particles);
            
            int curr_grid_size = h_map.GRID_WIDTH * h_map.GRID_HEIGHT;
            if (curr_grid_size != PRE_GRID_SIZE) {

                int MAX_DIST_IN_MAP = sqrt(pow(h_map.GRID_WIDTH, 2) + pow(h_map.GRID_HEIGHT, 2));
                //alloc_map_2d_var(d_2d_unique, h_2d_unique, h_map);
                resize_particles_vars(d_particles, h_measurements, MAX_DIST_IN_MAP);
                resize_unique_map_vars(d_unique_occupied, h_unique_occupied, h_map);
                resize_unique_map_vars(d_unique_free, h_unique_free, h_map);
                

                PRE_GRID_SIZE = curr_grid_size;
            }
            else {

                //reset_map_vars(d_map, h_map, pre_map);
                //reset_unique_map_vars(d_2d_unique);
            }
            thrust::fill(d_map.c_should_extend.begin(), d_map.c_should_extend.end(), 0);

            hvec_occupied_map_idx[1] = h_particles.OCCUPIED_LEN;
            hvec_free_map_idx[1] = 0;
            reset_unique_map_vars(d_unique_occupied, hvec_occupied_map_idx);
            reset_unique_map_vars(d_unique_free, hvec_free_map_idx);
            alloc_map_2d_var(d_2d_unique, h_2d_unique, h_map);


#if defined(GET_EXTRA_TRANSITION_WORLD_BODY)
            // printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
            // print_world_body(vec_arr_transition[curr_idx]);
            print_world_body(d_transition.c_world_body);
            // d_transition.c_world_body.assign(vec_arr_transition[curr_idx].begin(), vec_arr_transition[curr_idx].end());
#endif

            if (should_assert == true || (file_number - 1) % CHECK_STEP == 0) {
                if (should_assert == false) {
                    printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
                    printf("Iteration: %d\n", file_number);
                    printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
                }

                auto stop_alloc_init_step = std::chrono::high_resolution_clock::now();
                auto duration_alloc_init_step = std::chrono::duration_cast<std::chrono::microseconds>(stop_alloc_init_step - start_alloc_init_step);
                std::cout << std::endl;
                std::cout << "Time taken by function (Alloc & Init Step): " << duration_alloc_init_step.count() << " microseconds" << std::endl;
                std::cout << std::endl;
            }
            
            map_size_changed = false;
        }

        auto start_run_step = std::chrono::high_resolution_clock::now();
        test_robot(d_state, d_clone_state, d_map, d_transition, d_measurements, d_processed_measure, d_correlation, d_resampling,
            d_particles_transition, d_2d_unique, d_robot_particles, d_clone_robot_particles,
            h_map, h_measurements, h_particles_transition, h_resampling, h_robot_particles, h_clone_robot_particles, h_processed_measure,
            h_2d_unique, h_correlation, h_state, h_robot_state,
            pre_robot_particles, post_resampling_robot_particles, post_robot_move_state, pre_resampling, post_processed_measure, post_unique_robot_particles,
            post_state, pre_weights, post_loop_weights, general_info, should_assert);
        test_map(d_position, d_transition, d_particles, d_measurements, d_map,
            h_measurements, h_map, h_position, h_transition, d_unique_occupied, d_unique_free, h_unique_occupied, h_unique_free,
            pre_map, post_bg_map, post_map, post_particles, post_position, post_transition, h_particles, general_info, should_assert);

        PRE_GRID_SIZE = h_map.GRID_WIDTH * h_map.GRID_HEIGHT;

        if (should_assert == true || (file_number - 1) % CHECK_STEP == 0) {

            auto stop_run_step = std::chrono::high_resolution_clock::now();
            auto duration_run_step = std::chrono::duration_cast<std::chrono::microseconds>(stop_run_step - start_run_step);
            std::cout << std::endl;
            std::cout << "Time taken by function (Run Step): " << duration_run_step.count() << " microseconds" << std::endl;
            std::cout << std::endl;
        }
    }
}

#endif
