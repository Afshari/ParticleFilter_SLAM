#ifndef _TEST_ROBOT_EXTEND_H_
#define _TEST_ROBOT_EXTEND_H_

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

//#define ADD_ROBOT_MOVE
#define ADD_ROBOT

const int UNIQUE_COUNTER_LEN = NUM_PARTICLES + 1;

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

void test_robot_extend() {

    const int data_len = 6;
    int data[data_len] = { 1, 0, 2, 2, 1, 3 };
    int* d_data = NULL;
    gpuErrchk(cudaMalloc((void**)&d_data, data_len * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_data, data, data_len * sizeof(int), cudaMemcpyHostToDevice));

    thrust::exclusive_scan(thrust::host, data, data + 6, data, 0);
    thrust::exclusive_scan(thrust::device, d_data, d_data + 6, d_data, 0);

    printf("\n");
    printf("/****************************** ROBOT  ******************************/\n");

    //vector<int> ids({ 200, 300, 400, 500, 600, 700, 800, 900, 1000 });
    vector<int> ids({ 500, 600 });

    host_vector<float> pre_weights;
    host_vector<float> post_loop_weights;
    host_vector<float> post_weights;
    GeneralInfo general_info;

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
    HostParticlesPosition pre_particles_position;
    HostParticlesRotation pre_particles_rotation;
    HostTransition pre_transition;
    HostResampling pre_resampling;
    HostRobotState pre_robot_state;
    HostParticlesTransition pre_particles_transition;

    HostState h_state;
    HostMeasurements h_measurements;
    HostMap h_map;
    HostRobotParticles h_robot_particles;
    HostRobotParticles h_clone_robot_particles;
    HostCorrelation h_correlation;
    HostParticlesPosition h_particles_position;
    HostParticlesRotation h_particles_rotation;
    HostRobotState h_robot_state;
    HostParticlesTransition h_particles_transition;
    Host2DUniqueFinder h_2d_unique;
    HostProcessedMeasure h_processed_measure;
    HostResampling h_resampling;

    DeviceState d_state;
    DeviceState d_clone_state;
    DeviceMeasurements d_measurements;
    DeviceMap d_map;
    DeviceParticlesPosition d_particles_position;
    DeviceParticlesRotation d_particles_rotation;
    DeviceParticlesTransition d_particles_transition;
    DeviceTransition d_transition;
    DeviceProcessedMeasure d_processed_measure;
    DeviceRobotParticles d_robot_particles;
    DeviceRobotParticles d_clone_robot_particles;
    DeviceResampling d_resampling;

    Device2DUniqueFinder d_2d_unique;

    for (int i = 0; i < ids.size(); i++) {

        printf("/******************************** Index: %d *******************************/\n", ids[i]);

        DeviceCorrelation d_correlation;
        //Device2DUniqueFinder d_2d_unique;

        read_update_robot(ids[i], pre_map, pre_measurements, pre_robot_particles, pre_resampling_robot_particles,
            post_resampling_robot_particles, post_unique_robot_particles, post_processed_measure, pre_state,
            post_state, pre_resampling, pre_robot_state, pre_particles_transition,
            pre_weights, post_loop_weights, post_weights, general_info);


        int MEASURE_LEN = NUM_PARTICLES * pre_measurements.LEN;

        int negative_before_counter = getNegativeCounter(pre_robot_particles.f_x.data(), pre_robot_particles.f_y.data(), pre_robot_particles.LEN);
        int count_bigger_than_height = getGreaterThanCounter(pre_robot_particles.f_y.data(), pre_map.GRID_HEIGHT, pre_robot_particles.LEN);
        int negative_after_counter = getNegativeCounter(post_resampling_robot_particles.f_x.data(), post_resampling_robot_particles.f_y.data(), post_resampling_robot_particles.LEN);

        printf("~~$ GRID_WIDTH: \t\t%d\n", pre_map.GRID_WIDTH);
        printf("~~$ GRID_HEIGHT: \t\t%d\n", pre_map.GRID_HEIGHT);
        printf("~~$ negative_before_counter: \t%d\n", negative_before_counter);
        printf("~~$ negative_after_counter: \t%d\n", negative_after_counter);
        printf("~~$ count_bigger_than_height: \t%d\n", count_bigger_than_height);


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
        alloc_map_2d_var(d_2d_unique, h_2d_unique, h_map, (i == 0 || ids[i] == 800));
        //alloc_map_2d_var(d_2d_unique, h_2d_unique, h_map, true);
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
        if (should_assert == true) assert_update_weights(d_correlation, d_robot_particles, h_correlation,
            h_robot_particles, post_loop_weights);

        exec_resampling(d_correlation, d_resampling);
        reinit_particles_vars(d_state, d_robot_particles, d_resampling, d_clone_robot_particles, d_clone_state, h_robot_particles,
            h_state, h_last_len);
        if (should_assert == true) assert_resampling(d_resampling, h_resampling, pre_resampling, pre_state, post_state);

        exec_rearrangement(d_robot_particles, d_state, d_resampling, d_clone_robot_particles, d_clone_state, h_map,
            h_robot_particles, h_clone_robot_particles, h_last_len);
        exec_update_states(d_state, h_state, h_robot_state);
        auto stop_robot_particles_kernel = std::chrono::high_resolution_clock::now();

        assert_robot_final_results(d_robot_particles, d_correlation, h_robot_particles, h_correlation, h_robot_state,
            post_resampling_robot_particles, pre_robot_state, post_unique_robot_particles, post_loop_weights, negative_after_counter);


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

}


#endif
