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

    vector<int> ids({ 200, 300, 400, 500, 600, 700, 800, 900, 1000 });

    host_vector<float> weights_pre;
    host_vector<float> weights_new;
    host_vector<float> weights_updated;
    GeneralInfo general_info;

    HostMap h_map;
    HostMeasurements h_measurements;
    HostParticles h_particles;
    HostRobotParticles h_robot_particles;
    HostRobotParticles h_robot_particles_before_resampling;
    HostRobotParticles h_robot_particles_after_resampling;
    HostRobotParticles h_robot_particles_unique;
    HostProcessedMeasure h_processed_measure;
    HostState h_state;
    HostState h_state_updated;
    HostParticlesPosition h_particles_position;
    HostParticlesRotation h_particles_rotation;
    HostTransition h_transition;
    HostResampling h_resampling;
    HostRobotState h_robot_state;
    HostParticlesTransition h_particles_transition;

    HostState res_state;
    HostMeasurements res_measurements;
    HostMap res_map;
    HostRobotParticles res_robot_particles;
    HostRobotParticles res_clone_robot_particles;
    HostCorrelation res_correlation;
    HostParticlesPosition res_particles_position;
    HostParticlesRotation res_particles_rotation;
    HostRobotState res_robot_state;
    HostParticlesTransition res_particles_transition;
    Host2DUniqueFinder res_2d_unique;
    HostProcessedMeasure res_processed_measure;
    HostResampling res_resampling;

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


    for (int i = 0; i < ids.size(); i++) {

        printf("/******************************** Index: %d *******************************/\n", ids[i]);

        DeviceCorrelation d_correlation;
        Device2DUniqueFinder d_2d_unique;

        read_update_robot(ids[i], h_map, h_measurements, h_particles, h_robot_particles, h_robot_particles_before_resampling,
            h_robot_particles_after_resampling, h_robot_particles_unique, h_processed_measure, h_state,
            h_state_updated, h_particles_position, h_particles_rotation, h_resampling, h_robot_state, h_particles_transition,
            weights_pre, weights_new, weights_updated, general_info);


        int MEASURE_LEN = NUM_PARTICLES * h_measurements.LIDAR_COORDS_LEN;

        int negative_before_counter = getNegativeCounter(h_robot_particles.x.data(), h_robot_particles.y.data(), h_robot_particles.LEN);
        int count_bigger_than_height = getGreaterThanCounter(h_robot_particles.y.data(), h_map.GRID_HEIGHT, h_robot_particles.LEN);
        int negative_after_counter = getNegativeCounter(h_robot_particles_after_resampling.x.data(), h_robot_particles_after_resampling.y.data(), h_robot_particles_after_resampling.LEN);

        printf("~~$ GRID_WIDTH: \t\t%d\n", h_map.GRID_WIDTH);
        printf("~~$ GRID_HEIGHT: \t\t%d\n", h_map.GRID_HEIGHT);
        printf("~~$ negative_before_counter: \t%d\n", negative_before_counter);
        printf("~~$ negative_after_counter: \t%d\n", negative_after_counter);
        printf("~~$ count_bigger_than_height: \t%d\n", count_bigger_than_height);


        //auto start_robot_particles_alloc = std::chrono::high_resolution_clock::now();
        //alloc_init_state_vars(d_state, d_clone_state, res_state, res_robot_state, h_state);
        //alloc_init_measurement_vars(d_measurements, res_measurements, h_measurements);
        //alloc_init_map_vars(d_map, res_map, h_map);
        //alloc_init_particles_vars(d_robot_particles, res_robot_particles, h_robot_particles);
        //alloc_correlation_vars(d_correlation, res_correlation);
        //alloc_particles_transition_vars(d_particles_transition, res_particles_transition);
        //alloc_init_transition_vars(d_position, d_transition, res_position, res_transition,
        //    h_position, h_transition);
        //alloc_init_processed_measurement_vars(d_processed_measure, res_processed_measure, res_measurements);
        //alloc_map_2d_var(d_2d_unique, res_2d_unique, res_map);
        //alloc_resampling_vars(d_resampling, res_resampling, h_resampling);
        //auto stop_robot_particles_alloc = std::chrono::high_resolution_clock::now();

        auto start_robot_particles_alloc = std::chrono::high_resolution_clock::now();

        auto start_init_state = std::chrono::high_resolution_clock::now();
        alloc_init_state_vars(d_state, d_clone_state, res_state, res_robot_state, h_state);
        auto stop_init_state = std::chrono::high_resolution_clock::now();

        auto start_init_measurements = std::chrono::high_resolution_clock::now();
        alloc_init_measurement_vars(d_measurements, res_measurements, h_measurements);
        auto stop_init_measurements = std::chrono::high_resolution_clock::now();

        auto start_init_map = std::chrono::high_resolution_clock::now();
        alloc_init_map_vars(d_map, res_map, h_map);
        auto stop_init_map = std::chrono::high_resolution_clock::now();

        auto start_init_particles = std::chrono::high_resolution_clock::now();
        alloc_init_particles_vars(d_robot_particles, res_robot_particles, h_robot_particles);
        auto stop_init_particles = std::chrono::high_resolution_clock::now();

        auto start_init_correlation = std::chrono::high_resolution_clock::now();
        alloc_correlation_vars(d_correlation, res_correlation);
        auto stop_init_correlation = std::chrono::high_resolution_clock::now();

        auto start_init_particles_transition = std::chrono::high_resolution_clock::now();
        alloc_particles_transition_vars(d_particles_transition, d_particles_position, d_particles_rotation, 
            res_particles_transition, res_particles_position, res_particles_rotation);
        auto stop_init_particles_transition = std::chrono::high_resolution_clock::now();

        auto start_init_transition = std::chrono::high_resolution_clock::now();
        //alloc_init_transition_vars(d_position, d_transition, res_position, res_transition, h_position, h_transition);
        alloc_init_body_lidar(d_transition);
        auto stop_init_transition = std::chrono::high_resolution_clock::now();

        auto start_init_processed_measurement = std::chrono::high_resolution_clock::now();
        alloc_init_processed_measurement_vars(d_processed_measure, res_processed_measure, res_measurements);
        auto stop_init_processed_measurement = std::chrono::high_resolution_clock::now();

        auto start_init_map_2d = std::chrono::high_resolution_clock::now();
        alloc_map_2d_var(d_2d_unique, res_2d_unique, res_map);
        auto stop_init_map_2d = std::chrono::high_resolution_clock::now();

        auto start_init_resampling = std::chrono::high_resolution_clock::now();
        alloc_resampling_vars(d_resampling, res_resampling, h_resampling);
        auto stop_init_resampling = std::chrono::high_resolution_clock::now();

        auto stop_robot_particles_alloc = std::chrono::high_resolution_clock::now();

        int* res_last_len = (int*)malloc(sizeof(int));
        bool should_assert = false;

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
        if (should_assert == true) assert_resampling(d_resampling, res_resampling, h_resampling, h_state, h_state_updated);

        exec_rearrangement(d_robot_particles, d_state, d_resampling, d_clone_robot_particles, d_clone_state, res_map,
            res_robot_particles, res_clone_robot_particles, res_last_len);
        exec_update_states(d_state, res_state, res_robot_state);
        auto stop_robot_particles_kernel = std::chrono::high_resolution_clock::now();

        assert_robot_final_results(d_robot_particles, d_correlation, res_robot_particles, res_correlation, res_robot_state,
            h_robot_particles_after_resampling, h_robot_state, h_robot_particles_unique, weights_new, negative_after_counter);


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
