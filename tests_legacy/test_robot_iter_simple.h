#ifndef _TEST_ROBOT_ITER_SIMPLE_H_
#define _TEST_ROBOT_ITER_SIMPLE_H_


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
#include "device_set_reset_map.h"
#include "device_set_reset_robot.h"



///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

void test_robot_iter_simple() {

    const int UNIQUE_COUNTER_LEN = NUM_PARTICLES + 1;

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
    //vector<int> ids({ 500, 600, 700, 720, 800, 900, 1000 });
    vector<int> ids;
    string dir = "data/robot";
    getFiles(dir, ids);
    ids.erase(ids.begin() + 10, ids.end());

    GeneralInfo general_info;

    HostMap pre_map;
    HostMeasurements pre_measurements;
    HostParticles pre_particles;
    HostRobotParticles pre_robot_particles;
    HostRobotParticles pre_resampling_robot_particles;
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
    DeviceCorrelation d_correlation;

    Device2DUniqueFinder d_2d_unique;
    int PRE_GRID_SIZE = 0;
    int time_sum = 0;

    for (int i = 0; i < ids.size(); i++) {

        printf("/******************************** Index: %d *******************************/\n", ids[i]);

        read_robot_simple(ids[i], pre_map, pre_measurements,
            pre_robot_particles, pre_state, pre_resampling, general_info);

        int MEASURE_LEN = NUM_PARTICLES * pre_measurements.LEN;

        int negative_before_counter = getNegativeCounter(pre_robot_particles.f_x.data(), pre_robot_particles.f_y.data(), pre_robot_particles.LEN);
        int count_bigger_than_height = getGreaterThanCounter(pre_robot_particles.f_y.data(), pre_map.GRID_HEIGHT, pre_robot_particles.LEN);

        printf("~~$ GRID_WIDTH: \t\t%d\n", pre_map.GRID_WIDTH);
        printf("~~$ GRID_HEIGHT: \t\t%d\n", pre_map.GRID_HEIGHT);
        printf("~~$ negative_before_counter: \t%d\n", negative_before_counter);
        printf("~~$ count_bigger_than_height: \t%d\n", count_bigger_than_height);

        auto start_robot_particles_alloc = std::chrono::high_resolution_clock::now();

        if (i == 0) {

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
        }
        else {

            set_state(d_state, h_state, pre_state);
            set_measurement_vars(d_measurements, h_measurements, pre_measurements);
            set_robot_particles(d_robot_particles, h_robot_particles, pre_robot_particles);
            reset_correlation(d_correlation);
            reset_processed_measure(d_processed_measure, h_measurements);
            set_resampling(d_resampling, pre_resampling);

            int curr_grid_size = pre_map.GRID_WIDTH * pre_map.GRID_HEIGHT;
            if (curr_grid_size != PRE_GRID_SIZE) {

                alloc_init_map_vars(d_map, h_map, pre_map);
                alloc_map_2d_var(d_2d_unique, h_2d_unique, h_map);

                PRE_GRID_SIZE = curr_grid_size;
            }
            else {

                reset_map_vars(d_map, h_map, pre_map);
                reset_unique_map_vars(d_2d_unique);
            }
        }


        auto stop_robot_particles_alloc = std::chrono::high_resolution_clock::now();

        int* h_last_len = (int*)malloc(sizeof(int));

        auto start_robot_particles_kernel = std::chrono::high_resolution_clock::now();
        exec_calc_transition(d_particles_transition, d_state, d_transition, h_particles_transition);
        exec_process_measurements(d_processed_measure, d_particles_transition, d_measurements, h_map, h_measurements, general_info);

        exec_create_2d_map(d_2d_unique, d_robot_particles, h_map, h_robot_particles);

        exec_update_map(d_2d_unique, d_processed_measure, h_map, MEASURE_LEN);
        exec_particle_unique_cum_sum(d_2d_unique, h_map, h_2d_unique, h_robot_particles);

        reinit_map_vars(d_robot_particles, h_robot_particles);
        exec_map_restructure(d_robot_particles, d_2d_unique, h_map);

        exec_index_expansion(d_robot_particles, h_robot_particles);
        exec_correlation(d_map, d_robot_particles, d_correlation, h_map, h_robot_particles);

        exec_update_weights(d_robot_particles, d_correlation, h_robot_particles, h_correlation);

        exec_resampling(d_correlation, d_resampling);
        reinit_particles_vars(d_state, d_robot_particles, d_resampling, d_clone_robot_particles, d_clone_state, h_robot_particles,
            h_state, h_last_len);

        exec_rearrangement(d_robot_particles, d_state, d_resampling, d_clone_robot_particles, d_clone_state, h_map,
            h_robot_particles, h_clone_robot_particles, h_last_len);
        exec_update_states(d_state, d_transition, h_state, h_robot_state);
        auto stop_robot_particles_kernel = std::chrono::high_resolution_clock::now();

        PRE_GRID_SIZE = h_map.GRID_WIDTH * h_map.GRID_HEIGHT;

        auto duration_robot_particles_alloc = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_particles_alloc - start_robot_particles_alloc);
        auto duration_robot_particles_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_particles_kernel - start_robot_particles_kernel);
        auto duration_robot_particles_total = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_particles_kernel - start_robot_particles_alloc);

        std::cout << std::endl;
        std::cout << "Time taken by function (Allocation): " << duration_robot_particles_alloc.count() << " microseconds" << std::endl;
        std::cout << "Time taken by function (Kernel): " << duration_robot_particles_kernel.count() << " microseconds" << std::endl;
        std::cout << "Time taken by function (Total): " << duration_robot_particles_total.count() << " microseconds" << std::endl;

        time_sum += duration_robot_particles_kernel.count();
    }

    std::cout << std::endl;
    std::cout << "Time taken by function (Sum): " << time_sum << " microseconds" << std::endl;
    std::cout << "Time taken by function (Average): " << (time_sum / ids.size()) << " microseconds" << std::endl;
}


#endif
