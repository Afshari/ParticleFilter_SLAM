#ifndef _TEST_MAP_EXTEND_H_
#define _TEST_MAP_EXTEND_H_


#include "headers.h"
#include "host_asserts.h"
#include "host_utils.h"
#include "kernels.cuh"
#include "kernels_map.cuh"
#include "kernels_utils.cuh"
#include "kernels_robot.cuh"
#include "device_init_map.h"
#include "device_exec_map.h"
#include "device_assert_map.h"


///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////


void test_map_extend() {

    printf("/******************************** TEST MAP EXTEND *******************************/\n");

    HostMapData h_map_data;
    GeneralInfo general_info;
    HostMeasurements h_measurements;
    HostParticlesData h_particles_data;
    HostPositionTransition h_position_transition;

    HostMapData h_map_data_bg;
    HostMapData h_map_data_post;

    DevicePositionTransition d_position_transition;
    DeviceMeasurements d_measurements;
    DeviceParticlesData d_particles_data;

    HostPositionTransition res_position_transition;
    HostMeasurements res_measurements;
    HostParticlesData res_particles_data;
    HostMapData res_map_data;
    HostUniqueManager res_unique_manager;

    host_vector<int> hvec_occupied_map_idx(2, 0);
    host_vector<int> hvec_free_map_idx(2, 0);

    vector<int> ids({ 600, 700, 720, 800 });

    for (int i = 0; i < ids.size(); i++) {

        printf("/******************************** Index: %d *******************************/\n", ids[i]);

        DeviceMapData d_map_data;
        DeviceUniqueManager d_unique_manager;


        auto start_read_data_file = std::chrono::high_resolution_clock::now();
        read_update_map(ids[i], h_map_data, h_map_data_bg,
            h_map_data_post, general_info, h_measurements, h_particles_data, h_position_transition);
        auto stop_read_data_file = std::chrono::high_resolution_clock::now();

        auto duration_read_data_file = std::chrono::duration_cast<std::chrono::milliseconds>(stop_read_data_file - start_read_data_file);
        std::cout << "Time taken by function (Read Data File): " << duration_read_data_file.count() << " milliseconds" << std::endl;

        alloc_init_transition_vars(d_position_transition, res_position_transition, h_position_transition);
        alloc_init_measurement_vars(d_measurements, res_measurements, h_measurements);

        int MAX_DIST_IN_MAP = sqrt(pow(h_map_data.GRID_WIDTH, 2) + pow(h_map_data.GRID_HEIGHT, 2));
        alloc_init_particles_vars(d_particles_data, res_particles_data, h_measurements, h_particles_data, MAX_DIST_IN_MAP);
        alloc_init_map_vars(d_map_data, res_map_data, h_map_data);

        hvec_occupied_map_idx[1] = res_particles_data.PARTICLES_OCCUPIED_LEN;
        hvec_free_map_idx[1] = 0;

        alloc_init_unique_vars(d_unique_manager, res_unique_manager, res_map_data, hvec_occupied_map_idx, hvec_free_map_idx);


        auto start_world_to_image_transform_1 = std::chrono::high_resolution_clock::now();
        exec_world_to_image_transform_step_1(d_position_transition, d_particles_data, d_measurements, res_measurements);
        auto stop_world_to_image_transform_1 = std::chrono::high_resolution_clock::now();

        bool EXTEND = false;
        auto start_check_extend = std::chrono::high_resolution_clock::now();
        exec_map_extend(d_map_data, d_measurements, d_particles_data, d_unique_manager,
            res_map_data, res_measurements, res_unique_manager, general_info, EXTEND);
        auto stop_check_extend = std::chrono::high_resolution_clock::now();

        assert_map_extend(res_map_data, h_map_data, h_map_data_bg, h_map_data_post, EXTEND);

        auto start_world_to_image_transform_2 = std::chrono::high_resolution_clock::now();
        exec_world_to_image_transform_step_2(d_measurements, d_particles_data, d_position_transition,
            res_map_data, res_measurements, general_info);
        auto stop_world_to_image_transform_2 = std::chrono::high_resolution_clock::now();

        assert_world_to_image_transform(d_particles_data, d_position_transition,
            res_measurements, res_particles_data, res_position_transition, h_particles_data, h_position_transition);

        auto start_bresenham = std::chrono::high_resolution_clock::now();
        exec_bresenham(d_particles_data, d_position_transition, res_particles_data, MAX_DIST_IN_MAP);
        auto stop_bresenham = std::chrono::high_resolution_clock::now();

        assert_bresenham(d_particles_data, res_particles_data, res_measurements, d_measurements, h_particles_data);

        reinit_map_idx_vars(d_unique_manager, res_particles_data, res_unique_manager);

        auto start_create_map = std::chrono::high_resolution_clock::now();
        exec_create_map(d_particles_data, d_unique_manager, res_map_data, res_particles_data);
        auto stop_create_map = std::chrono::high_resolution_clock::now();

        reinit_map_vars(d_particles_data, d_unique_manager, res_particles_data, res_unique_manager);

        auto start_restructure_map = std::chrono::high_resolution_clock::now();
        exec_map_restructure(d_particles_data, d_unique_manager, res_map_data);
        auto stop_restructure_map = std::chrono::high_resolution_clock::now();

        assert_map_restructure(d_particles_data, res_particles_data, h_particles_data);

        auto start_update_map = std::chrono::high_resolution_clock::now();
        exec_log_odds(d_map_data, d_particles_data, res_map_data, res_particles_data, general_info);
        auto stop_update_map = std::chrono::high_resolution_clock::now();

        assert_log_odds(d_map_data, res_map_data, h_map_data, h_map_data_post);


        auto duration_world_to_image_transform_1 = std::chrono::duration_cast<std::chrono::microseconds>(stop_world_to_image_transform_1 - start_world_to_image_transform_1);
        auto duration_world_to_image_transform_2 = std::chrono::duration_cast<std::chrono::microseconds>(stop_world_to_image_transform_2 - start_world_to_image_transform_2);
        auto duration_bresenham = std::chrono::duration_cast<std::chrono::microseconds>(stop_bresenham - start_bresenham);
        auto duration_create_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_create_map - start_create_map);
        auto duration_restructure_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_restructure_map - start_restructure_map);
        auto duration_update_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_update_map - start_update_map);

        auto duration_total = duration_world_to_image_transform_1 + duration_world_to_image_transform_2 + duration_bresenham + duration_create_map +
            duration_restructure_map + duration_update_map;

        std::cout << "Time taken by function (World To Image Transform 1): " << duration_world_to_image_transform_1.count() << " microseconds" << std::endl;
        std::cout << "Time taken by function (World To Image Transform 2): " << duration_world_to_image_transform_2.count() << " microseconds" << std::endl;
        std::cout << "Time taken by function (Bresenham): " << duration_bresenham.count() << " microseconds" << std::endl;
        std::cout << "Time taken by function (Create Map): " << duration_create_map.count() << " microseconds" << std::endl;
        std::cout << "Time taken by function (Restructure Map): " << duration_restructure_map.count() << " microseconds" << std::endl;
        std::cout << "Time taken by function (Update Map): " << duration_update_map.count() << " microseconds" << std::endl;
        std::cout << "Time taken by function (Total): " << duration_total.count() << " microseconds" << std::endl;
        std::cout << std::endl;
    }

}

#endif
