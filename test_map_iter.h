#ifndef _TEST_MAP_ITER_H_
#define _TEST_MAP_ITER_H_


#include "headers.h"
#include "host_asserts.h"
#include "host_utils.h"
#include "kernels.cuh"
#include "kernels_map.cuh"
#include "kernels_utils.cuh"
#include "kernels_robot.cuh"
#include "device_init_map.h"
#include "device_init_common.h"
#include "device_exec_map.h"
#include "device_assert_map.h"
#include "device_set_reset_map.h"

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////


void test_map_iter() {

    printf("/******************************** TEST MAP EXTEND *******************************/\n");

    HostMap pre_map;
    GeneralInfo general_info;
    HostMeasurements pre_measurements;
    HostParticles pre_particles;
    HostParticles post_particles;
    HostPosition post_position;
    HostTransition pre_transition;
    HostTransition post_transition;

    HostMap post_bg_map;
    HostMap post_map;

    DeviceMap d_map;
    DevicePosition d_position;
    DeviceTransition d_transition;
    DeviceMeasurements d_measurements;
    DeviceParticles d_particles;
    Device2DUniqueFinder d_unique_occupied;
    Device2DUniqueFinder d_unique_free;

    HostPosition h_position;
    HostTransition h_transition;
    HostMeasurements h_measurements;
    HostParticles h_particles;
    HostMap h_map;
    Host2DUniqueFinder h_unique_occupied;
    Host2DUniqueFinder h_unique_free;

    host_vector<int> hvec_occupied_map_idx(2, 0);
    host_vector<int> hvec_free_map_idx(2, 0);

    // vector<int> ids({ 400, 500, 600, 700 });
    // vector<int> ids({ 600, 700, 720, 800, 900, 1000 });
    // vector<int> ids({ 700, 720, 800, 900, 1000 });
    // vector<int> ids({ 700, 800, 900, 1000 });
    vector<int> ids;
    string dir = "data/map";
    getFiles(dir, ids);

    int PRE_GRID_SIZE = 0;
    bool EXTEND = false;

    for (int i = 0; i < ids.size(); i++) {

        printf("/******************************** Index: %d *******************************/\n", ids[i]);

        //DeviceMap d_map;

        auto start_read_data_file = std::chrono::high_resolution_clock::now();
        read_update_map(ids[i], pre_map, post_bg_map,
            post_map, general_info, pre_measurements, pre_particles, post_particles, post_position, pre_transition, post_transition);
        auto stop_read_data_file = std::chrono::high_resolution_clock::now();

        auto duration_read_data_file = std::chrono::duration_cast<std::chrono::milliseconds>(stop_read_data_file - start_read_data_file);
        std::cout << "Time taken by function (Read Data File): " << duration_read_data_file.count() << " milliseconds" << std::endl;

        int MAX_DIST_IN_MAP = 0;

        if (i == 0) {
        
            alloc_init_transition_vars(d_position, d_transition, h_position, h_transition, pre_transition);
            alloc_init_body_lidar(d_transition);
            alloc_init_measurement_vars(d_measurements, h_measurements, pre_measurements);
            alloc_init_map_vars(d_map, h_map, pre_map);
            PRE_GRID_SIZE = h_map.GRID_WIDTH * h_map.GRID_HEIGHT;

            MAX_DIST_IN_MAP = sqrt(pow(h_map.GRID_WIDTH, 2) + pow(h_map.GRID_HEIGHT, 2));
            alloc_init_particles_vars(d_particles, h_particles, pre_measurements, pre_particles, MAX_DIST_IN_MAP);

            hvec_occupied_map_idx[1] = h_particles.OCCUPIED_LEN;
            hvec_free_map_idx[1] = 0;

            alloc_init_unique_map_vars(d_unique_occupied, h_unique_occupied, h_map, hvec_occupied_map_idx);
            alloc_init_unique_map_vars(d_unique_free, h_unique_free, h_map, hvec_free_map_idx);
        }
        else {

            set_transition_vars(d_transition, pre_transition);
            set_measurement_vars(d_measurements, h_measurements, pre_measurements);
            reset_map_vars(d_map, h_map, pre_map);

            MAX_DIST_IN_MAP = sqrt(pow(h_map.GRID_WIDTH, 2) + pow(h_map.GRID_HEIGHT, 2));
            h_particles.OCCUPIED_LEN = pre_particles.OCCUPIED_LEN;
            assert(h_particles.OCCUPIED_LEN == h_measurements.LEN);

            int curr_grid_size = h_map.GRID_WIDTH * h_map.GRID_HEIGHT;
            if (curr_grid_size != PRE_GRID_SIZE) {

                resize_particles_vars(d_particles, h_measurements, MAX_DIST_IN_MAP);
                resize_unique_map_vars(d_unique_occupied, h_unique_occupied, h_map);
                resize_unique_map_vars(d_unique_free, h_unique_free, h_map);

                PRE_GRID_SIZE = curr_grid_size;
            }

            hvec_occupied_map_idx[1] = h_particles.OCCUPIED_LEN;
            hvec_free_map_idx[1] = 0;

            reset_unique_map_vars(d_unique_occupied, hvec_occupied_map_idx);
            reset_unique_map_vars(d_unique_free, hvec_free_map_idx);
        }


        auto start_world_to_image_transform_1 = std::chrono::high_resolution_clock::now();
        exec_world_to_image_transform_step_1(d_position, d_transition, d_particles, d_measurements, h_measurements);
        auto stop_world_to_image_transform_1 = std::chrono::high_resolution_clock::now();

        EXTEND = false;
        auto start_check_extend = std::chrono::high_resolution_clock::now();
        exec_map_extend(d_map, d_measurements, d_particles, d_unique_occupied, d_unique_free,
            h_map, h_measurements, h_unique_occupied, h_unique_free, general_info, EXTEND);
        auto stop_check_extend = std::chrono::high_resolution_clock::now();

        assert_map_extend(h_map, pre_map, post_bg_map, post_map, EXTEND);

        auto start_world_to_image_transform_2 = std::chrono::high_resolution_clock::now();
        exec_world_to_image_transform_step_2(d_measurements, d_particles, d_position, d_transition,
            h_map, h_measurements, general_info);
        auto stop_world_to_image_transform_2 = std::chrono::high_resolution_clock::now();

        assert_world_to_image_transform(d_particles, d_position, d_transition,
            h_measurements, h_particles, h_position, h_transition, post_particles, post_position, post_transition);

        auto start_bresenham = std::chrono::high_resolution_clock::now();
        exec_bresenham(d_particles, d_position, d_transition, h_particles, MAX_DIST_IN_MAP);
        auto stop_bresenham = std::chrono::high_resolution_clock::now();

        assert_bresenham(d_particles, h_particles, h_measurements, d_measurements, post_particles);

        auto start_create_map = std::chrono::high_resolution_clock::now();
        reinit_map_idx_vars(d_unique_free, h_particles, h_unique_free);
        exec_create_map(d_particles, d_unique_occupied, d_unique_free, h_map, h_particles);
        auto stop_create_map = std::chrono::high_resolution_clock::now();

        reinit_map_vars(d_particles, d_unique_occupied, d_unique_free, h_particles, h_unique_occupied, h_unique_free);

        auto start_restructure_map = std::chrono::high_resolution_clock::now();
        exec_map_restructure(d_particles, d_unique_occupied, d_unique_free, h_map);
        auto stop_restructure_map = std::chrono::high_resolution_clock::now();

        assert_map_restructure(d_particles, h_particles, post_particles);

        auto start_update_map = std::chrono::high_resolution_clock::now();
        exec_log_odds(d_map, d_particles, h_map, h_particles, general_info);
        auto stop_update_map = std::chrono::high_resolution_clock::now();

        assert_log_odds(d_map, h_map, pre_map, post_map);

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
