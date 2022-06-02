#ifndef _DEVICE_EXEC_MAP_H_
#define _DEVICE_EXEC_MAP_H_

#include "headers.h"
#include "structures.h"
#include "kernels.cuh"
#include "kernels_map.cuh"
#include "kernels_utils.cuh"
#include "kernels_robot.cuh"

void exec_world_to_image_transform_step_1(
    DevicePosition& d_position, DeviceTransition& d_transition, DeviceParticles& d_particles, 
    DeviceMeasurements& d_measurements, HostMeasurements& h_measurements) {

    kernel_matrix_mul_3x3 << < 1, 1 >> > (
        THRUST_RAW_CAST(d_transition.c_world_body),
        THRUST_RAW_CAST(d_transition.c_body_lidar), THRUST_RAW_CAST(d_transition.c_world_lidar));
    cudaDeviceSynchronize();

    int threadsPerBlock = 1;
    int blocksPerGrid = h_measurements.LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles.v_world_x), THRUST_RAW_CAST(d_particles.v_world_y), SEP,
        THRUST_RAW_CAST(d_transition.c_world_lidar), THRUST_RAW_CAST(d_measurements.v_lidar_coords),
        h_measurements.LEN);
    cudaDeviceSynchronize();
}

void exec_map_extend(DeviceMap& d_map, DeviceMeasurements& d_measurements, DeviceParticles& d_particles,
    Device2DUniqueFinder& d_unique_occupied, Device2DUniqueFinder& d_unique_free, HostMap& h_map, HostMeasurements& h_measurements, 
    Host2DUniqueFinder& h_unique_occupied, Host2DUniqueFinder& h_unique_free,
    GeneralInfo& general_info, bool& EXTEND) {

    int threadsPerBlock = 256;
    int blocksPerGrid = (h_measurements.LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_check_map_extend_less << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_map.c_should_extend), SEP, THRUST_RAW_CAST(d_particles.v_world_x), h_map.xmin, 0, h_measurements.LEN);
    kernel_check_map_extend_less << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_map.c_should_extend), SEP, THRUST_RAW_CAST(d_particles.v_world_y), h_map.ymin, 1, h_measurements.LEN);

    kernel_check_map_extend_greater << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_map.c_should_extend), SEP, THRUST_RAW_CAST(d_particles.v_world_x), h_map.xmax, 2, h_measurements.LEN);
    kernel_check_map_extend_greater << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_map.c_should_extend), SEP, THRUST_RAW_CAST(d_particles.v_world_y), h_map.ymax, 3, h_measurements.LEN);
    cudaDeviceSynchronize();

    h_map.c_should_extend.assign(d_map.c_should_extend.begin(), d_map.c_should_extend.end());

    int xmin_pre = h_map.xmin;
    int ymax_pre = h_map.ymax;

    EXTEND = false;
    if (h_map.c_should_extend[0] != 0) {
        EXTEND = true;
        h_map.xmin = h_map.xmin * 2;
    }
    else if (h_map.c_should_extend[2] != 0) {
        EXTEND = true;
        h_map.xmax = h_map.xmax * 2;
    }
    else if (h_map.c_should_extend[1] != 0) {
        EXTEND = true;
        h_map.ymin = h_map.ymin * 2;
    }
    else if (h_map.c_should_extend[3] != 0) {
        EXTEND = true;
        h_map.ymax = h_map.ymax * 2;
    }

    //for (int i = 0; i < 4; i++)
    //    std::cout << "Should Extend: " << h_map.should_extend[i] << std::endl;

    if (EXTEND == true) {

        auto start_extend = std::chrono::high_resolution_clock::now();

        d_measurements.coord.resize(2);
        //HostMeasurements res_measurements;
        h_measurements.coord.resize(2);

        kernel_position_to_image << <1, 1 >> > (THRUST_RAW_CAST(d_measurements.coord), SEP,
            xmin_pre, ymax_pre, general_info.res, h_map.xmin, h_map.ymax);
        cudaDeviceSynchronize();

        h_measurements.coord.assign(d_measurements.coord.begin(), d_measurements.coord.end());

        device_vector<int> dvec_clone_grid_map(h_map.GRID_WIDTH * h_map.GRID_HEIGHT);
        dvec_clone_grid_map.assign(d_map.s_grid_map.begin(), d_map.s_grid_map.end());

        device_vector<float> dvec_clone_log_odds(h_map.GRID_WIDTH * h_map.GRID_HEIGHT);
        dvec_clone_log_odds.assign(d_map.s_log_odds.begin(), d_map.s_log_odds.end());

        const int PRE_GRID_WIDTH = h_map.GRID_WIDTH;
        const int PRE_GRID_HEIGHT = h_map.GRID_HEIGHT;
        h_map.GRID_WIDTH = ceil((h_map.ymax - h_map.ymin) / general_info.res + 1);
        h_map.GRID_HEIGHT = ceil((h_map.xmax - h_map.xmin) / general_info.res + 1);

        const int PRE_GRID_SIZE = PRE_GRID_WIDTH * PRE_GRID_HEIGHT;
        const int NEW_GRID_SIZE = h_map.GRID_WIDTH * h_map.GRID_HEIGHT;

        d_map.s_grid_map.clear();
        d_map.s_log_odds.clear();
        d_map.s_grid_map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);
        d_map.s_log_odds.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, LOG_ODD_PRIOR);

        h_map.s_grid_map.clear();
        h_map.s_log_odds.clear();
        h_map.s_grid_map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);
        h_map.s_log_odds.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);

        threadsPerBlock = 256;
        blocksPerGrid = (PRE_GRID_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        kernel_2d_copy_with_offset << <blocksPerGrid, threadsPerBlock >> > (
            THRUST_RAW_CAST(d_map.s_grid_map), THRUST_RAW_CAST(d_map.s_log_odds), SEP,
            THRUST_RAW_CAST(dvec_clone_grid_map), THRUST_RAW_CAST(dvec_clone_log_odds),
            h_measurements.coord[0], h_measurements.coord[1],
            PRE_GRID_HEIGHT, h_map.GRID_HEIGHT, PRE_GRID_SIZE);
        cudaDeviceSynchronize();

        d_unique_free.s_in_col.clear();
        d_unique_free.s_in_col.resize(h_map.GRID_WIDTH + 1);
        h_unique_free.s_in_col.resize(h_map.GRID_WIDTH + 1);

        d_unique_occupied.s_in_col.clear();
        d_unique_occupied.s_in_col.resize(h_map.GRID_WIDTH + 1);
        h_unique_occupied.s_in_col.resize(h_map.GRID_WIDTH + 1);

        d_unique_free.s_map.clear();
        d_unique_occupied.s_map.clear();
        d_unique_free.s_map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT);
        d_unique_occupied.s_map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT);

        auto stop_extend = std::chrono::high_resolution_clock::now();
        auto duration_extend = std::chrono::duration_cast<std::chrono::microseconds>(stop_extend - start_extend);

        std::cout << std::endl;
        std::cout << "Time taken by function (Extend): " << duration_extend.count() << " microseconds" << std::endl;

        h_map.s_grid_map.assign(d_map.s_grid_map.begin(), d_map.s_grid_map.end());
        h_map.s_log_odds.assign(d_map.s_log_odds.begin(), d_map.s_log_odds.end());
    }

    //for (int i = 0; i < 4; i++)
    //    std::cout << "Should Extend: " << h_map.should_extend[i] << std::endl;
}


void exec_world_to_image_transform_step_2(DeviceMeasurements& d_measurements, DeviceParticles& d_particles,
    DevicePosition& d_position, DeviceTransition& d_transition,
    HostMap& h_map, HostMeasurements& h_measurements, GeneralInfo& general_info) {

    int threadsPerBlock = 1;
    int blocksPerGrid = h_measurements.LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles.f_occupied_x), THRUST_RAW_CAST(d_particles.f_occupied_y), SEP,
        THRUST_RAW_CAST(d_transition.c_world_lidar), THRUST_RAW_CAST(d_measurements.v_lidar_coords),
        general_info.res, h_map.xmin, h_map.ymax, h_measurements.LEN);
    cudaDeviceSynchronize();

    kernel_position_to_image << < 1, 1 >> > (
        THRUST_RAW_CAST(d_position.c_image_body), SEP,
        THRUST_RAW_CAST(d_transition.c_world_lidar), general_info.res, h_map.xmin, h_map.ymax);
    cudaDeviceSynchronize();
}

void exec_bresenham(DeviceParticles& d_particles, DevicePosition& d_position, DeviceTransition& d_transition,
    HostParticles& h_particles, const int MAX_DIST_IN_MAP) {

    int PARTICLE_UNIQUE_COUNTER = h_particles.OCCUPIED_LEN + 1;

    int threadsPerBlock = 256;
    int blocksPerGrid = (h_particles.OCCUPIED_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_bresenham << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles.s_free_x_max), THRUST_RAW_CAST(d_particles.s_free_y_max), THRUST_RAW_CAST(d_particles.v_free_counter), SEP,
        THRUST_RAW_CAST(d_particles.f_occupied_x), THRUST_RAW_CAST(d_particles.f_occupied_y), THRUST_RAW_CAST(d_position.c_image_body),
        h_particles.OCCUPIED_LEN, MAX_DIST_IN_MAP);
    cudaDeviceSynchronize();
    thrust::exclusive_scan(thrust::device, d_particles.v_free_counter.begin(), d_particles.v_free_counter.end(), d_particles.v_free_counter.begin(), 0);

    h_particles.v_free_counter.assign(d_particles.v_free_counter.begin(), d_particles.v_free_counter.end());
    d_particles.v_free_idx.assign(d_particles.v_free_counter.begin(), d_particles.v_free_counter.end());

    h_particles.FREE_LEN = h_particles.v_free_counter[PARTICLE_UNIQUE_COUNTER - 1];
    //printf("^^^ FREE_LEN = %d\n", h_particles.FREE_LEN);

    d_particles.f_free_x.resize(h_particles.FREE_LEN);
    d_particles.f_free_y.resize(h_particles.FREE_LEN);


    kernel_bresenham_rearrange << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles.f_free_x), THRUST_RAW_CAST(d_particles.f_free_y), SEP,
        THRUST_RAW_CAST(d_particles.s_free_x_max), THRUST_RAW_CAST(d_particles.s_free_y_max),
        THRUST_RAW_CAST(d_particles.v_free_counter), MAX_DIST_IN_MAP, h_particles.OCCUPIED_LEN);
    cudaDeviceSynchronize();
}

void reinit_map_idx_vars(Device2DUniqueFinder& d_unique_free, HostParticles& h_particles, Host2DUniqueFinder& h_unique_free) {

    h_unique_free.c_idx[1] = h_particles.FREE_LEN;
    d_unique_free.c_idx.assign(h_unique_free.c_idx.begin(), h_unique_free.c_idx.end());
}

void exec_create_map(DeviceParticles& d_particles, Device2DUniqueFinder& d_unique_occupied, Device2DUniqueFinder& d_unique_free,
    HostMap& h_map, HostParticles& h_particles) {

    int threadsPerBlock = 256;
    int blocksPerGrid = 1;
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_unique_occupied.s_map), SEP,
        THRUST_RAW_CAST(d_particles.f_occupied_x), THRUST_RAW_CAST(d_particles.f_occupied_y),
        THRUST_RAW_CAST(d_unique_occupied.c_idx), h_particles.OCCUPIED_LEN,
        h_map.GRID_WIDTH, h_map.GRID_HEIGHT, NUM_PARTICLES);
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_unique_free.s_map), SEP,
        THRUST_RAW_CAST(d_particles.f_free_x), THRUST_RAW_CAST(d_particles.f_free_y),
        THRUST_RAW_CAST(d_unique_free.c_idx), h_particles.FREE_LEN, h_map.GRID_WIDTH, h_map.GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();

    threadsPerBlock = h_map.GRID_WIDTH;
    blocksPerGrid = 1;
    kernel_2d_map_counter << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_unique_occupied.c_in_map), THRUST_RAW_CAST(d_unique_occupied.s_in_col), SEP,
        THRUST_RAW_CAST(d_unique_occupied.s_map), h_map.GRID_WIDTH, h_map.GRID_HEIGHT);
    kernel_2d_map_counter << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_unique_free.c_in_map), THRUST_RAW_CAST(d_unique_free.s_in_col), SEP,
        THRUST_RAW_CAST(d_unique_free.s_map), h_map.GRID_WIDTH, h_map.GRID_HEIGHT);
    cudaDeviceSynchronize();

    threadsPerBlock = 1;
    blocksPerGrid = 1;
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (THRUST_RAW_CAST(d_unique_occupied.s_in_col), h_map.GRID_WIDTH);
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (THRUST_RAW_CAST(d_unique_free.s_in_col), h_map.GRID_WIDTH);
    cudaDeviceSynchronize();
}

void reinit_map_vars(DeviceParticles& d_particles, Device2DUniqueFinder& d_unique_occupied, Device2DUniqueFinder& d_unique_free,
    HostParticles& h_particles, Host2DUniqueFinder& h_unique_occupied, Host2DUniqueFinder& h_unique_free) {

    h_unique_occupied.c_in_map.assign(d_unique_occupied.c_in_map.begin(), d_unique_occupied.c_in_map.end());
    h_unique_free.c_in_map.assign(d_unique_free.c_in_map.begin(), d_unique_free.c_in_map.end());

    h_particles.OCCUPIED_UNIQUE_LEN = h_unique_occupied.c_in_map[0];
    h_particles.FREE_UNIQUE_LEN = h_unique_free.c_in_map[0];

    //h_particles.f_occupied_x.resize(h_particles.OCCUPIED_UNIQUE_LEN, 0);
    //h_particles.f_occupied_y.resize(h_particles.OCCUPIED_UNIQUE_LEN, 0);
    //h_particles.f_free_x.resize(h_particles.FREE_UNIQUE_LEN, 0);
    //h_particles.f_free_y.resize(h_particles.FREE_UNIQUE_LEN, 0);

    //d_particles.f_occupied_x.resize(h_particles.OCCUPIED_UNIQUE_LEN);
    //d_particles.f_occupied_y.resize(h_particles.OCCUPIED_UNIQUE_LEN);
    //d_particles.f_free_x.resize(h_particles.FREE_UNIQUE_LEN);
    //d_particles.f_free_y.resize(h_particles.FREE_UNIQUE_LEN);
}

void exec_map_restructure(DeviceParticles& d_particles, Device2DUniqueFinder& d_unique_occupied, 
    Device2DUniqueFinder& d_unique_free, HostMap& h_map) {

    int threadsPerBlock = h_map.GRID_WIDTH;
    int blocksPerGrid = 1;
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles.f_occupied_unique_x), THRUST_RAW_CAST(d_particles.f_occupied_unique_y), SEP,
        THRUST_RAW_CAST(d_unique_occupied.s_map), THRUST_RAW_CAST(d_unique_occupied.s_in_col), h_map.GRID_WIDTH, h_map.GRID_HEIGHT);
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles.f_free_unique_x), THRUST_RAW_CAST(d_particles.f_free_unique_y), SEP,
        THRUST_RAW_CAST(d_unique_free.s_map), THRUST_RAW_CAST(d_unique_free.s_in_col), h_map.GRID_WIDTH, h_map.GRID_HEIGHT);
    cudaDeviceSynchronize();
}

void exec_log_odds(DeviceMap& d_map, DeviceParticles& d_particles,
    HostMap& h_map, HostParticles& h_particles, GeneralInfo& general_info) {

    int threadsPerBlock = 256;
    int blocksPerGrid = (h_particles.OCCUPIED_UNIQUE_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_update_log_odds << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_map.s_log_odds), SEP,
        THRUST_RAW_CAST(d_particles.f_occupied_unique_x), THRUST_RAW_CAST(d_particles.f_occupied_unique_y),
        2 * general_info.log_t, h_map.GRID_WIDTH, h_map.GRID_HEIGHT, h_particles.OCCUPIED_UNIQUE_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = 256;
    blocksPerGrid = (h_particles.FREE_UNIQUE_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_update_log_odds << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_map.s_log_odds), SEP,
        THRUST_RAW_CAST(d_particles.f_free_unique_x), THRUST_RAW_CAST(d_particles.f_free_unique_y),
        (-1) * general_info.log_t, h_map.GRID_WIDTH, h_map.GRID_HEIGHT, h_particles.FREE_UNIQUE_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = 256;
    blocksPerGrid = ((h_map.GRID_WIDTH * h_map.GRID_HEIGHT) + threadsPerBlock - 1) / threadsPerBlock;
    kernel_update_map << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_map.s_grid_map), SEP, THRUST_RAW_CAST(d_map.s_log_odds), LOG_ODD_PRIOR, WALL, FREE, h_map.GRID_WIDTH * h_map.GRID_HEIGHT);
    cudaDeviceSynchronize();
}



#endif
