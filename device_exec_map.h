#ifndef _DEVICE_EXEC_MAP_H_
#define _DEVICE_EXEC_MAP_H_

#include "headers.h"
#include "structures.h"
#include "kernels.cuh"
#include "kernels_map.cuh"
#include "kernels_utils.cuh"
#include "kernels_robot.cuh"

void exec_world_to_image_transform_step_1(
    DevicePositionTransition& d_position_transition, DeviceParticlesData& d_particles_data, DeviceMeasurements& d_measurements,
    HostMeasurements& res_measurements) {

    kernel_matrix_mul_3x3 << < 1, 1 >> > (
        THRUST_RAW_CAST(d_position_transition.transition_single_world_body),
        THRUST_RAW_CAST(d_position_transition.transition_body_lidar), THRUST_RAW_CAST(d_position_transition.transition_single_world_lidar));
    cudaDeviceSynchronize();

    int threadsPerBlock = 1;
    int blocksPerGrid = res_measurements.LIDAR_COORDS_LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles_data.particles_world_x), THRUST_RAW_CAST(d_particles_data.particles_world_y), SEP,
        THRUST_RAW_CAST(d_position_transition.transition_single_world_lidar), THRUST_RAW_CAST(d_measurements.lidar_coords),
        res_measurements.LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();
}

void exec_map_extend(DeviceMapData& d_map_data, DeviceMeasurements& d_measurements, DeviceParticlesData& d_particles_data,
    DeviceUniqueManager& d_unique_manager, HostMapData& res_map_data, HostMeasurements& res_measurements, HostUniqueManager& res_unique_manager,
    GeneralInfo& general_info, bool& EXTEND) {

    int threadsPerBlock = 256;
    int blocksPerGrid = (res_measurements.LIDAR_COORDS_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_check_map_extend_less << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_map_data.should_extend), SEP, THRUST_RAW_CAST(d_particles_data.particles_world_x), res_map_data.xmin, 0, res_measurements.LIDAR_COORDS_LEN);
    kernel_check_map_extend_less << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_map_data.should_extend), SEP, THRUST_RAW_CAST(d_particles_data.particles_world_y), res_map_data.ymin, 1, res_measurements.LIDAR_COORDS_LEN);

    kernel_check_map_extend_greater << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_map_data.should_extend), SEP, THRUST_RAW_CAST(d_particles_data.particles_world_x), res_map_data.xmax, 2, res_measurements.LIDAR_COORDS_LEN);
    kernel_check_map_extend_greater << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_map_data.should_extend), SEP, THRUST_RAW_CAST(d_particles_data.particles_world_y), res_map_data.ymax, 3, res_measurements.LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    res_map_data.should_extend.assign(d_map_data.should_extend.begin(), d_map_data.should_extend.end());

    int xmin_pre = res_map_data.xmin;
    int ymax_pre = res_map_data.ymax;

    EXTEND = false;
    if (res_map_data.should_extend[0] != 0) {
        EXTEND = true;
        res_map_data.xmin = res_map_data.xmin * 2;
    }
    else if (res_map_data.should_extend[2] != 0) {
        EXTEND = true;
        res_map_data.xmax = res_map_data.xmax * 2;
    }
    else if (res_map_data.should_extend[1] != 0) {
        EXTEND = true;
        res_map_data.ymin = res_map_data.ymin * 2;
    }
    else if (res_map_data.should_extend[3] != 0) {
        EXTEND = true;
        res_map_data.ymax = res_map_data.ymax * 2;
    }

    for (int i = 0; i < 4; i++)
        std::cout << "Should Extend: " << res_map_data.should_extend[i] << std::endl;


    if (EXTEND == true) {

        auto start_extend = std::chrono::high_resolution_clock::now();

        d_measurements.coord.resize(2);
        //HostMeasurements res_measurements;
        res_measurements.coord.resize(2);

        kernel_position_to_image << <1, 1 >> > (THRUST_RAW_CAST(d_measurements.coord), SEP,
            xmin_pre, ymax_pre, general_info.res, res_map_data.xmin, res_map_data.ymax);
        cudaDeviceSynchronize();

        res_measurements.coord.assign(d_measurements.coord.begin(), d_measurements.coord.end());

        device_vector<int> dvec_clone_grid_map(res_map_data.GRID_WIDTH * res_map_data.GRID_HEIGHT);
        dvec_clone_grid_map.assign(d_map_data.grid_map.begin(), d_map_data.grid_map.end());

        device_vector<float> dvec_clone_log_odds(res_map_data.GRID_WIDTH * res_map_data.GRID_HEIGHT);
        dvec_clone_log_odds.assign(d_map_data.log_odds.begin(), d_map_data.log_odds.end());

        const int PRE_GRID_WIDTH = res_map_data.GRID_WIDTH;
        const int PRE_GRID_HEIGHT = res_map_data.GRID_HEIGHT;
        res_map_data.GRID_WIDTH = ceil((res_map_data.ymax - res_map_data.ymin) / general_info.res + 1);
        res_map_data.GRID_HEIGHT = ceil((res_map_data.xmax - res_map_data.xmin) / general_info.res + 1);

        const int PRE_GRID_SIZE = PRE_GRID_WIDTH * PRE_GRID_HEIGHT;
        const int NEW_GRID_SIZE = res_map_data.GRID_WIDTH * res_map_data.GRID_HEIGHT;

        d_map_data.grid_map.clear();
        d_map_data.log_odds.clear();
        d_map_data.grid_map.resize(res_map_data.GRID_WIDTH * res_map_data.GRID_HEIGHT, 0);
        d_map_data.log_odds.resize(res_map_data.GRID_WIDTH * res_map_data.GRID_HEIGHT, LOG_ODD_PRIOR);

        res_map_data.grid_map.clear();
        res_map_data.log_odds.clear();
        res_map_data.grid_map.resize(res_map_data.GRID_WIDTH * res_map_data.GRID_HEIGHT, 0);
        res_map_data.log_odds.resize(res_map_data.GRID_WIDTH * res_map_data.GRID_HEIGHT, 0);

        threadsPerBlock = 256;
        blocksPerGrid = (PRE_GRID_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        kernel_2d_copy_with_offset << <blocksPerGrid, threadsPerBlock >> > (
            THRUST_RAW_CAST(d_map_data.grid_map), THRUST_RAW_CAST(d_map_data.log_odds), SEP,
            THRUST_RAW_CAST(dvec_clone_grid_map), THRUST_RAW_CAST(dvec_clone_log_odds),
            res_measurements.coord[0], res_measurements.coord[1],
            PRE_GRID_HEIGHT, res_map_data.GRID_HEIGHT, PRE_GRID_SIZE);
        cudaDeviceSynchronize();

        d_unique_manager.free_unique_counter_col.clear();
        d_unique_manager.free_unique_counter_col.resize(res_map_data.GRID_WIDTH + 1);
        res_unique_manager.free_unique_counter_col.resize(res_map_data.GRID_WIDTH + 1);

        d_unique_manager.occupied_unique_counter_col.clear();
        d_unique_manager.occupied_unique_counter_col.resize(res_map_data.GRID_WIDTH + 1);
        res_unique_manager.occupied_unique_counter_col.resize(res_map_data.GRID_WIDTH + 1);

        d_unique_manager.free_map_2d.clear();
        d_unique_manager.occupied_map_2d.clear();
        d_unique_manager.free_map_2d.resize(res_map_data.GRID_WIDTH * res_map_data.GRID_HEIGHT);
        d_unique_manager.occupied_map_2d.resize(res_map_data.GRID_WIDTH * res_map_data.GRID_HEIGHT);

        auto stop_extend = std::chrono::high_resolution_clock::now();
        auto duration_extend = std::chrono::duration_cast<std::chrono::microseconds>(stop_extend - start_extend);

        std::cout << std::endl;
        std::cout << "Time taken by function (Extend): " << duration_extend.count() << " microseconds" << std::endl;

        res_map_data.grid_map.assign(d_map_data.grid_map.begin(), d_map_data.grid_map.end());
        res_map_data.log_odds.assign(d_map_data.log_odds.begin(), d_map_data.log_odds.end());
    }

    for (int i = 0; i < 4; i++)
        std::cout << "Should Extend: " << res_map_data.should_extend[i] << std::endl;
}


void exec_world_to_image_transform_step_2(DeviceMeasurements& d_measurements, DeviceParticlesData& d_particles_data,
    DevicePositionTransition& d_position_transition,
    HostMapData& res_map_data, HostMeasurements& res_measurements, GeneralInfo& general_info) {

    int threadsPerBlock = 1;
    int blocksPerGrid = res_measurements.LIDAR_COORDS_LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles_data.particles_occupied_x), THRUST_RAW_CAST(d_particles_data.particles_occupied_y), SEP,
        THRUST_RAW_CAST(d_position_transition.transition_single_world_lidar), THRUST_RAW_CAST(d_measurements.lidar_coords),
        general_info.res, res_map_data.xmin, res_map_data.ymax, res_measurements.LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    kernel_position_to_image << < 1, 1 >> > (
        THRUST_RAW_CAST(d_position_transition.position_image_body), SEP,
        THRUST_RAW_CAST(d_position_transition.transition_single_world_lidar), general_info.res, res_map_data.xmin, res_map_data.ymax);
    cudaDeviceSynchronize();
}

void exec_bresenham(DeviceParticlesData& d_particles_data, DevicePositionTransition& d_position_transition,
    HostParticlesData& res_particles_data, const int MAX_DIST_IN_MAP) {

    int PARTICLE_UNIQUE_COUNTER = res_particles_data.PARTICLES_OCCUPIED_LEN + 1;

    int threadsPerBlock = 256;
    int blocksPerGrid = (res_particles_data.PARTICLES_OCCUPIED_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_bresenham << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles_data.particles_free_x_max), THRUST_RAW_CAST(d_particles_data.particles_free_y_max), THRUST_RAW_CAST(d_particles_data.particles_free_counter), SEP,
        THRUST_RAW_CAST(d_particles_data.particles_occupied_x), THRUST_RAW_CAST(d_particles_data.particles_occupied_y), THRUST_RAW_CAST(d_position_transition.position_image_body),
        res_particles_data.PARTICLES_OCCUPIED_LEN, MAX_DIST_IN_MAP);
    cudaDeviceSynchronize();
    thrust::exclusive_scan(thrust::device, d_particles_data.particles_free_counter.begin(), d_particles_data.particles_free_counter.end(), d_particles_data.particles_free_counter.begin(), 0);

    res_particles_data.particles_free_counter.assign(d_particles_data.particles_free_counter.begin(), d_particles_data.particles_free_counter.end());
    d_particles_data.particles_free_idx.assign(d_particles_data.particles_free_counter.begin(), d_particles_data.particles_free_counter.end());

    res_particles_data.PARTICLES_FREE_LEN = res_particles_data.particles_free_counter[PARTICLE_UNIQUE_COUNTER - 1];
    printf("^^^ PARTICLES_FREE_LEN = %d\n", res_particles_data.PARTICLES_FREE_LEN);

    d_particles_data.particles_free_x.resize(res_particles_data.PARTICLES_FREE_LEN);
    d_particles_data.particles_free_y.resize(res_particles_data.PARTICLES_FREE_LEN);


    kernel_bresenham_rearrange << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles_data.particles_free_x), THRUST_RAW_CAST(d_particles_data.particles_free_y), SEP,
        THRUST_RAW_CAST(d_particles_data.particles_free_x_max), THRUST_RAW_CAST(d_particles_data.particles_free_y_max),
        THRUST_RAW_CAST(d_particles_data.particles_free_counter), MAX_DIST_IN_MAP, res_particles_data.PARTICLES_OCCUPIED_LEN);
    cudaDeviceSynchronize();
}

void reinit_map_idx_vars(DeviceUniqueManager& d_unique_manager, HostParticlesData& res_particles_data, HostUniqueManager& res_unique_manager) {

    res_unique_manager.free_map_idx[1] = res_particles_data.PARTICLES_FREE_LEN;
    d_unique_manager.free_map_idx.assign(res_unique_manager.free_map_idx.begin(), res_unique_manager.free_map_idx.end());
}

void exec_create_map(DeviceParticlesData& d_particles_data, DeviceUniqueManager& d_unique_manager,
    HostMapData& res_map_data, HostParticlesData& res_particles_data) {

    int threadsPerBlock = 256;
    int blocksPerGrid = 1;
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_unique_manager.occupied_map_2d), SEP,
        THRUST_RAW_CAST(d_particles_data.particles_occupied_x), THRUST_RAW_CAST(d_particles_data.particles_occupied_y),
        THRUST_RAW_CAST(d_unique_manager.occupied_map_idx), res_particles_data.PARTICLES_OCCUPIED_LEN,
        res_map_data.GRID_WIDTH, res_map_data.GRID_HEIGHT, NUM_PARTICLES);
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_unique_manager.free_map_2d), SEP,
        THRUST_RAW_CAST(d_particles_data.particles_free_x), THRUST_RAW_CAST(d_particles_data.particles_free_y),
        THRUST_RAW_CAST(d_unique_manager.free_map_idx), res_particles_data.PARTICLES_FREE_LEN, res_map_data.GRID_WIDTH, res_map_data.GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();

    threadsPerBlock = res_map_data.GRID_WIDTH;
    blocksPerGrid = 1;
    kernel_2d_map_counter << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_unique_manager.occupied_unique_counter), THRUST_RAW_CAST(d_unique_manager.occupied_unique_counter_col), SEP,
        THRUST_RAW_CAST(d_unique_manager.occupied_map_2d), res_map_data.GRID_WIDTH, res_map_data.GRID_HEIGHT);
    kernel_2d_map_counter << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_unique_manager.free_unique_counter), THRUST_RAW_CAST(d_unique_manager.free_unique_counter_col), SEP,
        THRUST_RAW_CAST(d_unique_manager.free_map_2d), res_map_data.GRID_WIDTH, res_map_data.GRID_HEIGHT);
    cudaDeviceSynchronize();

    threadsPerBlock = 1;
    blocksPerGrid = 1;
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (THRUST_RAW_CAST(d_unique_manager.occupied_unique_counter_col), res_map_data.GRID_WIDTH);
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (THRUST_RAW_CAST(d_unique_manager.free_unique_counter_col), res_map_data.GRID_WIDTH);
    cudaDeviceSynchronize();
}

void reinit_map_vars(DeviceParticlesData& d_particles_data, DeviceUniqueManager& d_unique_manager,
    HostParticlesData& res_particles_data, HostUniqueManager& res_unique_manager) {

    res_unique_manager.occupied_unique_counter.assign(d_unique_manager.occupied_unique_counter.begin(), d_unique_manager.occupied_unique_counter.end());
    res_unique_manager.free_unique_counter.assign(d_unique_manager.free_unique_counter.begin(), d_unique_manager.free_unique_counter.end());

    res_particles_data.PARTICLES_OCCUPIED_UNIQUE_LEN = res_unique_manager.occupied_unique_counter[0];
    res_particles_data.PARTICLES_FREE_UNIQUE_LEN = res_unique_manager.free_unique_counter[0];

    res_particles_data.particles_occupied_x.resize(res_particles_data.PARTICLES_OCCUPIED_UNIQUE_LEN, 0);
    res_particles_data.particles_occupied_y.resize(res_particles_data.PARTICLES_OCCUPIED_UNIQUE_LEN, 0);
    res_particles_data.particles_free_x.resize(res_particles_data.PARTICLES_FREE_UNIQUE_LEN, 0);
    res_particles_data.particles_free_y.resize(res_particles_data.PARTICLES_FREE_UNIQUE_LEN, 0);

    d_particles_data.particles_occupied_x.resize(res_particles_data.PARTICLES_OCCUPIED_UNIQUE_LEN);
    d_particles_data.particles_occupied_y.resize(res_particles_data.PARTICLES_OCCUPIED_UNIQUE_LEN);
    d_particles_data.particles_free_x.resize(res_particles_data.PARTICLES_FREE_UNIQUE_LEN);
    d_particles_data.particles_free_y.resize(res_particles_data.PARTICLES_FREE_UNIQUE_LEN);
}

void exec_map_restructure(DeviceParticlesData& d_particles_data, DeviceUniqueManager& d_unique_manager,
    HostMapData& res_map_data) {

    int threadsPerBlock = res_map_data.GRID_WIDTH;
    int blocksPerGrid = 1;
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles_data.particles_occupied_x), THRUST_RAW_CAST(d_particles_data.particles_occupied_y), SEP,
        THRUST_RAW_CAST(d_unique_manager.occupied_map_2d), THRUST_RAW_CAST(d_unique_manager.occupied_unique_counter_col), res_map_data.GRID_WIDTH, res_map_data.GRID_HEIGHT);
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles_data.particles_free_x), THRUST_RAW_CAST(d_particles_data.particles_free_y), SEP,
        THRUST_RAW_CAST(d_unique_manager.free_map_2d), THRUST_RAW_CAST(d_unique_manager.free_unique_counter_col), res_map_data.GRID_WIDTH, res_map_data.GRID_HEIGHT);
    cudaDeviceSynchronize();
}

void exec_log_odds(DeviceMapData& d_map_data, DeviceParticlesData& d_particles_data,
    HostMapData& res_map_data, HostParticlesData& res_particles_data, GeneralInfo& general_info) {

    int threadsPerBlock = 256;
    int blocksPerGrid = (res_particles_data.PARTICLES_OCCUPIED_UNIQUE_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_update_log_odds << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_map_data.log_odds), SEP,
        THRUST_RAW_CAST(d_particles_data.particles_occupied_x), THRUST_RAW_CAST(d_particles_data.particles_occupied_y),
        2 * general_info.log_t, res_map_data.GRID_WIDTH, res_map_data.GRID_HEIGHT, res_particles_data.PARTICLES_OCCUPIED_UNIQUE_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = 256;
    blocksPerGrid = (res_particles_data.PARTICLES_FREE_UNIQUE_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_update_log_odds << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_map_data.log_odds), SEP,
        THRUST_RAW_CAST(d_particles_data.particles_free_x), THRUST_RAW_CAST(d_particles_data.particles_free_y),
        (-1) * general_info.log_t, res_map_data.GRID_WIDTH, res_map_data.GRID_HEIGHT, res_particles_data.PARTICLES_FREE_UNIQUE_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = 256;
    blocksPerGrid = ((res_map_data.GRID_WIDTH * res_map_data.GRID_HEIGHT) + threadsPerBlock - 1) / threadsPerBlock;
    kernel_update_map << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_map_data.grid_map), SEP, THRUST_RAW_CAST(d_map_data.log_odds), LOG_ODD_PRIOR, WALL, FREE, res_map_data.GRID_WIDTH * res_map_data.GRID_HEIGHT);
    cudaDeviceSynchronize();
}



#endif
