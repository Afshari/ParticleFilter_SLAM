#ifndef _TEST_MODULAR_MAP_H_
#define _TEST_MODULAR_MAP_H_

#include "headers.h"
#include "host_asserts.h"
#include "host_utils.h"
#include "kernels.cuh"
#include "kernels_map.cuh"
#include "kernels_utils.cuh"
#include "kernels_robot.cuh"

host_vector<int> hvec_occupied_map_idx(2, 0);
host_vector<int> hvec_free_map_idx(2, 0);

HostMapData h_map_data;
HostMapData h_map_data_bg;
HostMapData h_map_data_post;
GeneralInfo general_info;
HostMeasurements h_measurements;
HostParticlesData h_particles_data;
HostPositionTransition h_position_transition;

void host_map();                                // Step Y
void test_map_func();

int threadsPerBlock = 1;
int blocksPerGrid = 1;

int test_map_main() {

    const int data_len = 6;
    int data[data_len] = { 1, 0, 2, 2, 1, 3 };
    int* d_data = NULL;
    gpuErrchk(cudaMalloc((void**)&d_data, data_len * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_data, data, data_len * sizeof(int), cudaMemcpyHostToDevice));

    thrust::exclusive_scan(thrust::host, data, data + 6, data, 0);
    thrust::exclusive_scan(thrust::device, d_data, d_data + 6, d_data, 0);

    auto start_read_data_file = std::chrono::high_resolution_clock::now();
    read_update_map(720, h_map_data, h_map_data_bg,
        h_map_data_post, general_info, h_measurements, h_particles_data, h_position_transition);
    auto stop_read_data_file = std::chrono::high_resolution_clock::now();

    auto duration_read_data_file = std::chrono::duration_cast<std::chrono::milliseconds>(stop_read_data_file - start_read_data_file);
    std::cout << "Time taken by function (Read Data File): " << duration_read_data_file.count() << " milliseconds" << std::endl;


    host_map();

    //test_map_func();

    return 0;
}

void alloc_init_transition_vars(DevicePositionTransition& d_position_transition, HostPositionTransition& res_position_transition) {

    d_position_transition.transition_body_lidar.resize(9);
    d_position_transition.transition_single_world_body.resize(9);
    d_position_transition.transition_single_world_lidar.resize(9);
    d_position_transition.transition_body_lidar.assign(hvec_transition_body_lidar.begin(), hvec_transition_body_lidar.end());
    d_position_transition.transition_single_world_body.assign(h_position_transition.transition_world_body.begin(), h_position_transition.transition_world_body.end());
    d_position_transition.position_image_body.resize(2);

    res_position_transition.transition_world_lidar.resize(9);
    res_position_transition.position_image_body.resize(2);
}

void alloc_init_map_vars(DeviceMapData& d_map_data, HostMapData& res_map_data, HostMapData& h_map_data) {

    res_map_data.GRID_WIDTH     = h_map_data.GRID_WIDTH;
    res_map_data.GRID_HEIGHT    = h_map_data.GRID_HEIGHT;
    res_map_data.xmin           = h_map_data.xmin;
    res_map_data.xmax           = h_map_data.xmax;
    res_map_data.ymin           = h_map_data.ymin;
    res_map_data.ymax           = h_map_data.ymax;

    d_map_data.grid_map.resize(res_map_data.GRID_WIDTH * res_map_data.GRID_HEIGHT, 0);
    d_map_data.grid_map.assign(h_map_data.grid_map.begin(), h_map_data.grid_map.end());
    d_map_data.log_odds.resize(res_map_data.GRID_WIDTH * res_map_data.GRID_HEIGHT);
    d_map_data.should_extend.resize(4, 0);
    d_map_data.log_odds.assign(h_map_data.log_odds.begin(), h_map_data.log_odds.end());

    res_map_data.grid_map.resize(res_map_data.GRID_WIDTH * res_map_data.GRID_HEIGHT, 0);
    res_map_data.log_odds.resize(res_map_data.GRID_WIDTH * res_map_data.GRID_HEIGHT, 0);
    res_map_data.should_extend.resize(4, 0);
}

void alloc_init_measurement_vars(DeviceMeasurements& d_measurements, HostMeasurements& res_measurements, HostMeasurements& h_measurements) {

    res_measurements.LIDAR_COORDS_LEN = h_measurements.LIDAR_COORDS_LEN;

    d_measurements.lidar_coords.resize(2 * res_measurements.LIDAR_COORDS_LEN);
    d_measurements.lidar_coords.assign(h_measurements.lidar_coords.begin(), h_measurements.lidar_coords.end());
}

void alloc_init_particles_vars(DeviceParticlesData& d_particles_data, HostParticlesData& res_particles_data,
    HostMeasurements& h_measurements, HostParticlesData& h_particles_data, const int MAX_DIST_IN_MAP) {

    int PARTICLE_UNIQUE_COUNTER = h_particles_data.PARTICLES_OCCUPIED_LEN + 1;

    res_particles_data.PARTICLES_OCCUPIED_LEN = h_particles_data.PARTICLES_OCCUPIED_LEN;
    res_particles_data.PARTICLES_FREE_LEN = h_particles_data.PARTICLES_FREE_LEN;

    d_particles_data.particles_occupied_x.resize(h_measurements.LIDAR_COORDS_LEN);
    d_particles_data.particles_occupied_y.resize(h_measurements.LIDAR_COORDS_LEN);
    d_particles_data.particles_world_x.resize(h_measurements.LIDAR_COORDS_LEN);
    d_particles_data.particles_world_y.resize(h_measurements.LIDAR_COORDS_LEN);
    d_particles_data.particles_free_x_max.resize(h_particles_data.PARTICLES_OCCUPIED_LEN * MAX_DIST_IN_MAP, 0);
    d_particles_data.particles_free_y_max.resize(h_particles_data.PARTICLES_OCCUPIED_LEN * MAX_DIST_IN_MAP, 0);
    d_particles_data.particles_free_counter.resize(PARTICLE_UNIQUE_COUNTER, 0);
    d_particles_data.particles_free_idx.resize(h_particles_data.PARTICLES_OCCUPIED_LEN);

    res_particles_data.particles_occupied_x.resize(h_measurements.LIDAR_COORDS_LEN);
    res_particles_data.particles_occupied_y.resize(h_measurements.LIDAR_COORDS_LEN);
    res_particles_data.particles_world_x.resize(h_measurements.LIDAR_COORDS_LEN);
    res_particles_data.particles_world_y.resize(h_measurements.LIDAR_COORDS_LEN);
    res_particles_data.particles_free_counter.resize(PARTICLE_UNIQUE_COUNTER);
}

void alloc_init_unique_vars(DeviceUniqueManager& d_unique_manager, HostUniqueManager& res_unique_manager,
    HostMapData& res_map_data, const host_vector<int>& hvec_occupied_map_idx, const host_vector<int>& hvec_free_map_idx) {

    d_unique_manager.free_map_2d.resize(res_map_data.GRID_WIDTH * res_map_data.GRID_HEIGHT);
    d_unique_manager.free_unique_counter.resize(1);
    d_unique_manager.free_unique_counter_col.resize(res_map_data.GRID_WIDTH + 1);
    d_unique_manager.occupied_map_2d.resize(res_map_data.GRID_WIDTH * res_map_data.GRID_HEIGHT);
    d_unique_manager.occupied_unique_counter.resize(1);
    d_unique_manager.occupied_unique_counter_col.resize(res_map_data.GRID_WIDTH + 1);
    d_unique_manager.occupied_map_idx.resize(2);
    d_unique_manager.free_map_idx.resize(2);

    d_unique_manager.occupied_map_idx.assign(hvec_occupied_map_idx.begin(), hvec_occupied_map_idx.end());
    d_unique_manager.free_map_idx.assign(hvec_free_map_idx.begin(), hvec_free_map_idx.end());

    res_unique_manager.free_unique_counter.resize(1);
    res_unique_manager.free_unique_counter_col.resize(res_map_data.GRID_WIDTH + 1);
    res_unique_manager.occupied_unique_counter.resize(1);
    res_unique_manager.occupied_unique_counter_col.resize(res_map_data.GRID_WIDTH + 1);
    res_unique_manager.occupied_map_idx.resize(2);
    res_unique_manager.free_map_idx.resize(2);
}

void exec_world_to_image_transform_step_1(
    DevicePositionTransition& d_position_transition, DeviceParticlesData& d_particles_data, DeviceMeasurements& d_measurements,
    HostMeasurements& res_measurements) {

    kernel_matrix_mul_3x3 << < 1, 1 >> > (
        THRUST_RAW_CAST(d_position_transition.transition_single_world_body),
        THRUST_RAW_CAST(d_position_transition.transition_body_lidar), THRUST_RAW_CAST(d_position_transition.transition_single_world_lidar));
    cudaDeviceSynchronize();

    threadsPerBlock = 1;
    blocksPerGrid = res_measurements.LIDAR_COORDS_LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles_data.particles_world_x), THRUST_RAW_CAST(d_particles_data.particles_world_y), SEP,
        THRUST_RAW_CAST(d_position_transition.transition_single_world_lidar), THRUST_RAW_CAST(d_measurements.lidar_coords), 
        res_measurements.LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();
}

void exec_map_extend(DeviceMapData& d_map_data, DeviceMeasurements& d_measurements, DeviceParticlesData& d_particles_data, 
    DeviceUniqueManager& d_unique_manager, HostMapData& res_map_data, HostMeasurements& res_measurements, HostUniqueManager& res_unique_manager) {

    threadsPerBlock = 256;
    blocksPerGrid = (res_measurements.LIDAR_COORDS_LEN + threadsPerBlock - 1) / threadsPerBlock;
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

    bool EXTEND = false;
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

    printf("xmin=%d, xmax=%d, ymin=%d, ymax=%d\n", res_map_data.xmin, res_map_data.xmax, res_map_data.ymin, res_map_data.ymax);
    printf("xmin=%d, xmax=%d, ymin=%d, ymax=%d\n", h_map_data_post.xmin, h_map_data_post.xmax, h_map_data_post.ymin, h_map_data_post.ymax);
    assert(EXTEND == h_map_data.b_should_extend);

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
        printf("GRID_WIDTH=%d, AF_GRID_WIDTH=%d, GRID_HEIGHT=%d, AF_GRID_HEIGHT=%d\n",
            res_map_data.GRID_WIDTH, res_map_data.GRID_HEIGHT, h_map_data_post.GRID_WIDTH, h_map_data_post.GRID_HEIGHT);
        assert(res_map_data.GRID_WIDTH == h_map_data_post.GRID_WIDTH);
        assert(res_map_data.GRID_HEIGHT == h_map_data_post.GRID_HEIGHT);

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

        int error_map = 0;
        int error_log = 0;
        for (int i = 0; i < (res_map_data.GRID_WIDTH * res_map_data.GRID_HEIGHT); i++) {
            if (res_map_data.grid_map[i] != h_map_data_bg.grid_map[i]) {
                error_map += 1;
                //printf("Grid Map: %d <> %d\n", res_map_data.grid_map[i], bg_grid_map[i]);
            }
            if (abs(res_map_data.log_odds[i] - h_map_data_bg.log_odds[i]) > 1e-4) {
                error_log += 1;
                //printf("Log Odds: (%d) %f <> %f\n", i, res_map_data.log_odds[i], bg_log_odds[i]);
            }
            if (error_log > 200)
                break;
        }
        printf("Map Erros: %d\n", error_map);
        printf("Log Erros: %d\n", error_log);
    }

    for (int i = 0; i < 4; i++)
        std::cout << "Should Extend: " << res_map_data.should_extend[i] << std::endl;
}

void exec_world_to_image_transform_step_2(DeviceMeasurements& d_measurements, DeviceParticlesData& d_particles_data, 
    DevicePositionTransition& d_position_transition,
    HostMapData& res_map_data, HostMeasurements& res_measurements, GeneralInfo& general_info) {

    threadsPerBlock = 1;
    blocksPerGrid = res_measurements.LIDAR_COORDS_LEN;
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

void assert_world_to_image_transform(DeviceParticlesData& d_particles_data, DevicePositionTransition& d_position_transition, 
    HostMeasurements& res_measurements, HostParticlesData& res_particles_data, HostPositionTransition& res_position_transition) {

    res_position_transition.transition_world_lidar.assign(d_position_transition.transition_single_world_lidar.begin(),
        d_position_transition.transition_single_world_lidar.end());
    res_particles_data.particles_occupied_x.assign(d_particles_data.particles_occupied_x.begin(), d_particles_data.particles_occupied_x.end());
    res_particles_data.particles_occupied_y.assign(d_particles_data.particles_occupied_y.begin(), d_particles_data.particles_occupied_y.end());
    res_particles_data.particles_world_x.assign(d_particles_data.particles_world_x.begin(), d_particles_data.particles_world_x.end());
    res_particles_data.particles_world_y.assign(d_particles_data.particles_world_y.begin(), d_particles_data.particles_world_y.end());
    res_position_transition.position_image_body.assign(d_position_transition.position_image_body.begin(),
        d_position_transition.position_image_body.end());

    ASSERT_transition_world_lidar(res_position_transition.transition_world_lidar.data(), h_position_transition.transition_world_lidar.data(), 9, false);
    ASSERT_particles_world_frame(res_particles_data.particles_world_x.data(), res_particles_data.particles_world_y.data(),
        h_particles_data.particles_world_x.data(), h_particles_data.particles_world_y.data(), res_measurements.LIDAR_COORDS_LEN, false);
    ASSERT_processed_measurements(res_particles_data.particles_occupied_x.data(), res_particles_data.particles_occupied_y.data(),
        h_particles_data.particles_occupied_x.data(), h_particles_data.particles_occupied_y.data(), res_measurements.LIDAR_COORDS_LEN);
    ASSERT_position_image_body(res_position_transition.position_image_body.data(), h_position_transition.position_image_body.data(), true, true);
}

void exec_bresenham(DeviceParticlesData& d_particles_data, DevicePositionTransition& d_position_transition,
    HostParticlesData& res_particles_data, const int MAX_DIST_IN_MAP) {

    int PARTICLE_UNIQUE_COUNTER = res_particles_data.PARTICLES_OCCUPIED_LEN + 1;

    threadsPerBlock = 256;
    blocksPerGrid = (res_particles_data.PARTICLES_OCCUPIED_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_bresenham << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles_data.particles_free_x_max), THRUST_RAW_CAST(d_particles_data.particles_free_y_max), THRUST_RAW_CAST(d_particles_data.particles_free_counter), SEP,
        THRUST_RAW_CAST(d_particles_data.particles_occupied_x), THRUST_RAW_CAST(d_particles_data.particles_occupied_y), THRUST_RAW_CAST(d_position_transition.position_image_body),
        res_particles_data.PARTICLES_OCCUPIED_LEN, MAX_DIST_IN_MAP);
    cudaDeviceSynchronize();
    thrust::exclusive_scan(thrust::device, d_particles_data.particles_free_counter.begin(), d_particles_data.particles_free_counter.end(), d_particles_data.particles_free_counter.begin(), 0); // in-place scan

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

void assert_bresenham(DeviceParticlesData& d_particles_data, 
    HostParticlesData& res_particles_data, HostMeasurements& res_measurements, DeviceMeasurements& d_measurements) {

    res_particles_data.particles_free_x.resize(res_particles_data.PARTICLES_FREE_LEN);
    res_particles_data.particles_free_y.resize(res_particles_data.PARTICLES_FREE_LEN);

    res_particles_data.particles_free_x.assign(d_particles_data.particles_free_x.begin(), d_particles_data.particles_free_x.end());
    res_particles_data.particles_free_y.assign(d_particles_data.particles_free_y.begin(), d_particles_data.particles_free_y.end());

    printf("~~$ PARTICLES_FREE_LEN = %d\n", res_particles_data.PARTICLES_FREE_LEN);

    ASSERT_particles_free_index(res_particles_data.particles_free_counter.data(), h_particles_data.particles_free_idx.data(),
        res_particles_data.PARTICLES_OCCUPIED_LEN, false);
    ASSERT_particles_free_new_len(res_particles_data.PARTICLES_FREE_LEN, h_particles_data.PARTICLES_FREE_LEN);
    ASSERT_particles_free(res_particles_data.particles_free_x.data(), res_particles_data.particles_free_y.data(),
        h_particles_data.particles_free_x.data(), h_particles_data.particles_free_y.data(), res_particles_data.PARTICLES_FREE_LEN);

}

void reinit_map_idx_vars(DeviceUniqueManager& d_unique_manager, HostParticlesData& res_particles_data, HostUniqueManager& res_unique_manager) {

    res_unique_manager.free_map_idx[1] = res_particles_data.PARTICLES_FREE_LEN;
    d_unique_manager.free_map_idx.assign(res_unique_manager.free_map_idx.begin(), res_unique_manager.free_map_idx.end());
}

void exec_create_map(DeviceParticlesData& d_particles_data, DeviceUniqueManager& d_unique_manager,
    HostMapData& res_map_data, HostParticlesData& res_particles_data) {

    threadsPerBlock = 256;
    blocksPerGrid = 1;
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

    printf("\n--> Occupied Unique: %d, %d\n", res_particles_data.PARTICLES_OCCUPIED_UNIQUE_LEN, h_particles_data.PARTICLES_OCCUPIED_UNIQUE_LEN);
    assert(res_particles_data.PARTICLES_OCCUPIED_UNIQUE_LEN == h_particles_data.PARTICLES_OCCUPIED_UNIQUE_LEN);
    printf("\n--> Free Unique: %d, %d\n", res_particles_data.PARTICLES_FREE_UNIQUE_LEN, h_particles_data.PARTICLES_FREE_UNIQUE_LEN);
    assert(res_particles_data.PARTICLES_FREE_UNIQUE_LEN == h_particles_data.PARTICLES_FREE_UNIQUE_LEN);

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

    threadsPerBlock = res_map_data.GRID_WIDTH;
    blocksPerGrid = 1;
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles_data.particles_occupied_x), THRUST_RAW_CAST(d_particles_data.particles_occupied_y), SEP,
        THRUST_RAW_CAST(d_unique_manager.occupied_map_2d), THRUST_RAW_CAST(d_unique_manager.occupied_unique_counter_col), res_map_data.GRID_WIDTH, res_map_data.GRID_HEIGHT);
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles_data.particles_free_x), THRUST_RAW_CAST(d_particles_data.particles_free_y), SEP,
        THRUST_RAW_CAST(d_unique_manager.free_map_2d), THRUST_RAW_CAST(d_unique_manager.free_unique_counter_col), res_map_data.GRID_WIDTH, res_map_data.GRID_HEIGHT);
    cudaDeviceSynchronize();
}

void assert_map_restructure(DeviceParticlesData& d_particles_data, HostParticlesData& res_particles_data) {

    res_particles_data.particles_occupied_x.assign(d_particles_data.particles_occupied_x.begin(), d_particles_data.particles_occupied_x.end());
    res_particles_data.particles_occupied_y.assign(d_particles_data.particles_occupied_y.begin(), d_particles_data.particles_occupied_y.end());
    res_particles_data.particles_free_x.assign(d_particles_data.particles_free_x.begin(), d_particles_data.particles_free_x.end());
    res_particles_data.particles_free_y.assign(d_particles_data.particles_free_y.begin(), d_particles_data.particles_free_y.end());

    ASSERT_particles_occupied(res_particles_data.particles_occupied_x.data(), res_particles_data.particles_occupied_y.data(),
        h_particles_data.particles_occupied_unique_x.data(), h_particles_data.particles_occupied_unique_y.data(),
        "Occupied", res_particles_data.PARTICLES_OCCUPIED_UNIQUE_LEN, false);
    ASSERT_particles_occupied(res_particles_data.particles_free_x.data(), res_particles_data.particles_free_y.data(),
        h_particles_data.particles_free_unique_x.data(), h_particles_data.particles_free_unique_y.data(),
        "Free", res_particles_data.PARTICLES_FREE_UNIQUE_LEN, false);
}

void exec_log_odds(DeviceMapData& d_map_data, DeviceParticlesData& d_particles_data, 
    HostMapData& res_map_data, HostParticlesData& res_particles_data, GeneralInfo& general_info) {

    threadsPerBlock = 256;
    blocksPerGrid = (res_particles_data.PARTICLES_OCCUPIED_UNIQUE_LEN + threadsPerBlock - 1) / threadsPerBlock;
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

void assert_log_odds(DeviceMapData& d_map_data, HostMapData& res_map_data) {

    thrust::fill(res_map_data.log_odds.begin(), res_map_data.log_odds.end(), 0);
    res_map_data.log_odds.assign(d_map_data.log_odds.begin(), d_map_data.log_odds.end());
    res_map_data.grid_map.assign(d_map_data.grid_map.begin(), d_map_data.grid_map.end());

    ASSERT_log_odds(res_map_data.log_odds.data(), h_map_data.log_odds.data(), h_map_data_post.log_odds.data(),
        (res_map_data.GRID_WIDTH * res_map_data.GRID_HEIGHT), (h_map_data.GRID_WIDTH * h_map_data.GRID_HEIGHT), false);
    ASSERT_log_odds_maps(res_map_data.grid_map.data(), h_map_data.grid_map.data(), h_map_data_post.grid_map.data(),
        (res_map_data.GRID_WIDTH * res_map_data.GRID_HEIGHT), (h_map_data.GRID_WIDTH * h_map_data.GRID_HEIGHT), false);
    printf("\n");
}

void host_map() {

    printf("/******************************** MAP *******************************/\n");

    int LIDAR_COORDS_LEN = h_measurements.LIDAR_COORDS_LEN;

    /********************* IMAGE TRANSFORM VARIABLES ********************/

    HostPositionTransition res_position_transition;
    DevicePositionTransition d_position_transition;
    alloc_init_transition_vars(d_position_transition, res_position_transition);


    DeviceMeasurements d_measurements;
    HostMeasurements res_measurements;
    alloc_init_measurement_vars(d_measurements, res_measurements, h_measurements);


    /************************ BRESENHAM VARIABLES ***********************/
    int MAX_DIST_IN_MAP = sqrt(pow(h_map_data.GRID_WIDTH, 2) + pow(h_map_data.GRID_HEIGHT, 2));

    HostParticlesData res_particles_data;
    DeviceParticlesData d_particles_data;
    alloc_init_particles_vars(d_particles_data, res_particles_data, h_measurements, h_particles_data, MAX_DIST_IN_MAP);


    /**************************** MAP VARIABLES *************************/

    /************************* LOG-ODDS VARIABLES ***********************/

    DeviceMapData d_map_data;
    HostMapData res_map_data;
    alloc_init_map_vars(d_map_data, res_map_data, h_map_data);

    hvec_occupied_map_idx[1] = res_particles_data.PARTICLES_OCCUPIED_LEN;
    hvec_free_map_idx[1] = 0;

    HostUniqueManager res_unique_manager;
    DeviceUniqueManager d_unique_manager;
    alloc_init_unique_vars(d_unique_manager, res_unique_manager, res_map_data, hvec_occupied_map_idx, hvec_free_map_idx);


    /************************************************************* KERNEL EXECUTION SCOPE *************************************************************/

    /***************** World to IMAGE TRANSFORM KERNEL ******************/
    auto start_world_to_image_transform_1 = std::chrono::high_resolution_clock::now();
    exec_world_to_image_transform_step_1(d_position_transition, d_particles_data, d_measurements, res_measurements);
    auto stop_world_to_image_transform_1 = std::chrono::high_resolution_clock::now();


    auto start_check_extend = std::chrono::high_resolution_clock::now();
    exec_map_extend(d_map_data, d_measurements, d_particles_data, d_unique_manager, 
        res_map_data, res_measurements, res_unique_manager);
    auto stop_check_extend = std::chrono::high_resolution_clock::now();

    auto start_world_to_image_transform_2 = std::chrono::high_resolution_clock::now();
    exec_world_to_image_transform_step_2(d_measurements, d_particles_data, d_position_transition, 
        res_map_data, res_measurements, general_info);
    auto stop_world_to_image_transform_2 = std::chrono::high_resolution_clock::now();

    assert_world_to_image_transform(d_particles_data, d_position_transition,
        res_measurements, res_particles_data, res_position_transition);

    /************************* BRESENHAM KERNEL *************************/
    auto start_bresenham = std::chrono::high_resolution_clock::now();
    exec_bresenham(d_particles_data, d_position_transition, res_particles_data, MAX_DIST_IN_MAP);
    auto stop_bresenham = std::chrono::high_resolution_clock::now();

    assert_bresenham(d_particles_data, res_particles_data, res_measurements, d_measurements);

    /*---------------------------------------------------------------------*/
    /*-------------------- REINITIALIZE MAP VARIABLES ---------------------*/
    //res_unique_manager.free_map_idx[1] = res_particles_data.PARTICLES_FREE_LEN;
    //d_unique_manager.free_map_idx.assign(res_unique_manager.free_map_idx.begin(), res_unique_manager.free_map_idx.end());
    reinit_map_idx_vars(d_unique_manager, res_particles_data, res_unique_manager);

    /************************** CREATE 2D MAP ***************************/
    auto start_create_map = std::chrono::high_resolution_clock::now();
    exec_create_map(d_particles_data, d_unique_manager, res_map_data, res_particles_data);
    auto stop_create_map = std::chrono::high_resolution_clock::now();

    auto start_unique_counter = std::chrono::high_resolution_clock::now();
    auto stop_unique_counter = std::chrono::high_resolution_clock::now();

    auto start_unique_sum = std::chrono::high_resolution_clock::now();
    auto stop_unique_sum = std::chrono::high_resolution_clock::now();

    /*---------------------------------------------------------------------*/
    /*-------------------- REINITIALIZE MAP VARIABLES ---------------------*/
    reinit_map_vars(d_particles_data, d_unique_manager, res_particles_data, res_unique_manager);

    auto start_restructure_map = std::chrono::high_resolution_clock::now();
    exec_map_restructure(d_particles_data, d_unique_manager, res_map_data);
    auto stop_restructure_map = std::chrono::high_resolution_clock::now();

    assert_map_restructure(d_particles_data, res_particles_data);

    /************************** LOG-ODDS KERNEL *************************/
    auto start_update_map = std::chrono::high_resolution_clock::now();
    exec_log_odds(d_map_data, d_particles_data, res_map_data, res_particles_data, general_info);
    auto stop_update_map = std::chrono::high_resolution_clock::now();


    assert_log_odds(d_map_data, res_map_data);


    auto duration_world_to_image_transform_1 = std::chrono::duration_cast<std::chrono::microseconds>(stop_world_to_image_transform_1 - start_world_to_image_transform_1);
    auto duration_world_to_image_transform_2 = std::chrono::duration_cast<std::chrono::microseconds>(stop_world_to_image_transform_2 - start_world_to_image_transform_2);
    auto duration_bresenham = std::chrono::duration_cast<std::chrono::microseconds>(stop_bresenham - start_bresenham);
    auto duration_create_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_create_map - start_create_map);
    auto duration_unique_counter = std::chrono::duration_cast<std::chrono::microseconds>(stop_unique_counter - start_unique_counter);
    auto duration_unique_sum = std::chrono::duration_cast<std::chrono::microseconds>(stop_unique_sum - start_unique_sum);
    auto duration_restructure_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_restructure_map - start_restructure_map);
    auto duration_update_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_update_map - start_update_map);

    auto duration_total = duration_world_to_image_transform_1 + duration_world_to_image_transform_2 + duration_bresenham + duration_create_map +
        duration_unique_counter + duration_unique_sum + duration_restructure_map + duration_update_map;

    std::cout << "Time taken by function (World To Image Transform 1): " << duration_world_to_image_transform_1.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (World To Image Transform 2): " << duration_world_to_image_transform_2.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Bresenham): " << duration_bresenham.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Create Map): " << duration_create_map.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Unique Counter): " << duration_unique_counter.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Unique Sum): " << duration_unique_sum.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Restructure Map): " << duration_restructure_map.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Update Map): " << duration_update_map.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Total): " << duration_total.count() << " microseconds" << std::endl;
    std::cout << std::endl;
}


#endif
