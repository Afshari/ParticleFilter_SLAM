#ifndef _TEST_MAP_H_
#define _TEST_MAP_H_

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

void host_update_map_init(HostMap&, HostMap&, HostMap&, GeneralInfo&, HostMeasurements&,
    HostParticles&, HostParticles&, HostPosition&, HostTransition&, HostTransition&);                           // Step 1
void host_bresenham(HostMap&, HostMap&, HostMap&, GeneralInfo&, HostMeasurements&,
    HostParticles&, HostParticles&, HostPosition&);                                                             // Step 2
void host_update_map(HostMap&, HostMap&, HostMap&, GeneralInfo&, HostMeasurements&,
    HostParticles&, HostParticles&);                                                                            // Step 3
void host_map(HostMap&, HostMap&, HostMap&, GeneralInfo&, HostMeasurements&,
    HostParticles&, HostParticles&, HostPosition&, HostTransition&, HostTransition&);                           // Step Y

void test_map_func(HostMap&, HostMap&, HostMap&, GeneralInfo&, HostMeasurements&,
    HostParticles&, HostParticles&, HostPosition&, HostTransition&, HostTransition&);

const int FILE_NUMBER = 600;

int test_map_main() {

    const int data_len = 6;
    int data[data_len] = { 1, 0, 2, 2, 1, 3 };
    int* d_data = NULL;
    gpuErrchk(cudaMalloc((void**)&d_data, data_len * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_data, data, data_len * sizeof(int), cudaMemcpyHostToDevice));

    thrust::exclusive_scan(thrust::host, data, data + 6, data, 0);
    thrust::exclusive_scan(thrust::device, d_data, d_data + 6, d_data, 0);

    HostMap pre_map;
    HostMap post_bg_map;
    HostMap post_map;
    GeneralInfo general_info;
    HostMeasurements pre_measurements;
    HostParticles pre_particles;
    HostParticles post_particles;
    HostPosition post_position;
    HostTransition pre_transition;
    HostTransition post_transition;

    printf("/************************** UPDATE MAP INIT *************************/\n");

    auto start_read_data_file = std::chrono::high_resolution_clock::now();
    read_update_map(FILE_NUMBER, pre_map, post_bg_map,
        post_map, general_info, pre_measurements, pre_particles, post_particles, post_position, pre_transition, post_transition);
    auto stop_read_data_file = std::chrono::high_resolution_clock::now();

    auto duration_read_data_file = std::chrono::duration_cast<std::chrono::milliseconds>(stop_read_data_file - start_read_data_file);
    std::cout << "Time taken by function (Read Data File): " << duration_read_data_file.count() << " milliseconds" << std::endl;


    host_update_map_init(pre_map, post_bg_map, post_map, general_info, pre_measurements, pre_particles, post_particles, post_position, pre_transition, post_transition);
    host_bresenham(pre_map, post_bg_map, post_map, general_info, pre_measurements, pre_particles, post_particles, post_position);
    host_update_map(pre_map, post_bg_map, post_map, general_info, pre_measurements, pre_particles, post_particles);
    host_map(pre_map, post_bg_map, post_map, general_info, pre_measurements, pre_particles, post_particles, post_position, pre_transition, post_transition);

    test_map_func(pre_map, post_bg_map, post_map, general_info, pre_measurements, pre_particles, post_particles, post_position, pre_transition, post_transition);
    //test_map_func(pre_map, post_bg_map, post_map, general_info, pre_measurements, pre_particles, post_particles, post_position, pre_transition, post_transition);

    return 0;
}

void host_update_map_init(HostMap& pre_map, HostMap& pre_map_bg,
    HostMap& pre_map_post, GeneralInfo& general_info, HostMeasurements& pre_measurements,
    HostParticles& pre_particles, HostParticles& post_particles, 
    HostPosition& post_position, HostTransition& pre_transition, HostTransition& post_transition) 
{

    printf("/************************** UPDATE MAP INIT *************************/\n");

    HostMap h_map;
    h_map.xmin = pre_map.xmin;
    h_map.xmax = pre_map.xmax;
    h_map.ymin = pre_map.ymin;
    h_map.ymax = pre_map.ymax;
    h_map.GRID_WIDTH = pre_map.GRID_WIDTH;
    h_map.GRID_HEIGHT = pre_map.GRID_HEIGHT;

    int xmin_pre = h_map.xmin;
    int ymax_pre = h_map.ymax;

    /********************* IMAGE TRANSFORM VARIABLES ********************/
    HostMeasurements h_measurements;
    h_measurements.LEN = pre_measurements.LEN;
    h_measurements.v_processed_measure_x.resize(h_measurements.MAX_LEN, 0);
    h_measurements.v_processed_measure_y.resize(h_measurements.MAX_LEN, 0);
    
    HostParticles h_particles;
    h_particles.v_world_x.resize(h_measurements.MAX_LEN, 0);
    h_particles.v_world_y.resize(h_measurements.MAX_LEN, 0);

    HostPosition h_position;
    h_position.c_image_body.resize(2, 0);
    HostTransition h_transition;
    h_transition.c_world_lidar.resize(9, 0);

    DeviceMeasurements d_measurements;
    d_measurements.v_lidar_coords.resize(2 * h_measurements.MAX_LEN, 0);
    d_measurements.v_processed_measure_x.resize(h_measurements.MAX_LEN, 0);
    d_measurements.v_processed_measure_y.resize(h_measurements.MAX_LEN, 0);
    thrust::copy(pre_measurements.v_lidar_coords.begin(), pre_measurements.v_lidar_coords.end(), d_measurements.v_lidar_coords.begin());

    DevicePosition d_position;
    d_position.c_image_body.resize(2, 0);

    DeviceTransition d_transition;
    d_transition.c_body_lidar.resize(9, 0);
    d_transition.c_world_body.resize(9, 0);
    d_transition.c_world_lidar.resize(9, 0);

    d_transition.c_body_lidar.assign(hvec_transition_body_lidar.begin(), hvec_transition_body_lidar.end());
    d_transition.c_world_body.assign(pre_transition.c_world_body.begin(), pre_transition.c_world_body.end());

    DeviceParticles d_particles;
    d_particles.v_world_x.resize(h_measurements.MAX_LEN, 0);
    d_particles.v_world_y.resize(h_measurements.MAX_LEN, 0);

    DeviceMap d_map;
    d_map.s_grid_map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);
    d_map.s_log_odds.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);
    d_map.s_grid_map.assign(pre_map.s_grid_map.begin(), pre_map.s_grid_map.end());
    d_map.s_log_odds.assign(pre_map.s_log_odds.begin(), pre_map.s_log_odds.end());

    h_map.s_grid_map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);
    h_map.s_log_odds.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);
    h_map.c_should_extend.resize(4, 0);

    /********************** IMAGE TRANSFORM KERNEL **********************/
    auto start_world_to_image_transform_1 = std::chrono::high_resolution_clock::now();
    kernel_matrix_mul_3x3 << < 1, 1 >> > (
        THRUST_RAW_CAST(d_transition.c_world_body),
        THRUST_RAW_CAST(d_transition.c_body_lidar), THRUST_RAW_CAST(d_transition.c_world_lidar));
    cudaDeviceSynchronize();

    int threadsPerBlock = 1;
    int blocksPerGrid = h_measurements.LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles.v_world_x), THRUST_RAW_CAST(d_particles.v_world_y), SEP,
        THRUST_RAW_CAST(d_transition.c_world_lidar), THRUST_RAW_CAST(d_measurements.v_lidar_coords), h_measurements.LEN);
    cudaDeviceSynchronize();
    auto stop_world_to_image_transform_1 = std::chrono::high_resolution_clock::now();

    auto start_check_extend = std::chrono::high_resolution_clock::now();
    threadsPerBlock = 256;
    blocksPerGrid = (h_measurements.LEN + threadsPerBlock - 1) / threadsPerBlock;
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

    bool EXTEND = false;
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
    auto stop_check_extend = std::chrono::high_resolution_clock::now();

    printf("xmin=%d, xmax=%d, ymin=%d, ymax=%d\n", h_map.xmin, h_map.xmax, h_map.ymin, h_map.ymax);
    for (int i = 0; i < 4; i++)
        std::cout << "Should Extend: " << h_map.c_should_extend[i] << std::endl;

    assert(EXTEND == pre_map.b_should_extend);

    if (EXTEND == true) {

        auto start_extend = std::chrono::high_resolution_clock::now();

        device_vector<int> dvec_coord(2);
        host_vector<int> hvec_coord(2);

        kernel_position_to_image << <1, 1 >> > (
            THRUST_RAW_CAST(dvec_coord), SEP, xmin_pre, ymax_pre, general_info.res, h_map.xmin, h_map.ymax);
        cudaDeviceSynchronize();

        hvec_coord.assign(dvec_coord.begin(), dvec_coord.end());

        device_vector<int> dvec_clone_grid_map(d_map.s_grid_map.size());
        dvec_clone_grid_map.assign(d_map.s_grid_map.begin(), d_map.s_grid_map.end());

        device_vector<float> dvec_clone_log_odds(d_map.s_log_odds.size());
        dvec_clone_log_odds.assign(d_map.s_log_odds.begin(), d_map.s_log_odds.end());

        const int PRE_GRID_WIDTH = h_map.GRID_WIDTH;
        const int PRE_GRID_HEIGHT = h_map.GRID_HEIGHT;
        h_map.GRID_WIDTH = ceil((h_map.ymax - h_map.ymin) / general_info.res + 1);
        h_map.GRID_HEIGHT = ceil((h_map.xmax - h_map.xmin) / general_info.res + 1);
        printf("GRID_WIDTH=%d, AF_GRID_WIDTH=%d, GRID_HEIGHT=%d, AF_GRID_HEIGHT=%d\n",
            h_map.GRID_WIDTH, h_map.GRID_HEIGHT, pre_map_post.GRID_WIDTH, pre_map_post.GRID_HEIGHT);
        assert(h_map.GRID_WIDTH == pre_map_post.GRID_WIDTH);
        assert(h_map.GRID_HEIGHT == pre_map_post.GRID_HEIGHT);

        const int PRE_GRID_SIZE = PRE_GRID_WIDTH * PRE_GRID_HEIGHT;
        const int NEW_GRID_SIZE = h_map.GRID_WIDTH * h_map.GRID_HEIGHT;

        d_map.s_grid_map.clear();
        d_map.s_log_odds.clear();
        d_map.s_grid_map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);
        d_map.s_log_odds.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, LOG_ODD_PRIOR);

        h_map.s_grid_map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);
        h_map.s_log_odds.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);

        threadsPerBlock = 256;
        blocksPerGrid = (PRE_GRID_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        kernel_2d_copy_with_offset << <blocksPerGrid, threadsPerBlock >> > (
            THRUST_RAW_CAST(d_map.s_grid_map), THRUST_RAW_CAST(d_map.s_log_odds), SEP,
            THRUST_RAW_CAST(dvec_clone_grid_map), THRUST_RAW_CAST(dvec_clone_log_odds), hvec_coord[0], hvec_coord[1],
            PRE_GRID_HEIGHT, h_map.GRID_HEIGHT, PRE_GRID_SIZE);
        cudaDeviceSynchronize();

        auto stop_extend = std::chrono::high_resolution_clock::now();
        auto duration_extend = std::chrono::duration_cast<std::chrono::microseconds>(stop_extend - start_extend);

        std::cout << std::endl;
        std::cout << "Time taken by function (Extend): " << duration_extend.count() << " microseconds" << std::endl;

        h_map.s_grid_map.assign(d_map.s_grid_map.begin(), d_map.s_grid_map.end());
        h_map.s_log_odds.assign(d_map.s_log_odds.begin(), d_map.s_log_odds.end());

        int error_map = 0;
        int error_log = 0;
        for (int i = 0; i < (h_map.GRID_WIDTH * h_map.GRID_HEIGHT); i++) {
            if (h_map.s_grid_map[i] != pre_map_bg.s_grid_map[i]) {
                error_map += 1;
                //printf("Grid Map: %d <> %d\n", hves_grid_map[i], bg_grid_map[i]);
            }
            if (abs(h_map.s_log_odds[i] - pre_map_bg.s_log_odds[i]) > 1e-4) {
                error_log += 1;
                printf("Log Odds: (%d) %f <> %f\n", i, h_map.s_log_odds[i], pre_map_bg.s_log_odds[i]);
            }
        }
        printf("Map Erros: %d\n", error_map);
        printf("Log Erros: %d\n", error_log);
    }

    auto start_world_to_image_transform_2 = std::chrono::high_resolution_clock::now();
    threadsPerBlock = 1;
    blocksPerGrid = h_measurements.LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_measurements.v_processed_measure_x), THRUST_RAW_CAST(d_measurements.v_processed_measure_y), SEP,
        THRUST_RAW_CAST(d_transition.c_world_lidar), THRUST_RAW_CAST(d_measurements.v_lidar_coords),
        general_info.res, h_map.xmin, h_map.ymax, h_measurements.LEN);
    cudaDeviceSynchronize();

    kernel_position_to_image << < 1, 1 >> > (THRUST_RAW_CAST(d_position.c_image_body), SEP,
        THRUST_RAW_CAST(d_transition.c_world_lidar), general_info.res, h_map.xmin, h_map.ymax);
    cudaDeviceSynchronize();
    auto stop_world_to_image_transform_2 = std::chrono::high_resolution_clock::now();

    h_transition.c_world_lidar.assign(d_transition.c_world_lidar.begin(), d_transition.c_world_lidar.end());
    h_position.c_image_body.assign(d_position.c_image_body.begin(), d_position.c_image_body.end());
    
    thrust::copy(d_measurements.v_processed_measure_x.begin(), d_measurements.v_processed_measure_x.begin() + h_measurements.LEN, h_measurements.v_processed_measure_x.begin());
    thrust::copy(d_measurements.v_processed_measure_y.begin(), d_measurements.v_processed_measure_y.begin() + h_measurements.LEN, h_measurements.v_processed_measure_y.begin());
    thrust::copy(d_particles.v_world_x.begin(), d_particles.v_world_x.begin() + h_measurements.LEN, h_particles.v_world_x.begin());
    thrust::copy(d_particles.v_world_y.begin(), d_particles.v_world_y.begin() + h_measurements.LEN, h_particles.v_world_y.begin());

    ASSERT_transition_world_lidar(h_transition.c_world_lidar.data(), post_transition.c_world_lidar.data(), 9, false);
    ASSERT_particles_world_frame(h_particles.v_world_x.data(), h_particles.v_world_y.data(),
        post_particles.v_world_x.data(), post_particles.v_world_y.data(), h_measurements.LEN, false);
    ASSERT_processed_measurements(h_measurements.v_processed_measure_x.data(), h_measurements.v_processed_measure_y.data(),
        post_particles.v_occupied_x.data(), post_particles.v_occupied_y.data(), h_measurements.LEN, false);
    ASSERT_position_image_body(h_position.c_image_body.data(), post_position.c_image_body.data(), true, true);

    auto duration_world_to_image_transform_1 = std::chrono::duration_cast<std::chrono::microseconds>(stop_world_to_image_transform_1 - start_world_to_image_transform_1);
    auto duration_world_to_image_transform_2 = std::chrono::duration_cast<std::chrono::microseconds>(stop_world_to_image_transform_2 - start_world_to_image_transform_2);
    std::cout << "Time taken by function (World To Image Transform 1): " << duration_world_to_image_transform_1.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (World To Image Transform 2): " << duration_world_to_image_transform_2.count() << " microseconds" << std::endl;
    std::cout << std::endl;
}

void host_bresenham(HostMap& pre_map, HostMap& pre_map_bg,
    HostMap& pre_map_post, GeneralInfo& general_info, HostMeasurements& pre_measurements,
    HostParticles& pre_particles, HostParticles& post_particles, 
    HostPosition& pre_position) {

    printf("/***************************** BRESENHAM ****************************/\n");

    HostMeasurements h_measurements;
    h_measurements.LEN = pre_measurements.LEN;

    HostMap h_map;
    h_map.GRID_WIDTH = pre_map.GRID_WIDTH;
    h_map.GRID_HEIGHT = pre_map.GRID_HEIGHT;

    printf("~~$ GRID_WIDTH = \t%d\n", h_map.GRID_WIDTH);
    printf("~~$ GRID_HEIGHT = \t%d\n", h_map.GRID_HEIGHT);
    int MAX_DIST_IN_MAP = sqrt(pow(h_map.GRID_WIDTH, 2) + pow(h_map.GRID_HEIGHT, 2));
    printf("~~$ MAX_DIST_IN_MAP = \t%d\n", MAX_DIST_IN_MAP);

    /************************ BRESENHAM VARIABLES ***********************/
    assert(pre_particles.OCCUPIED_LEN == h_measurements.LEN);
    int PARTICLE_UNIQUE_COUNTER = h_measurements.LEN + 1;

    HostParticles h_particles;
    h_particles.v_free_counter.resize(PARTICLE_UNIQUE_COUNTER, 0);
    
    DeviceParticles d_particles;
    d_particles.v_occupied_x.resize(h_measurements.MAX_LEN, 0);
    d_particles.v_occupied_y.resize(h_measurements.MAX_LEN, 0);
    d_particles.sv_free_x_max.resize(h_measurements.MAX_LEN * MAX_DIST_IN_MAP, 0);
    d_particles.sv_free_y_max.resize(h_measurements.MAX_LEN * MAX_DIST_IN_MAP, 0);
    d_particles.v_free_counter.resize(h_measurements.MAX_LEN, 0);
    d_particles.v_free_idx.resize(h_measurements.MAX_LEN, 0);

    d_particles.f_free_x.resize(h_particles.MAX_FREE_LEN, 0);
    d_particles.f_free_y.resize(h_particles.MAX_FREE_LEN, 0);
    h_particles.f_free_x.resize(h_particles.MAX_FREE_LEN, 0);
    h_particles.f_free_y.resize(h_particles.MAX_FREE_LEN, 0);

    thrust::copy(post_particles.v_occupied_x.begin(), post_particles.v_occupied_x.begin() + h_measurements.LEN, d_particles.v_occupied_x.begin());
    thrust::copy(post_particles.v_occupied_y.begin(), post_particles.v_occupied_y.begin() + h_measurements.LEN, d_particles.v_occupied_y.begin());
    thrust::copy(post_particles.v_free_idx.begin(), post_particles.v_free_idx.begin() + h_measurements.LEN, d_particles.v_free_idx.begin());

    DevicePosition d_position;
    d_position.c_image_body.resize(2, 0);
    d_position.c_image_body.assign(pre_position.c_image_body.begin(), pre_position.c_image_body.end());

    /************************* BRESENHAM KERNEL *************************/
    auto start_bresenham = std::chrono::high_resolution_clock::now();

    int threadsPerBlock = 256;
    int blocksPerGrid = (h_measurements.LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_bresenham << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles.sv_free_x_max), THRUST_RAW_CAST(d_particles.sv_free_y_max), THRUST_RAW_CAST(d_particles.v_free_counter), SEP,
        THRUST_RAW_CAST(d_particles.v_occupied_x), THRUST_RAW_CAST(d_particles.v_occupied_y), THRUST_RAW_CAST(d_position.c_image_body),
        h_measurements.LEN, MAX_DIST_IN_MAP);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, 
        d_particles.v_free_counter.begin(), d_particles.v_free_counter.end(), 
        d_particles.v_free_counter.begin(), 0);
    auto stop_bresenham = std::chrono::high_resolution_clock::now();

    auto start_bresenham_rearrange = std::chrono::high_resolution_clock::now();
    h_particles.v_free_counter.assign(d_particles.v_free_counter.begin(), d_particles.v_free_counter.end());

    h_particles.FREE_LEN = h_particles.v_free_counter[PARTICLE_UNIQUE_COUNTER - 1];

    kernel_bresenham_rearrange << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles.f_free_x), THRUST_RAW_CAST(d_particles.f_free_y), SEP,
        THRUST_RAW_CAST(d_particles.sv_free_x_max), THRUST_RAW_CAST(d_particles.sv_free_y_max),
        THRUST_RAW_CAST(d_particles.v_free_counter), MAX_DIST_IN_MAP, h_measurements.LEN);
    cudaDeviceSynchronize();
    auto stop_bresenham_rearrange = std::chrono::high_resolution_clock::now();

    thrust::copy(d_particles.f_free_x.begin(), d_particles.f_free_x.begin() + h_particles.FREE_LEN, h_particles.f_free_x.begin());
    thrust::copy(d_particles.f_free_y.begin(), d_particles.f_free_y.begin() + h_particles.FREE_LEN, h_particles.f_free_y.begin());

    ASSERT_particles_free_index(h_particles.v_free_counter.data(), post_particles.v_free_idx.data(), h_measurements.LEN, false);
    ASSERT_particles_free_new_len(h_particles.FREE_LEN, post_particles.FREE_LEN);
    ASSERT_particles_free(h_particles.f_free_x.data(), h_particles.f_free_y.data(),
        post_particles.f_free_x.data(), post_particles.f_free_y.data(), h_particles.FREE_LEN, true, true);

    auto duration_bresenham = std::chrono::duration_cast<std::chrono::microseconds>(stop_bresenham - start_bresenham);
    auto duration_bresenham_rearrange = std::chrono::duration_cast<std::chrono::microseconds>(stop_bresenham_rearrange - start_bresenham_rearrange);
    std::cout << "Time taken by function (Bresenham): " << duration_bresenham.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Bresenham Rearrange): " << duration_bresenham_rearrange.count() << " microseconds" << std::endl;
    std::cout << std::endl;
}

void host_update_map(HostMap& pre_map, HostMap& pre_map_bg,
    HostMap& pre_map_post, GeneralInfo& general_info, HostMeasurements& pre_measurements,
    HostParticles& pre_particles, HostParticles& post_particles) {

    printf("/**************************** UPDATE MAP ****************************/\n");

    host_vector<int> hvec_occupied_map_idx(2, 0);
    host_vector<int> hvec_free_map_idx(2, 0);

    HostMap h_map;
    h_map.xmin = pre_map.xmin;
    h_map.xmax = pre_map.xmax;
    h_map.ymin = pre_map.ymin;
    h_map.ymax = pre_map.ymax;

    h_map.GRID_WIDTH = pre_map_post.GRID_WIDTH;
    h_map.GRID_HEIGHT = pre_map_post.GRID_HEIGHT;

    HostMeasurements h_measurements;
    h_measurements.LEN = pre_measurements.LEN;

    assert(pre_particles.OCCUPIED_LEN == h_measurements.LEN);
    int PARTICLE_UNIQUE_COUNTER = h_measurements.LEN + 1;

    printf("~~$ GRID_WIDTH = \t%d\n", h_map.GRID_WIDTH);
    printf("~~$ GRID_HEIGHT = \t%d\n", h_map.GRID_HEIGHT);

    /**************************** MAP VARIABLES *************************/
    DeviceMap d_map;
    d_map.s_grid_map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);
    d_map.s_grid_map.assign(pre_map_bg.s_grid_map.begin(), pre_map_bg.s_grid_map.end());

    HostParticles h_particles;
    h_particles.FREE_LEN = post_particles.FREE_LEN;
    h_particles.v_occupied_x.resize(h_measurements.MAX_LEN, 0);
    h_particles.v_occupied_y.resize(h_measurements.MAX_LEN, 0);

    h_particles.f_occupied_unique_x.resize(h_particles.MAX_OCCUPIED_UNIQUE_LEN, 0);
    h_particles.f_occupied_unique_y.resize(h_particles.MAX_OCCUPIED_UNIQUE_LEN, 0);
    h_particles.f_free_unique_x.resize(h_particles.MAX_FREE_UNIQUE_LEN, 0);
    h_particles.f_free_unique_y.resize(h_particles.MAX_FREE_UNIQUE_LEN, 0);

    DeviceParticles d_particles;
    d_particles.v_occupied_x.resize(h_measurements.MAX_LEN, 0);
    d_particles.v_occupied_y.resize(h_measurements.MAX_LEN, 0);
    d_particles.f_free_x.resize(h_particles.MAX_FREE_LEN, 0);
    d_particles.f_free_y.resize(h_particles.MAX_FREE_LEN, 0);

    d_particles.f_occupied_unique_x.resize(h_particles.MAX_OCCUPIED_UNIQUE_LEN, 0);
    d_particles.f_occupied_unique_y.resize(h_particles.MAX_OCCUPIED_UNIQUE_LEN, 0);
    d_particles.f_free_unique_x.resize(h_particles.MAX_FREE_UNIQUE_LEN, 0);
    d_particles.f_free_unique_y.resize(h_particles.MAX_FREE_UNIQUE_LEN, 0);

    
    thrust::copy(post_particles.v_occupied_x.begin(), post_particles.v_occupied_x.end(), d_particles.v_occupied_x.begin());
    thrust::copy(post_particles.v_occupied_y.begin(), post_particles.v_occupied_y.end(), d_particles.v_occupied_y.begin());
    thrust::copy(post_particles.f_free_x.begin(), post_particles.f_free_x.end(), d_particles.f_free_x.begin());
    thrust::copy(post_particles.f_free_y.begin(), post_particles.f_free_y.end(), d_particles.f_free_y.begin());

    /************************* LOG-ODDS VARIABLES ***********************/

    Host2DUniqueFinder h_unique_occupied;
    Host2DUniqueFinder h_unique_free;
    h_unique_occupied.c_in_map.resize(1, 0);
    h_unique_occupied.s_in_col.resize(h_map.GRID_WIDTH + 1, 0);
    h_unique_free.c_in_map.resize(1, 0);
    h_unique_free.s_in_col.resize(h_map.GRID_WIDTH + 1, 0);

    hvec_occupied_map_idx[1] = h_measurements.LEN;
    hvec_free_map_idx[1] = h_particles.FREE_LEN;

    Device2DUniqueFinder d_unique_occupied;
    Device2DUniqueFinder d_unique_free;
    d_unique_occupied.s_map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);
    d_unique_free.s_map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);

    d_unique_occupied.c_in_map.resize(1, 0);
    d_unique_occupied.s_in_col.resize(h_map.GRID_WIDTH + 1, 0);
    d_unique_free.c_in_map.resize(1, 0);
    d_unique_free.s_in_col.resize(h_map.GRID_WIDTH + 1, 0);

    d_unique_occupied.c_idx.resize(2, 0);
    d_unique_free.c_idx.resize(2, 0);

    d_unique_occupied.c_idx.assign(hvec_occupied_map_idx.begin(), hvec_occupied_map_idx.end());
    d_unique_free.c_idx.assign(hvec_free_map_idx.begin(), hvec_free_map_idx.end());

    thrust::fill(d_unique_occupied.s_in_col.begin(), d_unique_occupied.s_in_col.end(), 0);
    thrust::fill(d_unique_free.s_in_col.begin(), d_unique_free.s_in_col.end(), 0);

    /**************************** CREATE MAP ****************************/
    auto start_create_map = std::chrono::high_resolution_clock::now();
    int threadsPerBlock = 256;
    int blocksPerGrid = 1;
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_unique_occupied.s_map), SEP,
        THRUST_RAW_CAST(d_particles.v_occupied_x), THRUST_RAW_CAST(d_particles.v_occupied_y),
        THRUST_RAW_CAST(d_unique_occupied.c_idx),
        h_particles.OCCUPIED_LEN, h_map.GRID_WIDTH, h_map.GRID_HEIGHT, NUM_PARTICLES);
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_unique_free.s_map), SEP,
        THRUST_RAW_CAST(d_particles.f_free_x), THRUST_RAW_CAST(d_particles.f_free_y),
        THRUST_RAW_CAST(d_unique_free.c_idx),
        h_particles.FREE_LEN, h_map.GRID_WIDTH, h_map.GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();
    auto stop_create_map = std::chrono::high_resolution_clock::now();


    auto start_unique_counter = std::chrono::high_resolution_clock::now();
    threadsPerBlock = h_map.GRID_WIDTH;
    blocksPerGrid = 1;
    kernel_2d_map_counter << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_unique_occupied.c_in_map), THRUST_RAW_CAST(d_unique_occupied.s_in_col), SEP,
        THRUST_RAW_CAST(d_unique_occupied.s_map), h_map.GRID_WIDTH, h_map.GRID_HEIGHT);
    kernel_2d_map_counter << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_unique_free.c_in_map), THRUST_RAW_CAST(d_unique_free.s_in_col), SEP,
        THRUST_RAW_CAST(d_unique_free.s_map), h_map.GRID_WIDTH, h_map.GRID_HEIGHT);
    cudaDeviceSynchronize();
    auto stop_unique_counter = std::chrono::high_resolution_clock::now();

    auto start_unique_sum = std::chrono::high_resolution_clock::now();
    threadsPerBlock = 1;
    blocksPerGrid = 1;
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (THRUST_RAW_CAST(d_unique_occupied.s_in_col), h_map.GRID_WIDTH);
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (THRUST_RAW_CAST(d_unique_free.s_in_col), h_map.GRID_WIDTH);
    cudaDeviceSynchronize();
    auto stop_unique_sum = std::chrono::high_resolution_clock::now();

    hvec_occupied_map_idx.assign(d_unique_occupied.c_idx.begin(), d_unique_occupied.c_idx.end());

    h_unique_occupied.c_in_map.assign(d_unique_occupied.c_in_map.begin(), d_unique_occupied.c_in_map.end());
    h_unique_free.c_in_map.assign(d_unique_free.c_in_map.begin(), d_unique_free.c_in_map.end());

    /*---------------------------------------------------------------------*/
    /*-------------------- REINITIALIZE MAP VARIABLES ---------------------*/
    h_particles.OCCUPIED_UNIQUE_LEN = h_unique_occupied.c_in_map[0];
    h_particles.FREE_UNIQUE_LEN = h_unique_free.c_in_map[0];

    printf("--> Occupied Unique: %d, %d\n", h_particles.OCCUPIED_UNIQUE_LEN, post_particles.OCCUPIED_UNIQUE_LEN);
    assert(h_particles.OCCUPIED_UNIQUE_LEN == post_particles.OCCUPIED_UNIQUE_LEN);
    printf("--> Free Unique: %d, %d\n", h_particles.FREE_UNIQUE_LEN, post_particles.FREE_UNIQUE_LEN);
    assert(h_particles.FREE_UNIQUE_LEN == post_particles.FREE_UNIQUE_LEN);

    thrust::fill(d_particles.f_occupied_unique_x.begin(), d_particles.f_occupied_unique_x.end(), 0);
    thrust::fill(d_particles.f_occupied_unique_y.begin(), d_particles.f_occupied_unique_y.end(), 0);
    thrust::fill(d_particles.f_free_unique_x.begin(), d_particles.f_free_unique_x.end(), 0);
    thrust::fill(d_particles.f_free_unique_y.begin(), d_particles.f_free_unique_y.end(), 0);


    auto start_restructure_map = std::chrono::high_resolution_clock::now();
    threadsPerBlock = h_map.GRID_WIDTH;
    blocksPerGrid = 1;
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles.f_occupied_unique_x), THRUST_RAW_CAST(d_particles.f_occupied_unique_y), SEP,
        THRUST_RAW_CAST(d_unique_occupied.s_map), THRUST_RAW_CAST(d_unique_occupied.s_in_col), h_map.GRID_WIDTH, h_map.GRID_HEIGHT);
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles.f_free_unique_x), THRUST_RAW_CAST(d_particles.f_free_unique_y), SEP,
        THRUST_RAW_CAST(d_unique_free.s_map), THRUST_RAW_CAST(d_unique_free.s_in_col), h_map.GRID_WIDTH, h_map.GRID_HEIGHT);
    cudaDeviceSynchronize();
    auto stop_restructure_map = std::chrono::high_resolution_clock::now();

    thrust::copy(d_particles.f_occupied_unique_x.begin(), d_particles.f_occupied_unique_x.begin() + h_particles.OCCUPIED_UNIQUE_LEN, h_particles.f_occupied_unique_x.begin());
    thrust::copy(d_particles.f_occupied_unique_y.begin(), d_particles.f_occupied_unique_y.begin() + h_particles.OCCUPIED_UNIQUE_LEN, h_particles.f_occupied_unique_y.begin());
    thrust::copy(d_particles.f_free_unique_x.begin(), d_particles.f_free_unique_x.begin() + h_particles.FREE_UNIQUE_LEN, h_particles.f_free_unique_x.begin());
    thrust::copy(d_particles.f_free_unique_y.begin(), d_particles.f_free_unique_y.begin() + h_particles.FREE_UNIQUE_LEN, h_particles.f_free_unique_y.begin());

    ASSERT_particles_occupied(h_particles.f_occupied_unique_x.data(), h_particles.f_occupied_unique_y.data(),
        post_particles.f_occupied_unique_x.data(), post_particles.f_occupied_unique_y.data(),
        "Occupied", h_particles.OCCUPIED_UNIQUE_LEN, false);
    ASSERT_particles_occupied(h_particles.f_free_unique_x.data(), h_particles.f_free_unique_y.data(),
        post_particles.f_free_unique_x.data(), post_particles.f_free_unique_y.data(),
        "Free", h_particles.FREE_UNIQUE_LEN, false);

    /************************* LOG-ODDS VARIABLES ***********************/
    h_map.s_grid_map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);
    h_map.s_log_odds.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);

    d_map.s_log_odds.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT);
    d_map.s_log_odds.assign(pre_map_bg.s_log_odds.begin(), pre_map_bg.s_log_odds.end());


    /************************** LOG-ODDS KERNEL *************************/
    auto start_update_map = std::chrono::high_resolution_clock::now();

    threadsPerBlock = 256;
    blocksPerGrid = (h_particles.OCCUPIED_UNIQUE_LEN + threadsPerBlock - 1) / threadsPerBlock;
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
        THRUST_RAW_CAST(d_map.s_grid_map), SEP,
        THRUST_RAW_CAST(d_map.s_log_odds), LOG_ODD_PRIOR, WALL, FREE, h_map.GRID_WIDTH * h_map.GRID_HEIGHT);
    cudaDeviceSynchronize();

    auto stop_update_map = std::chrono::high_resolution_clock::now();

    h_map.s_log_odds.assign(d_map.s_log_odds.begin(), d_map.s_log_odds.end());
    h_map.s_grid_map.assign(d_map.s_grid_map.begin(), d_map.s_grid_map.end());

    ASSERT_log_odds(h_map.s_log_odds.data(), pre_map.s_log_odds.data(), pre_map_post.s_log_odds.data(),
        (h_map.GRID_WIDTH * h_map.GRID_HEIGHT), (pre_map.GRID_WIDTH * pre_map.GRID_HEIGHT), false);
    ASSERT_log_odds_maps(h_map.s_grid_map.data(), pre_map.s_grid_map.data(), pre_map_post.s_grid_map.data(),
        (h_map.GRID_WIDTH * h_map.GRID_HEIGHT), (pre_map.GRID_WIDTH* pre_map.GRID_HEIGHT), false);
    printf("\n");

    auto duration_create_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_create_map - start_create_map);
    auto duration_unique_counter = std::chrono::duration_cast<std::chrono::microseconds>(stop_unique_counter - start_unique_counter);
    auto duration_unique_sum = std::chrono::duration_cast<std::chrono::microseconds>(stop_unique_sum - start_unique_sum);
    auto duration_restructure_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_restructure_map - start_restructure_map);
    auto duration_update_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_update_map - start_update_map);

    std::cout << "Time taken by function (Create Map): " << duration_create_map.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Unique Counter): " << duration_unique_counter.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Unique Sum): " << duration_unique_sum.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Restructure Map): " << duration_restructure_map.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Update Map): " << duration_update_map.count() << " microseconds" << std::endl;
    std::cout << std::endl;
}

void host_map(HostMap& pre_map, HostMap& pre_map_bg,
    HostMap& pre_map_post, GeneralInfo& general_info, HostMeasurements& pre_measurements,
    HostParticles& pre_particles, HostParticles& post_particles, 
    HostPosition& pre_position, HostTransition& pre_transition, HostTransition& post_transition) {

    printf("/******************************** MAP *******************************/\n");

    host_vector<int> hvec_occupied_map_idx(2, 0);
    host_vector<int> hvec_free_map_idx(2, 0);

    HostMap h_map;
    h_map.xmin = pre_map.xmin;
    h_map.xmax = pre_map.xmax;
    h_map.ymin = pre_map.ymin;
    h_map.ymax = pre_map.ymax;
    h_map.GRID_WIDTH = pre_map.GRID_WIDTH;
    h_map.GRID_HEIGHT = pre_map.GRID_HEIGHT;

    int xmin_pre = h_map.xmin;
    int ymax_pre = h_map.ymax;

    HostMeasurements h_measurements;
    h_measurements.LEN = pre_measurements.LEN;

    assert(h_measurements.LEN == pre_particles.OCCUPIED_LEN);

    /********************* IMAGE TRANSFORM VARIABLES ********************/

    HostPosition h_position;
    h_position.c_image_body.resize(2);
    HostTransition h_transition;
    h_transition.c_world_lidar.resize(9);
    

    HostParticles h_particles;
    h_particles.v_occupied_x.resize(h_measurements.MAX_LEN, 0);
    h_particles.v_occupied_y.resize(h_measurements.MAX_LEN, 0);
    h_particles.v_world_x.resize(h_measurements.MAX_LEN, 0);
    h_particles.v_world_y.resize(h_measurements.MAX_LEN, 0);
    
    DeviceMeasurements d_measurements;
    d_measurements.v_lidar_coords.resize(2 * h_measurements.MAX_LEN, 0);
    thrust::copy(pre_measurements.v_lidar_coords.begin(), pre_measurements.v_lidar_coords.end(), d_measurements.v_lidar_coords.begin());

    DeviceTransition d_transition;
    d_transition.c_body_lidar.resize(9, 0);
    d_transition.c_world_body.resize(9, 0);
    d_transition.c_world_lidar.resize(9, 0);
    d_transition.c_body_lidar.assign(hvec_transition_body_lidar.begin(), hvec_transition_body_lidar.end());
    d_transition.c_world_body.assign(pre_transition.c_world_body.begin(), pre_transition.c_world_body.end());
    
    DevicePosition d_position;
    d_position.c_image_body.resize(2);

    /************************ BRESENHAM VARIABLES ***********************/
    int MAX_DIST_IN_MAP = sqrt(pow(h_map.GRID_WIDTH, 2) + pow(h_map.GRID_HEIGHT, 2));
    int PARTICLE_UNIQUE_COUNTER = h_measurements.LEN + 1;

    DeviceParticles d_particles;
    d_particles.v_occupied_x.resize(h_measurements.MAX_LEN, 0);
    d_particles.v_occupied_y.resize(h_measurements.MAX_LEN, 0);
    d_particles.v_world_x.resize(h_measurements.MAX_LEN, 0);
    d_particles.v_world_y.resize(h_measurements.MAX_LEN, 0);
    d_particles.sv_free_x_max.resize(h_measurements.MAX_LEN * MAX_DIST_IN_MAP, 0);
    d_particles.sv_free_y_max.resize(h_measurements.MAX_LEN * MAX_DIST_IN_MAP, 0);
    d_particles.v_free_counter.resize(h_measurements.MAX_LEN, 0);      // PARTICLE_UNIQUE_COUNTER
    d_particles.v_free_idx.resize(h_measurements.MAX_LEN, 0);

    d_particles.f_free_x.resize(h_particles.MAX_FREE_LEN, 0);
    d_particles.f_free_y.resize(h_particles.MAX_FREE_LEN, 0);

    h_particles.f_free_x.resize(h_particles.MAX_FREE_LEN, 0);
    h_particles.f_free_y.resize(h_particles.MAX_FREE_LEN, 0);

    h_particles.f_occupied_unique_x.resize(h_particles.MAX_OCCUPIED_UNIQUE_LEN, 0);
    h_particles.f_occupied_unique_y.resize(h_particles.MAX_OCCUPIED_UNIQUE_LEN, 0);
    h_particles.f_free_unique_x.resize(h_particles.MAX_FREE_UNIQUE_LEN, 0);
    h_particles.f_free_unique_y.resize(h_particles.MAX_FREE_UNIQUE_LEN, 0);

    d_particles.f_occupied_unique_x.resize(h_particles.MAX_OCCUPIED_UNIQUE_LEN, 0);
    d_particles.f_occupied_unique_y.resize(h_particles.MAX_OCCUPIED_UNIQUE_LEN, 0);
    d_particles.f_free_unique_x.resize(h_particles.MAX_FREE_UNIQUE_LEN, 0);
    d_particles.f_free_unique_y.resize(h_particles.MAX_FREE_UNIQUE_LEN, 0);

    /**************************** MAP VARIABLES *************************/
    DeviceMap d_map;
    d_map.s_grid_map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);
    d_map.s_grid_map.assign(pre_map.s_grid_map.begin(), pre_map.s_grid_map.end());
    d_map.s_log_odds.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);
    d_map.c_should_extend.resize(4, 0);

    /************************* LOG-ODDS VARIABLES ***********************/

    h_particles.v_free_counter.resize(PARTICLE_UNIQUE_COUNTER);
    
    Host2DUniqueFinder h_unique_free;
    h_unique_free.c_in_map.resize(1, 0);
    h_unique_free.s_in_col.resize(h_map.GRID_WIDTH + 1, 0);
    h_unique_free.c_idx.resize(2, 0);

    Host2DUniqueFinder h_unique_occupied;
    h_unique_occupied.c_in_map.resize(1);
    h_unique_occupied.s_in_col.resize(h_map.GRID_WIDTH + 1);
    h_unique_occupied.c_idx.resize(2);
    

    Device2DUniqueFinder d_unique_free;
    d_unique_free.s_map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);
    d_unique_free.c_in_map.resize(1, 0);
    d_unique_free.s_in_col.resize(h_map.GRID_WIDTH + 1, 0);
    d_unique_free.c_idx.resize(2, 0);

    Device2DUniqueFinder d_unique_occupied;
    d_unique_occupied.s_map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);
    d_unique_occupied.c_in_map.resize(1, 0);
    d_unique_occupied.s_in_col.resize(h_map.GRID_WIDTH + 1, 0);
    d_unique_occupied.c_idx.resize(2, 0);
    
    h_map.s_grid_map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);
    h_map.s_log_odds.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);
    h_map.c_should_extend.resize(4, 0);

    hvec_occupied_map_idx[1] = h_measurements.LEN;
    hvec_free_map_idx[1] = 0;

    d_unique_occupied.c_idx.assign(hvec_occupied_map_idx.begin(), hvec_occupied_map_idx.end());
    d_unique_free.c_idx.assign(hvec_free_map_idx.begin(), hvec_free_map_idx.end());
    d_map.s_log_odds.assign(pre_map.s_log_odds.begin(), pre_map.s_log_odds.end());

    /************************************************************* KERNEL EXECUTION SCOPE *************************************************************/

    /***************** World to IMAGE TRANSFORM KERNEL ******************/
    auto start_world_to_image_transform_1 = std::chrono::high_resolution_clock::now();
    kernel_matrix_mul_3x3 << < 1, 1 >> > (
        THRUST_RAW_CAST(d_transition.c_world_body), 
        THRUST_RAW_CAST(d_transition.c_body_lidar), THRUST_RAW_CAST(d_transition.c_world_lidar));
    cudaDeviceSynchronize();

    int threadsPerBlock = 1;
    int blocksPerGrid = h_measurements.LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles.v_world_x), THRUST_RAW_CAST(d_particles.v_world_y), SEP,
        THRUST_RAW_CAST(d_transition.c_world_lidar), THRUST_RAW_CAST(d_measurements.v_lidar_coords), h_measurements.LEN);
    cudaDeviceSynchronize();
    auto stop_world_to_image_transform_1 = std::chrono::high_resolution_clock::now();


    auto start_check_extend = std::chrono::high_resolution_clock::now();
    threadsPerBlock = 256;
    blocksPerGrid = (h_measurements.LEN + threadsPerBlock - 1) / threadsPerBlock;
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

    bool EXTEND = false;
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
    auto stop_check_extend = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 4; i++)
        std::cout << "Should Extend: " << h_map.c_should_extend[i] << std::endl;

    printf("xmin=%d, xmax=%d, ymin=%d, ymax=%d\n", h_map.xmin, h_map.xmax, h_map.ymin, h_map.ymax);
    printf("xmin=%d, xmax=%d, ymin=%d, ymax=%d\n", pre_map_post.xmin, pre_map_post.xmax, pre_map_post.ymin, pre_map_post.ymax);
    assert(EXTEND == pre_map.b_should_extend);

    if (EXTEND == true) {

        auto start_extend = std::chrono::high_resolution_clock::now();

        d_measurements.c_coord.resize(2);
        HostMeasurements h_measurements;
        h_measurements.c_coord.resize(2);

        kernel_position_to_image << <1, 1 >> > (THRUST_RAW_CAST(d_measurements.c_coord), SEP, 
            xmin_pre, ymax_pre, general_info.res, h_map.xmin, h_map.ymax);
        cudaDeviceSynchronize();

        h_measurements.c_coord.assign(d_measurements.c_coord.begin(), d_measurements.c_coord.end());
        
        device_vector<int> dvec_clone_grid_map(h_map.GRID_WIDTH * h_map.GRID_HEIGHT);
        dvec_clone_grid_map.assign(d_map.s_grid_map.begin(), d_map.s_grid_map.end());

        device_vector<float> dvec_clone_log_odds(h_map.GRID_WIDTH * h_map.GRID_HEIGHT);
        dvec_clone_log_odds.assign(d_map.s_log_odds.begin(), d_map.s_log_odds.end());

        const int PRE_GRID_WIDTH = h_map.GRID_WIDTH;
        const int PRE_GRID_HEIGHT = h_map.GRID_HEIGHT;
        h_map.GRID_WIDTH = ceil((h_map.ymax - h_map.ymin) / general_info.res + 1);
        h_map.GRID_HEIGHT = ceil((h_map.xmax - h_map.xmin) / general_info.res + 1);
        printf("GRID_WIDTH=%d, AF_GRID_WIDTH=%d, GRID_HEIGHT=%d, AF_GRID_HEIGHT=%d\n", 
            h_map.GRID_WIDTH, h_map.GRID_HEIGHT, pre_map_post.GRID_WIDTH, pre_map_post.GRID_HEIGHT);
        assert(h_map.GRID_WIDTH == pre_map_post.GRID_WIDTH);
        assert(h_map.GRID_HEIGHT == pre_map_post.GRID_HEIGHT);

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
            h_measurements.c_coord[0], h_measurements.c_coord[1],
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

        int error_map = 0;
        int error_log = 0;
        for (int i = 0; i < (h_map.GRID_WIDTH * h_map.GRID_HEIGHT); i++) {
            if (h_map.s_grid_map[i] != pre_map_bg.s_grid_map[i]) {
                error_map += 1;
                //printf("Grid Map: %d <> %d\n", h_map.s_grid_map[i], bg_grid_map[i]);
            }
            if (abs(h_map.s_log_odds[i] - pre_map_bg.s_log_odds[i]) > 1e-4) {
                error_log += 1;
                //printf("Log Odds: (%d) %f <> %f\n", i, h_map.s_log_odds[i], bg_log_odds[i]);
            }
            if (error_log > 200)
                break;
        }
        printf("Map Erros: %d\n", error_map);
        printf("Log Erros: %d\n", error_log);
    }

    for (int i = 0; i < 4; i++)
        std::cout << "Should Extend: " << h_map.c_should_extend[i] << std::endl;

    auto start_world_to_image_transform_2 = std::chrono::high_resolution_clock::now();
    threadsPerBlock = 1;
    blocksPerGrid = h_measurements.LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles.v_occupied_x), THRUST_RAW_CAST(d_particles.v_occupied_y), SEP,
        THRUST_RAW_CAST(d_transition.c_world_lidar), THRUST_RAW_CAST(d_measurements.v_lidar_coords),
        general_info.res, h_map.xmin, h_map.ymax, h_measurements.LEN);
    cudaDeviceSynchronize();

    kernel_position_to_image << < 1, 1 >> > (
        THRUST_RAW_CAST(d_position.c_image_body), SEP, THRUST_RAW_CAST(d_transition.c_world_lidar), general_info.res, h_map.xmin, h_map.ymax);
    cudaDeviceSynchronize();
    auto stop_world_to_image_transform_2 = std::chrono::high_resolution_clock::now();

    h_transition.c_world_lidar.assign(d_transition.c_world_lidar.begin(), d_transition.c_world_lidar.end());
    h_position.c_image_body.assign(d_position.c_image_body.begin(), d_position.c_image_body.end());

    thrust::copy(d_particles.v_occupied_x.begin(), d_particles.v_occupied_x.begin() + h_measurements.LEN, h_particles.v_occupied_x.begin());
    thrust::copy(d_particles.v_occupied_y.begin(), d_particles.v_occupied_y.begin() + h_measurements.LEN, h_particles.v_occupied_y.begin());
    thrust::copy(d_particles.v_world_x.begin(), d_particles.v_world_x.begin() + h_measurements.LEN, h_particles.v_world_x.begin());
    thrust::copy(d_particles.v_world_y.begin(), d_particles.v_world_y.begin() + h_measurements.LEN, h_particles.v_world_y.begin());

    ASSERT_transition_world_lidar(h_transition.c_world_lidar.data(), post_transition.c_world_lidar.data(), 9, false);
    ASSERT_particles_world_frame(h_particles.v_world_x.data(), h_particles.v_world_y.data(),
        post_particles.v_world_x.data(), post_particles.v_world_y.data(), h_measurements.LEN, false);
    ASSERT_processed_measurements(h_particles.v_occupied_x.data(), h_particles.v_occupied_y.data(),
        post_particles.v_occupied_x.data(), post_particles.v_occupied_y.data(), h_measurements.LEN, false);
    ASSERT_position_image_body(h_position.c_image_body.data(), pre_position.c_image_body.data(), true, true);

    /************************* BRESENHAM KERNEL *************************/
    auto start_bresenham = std::chrono::high_resolution_clock::now();

    threadsPerBlock = 256;
    blocksPerGrid = (h_measurements.LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_bresenham << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles.sv_free_x_max), THRUST_RAW_CAST(d_particles.sv_free_y_max), THRUST_RAW_CAST(d_particles.v_free_counter), SEP,
        THRUST_RAW_CAST(d_particles.v_occupied_x), THRUST_RAW_CAST(d_particles.v_occupied_y), THRUST_RAW_CAST(d_position.c_image_body),
        h_measurements.LEN, MAX_DIST_IN_MAP);
    cudaDeviceSynchronize();
    thrust::exclusive_scan(thrust::device, d_particles.v_free_counter.begin(), d_particles.v_free_counter.begin() + PARTICLE_UNIQUE_COUNTER, 
        d_particles.v_free_counter.begin(), 0);
    auto stop_bresenham = std::chrono::high_resolution_clock::now();

    auto start_bresenham_rearrange = std::chrono::high_resolution_clock::now();

    thrust::copy(d_particles.v_free_counter.begin(), d_particles.v_free_counter.begin() + PARTICLE_UNIQUE_COUNTER, h_particles.v_free_counter.begin());
    thrust::copy(d_particles.v_free_counter.begin(), d_particles.v_free_counter.begin() + PARTICLE_UNIQUE_COUNTER, d_particles.v_free_idx.begin());

    h_particles.FREE_LEN = h_particles.v_free_counter[PARTICLE_UNIQUE_COUNTER - 1];
    printf("^^^ PARTICLES_FREE_LEN = %d\n", h_particles.FREE_LEN);


    kernel_bresenham_rearrange << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles.f_free_x), THRUST_RAW_CAST(d_particles.f_free_y), SEP,
        THRUST_RAW_CAST(d_particles.sv_free_x_max), THRUST_RAW_CAST(d_particles.sv_free_y_max),
        THRUST_RAW_CAST(d_particles.v_free_counter), MAX_DIST_IN_MAP, h_measurements.LEN);
    cudaDeviceSynchronize();
    auto stop_bresenham_rearrange = std::chrono::high_resolution_clock::now();

    thrust::copy(d_particles.f_free_x.begin(), d_particles.f_free_x.begin() + h_particles.FREE_LEN, h_particles.f_free_x.begin());
    thrust::copy(d_particles.f_free_y.begin(), d_particles.f_free_y.begin() + h_particles.FREE_LEN, h_particles.f_free_y.begin());

    printf("~~$ PARTICLES_FREE_LEN = %d\n", h_particles.FREE_LEN);

    ASSERT_particles_free_index(h_particles.v_free_counter.data(), post_particles.v_free_idx.data(), h_measurements.LEN, false);
    ASSERT_particles_free_new_len(h_particles.FREE_LEN, post_particles.FREE_LEN);
    ASSERT_particles_free(h_particles.f_free_x.data(), h_particles.f_free_y.data(),
        post_particles.f_free_x.data(), post_particles.f_free_y.data(), h_particles.FREE_LEN);

    /*---------------------------------------------------------------------*/
    /*-------------------- REINITIALIZE MAP VARIABLES ---------------------*/
    h_unique_free.c_idx[1] = h_particles.FREE_LEN;
    d_unique_free.c_idx.assign(h_unique_free.c_idx.begin(), h_unique_free.c_idx.end());

    /************************** CREATE 2D MAP ***************************/
    auto start_create_map = std::chrono::high_resolution_clock::now();
    threadsPerBlock = 256;
    blocksPerGrid = 1;
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_unique_occupied.s_map), SEP,
        THRUST_RAW_CAST(d_particles.v_occupied_x), THRUST_RAW_CAST(d_particles.v_occupied_y), 
        THRUST_RAW_CAST(d_unique_occupied.c_idx), h_measurements.LEN, h_map.GRID_WIDTH, h_map.GRID_HEIGHT, NUM_PARTICLES);
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_unique_free.s_map), SEP,
        THRUST_RAW_CAST(d_particles.f_free_x), THRUST_RAW_CAST(d_particles.f_free_y), 
        THRUST_RAW_CAST(d_unique_free.c_idx), h_particles.FREE_LEN, h_map.GRID_WIDTH, h_map.GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();
    auto stop_create_map = std::chrono::high_resolution_clock::now();

    auto start_unique_counter = std::chrono::high_resolution_clock::now();
    threadsPerBlock = h_map.GRID_WIDTH;
    blocksPerGrid = 1;
    kernel_2d_map_counter << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_unique_occupied.c_in_map), THRUST_RAW_CAST(d_unique_occupied.s_in_col), SEP,
        THRUST_RAW_CAST(d_unique_occupied.s_map), h_map.GRID_WIDTH, h_map.GRID_HEIGHT);
    kernel_2d_map_counter << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_unique_free.c_in_map), THRUST_RAW_CAST(d_unique_free.s_in_col), SEP,
        THRUST_RAW_CAST(d_unique_free.s_map), h_map.GRID_WIDTH, h_map.GRID_HEIGHT);
    cudaDeviceSynchronize();
    auto stop_unique_counter = std::chrono::high_resolution_clock::now();

    auto start_unique_sum = std::chrono::high_resolution_clock::now();
    threadsPerBlock = 1;
    blocksPerGrid = 1;
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (THRUST_RAW_CAST(d_unique_occupied.s_in_col), h_map.GRID_WIDTH);
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (THRUST_RAW_CAST(d_unique_free.s_in_col), h_map.GRID_WIDTH);
    cudaDeviceSynchronize();
    auto stop_unique_sum = std::chrono::high_resolution_clock::now();

    h_unique_occupied.c_in_map.assign(d_unique_occupied.c_in_map.begin(), d_unique_occupied.c_in_map.end());
    h_unique_free.c_in_map.assign(d_unique_free.c_in_map.begin(), d_unique_free.c_in_map.end());

    /*---------------------------------------------------------------------*/
    /*-------------------- REINITIALIZE MAP VARIABLES ---------------------*/
    h_particles.OCCUPIED_UNIQUE_LEN = h_unique_occupied.c_in_map[0];
    h_particles.FREE_UNIQUE_LEN = h_unique_free.c_in_map[0];

    printf("\n--> Occupied Unique: %d, %d\n", h_particles.OCCUPIED_UNIQUE_LEN, post_particles.OCCUPIED_UNIQUE_LEN);
    assert(h_particles.OCCUPIED_UNIQUE_LEN == post_particles.OCCUPIED_UNIQUE_LEN);
    printf("\n--> Free Unique: %d, %d\n", h_particles.FREE_UNIQUE_LEN, post_particles.FREE_UNIQUE_LEN);
    assert(h_particles.FREE_UNIQUE_LEN == post_particles.FREE_UNIQUE_LEN);

    thrust::fill(d_particles.f_occupied_unique_x.begin(), d_particles.f_occupied_unique_x.begin() + h_particles.OCCUPIED_UNIQUE_LEN, 0);
    thrust::fill(d_particles.f_occupied_unique_y.begin(), d_particles.f_occupied_unique_y.begin() + h_particles.OCCUPIED_UNIQUE_LEN, 0);
    thrust::fill(d_particles.f_free_unique_y.begin(), d_particles.f_free_unique_y.begin() + h_particles.FREE_UNIQUE_LEN, 0);
    thrust::fill(d_particles.f_free_unique_y.begin(), d_particles.f_free_unique_y.begin() + h_particles.FREE_UNIQUE_LEN, 0);

    auto start_restructure_map = std::chrono::high_resolution_clock::now();
    threadsPerBlock = h_map.GRID_WIDTH;
    blocksPerGrid = 1;
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles.f_occupied_unique_x), THRUST_RAW_CAST(d_particles.f_occupied_unique_y), SEP,
        THRUST_RAW_CAST(d_unique_occupied.s_map), THRUST_RAW_CAST(d_unique_occupied.s_in_col), h_map.GRID_WIDTH, h_map.GRID_HEIGHT);
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles.f_free_unique_x), THRUST_RAW_CAST(d_particles.f_free_unique_y), SEP,
        THRUST_RAW_CAST(d_unique_free.s_map), THRUST_RAW_CAST(d_unique_free.s_in_col), h_map.GRID_WIDTH, h_map.GRID_HEIGHT);
    cudaDeviceSynchronize();
    auto stop_restructure_map = std::chrono::high_resolution_clock::now();

    thrust::copy(d_particles.f_occupied_unique_x.begin(), d_particles.f_occupied_unique_x.begin() + h_particles.OCCUPIED_UNIQUE_LEN, h_particles.f_occupied_unique_x.begin());
    thrust::copy(d_particles.f_occupied_unique_y.begin(), d_particles.f_occupied_unique_y.begin() + h_particles.OCCUPIED_UNIQUE_LEN, h_particles.f_occupied_unique_y.begin());
    thrust::copy(d_particles.f_free_unique_x.begin(), d_particles.f_free_unique_x.begin() + h_particles.FREE_UNIQUE_LEN, h_particles.f_free_unique_x.begin());
    thrust::copy(d_particles.f_free_unique_y.begin(), d_particles.f_free_unique_y.begin() + h_particles.FREE_UNIQUE_LEN, h_particles.f_free_unique_y.begin());

    ASSERT_particles_occupied(h_particles.f_occupied_unique_x.data(), h_particles.f_occupied_unique_y.data(),
        post_particles.f_occupied_unique_x.data(), post_particles.f_occupied_unique_y.data(),
        "Occupied", h_particles.OCCUPIED_UNIQUE_LEN, false);
    ASSERT_particles_occupied(h_particles.f_free_unique_x.data(), h_particles.f_free_unique_y.data(),
        post_particles.f_free_unique_x.data(), post_particles.f_free_unique_y.data(),
        "Free", h_particles.FREE_UNIQUE_LEN, false);

    /************************** LOG-ODDS KERNEL *************************/
    auto start_update_map = std::chrono::high_resolution_clock::now();

    threadsPerBlock = 256;
    blocksPerGrid = (h_particles.OCCUPIED_UNIQUE_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_update_log_odds << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_map.s_log_odds), SEP, 
        THRUST_RAW_CAST(d_particles.f_occupied_unique_x), THRUST_RAW_CAST(d_particles.f_occupied_unique_y), 2 * general_info.log_t,
        h_map.GRID_WIDTH, h_map.GRID_HEIGHT, h_particles.OCCUPIED_UNIQUE_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = 256;
    blocksPerGrid = (h_particles.FREE_UNIQUE_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_update_log_odds << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_map.s_log_odds), SEP, 
        THRUST_RAW_CAST(d_particles.f_free_unique_x), THRUST_RAW_CAST(d_particles.f_free_unique_y), (-1) * general_info.log_t,
        h_map.GRID_WIDTH, h_map.GRID_HEIGHT, h_particles.FREE_UNIQUE_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = 256;
    blocksPerGrid = ((h_map.GRID_WIDTH * h_map.GRID_HEIGHT) + threadsPerBlock - 1) / threadsPerBlock;
    kernel_update_map << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_map.s_grid_map), SEP, THRUST_RAW_CAST(d_map.s_log_odds), LOG_ODD_PRIOR, WALL, FREE, 
        h_map.GRID_WIDTH * h_map.GRID_HEIGHT);
    cudaDeviceSynchronize();
    auto stop_update_map = std::chrono::high_resolution_clock::now();

    thrust::fill(h_map.s_log_odds.begin(), h_map.s_log_odds.end(), 0);
    h_map.s_log_odds.assign(d_map.s_log_odds.begin(), d_map.s_log_odds.end());
    h_map.s_grid_map.assign(d_map.s_grid_map.begin(), d_map.s_grid_map.end());

    ASSERT_log_odds(h_map.s_log_odds.data(), pre_map.s_log_odds.data(), pre_map_post.s_log_odds.data(),
        (h_map.GRID_WIDTH * h_map.GRID_HEIGHT), (pre_map.GRID_WIDTH* pre_map.GRID_HEIGHT), false);
    ASSERT_log_odds_maps(h_map.s_grid_map.data(), pre_map.s_grid_map.data(), pre_map_post.s_grid_map.data(),
        (h_map.GRID_WIDTH * h_map.GRID_HEIGHT), (pre_map.GRID_WIDTH* pre_map.GRID_HEIGHT), false);
    printf("\n");


    auto duration_world_to_image_transform_1 = std::chrono::duration_cast<std::chrono::microseconds>(stop_world_to_image_transform_1 - start_world_to_image_transform_1);
    auto duration_world_to_image_transform_2 = std::chrono::duration_cast<std::chrono::microseconds>(stop_world_to_image_transform_2 - start_world_to_image_transform_2);
    auto duration_bresenham = std::chrono::duration_cast<std::chrono::microseconds>(stop_bresenham - start_bresenham);
    auto duration_bresenham_rearrange = std::chrono::duration_cast<std::chrono::microseconds>(stop_bresenham_rearrange - start_bresenham_rearrange);
    auto duration_create_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_create_map - start_create_map);
    auto duration_unique_counter = std::chrono::duration_cast<std::chrono::microseconds>(stop_unique_counter - start_unique_counter);
    auto duration_unique_sum = std::chrono::duration_cast<std::chrono::microseconds>(stop_unique_sum - start_unique_sum);
    auto duration_restructure_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_restructure_map - start_restructure_map);
    auto duration_update_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_update_map - start_update_map);

    auto duration_total = duration_world_to_image_transform_1 + duration_world_to_image_transform_2 + duration_bresenham + duration_bresenham_rearrange + duration_create_map +
        duration_unique_counter + duration_unique_sum + duration_restructure_map + duration_update_map;

    std::cout << "Time taken by function (World To Image Transform 1): " << duration_world_to_image_transform_1.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (World To Image Transform 2): " << duration_world_to_image_transform_2.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Bresenham): " << duration_bresenham.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Bresenham Rearrange): " << duration_bresenham_rearrange.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Create Map): " << duration_create_map.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Unique Counter): " << duration_unique_counter.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Unique Sum): " << duration_unique_sum.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Restructure Map): " << duration_restructure_map.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Update Map): " << duration_update_map.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Total): " << duration_total.count() << " microseconds" << std::endl;
    std::cout << std::endl;
}


void test_map_func(HostMap& pre_map, HostMap& pre_map_bg,
    HostMap& pre_map_post, GeneralInfo& general_info, HostMeasurements& pre_measurements,
    HostParticles& pre_particles, HostParticles& post_particles, 
    HostPosition& post_position, HostTransition& pre_transition, HostTransition& post_transition) {

    host_vector<int> hvec_occupied_map_idx(2, 0);
    host_vector<int> hvec_free_map_idx(2, 0);

    printf("/******************************** Class Base MAP *******************************/\n");

    int LIDAR_COORDS_LEN = pre_measurements.LEN;

    HostPosition h_position;
    HostTransition h_transition;
    DevicePosition d_position;
    DeviceTransition d_transition;
    DeviceMeasurements d_measurements;
    HostMeasurements h_measurements;
    HostParticles h_particles;
    DeviceParticles d_particles;
    DeviceMap d_map;
    HostMap h_map;
    Host2DUniqueFinder h_unique_occupied;
    Host2DUniqueFinder h_unique_free;
    Device2DUniqueFinder d_unique_occupied;
    Device2DUniqueFinder d_unique_free;

    auto start_alloc_init = std::chrono::high_resolution_clock::now();
    alloc_init_transition_vars(d_position, d_transition, h_position, h_transition, pre_transition);
    alloc_init_body_lidar(d_transition);
    alloc_init_measurement_vars(d_measurements, h_measurements, pre_measurements);

    int MAX_DIST_IN_MAP = sqrt(pow(pre_map.GRID_WIDTH, 2) + pow(pre_map.GRID_HEIGHT, 2));

    alloc_init_particles_vars(d_particles, h_particles, pre_measurements, pre_particles, MAX_DIST_IN_MAP);
    alloc_init_map_vars(d_map, h_map, pre_map);

    hvec_occupied_map_idx[1] = h_particles.OCCUPIED_LEN;
    hvec_free_map_idx[1] = 0;

    alloc_init_unique_map_vars(d_unique_occupied, h_unique_occupied, h_map, hvec_occupied_map_idx);
    alloc_init_unique_map_vars(d_unique_free, h_unique_free, h_map, hvec_free_map_idx);
    auto stop_alloc_init = std::chrono::high_resolution_clock::now();

    auto start_world_to_image_transform_1 = std::chrono::high_resolution_clock::now();
    exec_world_to_image_transform_step_1(d_position, d_transition, d_particles, d_measurements, h_measurements);
    auto stop_world_to_image_transform_1 = std::chrono::high_resolution_clock::now();

    bool EXTEND = false;
    auto start_check_extend = std::chrono::high_resolution_clock::now();
    exec_map_extend(d_map, d_measurements, d_particles, d_unique_occupied, d_unique_free,
        h_map, h_measurements, h_unique_occupied, h_unique_free, general_info, EXTEND);
    auto stop_check_extend = std::chrono::high_resolution_clock::now();

    assert_map_extend(h_map, pre_map, pre_map_bg, pre_map_post, EXTEND);

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

    auto start_restructure_map = std::chrono::high_resolution_clock::now();
    reinit_map_vars(d_particles, d_unique_occupied, d_unique_free, h_particles, h_unique_occupied, h_unique_free);
    exec_map_restructure(d_particles, d_unique_occupied, d_unique_free, h_map);
    auto stop_restructure_map = std::chrono::high_resolution_clock::now();

    assert_map_restructure(d_particles, h_particles, post_particles);

    auto start_update_map = std::chrono::high_resolution_clock::now();
    exec_log_odds(d_map, d_particles, h_map, h_particles, general_info);
    auto stop_update_map = std::chrono::high_resolution_clock::now();

    assert_log_odds(d_map, h_map, pre_map, pre_map_post);

    auto duration_alloc_init = std::chrono::duration_cast<std::chrono::microseconds>(stop_alloc_init - start_alloc_init);
    auto duration_world_to_image_transform_1 = std::chrono::duration_cast<std::chrono::microseconds>(stop_world_to_image_transform_1 - start_world_to_image_transform_1);
    auto duration_world_to_image_transform_2 = std::chrono::duration_cast<std::chrono::microseconds>(stop_world_to_image_transform_2 - start_world_to_image_transform_2);
    auto duration_bresenham = std::chrono::duration_cast<std::chrono::microseconds>(stop_bresenham - start_bresenham);
    auto duration_create_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_create_map - start_create_map);
    auto duration_restructure_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_restructure_map - start_restructure_map);
    auto duration_update_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_update_map - start_update_map);

    auto duration_total = duration_world_to_image_transform_1 + duration_world_to_image_transform_2 + duration_bresenham + duration_create_map +
        duration_restructure_map + duration_update_map + duration_alloc_init;

    
    std::cout << "Time taken by function (Allocation & Init): " << duration_alloc_init.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (World To Image Transform 1): " << duration_world_to_image_transform_1.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (World To Image Transform 2): " << duration_world_to_image_transform_2.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Bresenham): " << duration_bresenham.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Create Map): " << duration_create_map.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Restructure Map): " << duration_restructure_map.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Update Map): " << duration_update_map.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Total): " << duration_total.count() << " microseconds" << std::endl;
    std::cout << std::endl;
}



#endif
