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
#include "device_exec_map.h"
#include "device_assert_map.h"

void host_update_map_init(HostMapData&, HostMapData&, HostMapData&, GeneralInfo&, HostMeasurements&,
    HostParticlesData&, HostPositionTransition&);                           // Step 1
void host_bresenham(HostMapData&, HostMapData&, HostMapData&, GeneralInfo&, HostMeasurements&,
    HostParticlesData&, HostPositionTransition&);                           // Step 2
void host_update_map(HostMapData&, HostMapData&, HostMapData&, GeneralInfo&, HostMeasurements&,
    HostParticlesData&, HostPositionTransition&);                           // Step 3
void host_map(HostMapData&, HostMapData&, HostMapData&, GeneralInfo&, HostMeasurements&,
    HostParticlesData&, HostPositionTransition&);                           // Step Y

void test_map_func(HostMapData&, HostMapData&, HostMapData&, GeneralInfo&, HostMeasurements&,
    HostParticlesData&, HostPositionTransition&);

const int FILE_NUMBER = 720;

int test_map_main() {

    const int data_len = 6;
    int data[data_len] = { 1, 0, 2, 2, 1, 3 };
    int* d_data = NULL;
    gpuErrchk(cudaMalloc((void**)&d_data, data_len * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_data, data, data_len * sizeof(int), cudaMemcpyHostToDevice));

    thrust::exclusive_scan(thrust::host, data, data + 6, data, 0);
    thrust::exclusive_scan(thrust::device, d_data, d_data + 6, d_data, 0);

    HostMapData h_map_data;
    HostMapData h_map_data_bg;
    HostMapData h_map_data_post;
    GeneralInfo general_info;
    HostMeasurements h_measurements;
    HostParticlesData h_particles_data;
    HostPositionTransition h_position_transition;

    printf("/************************** UPDATE MAP INIT *************************/\n");

    auto start_read_data_file = std::chrono::high_resolution_clock::now();
    read_update_map(FILE_NUMBER, h_map_data, h_map_data_bg,
        h_map_data_post, general_info, h_measurements, h_particles_data, h_position_transition);
    auto stop_read_data_file = std::chrono::high_resolution_clock::now();

    auto duration_read_data_file = std::chrono::duration_cast<std::chrono::milliseconds>(stop_read_data_file - start_read_data_file);
    std::cout << "Time taken by function (Read Data File): " << duration_read_data_file.count() << " milliseconds" << std::endl;


    host_update_map_init(h_map_data, h_map_data_bg, h_map_data_post, general_info, h_measurements, 
        h_particles_data, h_position_transition);
    host_bresenham(h_map_data, h_map_data_bg, h_map_data_post, general_info, h_measurements,
        h_particles_data, h_position_transition);
    host_update_map(h_map_data, h_map_data_bg, h_map_data_post, general_info, h_measurements,
        h_particles_data, h_position_transition);
    host_map(h_map_data, h_map_data_bg, h_map_data_post, general_info, h_measurements,
        h_particles_data, h_position_transition);

    test_map_func(h_map_data, h_map_data_bg, h_map_data_post, general_info, h_measurements,
        h_particles_data, h_position_transition);
    test_map_func(h_map_data, h_map_data_bg, h_map_data_post, general_info, h_measurements,
        h_particles_data, h_position_transition);

    return 0;
}

void host_update_map_init(HostMapData& h_map_data, HostMapData& h_map_data_bg,
    HostMapData& h_map_data_post, GeneralInfo& general_info, HostMeasurements& h_measurements,
    HostParticlesData& h_particles_data, HostPositionTransition& h_position_transition) {

    printf("/************************** UPDATE MAP INIT *************************/\n");

    int xmin = h_map_data.xmin;
    int xmax = h_map_data.xmax;
    int ymin = h_map_data.ymin;
    int ymax = h_map_data.ymax;

    int xmin_pre = xmin;
    int ymax_pre = ymax;

    int GRID_WIDTH = h_map_data.GRID_WIDTH;
    int GRID_HEIGHT = h_map_data.GRID_HEIGHT;
    int LIDAR_COORDS_LEN = h_measurements.LIDAR_COORDS_LEN;

    /********************* IMAGE TRANSFORM VARIABLES ********************/
    HostMeasurements res_measurements;
    res_measurements.processed_single_measure_x.resize(LIDAR_COORDS_LEN);
    res_measurements.processed_single_measure_y.resize(LIDAR_COORDS_LEN);
    
    HostParticlesData res_particles_data;
    res_particles_data.particles_world_x.resize(LIDAR_COORDS_LEN);
    res_particles_data.particles_world_y.resize(LIDAR_COORDS_LEN);
    
    HostPositionTransition res_position_transition;
    res_position_transition.transition_world_lidar.resize(9);
    res_position_transition.position_image_body.resize(2);


    DeviceMeasurements d_measurements;
    d_measurements.resize(2 * LIDAR_COORDS_LEN, 0);
    d_measurements.processed_single_measure_x.resize(LIDAR_COORDS_LEN);
    d_measurements.processed_single_measure_y.resize(LIDAR_COORDS_LEN);
    d_measurements.lidar_coords.assign(h_measurements.lidar_coords.begin(), h_measurements.lidar_coords.end());

    DevicePositionTransition d_position_transition;
    d_position_transition.transition_body_lidar.resize(9);
    d_position_transition.transition_world_body.resize(9);
    d_position_transition.transition_single_world_lidar.resize(9);
    d_position_transition.position_image_body.resize(2);
    d_position_transition.transition_single_world_body.assign(h_position_transition.transition_world_body.begin(), h_position_transition.transition_world_body.end());
    d_position_transition.transition_body_lidar.assign(hvec_transition_body_lidar.begin(), hvec_transition_body_lidar.end());


    DeviceParticlesData d_particles_data;
    d_particles_data.particles_world_x.resize(LIDAR_COORDS_LEN);
    d_particles_data.particles_world_y.resize(LIDAR_COORDS_LEN);    


    DeviceMapData d_map_data;
    d_map_data.grid_map.resize(GRID_WIDTH * GRID_HEIGHT);
    d_map_data.log_odds.resize(GRID_WIDTH * GRID_HEIGHT);
    d_map_data.grid_map.assign(h_map_data.grid_map.begin(), h_map_data.grid_map.end());
    d_map_data.log_odds.assign(h_map_data.log_odds.begin(), h_map_data.log_odds.end());


    HostMapData res_map_data;
    res_map_data.grid_map.resize(GRID_WIDTH * GRID_HEIGHT);
    res_map_data.log_odds.resize(GRID_WIDTH * GRID_HEIGHT);
    res_map_data.should_extend.resize(4, 0);

    

    /********************** IMAGE TRANSFORM KERNEL **********************/
    auto start_world_to_image_transform_1 = std::chrono::high_resolution_clock::now();
    kernel_matrix_mul_3x3 << < 1, 1 >> > (
        THRUST_RAW_CAST(d_position_transition.transition_single_world_body),
        THRUST_RAW_CAST(d_position_transition.transition_body_lidar), THRUST_RAW_CAST(d_position_transition.transition_single_world_lidar));
    cudaDeviceSynchronize();

    int threadsPerBlock = 1;
    int blocksPerGrid = LIDAR_COORDS_LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles_data.particles_world_x), THRUST_RAW_CAST(d_particles_data.particles_world_y), SEP,
        THRUST_RAW_CAST(d_position_transition.transition_single_world_lidar), THRUST_RAW_CAST(d_measurements.lidar_coords), LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();
    auto stop_world_to_image_transform_1 = std::chrono::high_resolution_clock::now();

    auto start_check_extend = std::chrono::high_resolution_clock::now();
    threadsPerBlock = 256;
    blocksPerGrid = (LIDAR_COORDS_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_check_map_extend_less << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_map_data.should_extend), SEP, THRUST_RAW_CAST(d_particles_data.particles_world_x), xmin, 0, LIDAR_COORDS_LEN);
    kernel_check_map_extend_less << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_map_data.should_extend), SEP, THRUST_RAW_CAST(d_particles_data.particles_world_y), ymin, 1, LIDAR_COORDS_LEN);

    kernel_check_map_extend_greater << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_map_data.should_extend), SEP, THRUST_RAW_CAST(d_particles_data.particles_world_x), xmax, 2, LIDAR_COORDS_LEN);
    kernel_check_map_extend_greater << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_map_data.should_extend), SEP, THRUST_RAW_CAST(d_particles_data.particles_world_y), ymax, 3, LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    res_map_data.should_extend.assign(d_map_data.should_extend.begin(), d_map_data.should_extend.end());

    bool EXTEND = false;
    if (res_map_data.should_extend[0] != 0) {
        EXTEND = true;
        xmin = xmin * 2;
    }
    else if (res_map_data.should_extend[2] != 0) {
        EXTEND = true;
        xmax = xmax * 2;
    }
    else if (res_map_data.should_extend[1] != 0) {
        EXTEND = true;
        ymin = ymin * 2;
    }
    else if (res_map_data.should_extend[3] != 0) {
        EXTEND = true;
        ymax = ymax * 2;
    }
    auto stop_check_extend = std::chrono::high_resolution_clock::now();

    printf("xmin=%d, xmax=%d, ymin=%d, ymax=%d\n", xmin, xmax, ymin, ymax);
    for (int i = 0; i < 4; i++)
        std::cout << "Should Extend: " << res_map_data.should_extend[i] << std::endl;

    assert(EXTEND == h_map_data.b_should_extend);

    if (EXTEND == true) {

        auto start_extend = std::chrono::high_resolution_clock::now();

        device_vector<int> dvec_coord(2);
        host_vector<int> hvec_coord(2);

        kernel_position_to_image << <1, 1 >> > (
            THRUST_RAW_CAST(dvec_coord), SEP, xmin_pre, ymax_pre, general_info.res, xmin, ymax);
        cudaDeviceSynchronize();

        hvec_coord.assign(dvec_coord.begin(), dvec_coord.end());

        device_vector<int> dvec_clone_grid_map(d_map_data.grid_map.size());
        dvec_clone_grid_map.assign(d_map_data.grid_map.begin(), d_map_data.grid_map.end());

        device_vector<float> dvec_clone_log_odds(d_map_data.log_odds.size());
        dvec_clone_log_odds.assign(d_map_data.log_odds.begin(), d_map_data.log_odds.end());

        const int PRE_GRID_WIDTH = GRID_WIDTH;
        const int PRE_GRID_HEIGHT = GRID_HEIGHT;
        GRID_WIDTH = ceil((ymax - ymin) / general_info.res + 1);
        GRID_HEIGHT = ceil((xmax - xmin) / general_info.res + 1);
        printf("GRID_WIDTH=%d, AF_GRID_WIDTH=%d, GRID_HEIGHT=%d, AF_GRID_HEIGHT=%d\n",
            GRID_WIDTH, GRID_HEIGHT, h_map_data_post.GRID_WIDTH, h_map_data_post.GRID_HEIGHT);
        assert(GRID_WIDTH == h_map_data_post.GRID_WIDTH);
        assert(GRID_HEIGHT == h_map_data_post.GRID_HEIGHT);

        const int PRE_GRID_SIZE = PRE_GRID_WIDTH * PRE_GRID_HEIGHT;
        const int NEW_GRID_SIZE = GRID_WIDTH * GRID_HEIGHT;

        d_map_data.grid_map.clear();
        d_map_data.log_odds.clear();
        d_map_data.grid_map.resize(GRID_WIDTH * GRID_HEIGHT, 0);
        d_map_data.log_odds.resize(GRID_WIDTH * GRID_HEIGHT, LOG_ODD_PRIOR);

        res_map_data.grid_map.resize(GRID_WIDTH * GRID_HEIGHT, 0);
        res_map_data.log_odds.resize(GRID_WIDTH * GRID_HEIGHT, 0);

        threadsPerBlock = 256;
        blocksPerGrid = (PRE_GRID_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        kernel_2d_copy_with_offset << <blocksPerGrid, threadsPerBlock >> > (
            THRUST_RAW_CAST(d_map_data.grid_map), THRUST_RAW_CAST(d_map_data.log_odds), SEP,
            THRUST_RAW_CAST(dvec_clone_grid_map), THRUST_RAW_CAST(dvec_clone_log_odds), hvec_coord[0], hvec_coord[1],
            PRE_GRID_HEIGHT, GRID_HEIGHT, PRE_GRID_SIZE);
        cudaDeviceSynchronize();

        auto stop_extend = std::chrono::high_resolution_clock::now();
        auto duration_extend = std::chrono::duration_cast<std::chrono::microseconds>(stop_extend - start_extend);

        std::cout << std::endl;
        std::cout << "Time taken by function (Extend): " << duration_extend.count() << " microseconds" << std::endl;

        res_map_data.grid_map.assign(d_map_data.grid_map.begin(), d_map_data.grid_map.end());
        res_map_data.log_odds.assign(d_map_data.log_odds.begin(), d_map_data.log_odds.end());

        int error_map = 0;
        int error_log = 0;
        for (int i = 0; i < (GRID_WIDTH * GRID_HEIGHT); i++) {
            if (res_map_data.grid_map[i] != h_map_data_bg.grid_map[i]) {
                error_map += 1;
                //printf("Grid Map: %d <> %d\n", hvec_grid_map[i], bg_grid_map[i]);
            }
            if (abs(res_map_data.log_odds[i] - h_map_data_bg.log_odds[i]) > 1e-4) {
                error_log += 1;
                printf("Log Odds: (%d) %f <> %f\n", i, res_map_data.log_odds[i], h_map_data_bg.log_odds[i]);
            }
        }
        printf("Map Erros: %d\n", error_map);
        printf("Log Erros: %d\n", error_log);
    }


    auto start_world_to_image_transform_2 = std::chrono::high_resolution_clock::now();
    threadsPerBlock = 1;
    blocksPerGrid = LIDAR_COORDS_LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_measurements.processed_single_measure_x), THRUST_RAW_CAST(d_measurements.processed_single_measure_y), SEP,
        THRUST_RAW_CAST(d_position_transition.transition_single_world_lidar), THRUST_RAW_CAST(d_measurements.lidar_coords),
        general_info.res, xmin, ymax, LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    kernel_position_to_image << < 1, 1 >> > (THRUST_RAW_CAST(d_position_transition.position_image_body), SEP,
        THRUST_RAW_CAST(d_position_transition.transition_single_world_lidar), general_info.res, xmin, ymax);
    cudaDeviceSynchronize();
    auto stop_world_to_image_transform_2 = std::chrono::high_resolution_clock::now();

    res_position_transition.transition_world_lidar.assign(d_position_transition.transition_single_world_lidar.begin(), d_position_transition.transition_single_world_lidar.end());
    res_position_transition.position_image_body.assign(d_position_transition.position_image_body.begin(), d_position_transition.position_image_body.end());
    res_measurements.processed_single_measure_x.assign(d_measurements.processed_single_measure_x.begin(), d_measurements.processed_single_measure_x.end());
    res_measurements.processed_single_measure_y.assign(d_measurements.processed_single_measure_y.begin(), d_measurements.processed_single_measure_y.end());
    res_particles_data.particles_world_x.assign(d_particles_data.particles_world_x.begin(), d_particles_data.particles_world_x.end());
    res_particles_data.particles_world_y.assign(d_particles_data.particles_world_y.begin(), d_particles_data.particles_world_y.end());


    ASSERT_transition_world_lidar(res_position_transition.transition_world_lidar.data(), h_position_transition.transition_world_lidar.data(), 9, false);
    ASSERT_particles_world_frame(res_particles_data.particles_world_x.data(), res_particles_data.particles_world_y.data(),
        h_particles_data.particles_world_x.data(), h_particles_data.particles_world_y.data(), LIDAR_COORDS_LEN, false);
    ASSERT_processed_measurements(res_measurements.processed_single_measure_x.data(), res_measurements.processed_single_measure_y.data(),
        h_particles_data.particles_occupied_x.data(), h_particles_data.particles_occupied_y.data(), LIDAR_COORDS_LEN);
    ASSERT_position_image_body(res_position_transition.position_image_body.data(), h_position_transition.position_image_body.data(), true, true);

    auto duration_world_to_image_transform_1 = std::chrono::duration_cast<std::chrono::microseconds>(stop_world_to_image_transform_1 - start_world_to_image_transform_1);
    auto duration_world_to_image_transform_2 = std::chrono::duration_cast<std::chrono::microseconds>(stop_world_to_image_transform_2 - start_world_to_image_transform_2);
    std::cout << "Time taken by function (World To Image Transform 1): " << duration_world_to_image_transform_1.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (World To Image Transform 2): " << duration_world_to_image_transform_2.count() << " microseconds" << std::endl;
    std::cout << std::endl;
}

void host_bresenham(HostMapData& h_map_data, HostMapData& h_map_data_bg,
    HostMapData& h_map_data_post, GeneralInfo& general_info, HostMeasurements& h_measurements,
    HostParticlesData& h_particles_data, HostPositionTransition& h_position_transition) {

    printf("/***************************** BRESENHAM ****************************/\n");

    int LIDAR_COORDS_LEN = h_measurements.LIDAR_COORDS_LEN;
    int GRID_WIDTH = h_map_data.GRID_WIDTH;
    int GRID_HEIGHT = h_map_data.GRID_HEIGHT;

    printf("~~$ GRID_WIDTH = \t%d\n", GRID_WIDTH);
    printf("~~$ GRID_HEIGHT = \t%d\n", GRID_HEIGHT);
    int MAX_DIST_IN_MAP = sqrt(pow(GRID_WIDTH, 2) + pow(GRID_HEIGHT, 2));
    printf("~~$ MAX_DIST_IN_MAP = \t%d\n", MAX_DIST_IN_MAP);

    /************************ BRESENHAM VARIABLES ***********************/
    int PARTICLES_OCCUPIED_LEN = h_particles_data.PARTICLES_OCCUPIED_LEN;
    int PARTICLES_FREE_LEN = h_particles_data.PARTICLES_FREE_LEN;
    int PARTICLE_UNIQUE_COUNTER = PARTICLES_OCCUPIED_LEN + 1;

    
    DeviceParticlesData d_particles_data;
    d_particles_data.particles_occupied_x.resize(PARTICLES_OCCUPIED_LEN);
    d_particles_data.particles_occupied_y.resize(PARTICLES_OCCUPIED_LEN);
    d_particles_data.particles_free_x_max.resize(PARTICLES_OCCUPIED_LEN * MAX_DIST_IN_MAP, 0);
    d_particles_data.particles_free_y_max.resize(PARTICLES_OCCUPIED_LEN * MAX_DIST_IN_MAP, 0);
    d_particles_data.particles_free_counter.resize(PARTICLE_UNIQUE_COUNTER, 0);
    d_particles_data.particles_free_idx.resize(PARTICLES_OCCUPIED_LEN);
    d_particles_data.particles_occupied_x.assign(h_particles_data.particles_occupied_x.begin(), h_particles_data.particles_occupied_x.end());
    d_particles_data.particles_occupied_y.assign(h_particles_data.particles_occupied_y.begin(), h_particles_data.particles_occupied_y.end());
    d_particles_data.particles_free_idx.assign(h_particles_data.particles_free_idx.begin(), h_particles_data.particles_free_idx.end());

    HostParticlesData res_particles_data;
    res_particles_data.particles_free_counter.resize(PARTICLE_UNIQUE_COUNTER, 0);

    //device_vector<int> dvec_position_image_body(2);
    DevicePositionTransition d_position_transition;
    d_position_transition.position_image_body.resize(2);
    d_position_transition.position_image_body.assign(h_position_transition.position_image_body.begin(), h_position_transition.position_image_body.end());

    /************************* BRESENHAM KERNEL *************************/
    auto start_bresenham = std::chrono::high_resolution_clock::now();

    int threadsPerBlock = 256;
    int blocksPerGrid = (PARTICLES_OCCUPIED_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_bresenham << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles_data.particles_free_x_max), THRUST_RAW_CAST(d_particles_data.particles_free_y_max), THRUST_RAW_CAST(d_particles_data.particles_free_counter), SEP,
        THRUST_RAW_CAST(d_particles_data.particles_occupied_x), THRUST_RAW_CAST(d_particles_data.particles_occupied_y), THRUST_RAW_CAST(d_position_transition.position_image_body),
        PARTICLES_OCCUPIED_LEN, MAX_DIST_IN_MAP);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, 
        d_particles_data.particles_free_counter.begin(), d_particles_data.particles_free_counter.end(), 
        d_particles_data.particles_free_counter.begin(), 0);
    auto stop_bresenham = std::chrono::high_resolution_clock::now();

    auto start_bresenham_rearrange = std::chrono::high_resolution_clock::now();
    res_particles_data.particles_free_counter.assign(d_particles_data.particles_free_counter.begin(), d_particles_data.particles_free_counter.end());

    PARTICLES_FREE_LEN = res_particles_data.particles_free_counter[PARTICLE_UNIQUE_COUNTER - 1];
    d_particles_data.particles_free_x.resize(PARTICLES_FREE_LEN);
    d_particles_data.particles_free_y.resize(PARTICLES_FREE_LEN);

    kernel_bresenham_rearrange << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles_data.particles_free_x), THRUST_RAW_CAST(d_particles_data.particles_free_y), SEP,
        THRUST_RAW_CAST(d_particles_data.particles_free_x_max), THRUST_RAW_CAST(d_particles_data.particles_free_y_max),
        THRUST_RAW_CAST(d_particles_data.particles_free_counter), MAX_DIST_IN_MAP, PARTICLES_OCCUPIED_LEN);
    cudaDeviceSynchronize();
    auto stop_bresenham_rearrange = std::chrono::high_resolution_clock::now();

    res_particles_data.particles_free_x.resize(PARTICLES_FREE_LEN);
    res_particles_data.particles_free_y.resize(PARTICLES_FREE_LEN);

    res_particles_data.particles_free_x.assign(d_particles_data.particles_free_x.begin(), d_particles_data.particles_free_x.end());
    res_particles_data.particles_free_y.assign(d_particles_data.particles_free_y.begin(), d_particles_data.particles_free_y.end());

    ASSERT_particles_free_index(res_particles_data.particles_free_counter.data(), h_particles_data.particles_free_idx.data(), PARTICLES_OCCUPIED_LEN, false);
    ASSERT_particles_free_new_len(PARTICLES_FREE_LEN, h_particles_data.PARTICLES_FREE_LEN);
    ASSERT_particles_free(res_particles_data.particles_free_x.data(), res_particles_data.particles_free_y.data(),
        h_particles_data.particles_free_x.data(), h_particles_data.particles_free_y.data(), PARTICLES_FREE_LEN, true, true);

    auto duration_bresenham = std::chrono::duration_cast<std::chrono::microseconds>(stop_bresenham - start_bresenham);
    auto duration_bresenham_rearrange = std::chrono::duration_cast<std::chrono::microseconds>(stop_bresenham_rearrange - start_bresenham_rearrange);
    std::cout << "Time taken by function (Bresenham): " << duration_bresenham.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Bresenham Rearrange): " << duration_bresenham_rearrange.count() << " microseconds" << std::endl;
    std::cout << std::endl;
}

void host_update_map(HostMapData& h_map_data, HostMapData& h_map_data_bg,
    HostMapData& h_map_data_post, GeneralInfo& general_info, HostMeasurements& h_measurements,
    HostParticlesData& h_particles_data, HostPositionTransition& h_position_transition) {

    printf("/**************************** UPDATE MAP ****************************/\n");

    host_vector<int> hvec_occupied_map_idx(2, 0);
    host_vector<int> hvec_free_map_idx(2, 0);

    int xmin = h_map_data.xmin;
    int xmax = h_map_data.xmax;
    int ymin = h_map_data.ymin;
    int ymax = h_map_data.ymax;

    int LIDAR_COORDS_LEN = h_measurements.LIDAR_COORDS_LEN;
    int GRID_WIDTH = h_map_data_post.GRID_WIDTH;
    int GRID_HEIGHT = h_map_data_post.GRID_HEIGHT;

    int PARTICLES_OCCUPIED_LEN = h_particles_data.PARTICLES_OCCUPIED_LEN;
    int PARTICLES_FREE_LEN = h_particles_data.PARTICLES_FREE_LEN;
    int PARTICLE_UNIQUE_COUNTER = PARTICLES_OCCUPIED_LEN + 1;

    printf("~~$ GRID_WIDTH = \t%d\n", GRID_WIDTH);
    printf("~~$ GRID_HEIGHT = \t%d\n", GRID_HEIGHT);

    /**************************** MAP VARIABLES *************************/
    DeviceMapData d_map_data;
    d_map_data.grid_map.resize(GRID_WIDTH * GRID_HEIGHT);
    d_map_data.grid_map.assign(h_map_data_bg.grid_map.begin(), h_map_data_bg.grid_map.end());

    DeviceParticlesData d_particles_data;
    d_particles_data.particles_occupied_x.resize(PARTICLES_OCCUPIED_LEN);
    d_particles_data.particles_occupied_y.resize(PARTICLES_OCCUPIED_LEN);
    d_particles_data.particles_free_x.resize(PARTICLES_FREE_LEN);
    d_particles_data.particles_free_y.resize(PARTICLES_FREE_LEN);
    
    d_particles_data.particles_occupied_x.assign(h_particles_data.particles_occupied_x.begin(), h_particles_data.particles_occupied_x.end());
    d_particles_data.particles_occupied_y.assign(h_particles_data.particles_occupied_y.begin(), h_particles_data.particles_occupied_y.end());
    d_particles_data.particles_free_x.assign(h_particles_data.particles_free_x.begin(), h_particles_data.particles_free_x.end());
    d_particles_data.particles_free_y.assign(h_particles_data.particles_free_y.begin(), h_particles_data.particles_free_y.end());

    /************************* LOG-ODDS VARIABLES ***********************/

    HostUniqueManager res_unique_manager;
    res_unique_manager.occupied_unique_counter.resize(1);
    res_unique_manager.free_unique_counter.resize(1);
    res_unique_manager.occupied_unique_counter_col.resize(GRID_WIDTH + 1);
    res_unique_manager.free_unique_counter_col.resize(GRID_WIDTH + 1);

    hvec_occupied_map_idx[1] = PARTICLES_OCCUPIED_LEN;
    hvec_free_map_idx[1] = PARTICLES_FREE_LEN;

    DeviceUniqueManager d_unique_manager;
    d_unique_manager.occupied_map_2d.resize(GRID_WIDTH * GRID_HEIGHT);
    d_unique_manager.free_map_2d.resize(GRID_WIDTH * GRID_HEIGHT);

    d_unique_manager.occupied_unique_counter.resize(1);
    d_unique_manager.occupied_unique_counter_col.resize(GRID_WIDTH + 1);
    d_unique_manager.free_unique_counter.resize(1);
    d_unique_manager.free_unique_counter_col.resize(GRID_WIDTH + 1);

    d_unique_manager.occupied_map_idx.resize(2, 0);
    d_unique_manager.free_map_idx.resize(2, 0);

    d_unique_manager.occupied_map_idx.assign(hvec_occupied_map_idx.begin(), hvec_occupied_map_idx.end());
    d_unique_manager.free_map_idx.assign(hvec_free_map_idx.begin(), hvec_free_map_idx.end());

    thrust::fill(d_unique_manager.occupied_unique_counter_col.begin(), d_unique_manager.occupied_unique_counter_col.end(), 0);
    thrust::fill(d_unique_manager.free_unique_counter_col.begin(), d_unique_manager.free_unique_counter_col.end(), 0);

    /**************************** CREATE MAP ****************************/
    auto start_create_map = std::chrono::high_resolution_clock::now();
    int threadsPerBlock = 256;
    int blocksPerGrid = 1;
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_unique_manager.occupied_map_2d), SEP,
        THRUST_RAW_CAST(d_particles_data.particles_occupied_x), THRUST_RAW_CAST(d_particles_data.particles_occupied_y),
        THRUST_RAW_CAST(d_unique_manager.occupied_map_idx),
        PARTICLES_OCCUPIED_LEN, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_unique_manager.free_map_2d), SEP,
        THRUST_RAW_CAST(d_particles_data.particles_free_x), THRUST_RAW_CAST(d_particles_data.particles_free_y),
        THRUST_RAW_CAST(d_unique_manager.free_map_idx),
        PARTICLES_FREE_LEN, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();
    auto stop_create_map = std::chrono::high_resolution_clock::now();


    auto start_unique_counter = std::chrono::high_resolution_clock::now();
    threadsPerBlock = GRID_WIDTH;
    blocksPerGrid = 1;
    kernel_2d_map_counter << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_unique_manager.occupied_unique_counter), THRUST_RAW_CAST(d_unique_manager.occupied_unique_counter_col), SEP,
        THRUST_RAW_CAST(d_unique_manager.occupied_map_2d), GRID_WIDTH, GRID_HEIGHT);
    kernel_2d_map_counter << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_unique_manager.free_unique_counter), THRUST_RAW_CAST(d_unique_manager.free_unique_counter_col), SEP,
        THRUST_RAW_CAST(d_unique_manager.free_map_2d), GRID_WIDTH, GRID_HEIGHT);
    cudaDeviceSynchronize();
    auto stop_unique_counter = std::chrono::high_resolution_clock::now();

    auto start_unique_sum = std::chrono::high_resolution_clock::now();
    threadsPerBlock = 1;
    blocksPerGrid = 1;
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (THRUST_RAW_CAST(d_unique_manager.occupied_unique_counter_col), GRID_WIDTH);
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (THRUST_RAW_CAST(d_unique_manager.free_unique_counter_col), GRID_WIDTH);
    cudaDeviceSynchronize();
    auto stop_unique_sum = std::chrono::high_resolution_clock::now();

    hvec_occupied_map_idx.assign(d_unique_manager.occupied_map_idx.begin(), d_unique_manager.occupied_map_idx.end());

    res_unique_manager.occupied_unique_counter.assign(d_unique_manager.occupied_unique_counter.begin(), d_unique_manager.occupied_unique_counter.end());
    res_unique_manager.free_unique_counter.assign(d_unique_manager.free_unique_counter.begin(), d_unique_manager.free_unique_counter.end());

    //gpuErrchk(cudaMemcpy(res_unique_occupied_counter_col, d_unique_occupied_counter_col, sz_unique_counter_col, cudaMemcpyDeviceToHost));
    //gpuErrchk(cudaMemcpy(res_unique_free_counter_col, d_unique_free_counter_col, sz_unique_counter_col, cudaMemcpyDeviceToHost));

    /*---------------------------------------------------------------------*/
    /*-------------------- REINITIALIZE MAP VARIABLES ---------------------*/
    int PARTICLES_OCCUPIED_UNIQUE_LEN = res_unique_manager.occupied_unique_counter[0];
    int PARTICLES_FREE_UNIQUE_LEN = res_unique_manager.free_unique_counter[0];

    printf("--> Occupied Unique: %d, %d\n", PARTICLES_OCCUPIED_UNIQUE_LEN, h_particles_data.PARTICLES_OCCUPIED_UNIQUE_LEN);
    assert(PARTICLES_OCCUPIED_UNIQUE_LEN == h_particles_data.PARTICLES_OCCUPIED_UNIQUE_LEN);
    printf("--> Free Unique: %d, %d\n", PARTICLES_FREE_UNIQUE_LEN, h_particles_data.PARTICLES_FREE_UNIQUE_LEN);
    assert(PARTICLES_FREE_UNIQUE_LEN == h_particles_data.PARTICLES_FREE_UNIQUE_LEN);

    HostParticlesData res_particles_data;
    res_particles_data.particles_occupied_x.resize(PARTICLES_OCCUPIED_UNIQUE_LEN);
    res_particles_data.particles_occupied_y.resize(PARTICLES_OCCUPIED_UNIQUE_LEN);
    res_particles_data.particles_free_x.resize(PARTICLES_FREE_UNIQUE_LEN);
    res_particles_data.particles_free_y.resize(PARTICLES_FREE_UNIQUE_LEN);

    d_particles_data.particles_occupied_x.clear();
    d_particles_data.particles_occupied_y.clear();
    d_particles_data.particles_free_x.clear();
    d_particles_data.particles_free_y.clear();

    d_particles_data.particles_occupied_x.resize(PARTICLES_OCCUPIED_UNIQUE_LEN, 0);
    d_particles_data.particles_occupied_y.resize(PARTICLES_OCCUPIED_UNIQUE_LEN, 0);
    d_particles_data.particles_free_x.resize(PARTICLES_FREE_UNIQUE_LEN, 0);
    d_particles_data.particles_free_y.resize(PARTICLES_FREE_UNIQUE_LEN, 0);

    auto start_restructure_map = std::chrono::high_resolution_clock::now();
    threadsPerBlock = GRID_WIDTH;
    blocksPerGrid = 1;
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles_data.particles_occupied_x), THRUST_RAW_CAST(d_particles_data.particles_occupied_y), SEP,
        THRUST_RAW_CAST(d_unique_manager.occupied_map_2d), THRUST_RAW_CAST(d_unique_manager.occupied_unique_counter_col), GRID_WIDTH, GRID_HEIGHT);
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles_data.particles_free_x), THRUST_RAW_CAST(d_particles_data.particles_free_y), SEP,
        THRUST_RAW_CAST(d_unique_manager.free_map_2d), THRUST_RAW_CAST(d_unique_manager.free_unique_counter_col), GRID_WIDTH, GRID_HEIGHT);
    cudaDeviceSynchronize();
    auto stop_restructure_map = std::chrono::high_resolution_clock::now();

    res_particles_data.particles_occupied_x.assign(d_particles_data.particles_occupied_x.begin(), d_particles_data.particles_occupied_x.end());
    res_particles_data.particles_occupied_y.assign(d_particles_data.particles_occupied_y.begin(), d_particles_data.particles_occupied_y.end());
    res_particles_data.particles_free_x.assign(d_particles_data.particles_free_x.begin(), d_particles_data.particles_free_x.end());
    res_particles_data.particles_free_y.assign(d_particles_data.particles_free_y.begin(), d_particles_data.particles_free_y.end());

    ASSERT_particles_occupied(res_particles_data.particles_occupied_x.data(), res_particles_data.particles_occupied_y.data(),
        h_particles_data.particles_occupied_unique_x.data(), h_particles_data.particles_occupied_unique_y.data(),
        "Occupied", PARTICLES_OCCUPIED_UNIQUE_LEN, false);
    ASSERT_particles_occupied(res_particles_data.particles_free_x.data(), res_particles_data.particles_free_y.data(),
        h_particles_data.particles_free_unique_x.data(), h_particles_data.particles_free_unique_y.data(),
        "Free", PARTICLES_FREE_UNIQUE_LEN, false);

    /************************* LOG-ODDS VARIABLES ***********************/
    HostMapData res_map_data;
    res_map_data.grid_map.resize(GRID_WIDTH* GRID_HEIGHT, 0);
    res_map_data.log_odds.resize(GRID_WIDTH* GRID_HEIGHT, 0);

    d_map_data.log_odds.resize(GRID_WIDTH* GRID_HEIGHT);
    d_map_data.log_odds.assign(h_map_data_bg.log_odds.begin(), h_map_data_bg.log_odds.end());


    /************************** LOG-ODDS KERNEL *************************/
    auto start_update_map = std::chrono::high_resolution_clock::now();

    threadsPerBlock = 256;
    blocksPerGrid = (PARTICLES_OCCUPIED_UNIQUE_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_update_log_odds << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_map_data.log_odds), SEP,
        THRUST_RAW_CAST(d_particles_data.particles_occupied_x), THRUST_RAW_CAST(d_particles_data.particles_occupied_y),
        2 * general_info.log_t, GRID_WIDTH, GRID_HEIGHT, PARTICLES_OCCUPIED_UNIQUE_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = 256;
    blocksPerGrid = (PARTICLES_FREE_UNIQUE_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_update_log_odds << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_map_data.log_odds), SEP,
        THRUST_RAW_CAST(d_particles_data.particles_free_x), THRUST_RAW_CAST(d_particles_data.particles_free_y),
        (-1) * general_info.log_t, GRID_WIDTH, GRID_HEIGHT, PARTICLES_FREE_UNIQUE_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = 256;
    blocksPerGrid = ((GRID_WIDTH * GRID_HEIGHT) + threadsPerBlock - 1) / threadsPerBlock;
    kernel_update_map << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_map_data.grid_map), SEP,
        THRUST_RAW_CAST(d_map_data.log_odds), LOG_ODD_PRIOR, WALL, FREE, GRID_WIDTH * GRID_HEIGHT);
    cudaDeviceSynchronize();

    auto stop_update_map = std::chrono::high_resolution_clock::now();

    res_map_data.log_odds.assign(d_map_data.log_odds.begin(), d_map_data.log_odds.end());
    res_map_data.grid_map.assign(d_map_data.grid_map.begin(), d_map_data.grid_map.end());

    ASSERT_log_odds(res_map_data.log_odds.data(), h_map_data.log_odds.data(), h_map_data_post.log_odds.data(),
        (GRID_WIDTH * GRID_HEIGHT), (h_map_data.GRID_WIDTH * h_map_data.GRID_HEIGHT), false);
    ASSERT_log_odds_maps(res_map_data.grid_map.data(), h_map_data.grid_map.data(), h_map_data_post.grid_map.data(),
        (GRID_WIDTH * GRID_HEIGHT), (h_map_data.GRID_WIDTH* h_map_data.GRID_HEIGHT), false);
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

void host_map(HostMapData& h_map_data, HostMapData& h_map_data_bg,
    HostMapData& h_map_data_post, GeneralInfo& general_info, HostMeasurements& h_measurements,
    HostParticlesData& h_particles_data, HostPositionTransition& h_position_transition) {

    printf("/******************************** MAP *******************************/\n");

    host_vector<int> hvec_occupied_map_idx(2, 0);
    host_vector<int> hvec_free_map_idx(2, 0);

    int xmin = h_map_data.xmin;
    int xmax = h_map_data.xmax;
    int ymin = h_map_data.ymin;
    int ymax = h_map_data.ymax;

    int xmin_pre = xmin;
    int ymax_pre = ymax;

    int GRID_WIDTH = h_map_data.GRID_WIDTH;
    int GRID_HEIGHT = h_map_data.GRID_HEIGHT;
    int LIDAR_COORDS_LEN = h_measurements.LIDAR_COORDS_LEN;

    int PARTICLES_OCCUPIED_LEN = h_particles_data.PARTICLES_OCCUPIED_LEN;
    int PARTICLES_FREE_LEN = h_particles_data.PARTICLES_FREE_LEN;

    /********************* IMAGE TRANSFORM VARIABLES ********************/

    HostPositionTransition res_position_transition;
    res_position_transition.transition_world_lidar.resize(9);
    res_position_transition.position_image_body.resize(2);

    HostParticlesData res_particles_data;
    res_particles_data.particles_occupied_x.resize(LIDAR_COORDS_LEN);
    res_particles_data.particles_occupied_y.resize(LIDAR_COORDS_LEN);
    res_particles_data.particles_world_x.resize(LIDAR_COORDS_LEN);
    res_particles_data.particles_world_y.resize(LIDAR_COORDS_LEN);
    
    DeviceMeasurements d_measurements;
    d_measurements.lidar_coords.resize(2 * LIDAR_COORDS_LEN);
    d_measurements.lidar_coords.assign(h_measurements.lidar_coords.begin(), h_measurements.lidar_coords.end());

    DevicePositionTransition d_position_transition;
    d_position_transition.transition_body_lidar.resize(9);
    d_position_transition.transition_single_world_body.resize(9);
    d_position_transition.transition_single_world_lidar.resize(9);
    d_position_transition.transition_body_lidar.assign(hvec_transition_body_lidar.begin(), hvec_transition_body_lidar.end());
    d_position_transition.transition_single_world_body.assign(h_position_transition.transition_world_body.begin(), h_position_transition.transition_world_body.end());
    d_position_transition.position_image_body.resize(2);


    /************************ BRESENHAM VARIABLES ***********************/
    int MAX_DIST_IN_MAP = sqrt(pow(GRID_WIDTH, 2) + pow(GRID_HEIGHT, 2));
    int PARTICLE_UNIQUE_COUNTER = PARTICLES_OCCUPIED_LEN + 1;

    DeviceParticlesData d_particles_data;
    d_particles_data.particles_occupied_x.resize(LIDAR_COORDS_LEN);
    d_particles_data.particles_occupied_y.resize(LIDAR_COORDS_LEN);
    d_particles_data.particles_world_x.resize(LIDAR_COORDS_LEN);
    d_particles_data.particles_world_y.resize(LIDAR_COORDS_LEN);
    d_particles_data.particles_free_x_max.resize(PARTICLES_OCCUPIED_LEN * MAX_DIST_IN_MAP, 0);
    d_particles_data.particles_free_y_max.resize(PARTICLES_OCCUPIED_LEN * MAX_DIST_IN_MAP, 0);
    d_particles_data.particles_free_counter.resize(PARTICLE_UNIQUE_COUNTER, 0);
    d_particles_data.particles_free_idx.resize(PARTICLES_OCCUPIED_LEN);


    /**************************** MAP VARIABLES *************************/
    DeviceMapData d_map_data;
    d_map_data.grid_map.resize(GRID_WIDTH* GRID_HEIGHT, 0);
    d_map_data.grid_map.assign(h_map_data.grid_map.begin(), h_map_data.grid_map.end());
    d_map_data.log_odds.resize(GRID_WIDTH * GRID_HEIGHT);
    d_map_data.should_extend.resize(4, 0);

    /************************* LOG-ODDS VARIABLES ***********************/

    res_particles_data.particles_free_counter.resize(PARTICLE_UNIQUE_COUNTER);
    
    HostUniqueManager res_unique_manager;
    res_unique_manager.free_unique_counter.resize(1);
    res_unique_manager.free_unique_counter_col.resize(GRID_WIDTH + 1);
    res_unique_manager.occupied_unique_counter.resize(1);
    res_unique_manager.occupied_unique_counter_col.resize(GRID_WIDTH + 1);
    res_unique_manager.occupied_map_idx.resize(2);
    res_unique_manager.free_map_idx.resize(2);


    DeviceUniqueManager d_unique_manager;
    d_unique_manager.free_map_2d.resize(GRID_WIDTH* GRID_HEIGHT);
    d_unique_manager.free_unique_counter.resize(1);
    d_unique_manager.free_unique_counter_col.resize(GRID_WIDTH + 1);
    d_unique_manager.occupied_map_2d.resize(GRID_WIDTH* GRID_HEIGHT);
    d_unique_manager.occupied_unique_counter.resize(1);
    d_unique_manager.occupied_unique_counter_col.resize(GRID_WIDTH + 1);
    d_unique_manager.occupied_map_idx.resize(2);
    d_unique_manager.free_map_idx.resize(2);

    HostMapData res_map_data;
    res_map_data.grid_map.resize(GRID_WIDTH* GRID_HEIGHT, 0);
    res_map_data.log_odds.resize(GRID_WIDTH* GRID_HEIGHT, 0);
    res_map_data.should_extend.resize(4, 0);

    hvec_occupied_map_idx[1] = PARTICLES_OCCUPIED_LEN;
    hvec_free_map_idx[1] = 0;

    d_unique_manager.occupied_map_idx.assign(hvec_occupied_map_idx.begin(), hvec_occupied_map_idx.end());
    d_unique_manager.free_map_idx.assign(hvec_free_map_idx.begin(), hvec_free_map_idx.end());
    d_map_data.log_odds.assign(h_map_data.log_odds.begin(), h_map_data.log_odds.end());

    /************************************************************* KERNEL EXECUTION SCOPE *************************************************************/

    /***************** World to IMAGE TRANSFORM KERNEL ******************/
    auto start_world_to_image_transform_1 = std::chrono::high_resolution_clock::now();
    kernel_matrix_mul_3x3 << < 1, 1 >> > (
        THRUST_RAW_CAST(d_position_transition.transition_single_world_body), 
        THRUST_RAW_CAST(d_position_transition.transition_body_lidar), THRUST_RAW_CAST(d_position_transition.transition_single_world_lidar));
    cudaDeviceSynchronize();

    int threadsPerBlock = 1;
    int blocksPerGrid = LIDAR_COORDS_LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles_data.particles_world_x), THRUST_RAW_CAST(d_particles_data.particles_world_y), SEP,
        THRUST_RAW_CAST(d_position_transition.transition_single_world_lidar), THRUST_RAW_CAST(d_measurements.lidar_coords), LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();
    auto stop_world_to_image_transform_1 = std::chrono::high_resolution_clock::now();


    auto start_check_extend = std::chrono::high_resolution_clock::now();
    threadsPerBlock = 256;
    blocksPerGrid = (LIDAR_COORDS_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_check_map_extend_less << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_map_data.should_extend), SEP, THRUST_RAW_CAST(d_particles_data.particles_world_x), xmin, 0, LIDAR_COORDS_LEN);
    kernel_check_map_extend_less << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_map_data.should_extend), SEP, THRUST_RAW_CAST(d_particles_data.particles_world_y), ymin, 1, LIDAR_COORDS_LEN);

    kernel_check_map_extend_greater << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_map_data.should_extend), SEP, THRUST_RAW_CAST(d_particles_data.particles_world_x), xmax, 2, LIDAR_COORDS_LEN);
    kernel_check_map_extend_greater << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_map_data.should_extend), SEP, THRUST_RAW_CAST(d_particles_data.particles_world_y), ymax, 3, LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    res_map_data.should_extend.assign(d_map_data.should_extend.begin(), d_map_data.should_extend.end());

    bool EXTEND = false;
    if (res_map_data.should_extend[0] != 0) {
        EXTEND = true;
        xmin = xmin * 2;
    }
    else if (res_map_data.should_extend[2] != 0) {
        EXTEND = true;
        xmax = xmax * 2;
    }
    else if (res_map_data.should_extend[1] != 0) {
        EXTEND = true;
        ymin = ymin * 2;
    }
    else if (res_map_data.should_extend[3] != 0) {
        EXTEND = true;
        ymax = ymax * 2;
    }
    auto stop_check_extend = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 4; i++)
        std::cout << "Should Extend: " << res_map_data.should_extend[i] << std::endl;

    printf("xmin=%d, xmax=%d, ymin=%d, ymax=%d\n", xmin, xmax, ymin, ymax);
    printf("xmin=%d, xmax=%d, ymin=%d, ymax=%d\n", h_map_data_post.xmin, h_map_data_post.xmax, h_map_data_post.ymin, h_map_data_post.ymax);
    assert(EXTEND == h_map_data.b_should_extend);

    if (EXTEND == true) {

        auto start_extend = std::chrono::high_resolution_clock::now();

        d_measurements.coord.resize(2);
        HostMeasurements res_measurements;
        res_measurements.coord.resize(2);

        kernel_position_to_image << <1, 1 >> > (THRUST_RAW_CAST(d_measurements.coord), SEP, 
            xmin_pre, ymax_pre, general_info.res, xmin, ymax);
        cudaDeviceSynchronize();

        res_measurements.coord.assign(d_measurements.coord.begin(), d_measurements.coord.end());
        
        device_vector<int> dvec_clone_grid_map(GRID_WIDTH * GRID_HEIGHT);
        dvec_clone_grid_map.assign(d_map_data.grid_map.begin(), d_map_data.grid_map.end());

        device_vector<float> dvec_clone_log_odds(GRID_WIDTH * GRID_HEIGHT);
        dvec_clone_log_odds.assign(d_map_data.log_odds.begin(), d_map_data.log_odds.end());

        const int PRE_GRID_WIDTH = GRID_WIDTH;
        const int PRE_GRID_HEIGHT = GRID_HEIGHT;
        GRID_WIDTH = ceil((ymax - ymin) / general_info.res + 1);
        GRID_HEIGHT = ceil((xmax - xmin) / general_info.res + 1);
        printf("GRID_WIDTH=%d, AF_GRID_WIDTH=%d, GRID_HEIGHT=%d, AF_GRID_HEIGHT=%d\n", 
            GRID_WIDTH, GRID_HEIGHT, h_map_data_post.GRID_WIDTH, h_map_data_post.GRID_HEIGHT);
        assert(GRID_WIDTH == h_map_data_post.GRID_WIDTH);
        assert(GRID_HEIGHT == h_map_data_post.GRID_HEIGHT);

        const int PRE_GRID_SIZE = PRE_GRID_WIDTH * PRE_GRID_HEIGHT;
        const int NEW_GRID_SIZE = GRID_WIDTH * GRID_HEIGHT;

        d_map_data.grid_map.clear();
        d_map_data.log_odds.clear();
        d_map_data.grid_map.resize(GRID_WIDTH * GRID_HEIGHT, 0);
        d_map_data.log_odds.resize(GRID_WIDTH * GRID_HEIGHT, LOG_ODD_PRIOR);

        res_map_data.grid_map.clear();
        res_map_data.log_odds.clear();
        res_map_data.grid_map.resize(GRID_WIDTH * GRID_HEIGHT, 0);
        res_map_data.log_odds.resize(GRID_WIDTH * GRID_HEIGHT, 0);

        threadsPerBlock = 256;
        blocksPerGrid = (PRE_GRID_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        kernel_2d_copy_with_offset << <blocksPerGrid, threadsPerBlock >> > (
            THRUST_RAW_CAST(d_map_data.grid_map), THRUST_RAW_CAST(d_map_data.log_odds), SEP,
            THRUST_RAW_CAST(dvec_clone_grid_map), THRUST_RAW_CAST(dvec_clone_log_odds), 
            res_measurements.coord[0], res_measurements.coord[1],
            PRE_GRID_HEIGHT, GRID_HEIGHT, PRE_GRID_SIZE);
        cudaDeviceSynchronize();

        d_unique_manager.free_unique_counter_col.clear();
        d_unique_manager.free_unique_counter_col.resize(GRID_WIDTH + 1);
        res_unique_manager.free_unique_counter_col.resize(GRID_WIDTH + 1);

        d_unique_manager.occupied_unique_counter_col.clear();
        d_unique_manager.occupied_unique_counter_col.resize(GRID_WIDTH + 1);
        res_unique_manager.occupied_unique_counter_col.resize(GRID_WIDTH + 1);

        d_unique_manager.free_map_2d.clear();
        d_unique_manager.occupied_map_2d.clear();
        d_unique_manager.free_map_2d.resize(GRID_WIDTH* GRID_HEIGHT);
        d_unique_manager.occupied_map_2d.resize(GRID_WIDTH* GRID_HEIGHT);

        auto stop_extend = std::chrono::high_resolution_clock::now();
        auto duration_extend = std::chrono::duration_cast<std::chrono::microseconds>(stop_extend - start_extend);

        std::cout << std::endl;
        std::cout << "Time taken by function (Extend): " << duration_extend.count() << " microseconds" << std::endl;

        res_map_data.grid_map.assign(d_map_data.grid_map.begin(), d_map_data.grid_map.end());
        res_map_data.log_odds.assign(d_map_data.log_odds.begin(), d_map_data.log_odds.end());

        int error_map = 0;
        int error_log = 0;
        for (int i = 0; i < (GRID_WIDTH * GRID_HEIGHT); i++) {
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

    auto start_world_to_image_transform_2 = std::chrono::high_resolution_clock::now();
    threadsPerBlock = 1;
    blocksPerGrid = LIDAR_COORDS_LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles_data.particles_occupied_x), THRUST_RAW_CAST(d_particles_data.particles_occupied_y), SEP,
        THRUST_RAW_CAST(d_position_transition.transition_single_world_lidar), THRUST_RAW_CAST(d_measurements.lidar_coords),
        general_info.res, xmin, ymax, LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    kernel_position_to_image << < 1, 1 >> > (
        THRUST_RAW_CAST(d_position_transition.position_image_body), SEP, THRUST_RAW_CAST(d_position_transition.transition_single_world_lidar), general_info.res, xmin, ymax);
    cudaDeviceSynchronize();
    auto stop_world_to_image_transform_2 = std::chrono::high_resolution_clock::now();

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
        h_particles_data.particles_world_x.data(), h_particles_data.particles_world_y.data(), LIDAR_COORDS_LEN, false);
    ASSERT_processed_measurements(res_particles_data.particles_occupied_x.data(), res_particles_data.particles_occupied_y.data(),
        h_particles_data.particles_occupied_x.data(), h_particles_data.particles_occupied_y.data(), LIDAR_COORDS_LEN);
    ASSERT_position_image_body(res_position_transition.position_image_body.data(), h_position_transition.position_image_body.data(), true, true);

    /************************* BRESENHAM KERNEL *************************/
    auto start_bresenham = std::chrono::high_resolution_clock::now();

    threadsPerBlock = 256;
    blocksPerGrid = (PARTICLES_OCCUPIED_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_bresenham << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles_data.particles_free_x_max), THRUST_RAW_CAST(d_particles_data.particles_free_y_max), THRUST_RAW_CAST(d_particles_data.particles_free_counter), SEP,
        THRUST_RAW_CAST(d_particles_data.particles_occupied_x), THRUST_RAW_CAST(d_particles_data.particles_occupied_y), THRUST_RAW_CAST(d_position_transition.position_image_body),
        PARTICLES_OCCUPIED_LEN, MAX_DIST_IN_MAP);
    cudaDeviceSynchronize();
    thrust::exclusive_scan(thrust::device, d_particles_data.particles_free_counter.begin(), d_particles_data.particles_free_counter.end(), d_particles_data.particles_free_counter.begin(), 0); // in-place scan
    auto stop_bresenham = std::chrono::high_resolution_clock::now();

    auto start_bresenham_rearrange = std::chrono::high_resolution_clock::now();

    res_particles_data.particles_free_counter.assign(d_particles_data.particles_free_counter.begin(), d_particles_data.particles_free_counter.end());
    d_particles_data.particles_free_idx.assign(d_particles_data.particles_free_counter.begin(), d_particles_data.particles_free_counter.end());

    PARTICLES_FREE_LEN = res_particles_data.particles_free_counter[PARTICLE_UNIQUE_COUNTER - 1];
    printf("^^^ PARTICLES_FREE_LEN = %d\n", PARTICLES_FREE_LEN);

    d_particles_data.particles_free_x.resize(PARTICLES_FREE_LEN);
    d_particles_data.particles_free_y.resize(PARTICLES_FREE_LEN);


    kernel_bresenham_rearrange << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles_data.particles_free_x), THRUST_RAW_CAST(d_particles_data.particles_free_y), SEP,
        THRUST_RAW_CAST(d_particles_data.particles_free_x_max), THRUST_RAW_CAST(d_particles_data.particles_free_y_max),
        THRUST_RAW_CAST(d_particles_data.particles_free_counter), MAX_DIST_IN_MAP, PARTICLES_OCCUPIED_LEN);
    cudaDeviceSynchronize();
    auto stop_bresenham_rearrange = std::chrono::high_resolution_clock::now();

    res_particles_data.particles_free_x.resize(PARTICLES_FREE_LEN);
    res_particles_data.particles_free_y.resize(PARTICLES_FREE_LEN);

    res_particles_data.particles_free_x.assign(d_particles_data.particles_free_x.begin(), d_particles_data.particles_free_x.end());
    res_particles_data.particles_free_y.assign(d_particles_data.particles_free_y.begin(), d_particles_data.particles_free_y.end());

    printf("~~$ PARTICLES_FREE_LEN = %d\n", PARTICLES_FREE_LEN);

    ASSERT_particles_free_index(res_particles_data.particles_free_counter.data(), h_particles_data.particles_free_idx.data(), PARTICLES_OCCUPIED_LEN, false);
    ASSERT_particles_free_new_len(PARTICLES_FREE_LEN, h_particles_data.PARTICLES_FREE_LEN);
    ASSERT_particles_free(res_particles_data.particles_free_x.data(), res_particles_data.particles_free_y.data(),
        h_particles_data.particles_free_x.data(), h_particles_data.particles_free_y.data(), PARTICLES_FREE_LEN);

    /*---------------------------------------------------------------------*/
    /*-------------------- REINITIALIZE MAP VARIABLES ---------------------*/
    res_unique_manager.free_map_idx[1] = PARTICLES_FREE_LEN;
    d_unique_manager.free_map_idx.assign(res_unique_manager.free_map_idx.begin(), res_unique_manager.free_map_idx.end());

    /************************** CREATE 2D MAP ***************************/
    auto start_create_map = std::chrono::high_resolution_clock::now();
    threadsPerBlock = 256;
    blocksPerGrid = 1;
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_unique_manager.occupied_map_2d), SEP,
        THRUST_RAW_CAST(d_particles_data.particles_occupied_x), THRUST_RAW_CAST(d_particles_data.particles_occupied_y), 
        THRUST_RAW_CAST(d_unique_manager.occupied_map_idx), PARTICLES_OCCUPIED_LEN, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_unique_manager.free_map_2d), SEP,
        THRUST_RAW_CAST(d_particles_data.particles_free_x), THRUST_RAW_CAST(d_particles_data.particles_free_y), 
        THRUST_RAW_CAST(d_unique_manager.free_map_idx), PARTICLES_FREE_LEN, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();
    auto stop_create_map = std::chrono::high_resolution_clock::now();

    auto start_unique_counter = std::chrono::high_resolution_clock::now();
    threadsPerBlock = GRID_WIDTH;
    blocksPerGrid = 1;
    kernel_2d_map_counter << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_unique_manager.occupied_unique_counter), THRUST_RAW_CAST(d_unique_manager.occupied_unique_counter_col), SEP,
        THRUST_RAW_CAST(d_unique_manager.occupied_map_2d), GRID_WIDTH, GRID_HEIGHT);
    kernel_2d_map_counter << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_unique_manager.free_unique_counter), THRUST_RAW_CAST(d_unique_manager.free_unique_counter_col), SEP,
        THRUST_RAW_CAST(d_unique_manager.free_map_2d), GRID_WIDTH, GRID_HEIGHT);
    cudaDeviceSynchronize();
    auto stop_unique_counter = std::chrono::high_resolution_clock::now();

    auto start_unique_sum = std::chrono::high_resolution_clock::now();
    threadsPerBlock = 1;
    blocksPerGrid = 1;
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (THRUST_RAW_CAST(d_unique_manager.occupied_unique_counter_col), GRID_WIDTH);
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (THRUST_RAW_CAST(d_unique_manager.free_unique_counter_col), GRID_WIDTH);
    cudaDeviceSynchronize();
    auto stop_unique_sum = std::chrono::high_resolution_clock::now();

    res_unique_manager.occupied_unique_counter.assign(d_unique_manager.occupied_unique_counter.begin(), d_unique_manager.occupied_unique_counter.end());
    res_unique_manager.free_unique_counter.assign(d_unique_manager.free_unique_counter.begin(), d_unique_manager.free_unique_counter.end());

    /*---------------------------------------------------------------------*/
    /*-------------------- REINITIALIZE MAP VARIABLES ---------------------*/
    int PARTICLES_OCCUPIED_UNIQUE_LEN = res_unique_manager.occupied_unique_counter[0];
    int PARTICLES_FREE_UNIQUE_LEN = res_unique_manager.free_unique_counter[0];

    printf("\n--> Occupied Unique: %d, %d\n", PARTICLES_OCCUPIED_UNIQUE_LEN, h_particles_data.PARTICLES_OCCUPIED_UNIQUE_LEN);
    assert(PARTICLES_OCCUPIED_UNIQUE_LEN == h_particles_data.PARTICLES_OCCUPIED_UNIQUE_LEN);
    printf("\n--> Free Unique: %d, %d\n", PARTICLES_FREE_UNIQUE_LEN, h_particles_data.PARTICLES_FREE_UNIQUE_LEN);
    assert(PARTICLES_FREE_UNIQUE_LEN == h_particles_data.PARTICLES_FREE_UNIQUE_LEN);

    res_particles_data.particles_occupied_x.resize(PARTICLES_OCCUPIED_UNIQUE_LEN, 0);
    res_particles_data.particles_occupied_y.resize(PARTICLES_OCCUPIED_UNIQUE_LEN, 0);
    res_particles_data.particles_free_x.resize(PARTICLES_FREE_UNIQUE_LEN, 0);
    res_particles_data.particles_free_y.resize(PARTICLES_FREE_UNIQUE_LEN, 0);
    
    d_particles_data.particles_occupied_x.resize(PARTICLES_OCCUPIED_UNIQUE_LEN);
    d_particles_data.particles_occupied_y.resize(PARTICLES_OCCUPIED_UNIQUE_LEN);
    d_particles_data.particles_free_x.resize(PARTICLES_FREE_UNIQUE_LEN);
    d_particles_data.particles_free_y.resize(PARTICLES_FREE_UNIQUE_LEN);

    auto start_restructure_map = std::chrono::high_resolution_clock::now();
    threadsPerBlock = GRID_WIDTH;
    blocksPerGrid = 1;
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles_data.particles_occupied_x), THRUST_RAW_CAST(d_particles_data.particles_occupied_y), SEP,
        THRUST_RAW_CAST(d_unique_manager.occupied_map_2d), THRUST_RAW_CAST(d_unique_manager.occupied_unique_counter_col), GRID_WIDTH, GRID_HEIGHT);
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles_data.particles_free_x), THRUST_RAW_CAST(d_particles_data.particles_free_y), SEP,
        THRUST_RAW_CAST(d_unique_manager.free_map_2d), THRUST_RAW_CAST(d_unique_manager.free_unique_counter_col), GRID_WIDTH, GRID_HEIGHT);
    cudaDeviceSynchronize();
    auto stop_restructure_map = std::chrono::high_resolution_clock::now();

    res_particles_data.particles_occupied_x.assign(d_particles_data.particles_occupied_x.begin(), d_particles_data.particles_occupied_x.end());
    res_particles_data.particles_occupied_y.assign(d_particles_data.particles_occupied_y.begin(), d_particles_data.particles_occupied_y.end());
    res_particles_data.particles_free_x.assign(d_particles_data.particles_free_x.begin(), d_particles_data.particles_free_x.end());
    res_particles_data.particles_free_y.assign(d_particles_data.particles_free_y.begin(), d_particles_data.particles_free_y.end());

    ASSERT_particles_occupied(res_particles_data.particles_occupied_x.data(), res_particles_data.particles_occupied_y.data(),
        h_particles_data.particles_occupied_unique_x.data(), h_particles_data.particles_occupied_unique_y.data(),
        "Occupied", PARTICLES_OCCUPIED_UNIQUE_LEN, false);
    ASSERT_particles_occupied(res_particles_data.particles_free_x.data(), res_particles_data.particles_free_y.data(),
        h_particles_data.particles_free_unique_x.data(), h_particles_data.particles_free_unique_y.data(),
        "Free", PARTICLES_FREE_UNIQUE_LEN, false);

    /************************** LOG-ODDS KERNEL *************************/
    auto start_update_map = std::chrono::high_resolution_clock::now();

    threadsPerBlock = 256;
    blocksPerGrid = (PARTICLES_OCCUPIED_UNIQUE_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_update_log_odds << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_map_data.log_odds), SEP, 
        THRUST_RAW_CAST(d_particles_data.particles_occupied_x), THRUST_RAW_CAST(d_particles_data.particles_occupied_y), 2 * general_info.log_t, GRID_WIDTH, GRID_HEIGHT, PARTICLES_OCCUPIED_UNIQUE_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = 256;
    blocksPerGrid = (PARTICLES_FREE_UNIQUE_LEN + threadsPerBlock - 1) / threadsPerBlock;
    kernel_update_log_odds << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_map_data.log_odds), SEP, 
        THRUST_RAW_CAST(d_particles_data.particles_free_x), THRUST_RAW_CAST(d_particles_data.particles_free_y), (-1) * general_info.log_t, GRID_WIDTH, GRID_HEIGHT, PARTICLES_FREE_UNIQUE_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = 256;
    blocksPerGrid = ((GRID_WIDTH * GRID_HEIGHT) + threadsPerBlock - 1) / threadsPerBlock;
    kernel_update_map << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_map_data.grid_map), SEP, THRUST_RAW_CAST(d_map_data.log_odds), LOG_ODD_PRIOR, WALL, FREE, GRID_WIDTH * GRID_HEIGHT);
    cudaDeviceSynchronize();
    auto stop_update_map = std::chrono::high_resolution_clock::now();

    thrust::fill(res_map_data.log_odds.begin(), res_map_data.log_odds.end(), 0);
    res_map_data.log_odds.assign(d_map_data.log_odds.begin(), d_map_data.log_odds.end());
    res_map_data.grid_map.assign(d_map_data.grid_map.begin(), d_map_data.grid_map.end());

    ASSERT_log_odds(res_map_data.log_odds.data(), h_map_data.log_odds.data(), h_map_data_post.log_odds.data(),
        (GRID_WIDTH * GRID_HEIGHT), (h_map_data.GRID_WIDTH* h_map_data.GRID_HEIGHT), false);
    ASSERT_log_odds_maps(res_map_data.grid_map.data(), h_map_data.grid_map.data(), h_map_data_post.grid_map.data(),
        (GRID_WIDTH * GRID_HEIGHT), (h_map_data.GRID_WIDTH* h_map_data.GRID_HEIGHT), false);
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


void test_map_func(HostMapData& h_map_data, HostMapData& h_map_data_bg,
    HostMapData& h_map_data_post, GeneralInfo& general_info, HostMeasurements& h_measurements,
    HostParticlesData& h_particles_data, HostPositionTransition& h_position_transition) {

    host_vector<int> hvec_occupied_map_idx(2, 0);
    host_vector<int> hvec_free_map_idx(2, 0);

    printf("/******************************** Class Base MAP *******************************/\n");

    int LIDAR_COORDS_LEN = h_measurements.LIDAR_COORDS_LEN;

    HostPositionTransition res_position_transition;
    DevicePositionTransition d_position_transition;
    alloc_init_transition_vars(d_position_transition, res_position_transition, h_position_transition);


    DeviceMeasurements d_measurements;
    HostMeasurements res_measurements;
    alloc_init_measurement_vars(d_measurements, res_measurements, h_measurements);


    int MAX_DIST_IN_MAP = sqrt(pow(h_map_data.GRID_WIDTH, 2) + pow(h_map_data.GRID_HEIGHT, 2));

    HostParticlesData res_particles_data;
    DeviceParticlesData d_particles_data;
    alloc_init_particles_vars(d_particles_data, res_particles_data, h_measurements, h_particles_data, MAX_DIST_IN_MAP);

    DeviceMapData d_map_data;
    HostMapData res_map_data;
    alloc_init_map_vars(d_map_data, res_map_data, h_map_data);

    hvec_occupied_map_idx[1] = res_particles_data.PARTICLES_OCCUPIED_LEN;
    hvec_free_map_idx[1] = 0;

    HostUniqueManager res_unique_manager;
    DeviceUniqueManager d_unique_manager;
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



#endif
