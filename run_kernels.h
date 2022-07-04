#ifndef _RUN_KERNELS_H_
#define _RUN_KERNELS_H_

#include "headers.h"
#include "host_utils.h"
#include "kernels.cuh"
#include "kernels_robot.cuh"
#include "kernels_map.cuh"
#include "kernels_utils.cuh"
#include "gl_draw_utils.h"
#include "device_init_common.h"
#include "device_init_robot.h"
#include "device_init_map.h"
#include "device_exec_robot.h"
#include "device_exec_map.h"
#include "device_assert_robot.h"
#include "device_assert_map.h"
#include "device_set_reset_map.h"
#include "device_set_reset_robot.h"

#define ST_nv   0.5
#define ST_nw   0.5
#define GET_EXTRA_TRANSITION_WORLD_BODY

const int UNIQUE_COUNTER_LEN = NUM_PARTICLES + 1;

int threadsPerBlock = 1;
int blocksPerGrid = 1;

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

Window mainWindow;
vector<Shader> shader_list;
Camera camera;

vector<Mesh*> freeList;
vector<Mesh*> wallList;


///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

//void test_setup(int& xmin, int& xmax, int& ymin, int& ymax, float& res, float& log_t,
//    int& LIDAR_COORDS_LEN, int& MEASURE_LEN, int& PARTICLES_ITEMS_LEN,
//    int& PARTICLES_OCCUPIED_LEN, const int FREE_LEN, int& PARTICLE_UNIQUE_COUNTER, int& MAX_DIST_IN_MAP,
//    int& GRID_WIDTH, int& GRID_HEIGHT,
//    const int NEW_GRID_WIDTH, const int NEW_GRID_HEIGHT, const int NEW_LIDAR_COORDS_LEN,
//    const int extra_xmin, const int extra_xmax, const int extra_ymin, const int extra_ymax,
//    const float extra_res, const float extra_log_t, const int EXTRA_PARTICLES_ITEMS_LEN) {
//
//    const int data_len = 6;
//    int data[data_len] = { 1, 0, 2, 2, 1, 3 };
//    int* d_data = NULL;
//    gpuErrchk(cudaMalloc((void**)&d_data, data_len * sizeof(int)));
//    gpuErrchk(cudaMemcpy(d_data, data, data_len * sizeof(int), cudaMemcpyHostToDevice));
//
//    thrust::exclusive_scan(thrust::host, data, data + 6, data, 0);
//    thrust::exclusive_scan(thrust::device, d_data, d_data + 6, d_data, 0);
//
//    GRID_WIDTH = NEW_GRID_WIDTH;
//    GRID_HEIGHT = NEW_GRID_HEIGHT;
//    LIDAR_COORDS_LEN = NEW_LIDAR_COORDS_LEN;
//
//    PARTICLES_ITEMS_LEN = EXTRA_PARTICLES_ITEMS_LEN;
//    MEASURE_LEN = NUM_PARTICLES * LIDAR_COORDS_LEN;
//
//    PARTICLES_OCCUPIED_LEN = NEW_LIDAR_COORDS_LEN;
//    MAX_DIST_IN_MAP = sqrt(pow(GRID_WIDTH, 2) + pow(GRID_HEIGHT, 2));
//    PARTICLE_UNIQUE_COUNTER = PARTICLES_OCCUPIED_LEN + 1;
//
//    xmin = extra_xmin;
//    xmax = extra_xmax;;
//    ymin = extra_ymin;
//    ymax = extra_ymax;
//
//    res = extra_res;
//    log_t = extra_log_t;
//
//    h_occupied_map_idx[1] = PARTICLES_OCCUPIED_LEN;
//    h_free_map_idx[1] = FREE_LEN;
//}
//
//void test_allocation_initialization(float* h_states_x, float* h_states_y, float* h_states_theta,
//    float* h_rnds_encoder_counts, float* h_rnds_yaws, float* h_particles_weight,
//    float* h_lidar_coords, int* h_grid_map, float* h_log_odds,
//    int* h_particles_x, int* h_particles_y, int* h_particles_idx,
//    float* h_rnds, float* h_transition_single_world_body,
//    const int LIDAR_COORDS_LEN, const int MEASURE_LEN,
//    const int PARTICLES_ITEMS_LEN, const int PARTICLES_OCCUPIED_LEN, const int PARTICLE_UNIQUE_COUNTER,
//    const int MAX_DIST_IN_MAP, const int GRID_WIDTH, const int GRID_HEIGHT) {
//
//    alloc_init_state_vars(h_states_x, h_states_y, h_states_theta);
//    alloc_init_movement_vars(h_rnds_encoder_counts, h_rnds_yaws, true);
//
//    alloc_init_lidar_coords_var(h_lidar_coords, LIDAR_COORDS_LEN);
//    alloc_init_grid_map(h_grid_map, GRID_WIDTH, GRID_HEIGHT);
//    alloc_init_particles_vars(h_particles_x, h_particles_y, h_particles_idx,
//        h_particles_weight, PARTICLES_ITEMS_LEN);
//    alloc_extended_idx(PARTICLES_ITEMS_LEN);
//    alloc_states_copy_vars();
//    alloc_correlation_vars();
//    alloc_init_transition_vars(h_transition_body_lidar);
//    alloc_init_processed_measurement_vars(LIDAR_COORDS_LEN);
//    alloc_map_2d_var(GRID_WIDTH, GRID_HEIGHT);
//    alloc_map_2d_unique_counter_vars(UNIQUE_COUNTER_LEN, GRID_WIDTH);
//    alloc_correlation_weights_vars();
//    alloc_resampling_vars(h_rnds, true);
//
//    alloc_init_transition_vars(h_transition_body_lidar, h_transition_single_world_body);
//    alloc_init_lidar_coords_var(h_lidar_coords, LIDAR_COORDS_LEN);
//    alloc_particles_world_vars(LIDAR_COORDS_LEN);
//    alloc_particles_free_vars(PARTICLES_OCCUPIED_LEN, PARTICLE_UNIQUE_COUNTER, MAX_DIST_IN_MAP);
//    alloc_particles_occupied_vars(LIDAR_COORDS_LEN);
//    alloc_bresenham_vars();
//    alloc_init_map_vars(h_grid_map, GRID_WIDTH, GRID_HEIGHT);
//    alloc_log_odds_vars(GRID_WIDTH, GRID_HEIGHT);
//    alloc_init_log_odds_free_vars(true, GRID_WIDTH, GRID_HEIGHT);
//    alloc_init_log_odds_occupied_vars(true, GRID_WIDTH, GRID_HEIGHT);
//    init_log_odds_vars(h_log_odds, PARTICLES_OCCUPIED_LEN);
//}
//
//void test_robot_move(float* extra_states_x, float* extra_states_y, float* extra_states_theta,
//                    bool check_result, float encoder_counts, float yaw, float dt) {
//
//    exec_robot_move(encoder_counts, yaw, dt, ST_nv, ST_nw);
//}
//
//void test_robot(float* extra_new_weights, float* vec_robot_transition_world_body,
//    float* vec_robot_state, float* vec_particles_weight_post,
//    bool check_result, const int xmin, const int ymax, const float res, const float log_t,
//    const int LIDAR_COORDS_LEN, const int MEASURE_LEN,
//    int& PARTICLES_ITEMS_LEN, int& C_PARTICLES_ITEMS_LEN,
//    const int GRID_WIDTH, const int GRID_HEIGHT) {
//
//    res_particles_x = (int*)malloc(sz_particles_pos);
//    res_particles_y = (int*)malloc(sz_particles_pos);
//
//    gpuErrchk(cudaMemcpy(res_particles_x, d_particles_x, sz_particles_pos, cudaMemcpyDeviceToHost));
//    gpuErrchk(cudaMemcpy(res_particles_y, d_particles_y, sz_particles_pos, cudaMemcpyDeviceToHost));
//
//    exec_calc_transition();
//    exec_process_measurements(res, xmin, ymax, LIDAR_COORDS_LEN);
//    exec_create_2d_map(PARTICLES_ITEMS_LEN, GRID_WIDTH, GRID_HEIGHT);
//    exec_update_map(MEASURE_LEN, GRID_WIDTH, GRID_HEIGHT);
//    exec_particle_unique_cum_sum(PARTICLES_ITEMS_LEN, C_PARTICLES_ITEMS_LEN, GRID_WIDTH);
//    reinit_map_vars(PARTICLES_ITEMS_LEN);
//    exec_map_restructure(GRID_WIDTH, GRID_HEIGHT);
//    exec_index_expansion(PARTICLES_ITEMS_LEN);
//    exec_correlation(PARTICLES_ITEMS_LEN, GRID_WIDTH, GRID_HEIGHT);
//    exec_update_weights();
//    exec_resampling();
//    reinit_particles_vars(PARTICLES_ITEMS_LEN);
//    exec_rearrangement(PARTICLES_ITEMS_LEN, C_PARTICLES_ITEMS_LEN, GRID_WIDTH, GRID_HEIGHT);
//    exec_update_states();
//}
//
//void test_map(int* extra_grid_map, float* extra_log_odds,
//    int* vec_grid_map, float* vec_log_odds,
//    bool check_result, int& xmin, int& xmax, int& ymin, int& ymax, const float res, const float log_t,
//    const int PARTICLES_OCCUPIED_LEN, int& OCCUPIED_UNIQUE_LEN,
//    int& FREE_LEN, int& FREE_UNIQUE_LEN,
//    const int PARTICLE_UNIQUE_COUNTER, const int MAX_DIST_IN_MAP,
//    const int LIDAR_COORDS_LEN, int& GRID_WIDTH, int& GRID_HEIGHT) {
//
//    exec_world_to_image_transform_step_1(xmin, ymax, res, LIDAR_COORDS_LEN);
//    exec_map_extend(xmin, xmax, ymin, ymax, res, LIDAR_COORDS_LEN, GRID_WIDTH, GRID_HEIGHT);
//    exec_world_to_image_transform_step_2(xmin, ymax, res, LIDAR_COORDS_LEN);
//    exec_bresenham(PARTICLES_OCCUPIED_LEN, FREE_LEN, PARTICLE_UNIQUE_COUNTER, MAX_DIST_IN_MAP);
//    reinit_map_idx_vars(PARTICLES_OCCUPIED_LEN, FREE_LEN);
//
//    exec_create_map(PARTICLES_OCCUPIED_LEN, FREE_LEN, GRID_WIDTH, GRID_HEIGHT);
//    reinit_map_vars(OCCUPIED_UNIQUE_LEN, FREE_UNIQUE_LEN, GRID_WIDTH, GRID_HEIGHT);
//
//    exec_log_odds(log_t, OCCUPIED_UNIQUE_LEN, FREE_UNIQUE_LEN, GRID_WIDTH, GRID_HEIGHT);
//}
//
//void resetMiddleVariables(int LIDAR_COORDS_LEN, int GRID_WIDTH, int GRID_HEIGHT) {
//
//    int num_items = 25 * NUM_PARTICLES;
//    threadsPerBlock = 256;
//    blocksPerGrid = (num_items + threadsPerBlock - 1) / threadsPerBlock;
//    kernel_index_arr_const << <blocksPerGrid, threadsPerBlock >> > (d_correlation_weights_raw, 0, num_items);
//
//    num_items = NUM_PARTICLES * LIDAR_COORDS_LEN;
//    threadsPerBlock = 256;
//    blocksPerGrid = (num_items + threadsPerBlock - 1) / threadsPerBlock;
//    kernel_index_arr_const << <blocksPerGrid, threadsPerBlock >> > (d_processed_measure_x, 0, num_items);
//    kernel_index_arr_const << <blocksPerGrid, threadsPerBlock >> > (d_processed_measure_y, 0, num_items);
//    kernel_index_arr_const << <blocksPerGrid, threadsPerBlock >> > (d_processed_measure_idx, 0, num_items);
//
//    num_items = GRID_WIDTH * GRID_HEIGHT * NUM_PARTICLES;
//    threadsPerBlock = 256;
//    blocksPerGrid = (num_items + threadsPerBlock - 1) / threadsPerBlock;
//    kernel_index_arr_const << <blocksPerGrid, threadsPerBlock >> > (d_map_2d, 0, num_items);
//
//    num_items = UNIQUE_COUNTER_LEN;
//    threadsPerBlock = 256;
//    blocksPerGrid = (num_items + threadsPerBlock - 1) / threadsPerBlock;
//    kernel_index_arr_const << <blocksPerGrid, threadsPerBlock >> > (d_unique_in_particle, 0, num_items);
//
//    num_items = UNIQUE_COUNTER_LEN * GRID_WIDTH;
//    threadsPerBlock = 256;
//    blocksPerGrid = (num_items + threadsPerBlock - 1) / threadsPerBlock;
//    kernel_index_arr_const << <blocksPerGrid, threadsPerBlock >> > (d_unique_in_particle_col, 0, num_items);
//
//    num_items = 1;
//    threadsPerBlock = 1;
//    blocksPerGrid = 1;
//    kernel_index_arr_const << <blocksPerGrid, threadsPerBlock >> > (d_correlation_sum_exp, 0, num_items);
//
//    num_items = 1;
//    threadsPerBlock = 1;
//    blocksPerGrid = 1;
//    kernel_index_arr_const << <blocksPerGrid, threadsPerBlock >> > (d_correlation_weights_max, 0, num_items);
//
//    num_items = NUM_PARTICLES;
//    threadsPerBlock = 256;
//    blocksPerGrid = (num_items + threadsPerBlock - 1) / threadsPerBlock;
//    kernel_index_arr_const << <blocksPerGrid, threadsPerBlock >> > (d_resampling_js, 0, num_items);
//    cudaDeviceSynchronize();
//
//
//    num_items = NUM_PARTICLES;
//    threadsPerBlock = 256;
//    blocksPerGrid = (num_items + threadsPerBlock - 1) / threadsPerBlock;
//    kernel_index_arr_const << <blocksPerGrid, threadsPerBlock >> > (d_particles_weight, 1, num_items);
//    cudaDeviceSynchronize();
//}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

void test_alloc_init_robot(DeviceState& d_state, DeviceState& d_clone_state, DeviceMeasurements& d_measurements,
    DeviceMap& d_map, DeviceRobotParticles& d_robot_particles, DeviceRobotParticles& d_clone_robot_particles, DeviceCorrelation& d_correlation, 
    DeviceParticlesTransition& d_particles_transition,
    DeviceParticlesPosition& d_particles_position, DeviceParticlesRotation& d_particles_rotation, DeviceTransition& d_transition,
    DeviceProcessedMeasure& d_processed_measure, Device2DUniqueFinder& d_2d_unique, DeviceResampling& d_resampling,
    HostState& h_state, HostRobotState& h_robot_state, HostMeasurements& h_measurements, HostMap& h_map, HostRobotParticles& h_robot_particles,
    HostCorrelation& h_correlation, HostParticlesTransition& h_particles_transition, HostParticlesPosition& h_particles_position,
    HostParticlesRotation& h_particles_rotation, HostProcessedMeasure& h_processed_measure, Host2DUniqueFinder& h_2d_unique, HostResampling& h_resampling,
    HostState& pre_state, HostMeasurements& pre_measurements, HostMap& pre_map, HostRobotParticles& pre_robot_particles, HostResampling& pre_resampling) {

#ifdef VERBOSE_BANNER
    printf("/****************** ALLOCATIONS & INITIALIZATIONS  ******************/\n");
#endif

#ifdef VERBOSE_TOTAL_INFO
    printf("~~$ GRID_WIDTH: \t\t%d\n", GRID_WIDTH);
    printf("~~$ GRID_HEIGHT: \t\t%d\n", GRID_HEIGHT);

    printf("~~$ PARTICLES_OCCUPIED_LEN: \t%d\n", OCCUPIED_LEN);
    printf("~~$ PARTICLE_UNIQUE_COUNTER: \t%d\n", PARTICLE_UNIQUE_COUNTER);
    printf("~~$ MAX_DIST_IN_MAP: \t\t%d\n", MAX_DIST_IN_MAP);
    printf("~~$ LIDAR_COORDS_LEN: \t\t%d\n", LIDAR_COORDS_LEN);
    printf("~~$ PARTICLES_ITEMS_LEN: \t%d\n", PARTICLES_ITEMS_LEN);
#endif

    auto start_robot_particles_alloc = std::chrono::high_resolution_clock::now();
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
    auto stop_robot_particles_alloc = std::chrono::high_resolution_clock::now();


#ifdef VERBOSE_EXECUTION_TIME
    auto duration_robot_particles_alloc = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_particles_alloc - start_robot_particles_alloc);
    auto duration_mapping_alloc = std::chrono::duration_cast<std::chrono::microseconds>(stop_mapping_alloc - start_mapping_alloc);

    std::cout << "Time taken by function (Particles Allocation): " << duration_robot_particles_alloc.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Mapping Allocation): " << duration_mapping_alloc.count() << " microseconds" << std::endl;
    std::cout << std::endl;
#endif
}

void test_alloc_init_map(DevicePosition& d_position, DeviceTransition& d_transition, DeviceParticles& d_particles,
    Device2DUniqueFinder& d_unique_occupied, Device2DUniqueFinder& d_unique_free,
    HostMap& h_map, HostPosition& h_position, HostTransition& h_transition, HostParticles& h_particles,
    HostMeasurements& h_measurements, Host2DUniqueFinder& h_unique_occupied, Host2DUniqueFinder& h_unique_free,
    HostTransition& pre_transition, HostMap& pre_map, HostParticles& pre_particles,
    host_vector<int>& hvec_occupied_map_idx, host_vector<int>& hvec_free_map_idx) {

    hvec_occupied_map_idx.resize(2, 0);
    hvec_free_map_idx.resize(2, 0);

    auto start_mapping_alloc = std::chrono::high_resolution_clock::now();
    alloc_init_transition_vars(d_position, d_transition, h_position, h_transition, pre_transition);
    int MAX_DIST_IN_MAP = sqrt(pow(pre_map.GRID_WIDTH, 2) + pow(pre_map.GRID_HEIGHT, 2));
    alloc_init_particles_vars(d_particles, h_particles, h_measurements, pre_particles, MAX_DIST_IN_MAP);
    hvec_occupied_map_idx[1] = h_particles.OCCUPIED_LEN;
    hvec_free_map_idx[1] = 0;
    alloc_init_unique_map_vars(d_unique_occupied, h_unique_occupied, h_map, hvec_occupied_map_idx);
    alloc_init_unique_map_vars(d_unique_free, h_unique_free, h_map, hvec_free_map_idx);
    auto stop_mapping_alloc = std::chrono::high_resolution_clock::now();

#ifdef VERBOSE_EXECUTION_TIME
    auto duration_mapping_alloc = std::chrono::duration_cast<std::chrono::microseconds>(stop_mapping_alloc - start_mapping_alloc);
    std::cout << "Time taken by function (Mapping Allocation): " << duration_mapping_alloc.count() << " microseconds" << std::endl;
    std::cout << std::endl;
#endif
}


void test_robot(DeviceState& d_state, DeviceState& d_clone_state, DeviceMap& d_map, DeviceTransition& d_transition, DeviceMeasurements& d_measurements,
    DeviceProcessedMeasure& d_processed_measure, DeviceCorrelation& d_correlation, DeviceResampling& d_resampling,
    DeviceParticlesTransition& d_particles_transition, Device2DUniqueFinder& d_2d_unique,
    DeviceRobotParticles& d_robot_particles, DeviceRobotParticles& d_clone_robot_particles,
    HostMap& h_map, HostMeasurements& h_measurements, HostParticlesTransition& h_particles_transition, HostResampling& h_resampling,
    HostRobotParticles& h_robot_particles, HostRobotParticles& h_clone_robot_particles, HostProcessedMeasure& h_processed_measure,
    Host2DUniqueFinder& h_2d_unique, HostCorrelation& h_correlation, HostState& h_state, HostRobotState& h_robot_state,
    GeneralInfo& general_info) {

#ifdef VERBOSE_BANNER
    printf("/****************************** ROBOT *******************************/\n");
#endif

    const int MEASURE_LEN = NUM_PARTICLES * h_measurements.LEN;
    int* h_last_len = (int*)malloc(sizeof(int));


    auto start_robot_particles_kernel = std::chrono::high_resolution_clock::now();
    exec_robot_move(d_state, h_state);

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
    reinit_particles_vars(d_state, d_robot_particles, d_resampling, d_clone_robot_particles, d_clone_state, h_robot_particles, h_state, h_last_len);

    exec_rearrangement(d_robot_particles, d_state, d_resampling, d_clone_robot_particles, d_clone_state, h_map,
        h_robot_particles, h_clone_robot_particles, h_last_len);
    exec_update_states(d_state, d_transition, h_state, h_robot_state);
    auto stop_robot_particles_kernel = std::chrono::high_resolution_clock::now();

#ifdef VERBOSE_EXECUTION_TIME
    auto duration_robot_particles_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_particles_kernel - start_robot_particles_kernel);

    std::cout << std::endl;
    std::cout << "Time taken by function (Kernel): " << duration_robot_particles_kernel.count() << " microseconds" << std::endl;
    std::cout << std::endl;
#endif
}

void test_map(DevicePosition& d_position, DeviceTransition& d_transition, DeviceParticles& d_particles,
    DeviceMeasurements& d_measurements, DeviceMap& d_map,
    HostMeasurements& h_measurements, HostMap& h_map, HostPosition& h_position, HostTransition& h_transition,
    Device2DUniqueFinder& d_unique_occupied, Device2DUniqueFinder& d_unique_free,
    Host2DUniqueFinder& h_unique_occupied, Host2DUniqueFinder& h_unique_free,
    HostParticles& h_particles, GeneralInfo& general_info) {

#ifdef VERBOSE_BANNER
    printf("/****************************** MAP MAIN ****************************/\n");
#endif

    auto start_mapping_kernel = std::chrono::high_resolution_clock::now();

    exec_world_to_image_transform_step_1(d_position, d_transition, d_particles, d_measurements, h_measurements);

    bool EXTEND = false;
    exec_map_extend(d_map, d_measurements, d_particles, d_unique_occupied, d_unique_free,
        h_map, h_measurements, h_unique_occupied, h_unique_free, general_info, EXTEND);

    exec_world_to_image_transform_step_2(d_measurements, d_particles, d_position, d_transition,
        h_map, h_measurements, general_info);

    int MAX_DIST_IN_MAP = sqrt(pow(h_map.GRID_WIDTH, 2) + pow(h_map.GRID_HEIGHT, 2));

    exec_bresenham(d_particles, d_position, d_transition, h_particles, MAX_DIST_IN_MAP);

    reinit_map_idx_vars(d_unique_free, h_particles, h_unique_free);
    exec_create_map(d_particles, d_unique_occupied, d_unique_free, h_map, h_particles);

    reinit_map_vars(d_particles, d_unique_occupied, d_unique_free, h_particles, h_unique_occupied, h_unique_free);
    exec_map_restructure(d_particles, d_unique_occupied, d_unique_free, h_map);

    exec_log_odds(d_map, d_particles, h_map, h_particles, general_info);

    auto stop_mapping_kernel = std::chrono::high_resolution_clock::now();

#ifdef VERBOSE_EXECUTION_TIME
    auto duration_mapping_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_mapping_kernel - start_mapping_kernel);

    std::cout << std::endl;
    std::cout << "Time taken by function (Mapping Kernel): " << duration_mapping_kernel.count() << " microseconds" << std::endl;
    std::cout << std::endl;
#endif
}



//void resetMiddleVariables(DeviceCorrelation& d_correlation, DeviceProcessedMeasure& d_processed_measure, DeviceResampling& d_resampling,
//    Device2DUniqueFinder& d_2d_unique, DeviceRobotParticles& d_robot_particles,
//    HostMap& h_map, HostMeasurements& h_measurements, HostProcessedMeasure& h_processed_measure) {
//
//    int num_items = NUM_PARTICLES * h_measurements.LEN;
//    d_processed_measure.x.clear();
//    d_processed_measure.y.clear();
//    d_processed_measure.idx.clear();
//    d_processed_measure.x.resize(num_items, 0);
//    d_processed_measure.y.resize(num_items, 0);
//    d_processed_measure.idx.resize(num_items, 0);
//
//    h_processed_measure.x.clear();
//    h_processed_measure.y.clear();
//    h_processed_measure.idx.clear();
//    h_processed_measure.x.resize(num_items, 0);
//    h_processed_measure.y.resize(num_items, 0);
//    h_processed_measure.idx.resize(num_items, 0);
//
//    thrust::fill(d_2d_unique.s_map.begin(), d_2d_unique.s_map.end(), 0);
//    thrust::fill(d_2d_unique.c_in_map.begin(), d_2d_unique.c_in_map.end(), 0);
//    thrust::fill(d_2d_unique.s_in_col.begin(), d_2d_unique.s_in_col.end(), 0);
//
//    thrust::fill(d_correlation.raw.begin(), d_correlation.raw.end(), 0);
//    thrust::fill(d_correlation.sum_exp.begin(), d_correlation.sum_exp.end(), 0);
//    thrust::fill(d_correlation.max.begin(), d_correlation.max.end(), 0);
//
//    thrust::fill(d_resampling.js.begin(), d_resampling.js.end(), 0);
//    thrust::fill(d_robot_particles.weight.begin(), d_robot_particles.weight.end(), 0);
//}
//
//

// [ ] - Define THR_GRID_WIDTH, THR_GRID_HEIGHT
// [ ] - Define CURR_GRID_WIDTH, CURR_GRID_HEIGHT
// [ ] - Define curr_grid_map
// [ ] - Define draw thread function
// [ ] - Define mutex for synchronization

int THR_GRID_WIDTH = 0;
int THR_GRID_HEIGHT = 0;
HostMap thr_map;

timed_mutex timed_mutex_draw;

void thread_draw() {

    GLfloat delta_time = 0.0f;
    GLfloat last_time = 0.0f;

    Light main_light;

    int CURR_GRID_WIDTH = 0;
    int CURR_GRID_HEIGHT = 0;

    while (timed_mutex_draw.try_lock_until(std::chrono::steady_clock::now() + std::chrono::seconds(1)) == false);
    printf("Draw Thread Started ...\n");

    CURR_GRID_WIDTH = THR_GRID_WIDTH;
    CURR_GRID_HEIGHT = THR_GRID_HEIGHT;
    printf("CURR_GRID_WIDTH: %d, CURR_GRID_HEIGHT: %d\n", CURR_GRID_WIDTH, CURR_GRID_HEIGHT);

    // Vertex Shader
    static const char* vShader = "Shaders/shader.vert";

    // Fragment Shader
    static const char* fShader = "Shaders/shader.frag";

    mainWindow.initialize();

    //gpuErrchk(cudaMemcpy(res_grid_map, d_grid_map, sz_grid_map, cudaMemcpyDeviceToHost));

    CreateObjects(freeList, thr_map.s_grid_map.data(), CURR_GRID_WIDTH * CURR_GRID_HEIGHT, 1, 0.1f, CURR_GRID_HEIGHT);
    CreateObjects(wallList, thr_map.s_grid_map.data(), CURR_GRID_WIDTH * CURR_GRID_HEIGHT, 2, 0.5f, CURR_GRID_HEIGHT);
    CreateShaders(shader_list, vShader, fShader);

    camera = Camera(glm::vec3(-2.0f, 4.0f, 12.0f), glm::vec3(0.0f, 1.0f, 0.0f), -90.0f, -45.0f, 5.0f, 0.1f);

    main_light = Light(1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    GLuint uniformModel = 0, uniformProjection = 0, uniformView = 0, uniformColor = 0,
        uniformAmbientIntensity = 0, uniformAmbientColor = 0;
    glm::mat4 projection = glm::perspective(45.0f,
        (GLfloat)mainWindow.getBufferWidth() / (GLfloat)mainWindow.getBufferHeight(), 0.1f, 90.0f);


    // Loop until windows closed
    while (!mainWindow.getShouldClose()) {

        if (timed_mutex_draw.try_lock_until(std::chrono::steady_clock::now() + std::chrono::milliseconds(10)) == true) {
            
            CURR_GRID_WIDTH = THR_GRID_WIDTH;
            CURR_GRID_HEIGHT = THR_GRID_HEIGHT;
            printf("CURR_GRID_WIDTH: %d, CURR_GRID_HEIGHT: %d\n", CURR_GRID_WIDTH, CURR_GRID_HEIGHT);

            //gpuErrchk(cudaMemcpy(res_grid_map, d_grid_map, sz_grid_map, cudaMemcpyDeviceToHost));

            freeList.clear();
            wallList.clear();

            //CreateObjects(freeList, res_grid_map, CURR_GRID_WIDTH * CURR_GRID_HEIGHT, 1, 0.1f, CURR_GRID_HEIGHT);
            //CreateObjects(wallList, res_grid_map, CURR_GRID_WIDTH * CURR_GRID_HEIGHT, 2, 0.5f, CURR_GRID_HEIGHT);
            CreateObjects(freeList, thr_map.s_grid_map.data(), CURR_GRID_WIDTH * CURR_GRID_HEIGHT, 1, 0.1f, CURR_GRID_HEIGHT);
            CreateObjects(wallList, thr_map.s_grid_map.data(), CURR_GRID_WIDTH * CURR_GRID_HEIGHT, 2, 0.5f, CURR_GRID_HEIGHT);

            //CreateShaders(shader_list, vShader, fShader);
        }

        GLfloat now = glfwGetTime();
        delta_time = now - last_time;
        last_time = now;

        // Get+Handle user inputs
        glfwPollEvents();

        camera.keyControl(mainWindow.getskeys(), delta_time);
        camera.mouseControl(mainWindow.getXChange(), mainWindow.getYChange(), delta_time);

        // Clear window
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        shader_list[0].UseShader();
        uniformModel = shader_list[0].GetModelLocation();
        uniformProjection = shader_list[0].GetProjectionLocation();
        uniformView = shader_list[0].GetViewLocation();
        uniformColor = shader_list[0].GetColorLocation();
        uniformAmbientColor = shader_list[0].GetAmbientColorLocation();
        uniformAmbientIntensity = shader_list[0].GetAmbientIntensityLocation();

        main_light.UseLight(uniformAmbientIntensity, uniformAmbientColor, 0.0f, 0.0f);

        glm::mat4 model = glm::identity<glm::mat4>();

        model = glm::translate(model, glm::vec3(0.0f, 0.0f, -5.0f));
        //model = glm::scale(model, glm::vec3(0.4f, 0.4f, 0.4f));
        glUniformMatrix4fv(uniformModel, 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(uniformProjection, 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(uniformView, 1, GL_FALSE, glm::value_ptr(camera.calculateViewMatrix()));

        glUniform4f(uniformColor, 0.8f, 0.8f, 0.8f, 1.0f);
        for (int i = 0; i < freeList.size(); i++) {
            freeList[i]->RenderMesh();
        }

        glUniform4f(uniformColor, 0.0f, 0.2f, 0.9f, 1.0f);
        for (int i = 0; i < wallList.size(); i++) {
            wallList[i]->RenderMesh();
        }

        glUseProgram(0);

        mainWindow.swapBuffers();
    }
}

void resetMiddleVariables(DeviceCorrelation& d_correlation, DeviceProcessedMeasure& d_processed_measure, DeviceResampling& d_resampling,
    Device2DUniqueFinder& d_2d_unique, DeviceRobotParticles& d_robot_particles,
    HostMap& h_map, HostMeasurements& h_measurements, HostProcessedMeasure& h_processed_measure) {

    int num_items = NUM_PARTICLES * h_measurements.LEN;
    reset_processed_measure(d_processed_measure, h_measurements);

    thrust::fill(d_2d_unique.s_map.begin(), d_2d_unique.s_map.end(), 0);
    thrust::fill(d_2d_unique.c_in_map.begin(), d_2d_unique.c_in_map.end(), 0);
    thrust::fill(d_2d_unique.s_in_col.begin(), d_2d_unique.s_in_col.end(), 0);

    reset_correlation(d_correlation);

    thrust::fill(d_resampling.c_js.begin(), d_resampling.c_js.end(), 0);
    thrust::fill(d_robot_particles.c_weight.begin(), d_robot_particles.c_weight.end(), 0);
}

void run_main() {

    std::cout << "Run Application" << std::endl;


    // vector<vector<float>> vec_arr_transition;

    //vector<vector<float>> vec_arr_rnds_encoder_counts;
    //vector<vector<float>> vec_arr_rnds;
    //vector<vector<float>> vec_arr_rnds_yaws;
    vector<float> vec_rnds_encoder_counts;
    vector<float> vec_rnds_yaws;
    vector<float> vec_rnds;


    vector<vector<float>> vec_arr_lidar_coords;
    vector<float> vec_encoder_counts;
    vector<float> vec_yaws;
    vector<float> vec_dt;

    printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
    printf("Reading Data Files\n");
    printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");

    // bool should_assert = false;

    HostState pre_state;
    HostState post_robot_move_state;
    HostState post_state;
    HostRobotParticles pre_robot_particles;
    HostRobotParticles post_unique_robot_particles;
    HostRobotParticles pre_resampling_robot_particles;
    HostRobotParticles post_resampling_robot_particles;
    HostProcessedMeasure post_processed_measure;
    HostParticlesTransition post_particles_transition;
    HostResampling pre_resampling;
    HostRobotState post_robot_state;
    HostMap pre_map;
    HostMap post_bg_map;
    HostMap post_map;
    HostMeasurements pre_measurements;
    HostPosition post_position;
    HostTransition pre_transition;
    HostTransition post_transition;
    HostParticles pre_particles;
    HostParticles post_particles;
    GeneralInfo general_info;

    DeviceState d_state;
    DeviceState d_clone_state;
    DeviceMeasurements d_measurements;
    DeviceMap d_map;
    DeviceRobotParticles d_robot_particles;
    DeviceRobotParticles d_clone_robot_particles;
    DeviceCorrelation d_correlation;
    DeviceParticlesTransition d_particles_transition;
    DeviceParticlesPosition d_particles_position;
    DeviceParticlesRotation d_particles_rotation;
    DeviceProcessedMeasure d_processed_measure;
    Device2DUniqueFinder d_2d_unique;
    DeviceResampling d_resampling;
    DeviceTransition d_transition;
    DevicePosition d_position;
    DeviceParticles d_particles;
    Device2DUniqueFinder d_unique_occupied;
    Device2DUniqueFinder d_unique_free;


    HostState h_state;
    HostRobotState h_robot_state;
    HostMeasurements h_measurements;
    HostMap h_map;
    HostRobotParticles h_robot_particles;
    HostRobotParticles h_clone_robot_particles;
    HostCorrelation h_correlation;
    HostParticlesTransition h_particles_transition;
    HostParticlesPosition h_particles_position;
    HostParticlesRotation h_particles_rotation;
    HostProcessedMeasure h_processed_measure;
    Host2DUniqueFinder h_2d_unique;
    HostResampling h_resampling;
    HostPosition h_position;
    HostTransition h_transition;
    HostParticles h_particles;
    Host2DUniqueFinder h_unique_occupied;
    Host2DUniqueFinder h_unique_free;

    host_vector<float> pre_weights;
    host_vector<float> post_loop_weights;
    host_vector<float> post_weights;

    host_vector<int> hvec_occupied_map_idx;
    host_vector<int> hvec_free_map_idx;

    const int LOOP_LEN = 4800;
    // const int LOOP_LEN = 1400;
    const int ST_FILE_NUMBER = 0;
    const int INFO_STEP = 100;
    const int DIFF_FROM_START = ST_FILE_NUMBER - 0;

    int PRE_GRID_SIZE = 0;

    bool map_size_changed = false;
    bool check_assert = false;

    std::thread t(thread_draw);
    lock_guard<timed_mutex> l(timed_mutex_draw);

    const int INPUT_VEC_SIZE = 4900;
    // const int INPUT_VEC_SIZE = 1600;
    read_small_steps_vec("encoder_counts", vec_encoder_counts, INPUT_VEC_SIZE);
    read_small_steps_vec_arr("lidar_coords", vec_arr_lidar_coords, INPUT_VEC_SIZE);
    read_small_steps_vec("yaws", vec_yaws, INPUT_VEC_SIZE);
    read_small_steps_vec("dt", vec_dt, INPUT_VEC_SIZE);

    // read_small_steps_vec_arr("transition", vec_arr_transition, INPUT_VEC_SIZE);

    //read_small_steps_vec_arr("rnds_encoder_counts", vec_arr_rnds_encoder_counts, INPUT_VEC_SIZE);
    //read_small_steps_vec_arr("rnds", vec_arr_rnds, INPUT_VEC_SIZE);
    //read_small_steps_vec_arr("rnds_yaws", vec_arr_rnds_yaws, INPUT_VEC_SIZE);

    //for (int i = 0; i < 10; i++) {
    //    printf("vec_arr_rnds_encoder_counts size: %d, vec_arr_rnds_encoder_counts[%d] size: %d\n", vec_arr_rnds_encoder_counts.size(), i, vec_arr_rnds_encoder_counts[i].size());
    //    printf("vec_arr_rnds size: %d, v[%d] size: %d\n", vec_arr_rnds_encoder_counts.size(), i, vec_arr_rnds_encoder_counts[i].size());
    //    printf("rnds_yaws size: %d, rnds_yaws[%d] size: %d\n", vec_arr_rnds_encoder_counts.size(), i, vec_arr_rnds_encoder_counts[i].size());
    //    std::cout << std::endl;
    //}


    for (int file_number = ST_FILE_NUMBER; file_number < ST_FILE_NUMBER + LOOP_LEN; file_number += 1) {

        if (file_number == ST_FILE_NUMBER) {

            printf("Iteration: %d\n", file_number);

            auto start_read_file = std::chrono::high_resolution_clock::now();
            read_iteration(file_number, pre_state, post_robot_move_state, post_state,
                pre_robot_particles, post_unique_robot_particles,
                pre_resampling_robot_particles, post_resampling_robot_particles,
                post_processed_measure, post_particles_transition,
                pre_resampling, post_robot_state,
                pre_map, post_bg_map, post_map, pre_measurements,
                post_position, pre_transition, post_transition,
                pre_particles, post_particles, general_info,
                pre_weights, post_loop_weights, post_weights);
            auto stop_read_file = std::chrono::high_resolution_clock::now();

            test_alloc_init_robot(d_state, d_clone_state, d_measurements,
                d_map, d_robot_particles, d_clone_robot_particles, d_correlation, d_particles_transition,
                d_particles_position, d_particles_rotation, d_transition,
                d_processed_measure, d_2d_unique, d_resampling,
                h_state, h_robot_state, h_measurements, h_map, h_robot_particles,
                h_correlation, h_particles_transition, h_particles_position,
                h_particles_rotation, h_processed_measure, h_2d_unique, h_resampling,
                pre_state, pre_measurements, pre_map, pre_robot_particles, pre_resampling);

            test_alloc_init_map(d_position, d_transition, d_particles,
                d_unique_occupied, d_unique_free,
                h_map, h_position, h_transition, h_particles,
                h_measurements, h_unique_occupied, h_unique_free,
                pre_transition, pre_map, pre_particles,
                hvec_occupied_map_idx, hvec_free_map_idx);

            auto duration_read_file = std::chrono::duration_cast<std::chrono::microseconds>(stop_read_file - start_read_file);
            std::cout << std::endl;
            std::cout << "Time taken by function (Read Data Files): " << duration_read_file.count() << " microseconds" << std::endl;
            std::cout << std::endl;
        }
        else {

            if (file_number % INFO_STEP == 0) {
                printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
                printf("Iteration: %d\n", file_number);
            }

            int curr_idx = file_number - ST_FILE_NUMBER + DIFF_FROM_START;

            auto start_alloc_init_step = std::chrono::high_resolution_clock::now();

            h_measurements.LEN = vec_arr_lidar_coords[curr_idx].size() / 2;
            int MEASURE_LEN = NUM_PARTICLES * h_measurements.LEN;
            h_particles.OCCUPIED_LEN = h_measurements.LEN;
            int PARTICLE_UNIQUE_COUNTER = h_particles.OCCUPIED_LEN + 1;

            set_measurement_vars(d_measurements, h_measurements, vec_arr_lidar_coords[curr_idx], vec_arr_lidar_coords[curr_idx].size() / 2);
            reset_processed_measure(d_processed_measure, h_measurements);
            reset_correlation(d_correlation);
            thrust::fill(d_robot_particles.c_weight.begin(), d_robot_particles.c_weight.end(), 0);
            gen_normal_numbers(vec_rnds_encoder_counts);
            gen_normal_numbers(vec_rnds_yaws);
            
            set_state(d_state, h_state, pre_state, vec_rnds_encoder_counts,
                vec_rnds_yaws, vec_encoder_counts[curr_idx], vec_yaws[curr_idx], vec_dt[curr_idx]);
            // set_resampling(d_resampling, vec_arr_rnds[curr_idx]);

            gen_uniform_numbers(vec_rnds);
            set_resampling(d_resampling, vec_rnds);
            

            int MAX_DIST_IN_MAP = sqrt(pow(pre_map.GRID_WIDTH, 2) + pow(pre_map.GRID_HEIGHT, 2));

            int curr_grid_size = h_map.GRID_WIDTH * h_map.GRID_HEIGHT;
            if (curr_grid_size != PRE_GRID_SIZE) {

                int MAX_DIST_IN_MAP = sqrt(pow(h_map.GRID_WIDTH, 2) + pow(h_map.GRID_HEIGHT, 2));
                resize_particles_vars(d_particles, h_measurements, MAX_DIST_IN_MAP);
                resize_unique_map_vars(d_unique_occupied, h_unique_occupied, h_map);
                resize_unique_map_vars(d_unique_free, h_unique_free, h_map);


                PRE_GRID_SIZE = curr_grid_size;
            }
            else {

                //reset_map_vars(d_map, h_map, pre_map);
                //reset_unique_map_vars(d_2d_unique);
            }
            thrust::fill(d_map.c_should_extend.begin(), d_map.c_should_extend.end(), 0);

            hvec_occupied_map_idx[1] = h_particles.OCCUPIED_LEN;
            hvec_free_map_idx[1] = 0;
            reset_unique_map_vars(d_unique_occupied, hvec_occupied_map_idx);
            reset_unique_map_vars(d_unique_free, hvec_free_map_idx);
            alloc_map_2d_var(d_2d_unique, h_2d_unique, h_map);


#if defined(GET_EXTRA_TRANSITION_WORLD_BODY)
            //if (file_number % INFO_STEP == 0) {
            //    print_world_body(vec_arr_transition[curr_idx]);
            //    print_world_body(d_transition.c_world_body);
            //}
            //d_transition.c_world_body.assign(vec_arr_transition[curr_idx].begin(), vec_arr_transition[curr_idx].end());
#endif

            if (file_number % INFO_STEP == 0) {

                auto stop_alloc_init_step = std::chrono::high_resolution_clock::now();
                auto duration_alloc_init_step = std::chrono::duration_cast<std::chrono::microseconds>(stop_alloc_init_step - start_alloc_init_step);
                std::cout << std::endl;
                std::cout << "Time taken by function (Alloc & Init Step): " << duration_alloc_init_step.count() << " microseconds" << std::endl;
                std::cout << std::endl;
            }
        }

        auto start_run_step = std::chrono::high_resolution_clock::now();
        test_robot(d_state, d_clone_state, d_map, d_transition, d_measurements, d_processed_measure, d_correlation, d_resampling,
            d_particles_transition, d_2d_unique, d_robot_particles, d_clone_robot_particles,
            h_map, h_measurements, h_particles_transition, h_resampling, h_robot_particles, h_clone_robot_particles, h_processed_measure,
            h_2d_unique, h_correlation, h_state, h_robot_state,
            general_info);
        test_map(d_position, d_transition, d_particles, d_measurements, d_map,
            h_measurements, h_map, h_position, h_transition, d_unique_occupied, d_unique_free, h_unique_occupied, h_unique_free,
            h_particles, general_info);

        PRE_GRID_SIZE = h_map.GRID_WIDTH * h_map.GRID_HEIGHT;

        if (file_number % INFO_STEP == 0) {

            auto stop_run_step = std::chrono::high_resolution_clock::now();

            // printf("Iteration: %d\n", file_number);

            auto duration_run_step = std::chrono::duration_cast<std::chrono::microseconds>(stop_run_step - start_run_step);
            std::cout << std::endl;
            std::cout << "Time taken by function (Run Step): " << duration_run_step.count() << " microseconds" << std::endl;
            std::cout << std::endl;

            THR_GRID_WIDTH = h_map.GRID_WIDTH;
            THR_GRID_HEIGHT = h_map.GRID_HEIGHT;
            thr_map.s_grid_map.clear();
            thr_map.s_grid_map.resize(THR_GRID_WIDTH* THR_GRID_HEIGHT, 0);
            thr_map.s_grid_map.assign(d_map.s_grid_map.begin(), d_map.s_grid_map.end());
            timed_mutex_draw.unlock();
        }
    }

    printf("Execution Finished\n\n");

    t.join();
}

#endif

