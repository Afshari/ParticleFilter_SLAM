#ifndef _DEVICE_EXEC_ROBOT_H_
#define _DEVICE_EXEC_ROBOT_H_

#include "headers.h"
#include "structures.h"
#include "kernels.cuh"
#include "kernels_utils.cuh"
#include "kernels_robot.cuh"


void exec_robot_move(DeviceState& d_state, HostState& res_state) {

	int threadsPerBlock = NUM_PARTICLES;
	int blocksPerGrid = 1;
	kernel_robot_advance << <blocksPerGrid, threadsPerBlock >> > (
		THRUST_RAW_CAST(d_state.x), THRUST_RAW_CAST(d_state.y), THRUST_RAW_CAST(d_state.theta), SEP,
		THRUST_RAW_CAST(d_state.rnds_encoder_counts), THRUST_RAW_CAST(d_state.rnds_yaws),
		res_state.encoder_counts, res_state.yaw, res_state.dt, res_state.nv, res_state.nw, NUM_PARTICLES);
	cudaDeviceSynchronize();
}

void exec_calc_transition(DeviceParticlesTransition& d_particles_transition, DeviceState& d_state,
    DeviceTransition& d_transition, HostParticlesTransition& res_particles_transition) {

    int threadsPerBlock = NUM_PARTICLES;
    int blocksPerGrid = 1;
    kernel_update_particles_states << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_particles_transition.world_body),
        THRUST_RAW_CAST(d_particles_transition.world_lidar), SEP,
        THRUST_RAW_CAST(d_state.x), THRUST_RAW_CAST(d_state.y), THRUST_RAW_CAST(d_state.theta),
        THRUST_RAW_CAST(d_transition.body_lidar), NUM_PARTICLES);
    cudaDeviceSynchronize();
}

void exec_process_measurements(DeviceProcessedMeasure& d_processed_measure, DeviceParticlesTransition& d_particles_transition,
    DeviceMeasurements& d_measurements, HostMap& h_map, HostMeasurements& res_measurements, GeneralInfo& general_info) {

    int threadsPerBlock = NUM_PARTICLES;
    int blocksPerGrid = res_measurements.LIDAR_COORDS_LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_processed_measure.x), THRUST_RAW_CAST(d_processed_measure.y), SEP,
        THRUST_RAW_CAST(d_particles_transition.world_lidar), THRUST_RAW_CAST(d_measurements.lidar_coords),
        general_info.res, h_map.xmin, h_map.ymax, res_measurements.LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_index_init_const << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_processed_measure.idx), res_measurements.LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, d_processed_measure.idx.begin(), d_processed_measure.idx.end(), d_processed_measure.idx.begin(), 0);
}


void exec_create_2d_map(Device2DUniqueFinder& d_2d_unique, DeviceRobotParticles& d_robot_particles, HostMap& h_map,
    HostRobotParticles& res_robot_particles) {

    int threadsPerBlock = 100;
    int blocksPerGrid = NUM_PARTICLES;
    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_2d_unique.map), THRUST_RAW_CAST(d_2d_unique.in_map), THRUST_RAW_CAST(d_2d_unique.in_col), SEP,
        THRUST_RAW_CAST(d_robot_particles.x), THRUST_RAW_CAST(d_robot_particles.y), THRUST_RAW_CAST(d_robot_particles.idx),
        res_robot_particles.LEN, h_map.GRID_WIDTH, h_map.GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();
}


void exec_update_map(Device2DUniqueFinder& d_2d_unique, DeviceProcessedMeasure& d_processed_measure, HostMap& h_map,
    const int MEASURE_LEN) {

    int threadsPerBlock = NUM_PARTICLES;
    int blocksPerGrid = 1;

    kernel_update_2d_map_with_measure << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_2d_unique.map), THRUST_RAW_CAST(d_2d_unique.in_map), THRUST_RAW_CAST(d_2d_unique.in_col), SEP,
        THRUST_RAW_CAST(d_processed_measure.x), THRUST_RAW_CAST(d_processed_measure.y), THRUST_RAW_CAST(d_processed_measure.idx),
        MEASURE_LEN, h_map.GRID_WIDTH, h_map.GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();
}

void exec_particle_unique_cum_sum(Device2DUniqueFinder& d_2d_unique, HostMap& h_map, Host2DUniqueFinder& res_2d_unique,
    HostRobotParticles& res_robot_particles) {

    const int UNIQUE_COUNTER_LEN = NUM_PARTICLES + 1;

    int threadsPerBlock = UNIQUE_COUNTER_LEN;
    int blocksPerGrid = 1;
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_2d_unique.in_col), h_map.GRID_WIDTH);
    thrust::exclusive_scan(thrust::device, d_2d_unique.in_map.begin(), d_2d_unique.in_map.end(), d_2d_unique.in_map.begin(), 0);
    cudaDeviceSynchronize();

    res_2d_unique.in_map.assign(d_2d_unique.in_map.begin(), d_2d_unique.in_map.end());

    res_robot_particles.LEN = res_2d_unique.in_map[UNIQUE_COUNTER_LEN - 1];
}


void reinit_map_vars(DeviceRobotParticles& d_robot_particles, HostRobotParticles& res_robot_particles) {

    d_robot_particles.x.clear();
    d_robot_particles.x.resize(res_robot_particles.LEN, 0);
    d_robot_particles.y.clear();
    d_robot_particles.y.resize(res_robot_particles.LEN, 0);
    d_robot_particles.extended_idx.clear();
    d_robot_particles.extended_idx.resize(res_robot_particles.LEN, 0);

    res_robot_particles.x.clear();
    res_robot_particles.x.resize(res_robot_particles.LEN, 0);
    res_robot_particles.y.clear();
    res_robot_particles.y.resize(res_robot_particles.LEN, 0);
}

void exec_map_restructure(DeviceRobotParticles& d_robot_particles, Device2DUniqueFinder& d_2d_unique,
    HostMap& h_map) {

    int threadsPerBlock = h_map.GRID_WIDTH;
    int blocksPerGrid = NUM_PARTICLES;

    thrust::fill(d_robot_particles.idx.begin(), d_robot_particles.idx.end(), 0);
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_robot_particles.x), THRUST_RAW_CAST(d_robot_particles.y), THRUST_RAW_CAST(d_robot_particles.idx), SEP,
        THRUST_RAW_CAST(d_2d_unique.map), THRUST_RAW_CAST(d_2d_unique.in_map),
        THRUST_RAW_CAST(d_2d_unique.in_col), h_map.GRID_WIDTH, h_map.GRID_HEIGHT);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, d_robot_particles.idx.begin(), d_robot_particles.idx.end(), d_robot_particles.idx.begin(), 0);
}


void exec_index_expansion(DeviceRobotParticles& d_robot_particles, HostRobotParticles& res_robot_particles) {

    int threadsPerBlock = 100;
    int blocksPerGrid = NUM_PARTICLES;
    kernel_index_expansion << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_robot_particles.extended_idx), THRUST_RAW_CAST(d_robot_particles.idx), res_robot_particles.LEN);
    cudaDeviceSynchronize();

    res_robot_particles.extended_idx.clear();
    res_robot_particles.extended_idx.resize(res_robot_particles.LEN, 0);
    res_robot_particles.extended_idx.assign(d_robot_particles.extended_idx.begin(), d_robot_particles.extended_idx.end());
    res_robot_particles.idx.assign(d_robot_particles.idx.begin(), d_robot_particles.idx.end());
}

void exec_correlation(DeviceMap& d_map, DeviceRobotParticles& d_robot_particles, DeviceCorrelation& d_correlation,
    HostMap& h_map, HostRobotParticles& res_robot_particles) {

    int threadsPerBlock = 256;
    int blocksPerGrid = (res_robot_particles.LEN + threadsPerBlock - 1) / threadsPerBlock;

    kernel_correlation << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.raw), SEP,
        THRUST_RAW_CAST(d_map.grid_map), THRUST_RAW_CAST(d_robot_particles.x), THRUST_RAW_CAST(d_robot_particles.y),
        THRUST_RAW_CAST(d_robot_particles.extended_idx), h_map.GRID_WIDTH, h_map.GRID_HEIGHT, res_robot_particles.LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_correlation_max << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.weight), THRUST_RAW_CAST(d_correlation.raw), NUM_PARTICLES);
    cudaDeviceSynchronize();

    res_robot_particles.extended_idx.assign(d_robot_particles.extended_idx.begin(), d_robot_particles.extended_idx.end());
}

void exec_update_weights(DeviceRobotParticles& d_robot_particles, DeviceCorrelation& d_correlation,
    HostRobotParticles& res_robot_particles, HostCorrelation& res_correlation) {

    int threadsPerBlock = 1;
    int blocksPerGrid = 1;

    kernel_arr_max << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.weight), THRUST_RAW_CAST(d_correlation.max), NUM_PARTICLES);
    cudaDeviceSynchronize();

    res_correlation.max.assign(d_correlation.max.begin(), d_correlation.max.end());

    float norm_value = -res_correlation.max[0] + 50;

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_arr_increase << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.weight), norm_value, 0);
    cudaDeviceSynchronize();

    threadsPerBlock = 1;
    blocksPerGrid = 1;
    kernel_arr_sum_exp << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.sum_exp), THRUST_RAW_CAST(d_correlation.weight), NUM_PARTICLES);
    cudaDeviceSynchronize();

    res_correlation.sum_exp.assign(d_correlation.sum_exp.begin(), d_correlation.sum_exp.end());

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_arr_normalize << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_correlation.weight), res_correlation.sum_exp[0]);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_arr_mult << < blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_robot_particles.weight), THRUST_RAW_CAST(d_correlation.weight));
    cudaDeviceSynchronize();

    res_correlation.weight.assign(d_correlation.weight.begin(), d_correlation.weight.end());
    res_robot_particles.weight.assign(d_robot_particles.weight.begin(), d_robot_particles.weight.end());
}


void exec_resampling(DeviceCorrelation& d_correlation, DeviceResampling& d_resampling) {

    int threadsPerBlock = NUM_PARTICLES;
    int blocksPerGrid = 1;

    kernel_resampling << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_resampling.js), THRUST_RAW_CAST(d_correlation.weight),
        THRUST_RAW_CAST(d_resampling.rnds), NUM_PARTICLES);
    cudaDeviceSynchronize();
}

void reinit_particles_vars(DeviceState& d_state, DeviceRobotParticles& d_robot_particles, DeviceResampling& d_resampling,
    DeviceRobotParticles& d_clone_robot_particles, DeviceState& d_clone_state, HostRobotParticles& res_robot_particles,
    HostState& res_state, int* res_last_len) {

    int* d_last_len = NULL;
    gpuErrchk(cudaMalloc((void**)&d_last_len, sizeof(int)));

    d_clone_robot_particles.x.resize(res_robot_particles.LEN, 0);
    d_clone_robot_particles.y.resize(res_robot_particles.LEN, 0);
    d_clone_robot_particles.idx.resize(res_robot_particles.LEN, 0);

    d_clone_robot_particles.x.assign(d_robot_particles.x.begin(), d_robot_particles.x.end());
    d_clone_robot_particles.y.assign(d_robot_particles.y.begin(), d_robot_particles.y.end());
    d_clone_robot_particles.idx.assign(d_robot_particles.idx.begin(), d_robot_particles.idx.end());

    d_clone_state.x.assign(d_state.x.begin(), d_state.x.end());
    d_clone_state.y.assign(d_state.y.begin(), d_state.y.end());
    d_clone_state.theta.assign(d_state.theta.begin(), d_state.theta.end());

    int threadsPerBlock = NUM_PARTICLES;
    int blocksPerGrid = 1;
    kernel_rearrange_indecies << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_robot_particles.idx), d_last_len, SEP,
        THRUST_RAW_CAST(d_clone_robot_particles.idx), THRUST_RAW_CAST(d_resampling.js), res_robot_particles.LEN);
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(res_last_len, d_last_len, sizeof(int), cudaMemcpyDeviceToHost));
    res_robot_particles.idx.assign(d_robot_particles.idx.begin(), d_robot_particles.idx.end());
}

void exec_rearrangement(DeviceRobotParticles& d_robot_particles, DeviceState& d_state, DeviceResampling& d_resampling,
    DeviceRobotParticles& d_clone_robot_particles, DeviceState& d_clone_state, HostMap& h_map,
    HostRobotParticles& res_robot_particles, HostRobotParticles& res_clone_robot_particles, int* res_last_len) {

    thrust::exclusive_scan(thrust::device, d_robot_particles.idx.begin(), d_robot_particles.idx.end(), d_robot_particles.idx.begin(), 0);

    res_robot_particles.idx.assign(d_robot_particles.idx.begin(), d_robot_particles.idx.end());

    res_clone_robot_particles.LEN = res_robot_particles.LEN;
    res_robot_particles.LEN = res_robot_particles.idx[NUM_PARTICLES - 1] + res_last_len[0];

    int threadsPerBlock = 100;
    int blocksPerGrid = NUM_PARTICLES;
    kernel_rearrange_particles << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_robot_particles.x), THRUST_RAW_CAST(d_robot_particles.y), SEP,
        THRUST_RAW_CAST(d_robot_particles.idx), THRUST_RAW_CAST(d_clone_robot_particles.x), THRUST_RAW_CAST(d_clone_robot_particles.y),
        THRUST_RAW_CAST(d_clone_robot_particles.idx), THRUST_RAW_CAST(d_resampling.js),
        h_map.GRID_WIDTH, h_map.GRID_HEIGHT, NUM_PARTICLES, res_robot_particles.LEN, res_clone_robot_particles.LEN);

    kernel_rearrange_states << <blocksPerGrid, threadsPerBlock >> > (
        THRUST_RAW_CAST(d_state.x), THRUST_RAW_CAST(d_state.y), THRUST_RAW_CAST(d_state.theta), SEP,
        THRUST_RAW_CAST(d_clone_state.x), THRUST_RAW_CAST(d_clone_state.y), THRUST_RAW_CAST(d_clone_state.theta),
        THRUST_RAW_CAST(d_resampling.js));
    cudaDeviceSynchronize();
}


void exec_update_states(DeviceState& d_state, HostState& res_state, HostRobotState& res_robot_state) {

    res_state.x.assign(d_state.x.begin(), d_state.x.end());
    res_state.y.assign(d_state.y.begin(), d_state.y.end());
    res_state.theta.assign(d_state.theta.begin(), d_state.theta.end());

    std::vector<float> std_vec_states_x(res_state.x.begin(), res_state.x.end());
    std::vector<float> std_vec_states_y(res_state.y.begin(), res_state.y.end());
    std::vector<float> std_vec_states_theta(res_state.theta.begin(), res_state.theta.end());


    std::map<std::tuple<float, float, float>, int> states;

    for (int i = 0; i < NUM_PARTICLES; i++) {

        if (states.find(std::make_tuple(std_vec_states_x[i], std_vec_states_y[i], std_vec_states_theta[i])) == states.end()) {
            states.insert({ std::make_tuple(std_vec_states_x[i], std_vec_states_y[i], std_vec_states_theta[i]), 1 });
        }
        else {
            states[std::make_tuple(std_vec_states_x[i], std_vec_states_y[i], std_vec_states_theta[i])] += 1;
        }
    }

    std::map<std::tuple<float, float, float>, int>::iterator best
        = std::max_element(states.begin(), states.end(), [](const std::pair<std::tuple<float, float, float>, int>& a,
            const std::pair<std::tuple<float, float, float>, int>& b)->bool { return a.second < b.second; });

    auto key = best->first;
    //printf("~~$ Max Weight: %d\n", best->second);

    float theta = std::get<2>(key);
    float res_transition_world_body[] = { cos(theta), -sin(theta), std::get<0>(key),
                        sin(theta),  cos(theta), std::get<1>(key),
                        0, 0, 1 };

    res_robot_state.transition_world_body.resize(9, 0);
    res_robot_state.state.resize(3, 0);

    res_robot_state.transition_world_body[0] = cos(theta);	res_robot_state.transition_world_body[1] = -sin(theta);	res_robot_state.transition_world_body[2] = std::get<0>(key);
    res_robot_state.transition_world_body[3] = sin(theta);   res_robot_state.transition_world_body[4] = cos(theta);	res_robot_state.transition_world_body[5] = std::get<1>(key);
    res_robot_state.transition_world_body[6] = 0;			res_robot_state.transition_world_body[7] = 0;			res_robot_state.transition_world_body[8] = 1;

    res_robot_state.state[0] = std::get<0>(key); res_robot_state.state[1] = std::get<1>(key); res_robot_state.state[2] = std::get<2>(key);
}



#endif
