#ifndef _DEVICE_SET_RESET_MAP_H_
#define _DEVICE_SET_RESET_MAP_H_

#include "headers.h"
#include "structures.h"

void resize_particles_vars(DeviceParticles& d_particles, HostMeasurements& h_measurements, const int MAX_DIST_IN_MAP) {

    d_particles.sv_free_x_max.clear();
    d_particles.sv_free_y_max.clear();
    d_particles.sv_free_x_max.resize(h_measurements.MAX_LEN * MAX_DIST_IN_MAP, 0);
    d_particles.sv_free_y_max.resize(h_measurements.MAX_LEN * MAX_DIST_IN_MAP, 0);
}

void resize_unique_map_vars(Device2DUniqueFinder& d_unique, Host2DUniqueFinder& h_unique, HostMap& h_map) {

    d_unique.s_map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);
    d_unique.s_in_col.resize(h_map.GRID_WIDTH + 1, 0);
    h_unique.s_map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);
    h_unique.s_in_col.resize(h_map.GRID_WIDTH + 1, 0);
}

void reset_unique_map_vars(Device2DUniqueFinder& d_unique, host_vector<int>& hvec_map_idx) {

    thrust::fill(d_unique.s_map.begin(),    d_unique.s_map.end(), 0);
    thrust::fill(d_unique.c_in_map.begin(), d_unique.c_in_map.end(), 0);
    thrust::fill(d_unique.s_in_col.begin(), d_unique.s_in_col.end(), 0);

    d_unique.c_idx.assign(hvec_map_idx.begin(), hvec_map_idx.end());
}

void reset_unique_map_vars(Device2DUniqueFinder& d_unique) {

    thrust::fill(d_unique.s_map.begin(), d_unique.s_map.end(), 0);
    thrust::fill(d_unique.c_in_map.begin(), d_unique.c_in_map.end(), 0);
    thrust::fill(d_unique.s_in_col.begin(), d_unique.s_in_col.end(), 0);
}

void reset_map_vars(DeviceMap& d_map, HostMap& h_map, HostMap& pre_map) {

    h_map.b_should_extend = pre_map.b_should_extend;
    d_map.s_grid_map.assign(pre_map.s_grid_map.begin(), pre_map.s_grid_map.end());
    d_map.s_log_odds.assign(pre_map.s_log_odds.begin(), pre_map.s_log_odds.end());
    thrust::fill(d_map.c_should_extend.begin(), d_map.c_should_extend.end(), 0);
}

void set_measurement_vars(DeviceMeasurements& d_measurements, HostMeasurements& h_measurements, 
    HostMeasurements& pre_measurements) {

    h_measurements.LEN = pre_measurements.LEN;
    thrust::copy(pre_measurements.v_lidar_coords.begin(), pre_measurements.v_lidar_coords.end(), d_measurements.v_lidar_coords.begin());
}

void set_measurement_vars(DeviceMeasurements& d_measurements, HostMeasurements& h_measurements, std::vector<float> v_lidar_coords, 
    int LEN) {

    h_measurements.LEN = LEN;
    thrust::copy(v_lidar_coords.begin(), v_lidar_coords.end(), d_measurements.v_lidar_coords.begin());
}

void set_transition_vars(DeviceTransition& d_transition, HostTransition& pre_transition) {

    d_transition.c_world_body.assign(pre_transition.c_world_body.begin(), pre_transition.c_world_body.end());
    d_transition.c_body_lidar.assign(hvec_transition_body_lidar.begin(), hvec_transition_body_lidar.end());
}

#endif
