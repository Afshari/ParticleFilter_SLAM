#ifndef _DEVICE_INIT_COMMON_H_
#define _DEVICE_INIT_COMMON_H_

#include "structures.h"

void alloc_init_measurement_vars(DeviceMeasurements& d_measurements, HostMeasurements& h_measurements, HostMeasurements& pre_measurements) {

    h_measurements.LEN = pre_measurements.LEN;

    d_measurements.v_lidar_coords.resize(2 * h_measurements.MAX_LEN);
    //d_measurements.v_lidar_coords.assign(pre_measurements.v_lidar_coords.begin(), pre_measurements.v_lidar_coords.end());
    thrust::copy(pre_measurements.v_lidar_coords.begin(), pre_measurements.v_lidar_coords.end(), d_measurements.v_lidar_coords.begin());
}

void reset_measurement_vars(DeviceMeasurements& d_measurements, HostMeasurements& h_measurements, host_vector<float> hvec_lidar_coords) {

    h_measurements.LEN = hvec_lidar_coords.size();

    d_measurements.v_lidar_coords.resize(2 * h_measurements.LEN);
    d_measurements.v_lidar_coords.assign(hvec_lidar_coords.begin(), hvec_lidar_coords.end());
}

void alloc_init_map_vars(DeviceMap& d_map, HostMap& h_map, HostMap& pre_map) {

    h_map.GRID_WIDTH = pre_map.GRID_WIDTH;
    h_map.GRID_HEIGHT = pre_map.GRID_HEIGHT;
    h_map.xmin = pre_map.xmin;
    h_map.xmax = pre_map.xmax;
    h_map.ymin = pre_map.ymin;
    h_map.ymax = pre_map.ymax;
    h_map.b_should_extend = pre_map.b_should_extend;

    d_map.c_should_extend.resize(4, 0);
    d_map.s_grid_map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);
    d_map.s_grid_map.assign(pre_map.s_grid_map.begin(), pre_map.s_grid_map.end());
    d_map.s_log_odds.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT);
    d_map.s_log_odds.assign(pre_map.s_log_odds.begin(), pre_map.s_log_odds.end());

    h_map.s_grid_map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);
    h_map.s_log_odds.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);
    h_map.c_should_extend.resize(4, 0);
}

void alloc_init_body_lidar(DeviceTransition& d_transition) {

    d_transition.c_body_lidar.resize(9);
    d_transition.c_body_lidar.assign(hvec_transition_body_lidar.begin(), hvec_transition_body_lidar.end());
}


#endif
