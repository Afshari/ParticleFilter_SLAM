#ifndef _DEVICE_INIT_COMMON_H_
#define _DEVICE_INIT_COMMON_H_

#include "structures.h"

void alloc_init_measurement_vars(DeviceMeasurements& d_measurements, HostMeasurements& res_measurements, HostMeasurements& h_measurements) {

    res_measurements.LIDAR_COORDS_LEN = h_measurements.LIDAR_COORDS_LEN;

    d_measurements.lidar_coords.resize(2 * res_measurements.LIDAR_COORDS_LEN);
    d_measurements.lidar_coords.assign(h_measurements.lidar_coords.begin(), h_measurements.lidar_coords.end());
}

void alloc_init_map_vars(DeviceMap& d_map, HostMap& h_map, HostMap& pre_map) {

    h_map.GRID_WIDTH = pre_map.GRID_WIDTH;
    h_map.GRID_HEIGHT = pre_map.GRID_HEIGHT;
    h_map.xmin = pre_map.xmin;
    h_map.xmax = pre_map.xmax;
    h_map.ymin = pre_map.ymin;
    h_map.ymax = pre_map.ymax;
    h_map.b_should_extend = pre_map.b_should_extend;

    d_map.should_extend.resize(4, 0);
    d_map.grid_map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);
    d_map.grid_map.assign(pre_map.grid_map.begin(), pre_map.grid_map.end());
    d_map.log_odds.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT);
    d_map.log_odds.assign(pre_map.log_odds.begin(), pre_map.log_odds.end());

    h_map.grid_map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);
    h_map.log_odds.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);
    h_map.should_extend.resize(4, 0);
}

void alloc_init_body_lidar(DeviceTransition& d_transition) {

    d_transition.body_lidar.resize(9);
    d_transition.body_lidar.assign(hvec_transition_body_lidar.begin(), hvec_transition_body_lidar.end());
}


#endif
