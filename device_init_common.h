#ifndef _DEVICE_INIT_COMMON_H_
#define _DEVICE_INIT_COMMON_H_

#include "structures.h"

void alloc_init_measurement_vars(DeviceMeasurements& d_measurements, HostMeasurements& res_measurements, HostMeasurements& h_measurements) {

    res_measurements.LIDAR_COORDS_LEN = h_measurements.LIDAR_COORDS_LEN;

    d_measurements.lidar_coords.resize(2 * res_measurements.LIDAR_COORDS_LEN);
    d_measurements.lidar_coords.assign(h_measurements.lidar_coords.begin(), h_measurements.lidar_coords.end());
}

void alloc_init_map_vars(DeviceMap& d_map, HostMap& res_map, HostMap& h_map) {

    res_map.GRID_WIDTH = h_map.GRID_WIDTH;
    res_map.GRID_HEIGHT = h_map.GRID_HEIGHT;
    res_map.xmin = h_map.xmin;
    res_map.xmax = h_map.xmax;
    res_map.ymin = h_map.ymin;
    res_map.ymax = h_map.ymax;
    res_map.b_should_extend = h_map.b_should_extend;

    d_map.should_extend.resize(4, 0);
    d_map.grid_map.resize(res_map.GRID_WIDTH * res_map.GRID_HEIGHT, 0);
    d_map.grid_map.assign(h_map.grid_map.begin(), h_map.grid_map.end());
    d_map.log_odds.resize(res_map.GRID_WIDTH * res_map.GRID_HEIGHT);
    d_map.log_odds.assign(h_map.log_odds.begin(), h_map.log_odds.end());

    res_map.grid_map.resize(res_map.GRID_WIDTH * res_map.GRID_HEIGHT, 0);
    res_map.log_odds.resize(res_map.GRID_WIDTH * res_map.GRID_HEIGHT, 0);
    res_map.should_extend.resize(4, 0);
}

void alloc_init_body_lidar(DeviceTransition& d_transition) {

    d_transition.body_lidar.resize(9);
    d_transition.body_lidar.assign(hvec_transition_body_lidar.begin(), hvec_transition_body_lidar.end());
}


#endif
