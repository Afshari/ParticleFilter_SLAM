#ifndef _DEVICE_INIT_MAP_H_
#define _DEVICE_INIT_MAP_H_

#include "headers.h"
#include "structures.h"

void alloc_init_transition_vars(DevicePositionTransition& d_position_transition, HostPositionTransition& res_position_transition,
    HostPositionTransition& h_position_transition) {

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

    res_map_data.GRID_WIDTH = h_map_data.GRID_WIDTH;
    res_map_data.GRID_HEIGHT = h_map_data.GRID_HEIGHT;
    res_map_data.xmin = h_map_data.xmin;
    res_map_data.xmax = h_map_data.xmax;
    res_map_data.ymin = h_map_data.ymin;
    res_map_data.ymax = h_map_data.ymax;
    res_map_data.b_should_extend = h_map_data.b_should_extend;

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



#endif
