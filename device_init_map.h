#ifndef _DEVICE_INIT_MAP_H_
#define _DEVICE_INIT_MAP_H_

#include "headers.h"
#include "structures.h"


void alloc_init_particles_vars(DeviceParticles& d_particles, HostParticles& res_particles,
    HostMeasurements& h_measurements, HostParticles& h_particles_data, const int MAX_DIST_IN_MAP) {

    int PARTICLE_UNIQUE_COUNTER = h_particles_data.PARTICLES_OCCUPIED_LEN + 1;

    res_particles.PARTICLES_OCCUPIED_LEN = h_particles_data.PARTICLES_OCCUPIED_LEN;
    res_particles.PARTICLES_FREE_LEN = h_particles_data.PARTICLES_FREE_LEN;

    d_particles.particles_occupied_x.resize(h_measurements.LIDAR_COORDS_LEN);
    d_particles.particles_occupied_y.resize(h_measurements.LIDAR_COORDS_LEN);
    d_particles.particles_world_x.resize(h_measurements.LIDAR_COORDS_LEN);
    d_particles.particles_world_y.resize(h_measurements.LIDAR_COORDS_LEN);
    d_particles.particles_free_x_max.resize(h_particles_data.PARTICLES_OCCUPIED_LEN * MAX_DIST_IN_MAP, 0);
    d_particles.particles_free_y_max.resize(h_particles_data.PARTICLES_OCCUPIED_LEN * MAX_DIST_IN_MAP, 0);
    d_particles.particles_free_counter.resize(PARTICLE_UNIQUE_COUNTER, 0);
    d_particles.particles_free_idx.resize(h_particles_data.PARTICLES_OCCUPIED_LEN);

    res_particles.particles_occupied_x.resize(h_measurements.LIDAR_COORDS_LEN);
    res_particles.particles_occupied_y.resize(h_measurements.LIDAR_COORDS_LEN);
    res_particles.particles_world_x.resize(h_measurements.LIDAR_COORDS_LEN);
    res_particles.particles_world_y.resize(h_measurements.LIDAR_COORDS_LEN);
    res_particles.particles_free_counter.resize(PARTICLE_UNIQUE_COUNTER);
}

void alloc_init_transition_vars(DevicePosition& d_position, DeviceTransition& d_transition,
    HostPosition& res_position, HostTransition& res_transition,
    HostPosition& h_position, HostTransition& h_transition) {

    d_transition.single_world_body.resize(9);
    d_transition.single_world_lidar.resize(9);

    d_transition.single_world_body.assign(h_transition.world_body.begin(), h_transition.world_body.end());

    d_position.image_body.resize(2);

    res_transition.world_lidar.resize(9);
    res_position.image_body.resize(2);
}

void alloc_init_unique_map_vars(Device2DUniqueFinder& d_unique,
    Host2DUniqueFinder& res_unique, HostMap& res_map, const host_vector<int>& hvec_map_idx) {

    d_unique.map.resize(res_map.GRID_WIDTH * res_map.GRID_HEIGHT);
    d_unique.in_map.resize(1);
    d_unique.in_col.resize(res_map.GRID_WIDTH + 1);
    d_unique.idx.resize(2);

    d_unique.idx.assign(hvec_map_idx.begin(), hvec_map_idx.end());

    res_unique.in_map.resize(1);
    res_unique.in_col.resize(res_map.GRID_WIDTH + 1);
    res_unique.idx.resize(2);
}

#endif
