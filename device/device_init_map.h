#ifndef _DEVICE_INIT_MAP_H_
#define _DEVICE_INIT_MAP_H_

#include "headers.h"
#include "structures.h"


void alloc_init_particles_vars(DeviceParticles& d_particles, HostParticles& h_particles,
    HostMeasurements& h_measurements, HostParticles& pre_particles, const int MAX_DIST_IN_MAP) {

    int PARTICLE_UNIQUE_COUNTER = pre_particles.OCCUPIED_LEN + 1;

    h_particles.OCCUPIED_LEN = pre_particles.OCCUPIED_LEN;
    h_particles.FREE_LEN = 0; //pre_particles.FREE_LEN;

    d_particles.v_occupied_x.resize(h_measurements.MAX_LEN, 0);
    d_particles.v_occupied_y.resize(h_measurements.MAX_LEN, 0);
    d_particles.v_world_x.resize(h_measurements.MAX_LEN, 0);
    d_particles.v_world_y.resize(h_measurements.MAX_LEN, 0);
    d_particles.v_free_counter.resize(h_measurements.MAX_LEN, 0); // PARTICLE_UNIQUE_COUNTER = pre_particles.OCCUPIED_LEN + 1;
    d_particles.v_free_idx.resize(h_measurements.MAX_LEN, 0);

    d_particles.sv_free_x_max.resize(h_measurements.MAX_LEN * MAX_DIST_IN_MAP, 0);
    d_particles.sv_free_y_max.resize(h_measurements.MAX_LEN * MAX_DIST_IN_MAP, 0);

    h_particles.v_occupied_x.resize(h_measurements.MAX_LEN, 0);
    h_particles.v_occupied_y.resize(h_measurements.MAX_LEN, 0);
    h_particles.v_world_x.resize(h_measurements.MAX_LEN, 0);
    h_particles.v_world_y.resize(h_measurements.MAX_LEN, 0);
    h_particles.v_free_counter.resize(h_measurements.MAX_LEN, 0);

    h_particles.f_occupied_unique_x.resize(h_particles.MAX_OCCUPIED_UNIQUE_LEN, 0);
    h_particles.f_occupied_unique_y.resize(h_particles.MAX_OCCUPIED_UNIQUE_LEN, 0);
    h_particles.f_free_unique_x.resize(h_particles.MAX_FREE_UNIQUE_LEN, 0);
    h_particles.f_free_unique_y.resize(h_particles.MAX_FREE_UNIQUE_LEN, 0);

    d_particles.f_free_x.resize(h_particles.MAX_FREE_LEN, 0);
    d_particles.f_free_y.resize(h_particles.MAX_FREE_LEN, 0);
   
    h_particles.f_free_x.resize(h_particles.MAX_FREE_LEN, 0);
    h_particles.f_free_y.resize(h_particles.MAX_FREE_LEN, 0);

    d_particles.f_occupied_unique_x.resize(h_particles.MAX_OCCUPIED_UNIQUE_LEN, 0);
    d_particles.f_occupied_unique_y.resize(h_particles.MAX_OCCUPIED_UNIQUE_LEN, 0);
    d_particles.f_free_unique_x.resize(h_particles.MAX_FREE_UNIQUE_LEN, 0);
    d_particles.f_free_unique_y.resize(h_particles.MAX_FREE_UNIQUE_LEN, 0);
}

void alloc_init_transition_vars(DevicePosition& d_position, DeviceTransition& d_transition,
    HostPosition& h_position, HostTransition& h_transition,
    HostTransition& pre_transition) {

    d_transition.c_world_body.resize(9);
    d_transition.c_world_lidar.resize(9);

    d_transition.c_world_body.assign(pre_transition.c_world_body.begin(), pre_transition.c_world_body.end());

    d_position.c_image_body.resize(2);

    h_transition.c_world_lidar.resize(9);
    h_position.c_image_body.resize(2);
}

void alloc_init_unique_map_vars(Device2DUniqueFinder& d_unique,
    Host2DUniqueFinder& h_unique, HostMap& h_map, const host_vector<int>& hvec_map_idx) {

    d_unique.s_map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);
    d_unique.c_in_map.resize(1, 0);
    d_unique.s_in_col.resize(h_map.GRID_WIDTH + 1, 0);
    d_unique.c_idx.resize(2, 0);

    d_unique.c_idx.assign(hvec_map_idx.begin(), hvec_map_idx.end());

    h_unique.s_map.resize(h_map.GRID_WIDTH * h_map.GRID_HEIGHT, 0);
    h_unique.c_in_map.resize(1, 0);
    h_unique.s_in_col.resize(h_map.GRID_WIDTH + 1, 0);
    h_unique.c_idx.resize(2, 0);
}

#endif
