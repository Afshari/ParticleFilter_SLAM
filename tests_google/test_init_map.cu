
#include "gtest/gtest.h"
#include "../device_init_map.h"


TEST(DeviceInitMap, AllocInitParticlesVars) {

    DeviceParticles d_particles; 
    HostParticles h_particles;
    HostMeasurements h_measurements; 
    HostParticles pre_particles; 
    int MAX_DIST_IN_MAP = 300;

    pre_particles.OCCUPIED_LEN = 100;

    alloc_init_particles_vars(d_particles, h_particles, h_measurements, pre_particles, MAX_DIST_IN_MAP);

    EXPECT_EQ(h_particles.OCCUPIED_LEN, pre_particles.OCCUPIED_LEN);
    EXPECT_EQ(h_particles.FREE_LEN, 0);

    EXPECT_EQ(d_particles.v_occupied_x.size(), h_measurements.MAX_LEN);
    EXPECT_EQ(d_particles.v_occupied_y.size(), h_measurements.MAX_LEN);
    EXPECT_EQ(d_particles.v_world_x.size(), h_measurements.MAX_LEN);
    EXPECT_EQ(d_particles.v_world_y.size(), h_measurements.MAX_LEN);
    EXPECT_EQ(d_particles.v_free_counter.size(), h_measurements.MAX_LEN);
    EXPECT_EQ(d_particles.v_free_idx.size(), h_measurements.MAX_LEN);

    EXPECT_EQ(d_particles.sv_free_x_max.size(), h_measurements.MAX_LEN * MAX_DIST_IN_MAP);
    EXPECT_EQ(d_particles.sv_free_y_max.size(), h_measurements.MAX_LEN * MAX_DIST_IN_MAP);

    EXPECT_EQ(h_particles.v_occupied_x.size(), h_measurements.MAX_LEN);
    EXPECT_EQ(h_particles.v_occupied_y.size(), h_measurements.MAX_LEN);
    EXPECT_EQ(h_particles.v_world_x.size(), h_measurements.MAX_LEN);
    EXPECT_EQ(h_particles.v_world_y.size(), h_measurements.MAX_LEN);
    EXPECT_EQ(h_particles.v_free_counter.size(), h_measurements.MAX_LEN);

    EXPECT_EQ(h_particles.f_occupied_unique_x.size(), h_particles.MAX_OCCUPIED_UNIQUE_LEN);
    EXPECT_EQ(h_particles.f_occupied_unique_y.size(), h_particles.MAX_OCCUPIED_UNIQUE_LEN);
    EXPECT_EQ(h_particles.f_free_unique_x.size(), h_particles.MAX_FREE_UNIQUE_LEN);
    EXPECT_EQ(h_particles.f_free_unique_y.size(), h_particles.MAX_FREE_UNIQUE_LEN);

    EXPECT_EQ(d_particles.f_free_x.size(), h_particles.MAX_FREE_LEN);
    EXPECT_EQ(d_particles.f_free_y.size(), h_particles.MAX_FREE_LEN);

    EXPECT_EQ(h_particles.f_free_x.size(), h_particles.MAX_FREE_LEN);
    EXPECT_EQ(h_particles.f_free_y.size(), h_particles.MAX_FREE_LEN);

    EXPECT_EQ(d_particles.f_occupied_unique_x.size(), h_particles.MAX_OCCUPIED_UNIQUE_LEN);
    EXPECT_EQ(d_particles.f_occupied_unique_y.size(), h_particles.MAX_OCCUPIED_UNIQUE_LEN);
    EXPECT_EQ(d_particles.f_free_unique_x.size(), h_particles.MAX_FREE_UNIQUE_LEN);
    EXPECT_EQ(d_particles.f_free_unique_y.size(), h_particles.MAX_FREE_UNIQUE_LEN);
}

TEST(DeviceInitMap, AllocInitTransitionVars) {

    DevicePosition d_position; 
    DeviceTransition d_transition;
    HostPosition h_position; 
    HostTransition h_transition;
    HostTransition pre_transition;

    pre_transition.c_world_body.resize(9, 0);

    alloc_init_transition_vars(d_position, d_transition, h_position, h_transition, pre_transition);

    EXPECT_EQ(d_transition.c_world_body.size(), 9);
    EXPECT_EQ(d_transition.c_world_lidar.size(), 9);

    EXPECT_EQ(d_position.c_image_body.size(), 2);

    EXPECT_EQ(h_transition.c_world_lidar.size(), 9);
    EXPECT_EQ(h_position.c_image_body.size(), 2);
}

TEST(DeviceInitMap, AllocInitUniqueMapVars) {

    Device2DUniqueFinder d_unique;
    Host2DUniqueFinder h_unique; 
    HostMap h_map; 
    host_vector<int> hvec_map_idx;

    h_map.GRID_WIDTH = 100;
    h_map.GRID_HEIGHT = 100;
    hvec_map_idx.resize(2, 0);

    alloc_init_unique_map_vars(d_unique, h_unique, h_map, hvec_map_idx);

    EXPECT_EQ(d_unique.s_map.size(), h_map.GRID_WIDTH * h_map.GRID_HEIGHT);
    EXPECT_EQ(d_unique.c_in_map.size(), 1);
    EXPECT_EQ(d_unique.s_in_col.size(), h_map.GRID_WIDTH + 1);
    EXPECT_EQ(d_unique.c_idx.size(), 2);

    EXPECT_EQ(h_unique.s_map.size(), h_map.GRID_WIDTH * h_map.GRID_HEIGHT);
    EXPECT_EQ(h_unique.c_in_map.size(), 1);
    EXPECT_EQ(h_unique.s_in_col.size(), h_map.GRID_WIDTH + 1);
    EXPECT_EQ(h_unique.c_idx.size(), 2);
}
