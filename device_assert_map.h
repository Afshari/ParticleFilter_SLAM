#ifndef _DEVICE_ASSERT_MAP_H_
#define _DEVICE_ASSERT_MAP_H_

#include "headers.h"
#include "structures.h"
#include "host_asserts.h"

void assert_map_extend(HostMap& h_map, HostMap& pre_map, HostMap& post_bg_map,
    HostMap& post_map, bool EXTEND) {

    printf("xmin=%d, xmax=%d, ymin=%d, ymax=%d\n", h_map.xmin, h_map.xmax, h_map.ymin, h_map.ymax);
    printf("xmin=%d, xmax=%d, ymin=%d, ymax=%d\n", post_map.xmin, post_map.xmax, post_map.ymin, post_map.ymax);
    assert(EXTEND == pre_map.b_should_extend);

    if (EXTEND == true) {

        printf("GRID_WIDTH=%d, AF_GRID_WIDTH=%d, GRID_HEIGHT=%d, AF_GRID_HEIGHT=%d\n",
            h_map.GRID_WIDTH, h_map.GRID_HEIGHT, post_map.GRID_WIDTH, post_map.GRID_HEIGHT);
        assert(h_map.GRID_WIDTH == post_map.GRID_WIDTH);
        assert(h_map.GRID_HEIGHT == post_map.GRID_HEIGHT);

        int error_map = 0;
        int error_log = 0;
        for (int i = 0; i < (h_map.GRID_WIDTH * h_map.GRID_HEIGHT); i++) {
            if (h_map.grid_map[i] != post_bg_map.grid_map[i]) {
                error_map += 1;
                //printf("Grid Map: %d <> %d\n", h_map.grid_map[i], bg_grid_map[i]);
            }
            if (abs(h_map.log_odds[i] - post_bg_map.log_odds[i]) > 1e-4) {
                error_log += 1;
                //printf("Log Odds: (%d) %f <> %f\n", i, h_map.log_odds[i], bg_log_odds[i]);
            }
            if (error_log > 200)
                break;
        }
        printf("Map Erros: %d\n", error_map);
        printf("Log Erros: %d\n", error_log);
    }
}

void assert_world_to_image_transform(DeviceParticles& d_particles, DevicePosition& d_position, DeviceTransition& d_transition,
    HostMeasurements& h_measurements, HostParticles& h_particles, HostPosition& h_position, HostTransition& h_transition,
    HostParticles& post_particles, HostPosition& post_position, HostTransition& post_transition) {

    h_transition.world_lidar.assign(d_transition.single_world_lidar.begin(), d_transition.single_world_lidar.end());
    h_particles.particles_occupied_x.assign(d_particles.particles_occupied_x.begin(), d_particles.particles_occupied_x.end());
    h_particles.particles_occupied_y.assign(d_particles.particles_occupied_y.begin(), d_particles.particles_occupied_y.end());
    h_particles.particles_world_x.assign(d_particles.particles_world_x.begin(), d_particles.particles_world_x.end());
    h_particles.particles_world_y.assign(d_particles.particles_world_y.begin(), d_particles.particles_world_y.end());
    h_position.image_body.assign(d_position.image_body.begin(), d_position.image_body.end());

    ASSERT_transition_world_lidar(h_transition.world_lidar.data(), post_transition.world_lidar.data(), 9, false);
    ASSERT_particles_world_frame(h_particles.particles_world_x.data(), h_particles.particles_world_y.data(),
        post_particles.particles_world_x.data(), post_particles.particles_world_y.data(), h_measurements.LIDAR_COORDS_LEN, false);
    ASSERT_processed_measurements(h_particles.particles_occupied_x.data(), h_particles.particles_occupied_y.data(),
        post_particles.particles_occupied_x.data(), post_particles.particles_occupied_y.data(), h_measurements.LIDAR_COORDS_LEN);
    ASSERT_position_image_body(h_position.image_body.data(), post_position.image_body.data(), true, true);
}

void assert_bresenham(DeviceParticles& d_particles,
    HostParticles& h_particles, HostMeasurements& h_measurements, DeviceMeasurements& d_measurements,
    HostParticles& post_particles) {

    h_particles.particles_free_x.resize(h_particles.PARTICLES_FREE_LEN);
    h_particles.particles_free_y.resize(h_particles.PARTICLES_FREE_LEN);

    h_particles.particles_free_x.assign(d_particles.particles_free_x.begin(), d_particles.particles_free_x.end());
    h_particles.particles_free_y.assign(d_particles.particles_free_y.begin(), d_particles.particles_free_y.end());

    printf("~~$ PARTICLES_FREE_LEN = %d\n", h_particles.PARTICLES_FREE_LEN);

    ASSERT_particles_free_index(h_particles.particles_free_counter.data(), post_particles.particles_free_idx.data(),
        h_particles.PARTICLES_OCCUPIED_LEN, false);
    ASSERT_particles_free_new_len(h_particles.PARTICLES_FREE_LEN, post_particles.PARTICLES_FREE_LEN);
    ASSERT_particles_free(h_particles.particles_free_x.data(), h_particles.particles_free_y.data(),
        post_particles.particles_free_x.data(), post_particles.particles_free_y.data(), h_particles.PARTICLES_FREE_LEN);
}

void assert_map_restructure(DeviceParticles& d_particles, HostParticles& h_particles, HostParticles& post_particles) {

    h_particles.particles_occupied_x.assign(d_particles.particles_occupied_x.begin(), d_particles.particles_occupied_x.end());
    h_particles.particles_occupied_y.assign(d_particles.particles_occupied_y.begin(), d_particles.particles_occupied_y.end());
    h_particles.particles_free_x.assign(d_particles.particles_free_x.begin(), d_particles.particles_free_x.end());
    h_particles.particles_free_y.assign(d_particles.particles_free_y.begin(), d_particles.particles_free_y.end());

    printf("\n--> Occupied Unique: %d, %d\n", h_particles.PARTICLES_OCCUPIED_UNIQUE_LEN, post_particles.PARTICLES_OCCUPIED_UNIQUE_LEN);
    assert(h_particles.PARTICLES_OCCUPIED_UNIQUE_LEN == post_particles.PARTICLES_OCCUPIED_UNIQUE_LEN);
    printf("\n--> Free Unique: %d, %d\n", h_particles.PARTICLES_FREE_UNIQUE_LEN, post_particles.PARTICLES_FREE_UNIQUE_LEN);
    assert(h_particles.PARTICLES_FREE_UNIQUE_LEN == post_particles.PARTICLES_FREE_UNIQUE_LEN);

    ASSERT_particles_occupied(h_particles.particles_occupied_x.data(), h_particles.particles_occupied_y.data(),
        post_particles.particles_occupied_unique_x.data(), post_particles.particles_occupied_unique_y.data(),
        "Occupied", h_particles.PARTICLES_OCCUPIED_UNIQUE_LEN, false);
    ASSERT_particles_occupied(h_particles.particles_free_x.data(), h_particles.particles_free_y.data(),
        post_particles.particles_free_unique_x.data(), post_particles.particles_free_unique_y.data(),
        "Free", h_particles.PARTICLES_FREE_UNIQUE_LEN, false);
}

void assert_log_odds(DeviceMap& d_map_data, HostMap& h_map, HostMap& pre_map, HostMap& post_map) {

    thrust::fill(h_map.log_odds.begin(), h_map.log_odds.end(), 0);
    h_map.log_odds.assign(d_map_data.log_odds.begin(), d_map_data.log_odds.end());
    h_map.grid_map.assign(d_map_data.grid_map.begin(), d_map_data.grid_map.end());

    ASSERT_log_odds(h_map.log_odds.data(), pre_map.log_odds.data(), post_map.log_odds.data(),
        (h_map.GRID_WIDTH * h_map.GRID_HEIGHT), (pre_map.GRID_WIDTH * pre_map.GRID_HEIGHT), false);
    ASSERT_log_odds_maps(h_map.grid_map.data(), pre_map.grid_map.data(), post_map.grid_map.data(),
        (h_map.GRID_WIDTH * h_map.GRID_HEIGHT), (pre_map.GRID_WIDTH * pre_map.GRID_HEIGHT), false);
    printf("\n");
}

#endif