#ifndef _DEVICE_ASSERT_MAP_H_
#define _DEVICE_ASSERT_MAP_H_

#include "headers.h"
#include "structures.h"
#include "host_asserts.h"

void assert_map_extend(HostMapData& res_map_data, HostMapData& h_map_data, HostMapData& h_map_data_bg,
    HostMapData& h_map_data_post, bool EXTEND) {

    printf("xmin=%d, xmax=%d, ymin=%d, ymax=%d\n", res_map_data.xmin, res_map_data.xmax, res_map_data.ymin, res_map_data.ymax);
    printf("xmin=%d, xmax=%d, ymin=%d, ymax=%d\n", h_map_data_post.xmin, h_map_data_post.xmax, h_map_data_post.ymin, h_map_data_post.ymax);
    assert(EXTEND == h_map_data.b_should_extend);

    if (EXTEND == true) {

        printf("GRID_WIDTH=%d, AF_GRID_WIDTH=%d, GRID_HEIGHT=%d, AF_GRID_HEIGHT=%d\n",
            res_map_data.GRID_WIDTH, res_map_data.GRID_HEIGHT, h_map_data_post.GRID_WIDTH, h_map_data_post.GRID_HEIGHT);
        assert(res_map_data.GRID_WIDTH == h_map_data_post.GRID_WIDTH);
        assert(res_map_data.GRID_HEIGHT == h_map_data_post.GRID_HEIGHT);

        int error_map = 0;
        int error_log = 0;
        for (int i = 0; i < (res_map_data.GRID_WIDTH * res_map_data.GRID_HEIGHT); i++) {
            if (res_map_data.grid_map[i] != h_map_data_bg.grid_map[i]) {
                error_map += 1;
                //printf("Grid Map: %d <> %d\n", res_map_data.grid_map[i], bg_grid_map[i]);
            }
            if (abs(res_map_data.log_odds[i] - h_map_data_bg.log_odds[i]) > 1e-4) {
                error_log += 1;
                //printf("Log Odds: (%d) %f <> %f\n", i, res_map_data.log_odds[i], bg_log_odds[i]);
            }
            if (error_log > 200)
                break;
        }
        printf("Map Erros: %d\n", error_map);
        printf("Log Erros: %d\n", error_log);
    }
}

void assert_world_to_image_transform(DeviceParticlesData& d_particles_data, DevicePositionTransition& d_position_transition,
    HostMeasurements& res_measurements, HostParticlesData& res_particles_data, HostPositionTransition& res_position_transition,
    HostParticlesData& h_particles_data, HostPositionTransition& h_position_transition) {

    res_position_transition.transition_world_lidar.assign(d_position_transition.transition_single_world_lidar.begin(),
        d_position_transition.transition_single_world_lidar.end());
    res_particles_data.particles_occupied_x.assign(d_particles_data.particles_occupied_x.begin(), d_particles_data.particles_occupied_x.end());
    res_particles_data.particles_occupied_y.assign(d_particles_data.particles_occupied_y.begin(), d_particles_data.particles_occupied_y.end());
    res_particles_data.particles_world_x.assign(d_particles_data.particles_world_x.begin(), d_particles_data.particles_world_x.end());
    res_particles_data.particles_world_y.assign(d_particles_data.particles_world_y.begin(), d_particles_data.particles_world_y.end());
    res_position_transition.position_image_body.assign(d_position_transition.position_image_body.begin(),
        d_position_transition.position_image_body.end());

    ASSERT_transition_world_lidar(res_position_transition.transition_world_lidar.data(), h_position_transition.transition_world_lidar.data(), 9, false);
    ASSERT_particles_world_frame(res_particles_data.particles_world_x.data(), res_particles_data.particles_world_y.data(),
        h_particles_data.particles_world_x.data(), h_particles_data.particles_world_y.data(), res_measurements.LIDAR_COORDS_LEN, false);
    ASSERT_processed_measurements(res_particles_data.particles_occupied_x.data(), res_particles_data.particles_occupied_y.data(),
        h_particles_data.particles_occupied_x.data(), h_particles_data.particles_occupied_y.data(), res_measurements.LIDAR_COORDS_LEN);
    ASSERT_position_image_body(res_position_transition.position_image_body.data(), h_position_transition.position_image_body.data(), true, true);
}

void assert_bresenham(DeviceParticlesData& d_particles_data,
    HostParticlesData& res_particles_data, HostMeasurements& res_measurements, DeviceMeasurements& d_measurements,
    HostParticlesData& h_particles_data) {

    res_particles_data.particles_free_x.resize(res_particles_data.PARTICLES_FREE_LEN);
    res_particles_data.particles_free_y.resize(res_particles_data.PARTICLES_FREE_LEN);

    res_particles_data.particles_free_x.assign(d_particles_data.particles_free_x.begin(), d_particles_data.particles_free_x.end());
    res_particles_data.particles_free_y.assign(d_particles_data.particles_free_y.begin(), d_particles_data.particles_free_y.end());

    printf("~~$ PARTICLES_FREE_LEN = %d\n", res_particles_data.PARTICLES_FREE_LEN);

    ASSERT_particles_free_index(res_particles_data.particles_free_counter.data(), h_particles_data.particles_free_idx.data(),
        res_particles_data.PARTICLES_OCCUPIED_LEN, false);
    ASSERT_particles_free_new_len(res_particles_data.PARTICLES_FREE_LEN, h_particles_data.PARTICLES_FREE_LEN);
    ASSERT_particles_free(res_particles_data.particles_free_x.data(), res_particles_data.particles_free_y.data(),
        h_particles_data.particles_free_x.data(), h_particles_data.particles_free_y.data(), res_particles_data.PARTICLES_FREE_LEN);

}

void assert_map_restructure(DeviceParticlesData& d_particles_data, HostParticlesData& res_particles_data, HostParticlesData& h_particles_data) {

    res_particles_data.particles_occupied_x.assign(d_particles_data.particles_occupied_x.begin(), d_particles_data.particles_occupied_x.end());
    res_particles_data.particles_occupied_y.assign(d_particles_data.particles_occupied_y.begin(), d_particles_data.particles_occupied_y.end());
    res_particles_data.particles_free_x.assign(d_particles_data.particles_free_x.begin(), d_particles_data.particles_free_x.end());
    res_particles_data.particles_free_y.assign(d_particles_data.particles_free_y.begin(), d_particles_data.particles_free_y.end());

    printf("\n--> Occupied Unique: %d, %d\n", res_particles_data.PARTICLES_OCCUPIED_UNIQUE_LEN, h_particles_data.PARTICLES_OCCUPIED_UNIQUE_LEN);
    assert(res_particles_data.PARTICLES_OCCUPIED_UNIQUE_LEN == h_particles_data.PARTICLES_OCCUPIED_UNIQUE_LEN);
    printf("\n--> Free Unique: %d, %d\n", res_particles_data.PARTICLES_FREE_UNIQUE_LEN, h_particles_data.PARTICLES_FREE_UNIQUE_LEN);
    assert(res_particles_data.PARTICLES_FREE_UNIQUE_LEN == h_particles_data.PARTICLES_FREE_UNIQUE_LEN);

    ASSERT_particles_occupied(res_particles_data.particles_occupied_x.data(), res_particles_data.particles_occupied_y.data(),
        h_particles_data.particles_occupied_unique_x.data(), h_particles_data.particles_occupied_unique_y.data(),
        "Occupied", res_particles_data.PARTICLES_OCCUPIED_UNIQUE_LEN, false);
    ASSERT_particles_occupied(res_particles_data.particles_free_x.data(), res_particles_data.particles_free_y.data(),
        h_particles_data.particles_free_unique_x.data(), h_particles_data.particles_free_unique_y.data(),
        "Free", res_particles_data.PARTICLES_FREE_UNIQUE_LEN, false);
}

void assert_log_odds(DeviceMapData& d_map_data, HostMapData& res_map_data, HostMapData& h_map_data, HostMapData& h_map_data_post) {

    thrust::fill(res_map_data.log_odds.begin(), res_map_data.log_odds.end(), 0);
    res_map_data.log_odds.assign(d_map_data.log_odds.begin(), d_map_data.log_odds.end());
    res_map_data.grid_map.assign(d_map_data.grid_map.begin(), d_map_data.grid_map.end());

    ASSERT_log_odds(res_map_data.log_odds.data(), h_map_data.log_odds.data(), h_map_data_post.log_odds.data(),
        (res_map_data.GRID_WIDTH * res_map_data.GRID_HEIGHT), (h_map_data.GRID_WIDTH * h_map_data.GRID_HEIGHT), false);
    ASSERT_log_odds_maps(res_map_data.grid_map.data(), h_map_data.grid_map.data(), h_map_data_post.grid_map.data(),
        (res_map_data.GRID_WIDTH * res_map_data.GRID_HEIGHT), (h_map_data.GRID_WIDTH * h_map_data.GRID_HEIGHT), false);
    printf("\n");
}

#endif