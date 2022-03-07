#pragma once

#include "headers.h"


void ASSERT_transition_frames(float* res_transition_body_frame, float* res_transition_world_frame, 
								float* h_transition_body_frame, float* h_transition_world_frame, const int _NUM_PARTICLES, bool printVerbose) {

    printf("\n--> Start Checking Body Frame\n");
    for (int i = 0; i < 9 * _NUM_PARTICLES; i++) {
        if (printVerbose == true) printf("%f, %f | ", res_transition_body_frame[i], h_transition_body_frame[i]);
        assert(abs(res_transition_body_frame[i] - h_transition_body_frame[i]) < 1e-5);
    }
    printf("--> Start Checking World Frame\n");
    for (int i = 0; i < 9 * _NUM_PARTICLES; i++) {
        if (printVerbose == true) printf("%f, %f |  ", res_transition_world_frame[i], h_transition_world_frame[i]);
        assert(abs(res_transition_world_frame[i] - h_transition_world_frame[i]) < 1e-5);
    }
    printf("--> All Body Frame & World Frame Passed\n\n");

}


void ASSERT_processed_measurements(int* res_processed_measure_x, int* res_processed_measure_y, float* processed_measure, const int _NUM_PARTICLES, const int _LIDAR_COORDS_LEN) {

    printf("--> Measurement Check\n");
    int notEqualCounter = 0;
    int allItems = 0;
    for (int i = 0; i < _NUM_PARTICLES; i++) {
        int h_idx = 2 * i * _LIDAR_COORDS_LEN;
        int res_idx = i * _LIDAR_COORDS_LEN;
        for (int j = 0; j < _LIDAR_COORDS_LEN; j++) {
            if (abs(res_processed_measure_x[j + res_idx] - processed_measure[j + h_idx]) > 1e-5) {
                printf("(x) index=%d, result=%d, expect=%d  |  ", (j + res_idx), res_processed_measure_x[j + res_idx], (int)processed_measure[j + h_idx]);
                notEqualCounter += 1;
                if (notEqualCounter > 50)
                    exit(-1);
            }
            allItems += 1;
        }
        h_idx += _LIDAR_COORDS_LEN;
        for (int j = 0; j < _LIDAR_COORDS_LEN; j++) {
            if (abs(res_processed_measure_y[j + res_idx] - processed_measure[j + h_idx]) > 1e-5) {
                printf("(y) index=%d, result=%d, expect=%d  |  ", (j + res_idx), res_processed_measure_y[j + res_idx], (int)processed_measure[j + h_idx]);
                notEqualCounter += 1;
                if (notEqualCounter > 50)
                    exit(-1);
            }
            allItems += 1;
        }
    }
    printf("--> Processed Measure Error Count: %d of Items: %d\n\n", notEqualCounter, allItems);
}


void ASSERT_create_2d_map_elements(uint8_t* res_map_2d, const int negative_before_counter, const int _GRID_WIDTH, const int _GRID_HEIGHT, const int _NUM_PARTICLES, const int _ELEMS_PARTICLES_START) {

    int nonZeroCounter = 0;
    for (int i = 0; i < _GRID_WIDTH * _GRID_HEIGHT * _NUM_PARTICLES; i++) {
        if (res_map_2d[i] == 1)
            nonZeroCounter += 1;
    }
    printf("\n--> Non Zero: %d, ELEMS_PARTICLES_START=%d\n", nonZeroCounter, _ELEMS_PARTICLES_START);
    printf("--> diff=%d, negative_before_counter=%d\n\n", (_ELEMS_PARTICLES_START - nonZeroCounter), negative_before_counter);
    assert(nonZeroCounter + negative_before_counter == _ELEMS_PARTICLES_START);
}


void ASSERT_particles_pos_unique(int* res_particles_x, int* res_particles_y, int* h_particles_x_after_unique, int* h_particles_y_after_unique, const int LEN) {

    for (int i = 0, j = 0; i < LEN; i++) {

        if (h_particles_x_after_unique[i] > 0 && h_particles_y_after_unique[i] > 0) {

            assert(h_particles_x_after_unique[i] == res_particles_x[j]);
            assert(h_particles_y_after_unique[i] == res_particles_y[j]);
            j += 1;
        }
    }
    printf("\n--> Particles Pose (x & y) are OK\n\n");
}


void ASSERT_new_len_calculation(const int NEW_LEN, const int _ELEMS_PARTICLES_AFTER, const int negative_after_counter) {

    printf("--> NEW_LEN: %d == %d, diff=%d\n\n", NEW_LEN, _ELEMS_PARTICLES_AFTER, (_ELEMS_PARTICLES_AFTER - NEW_LEN));
    assert(NEW_LEN + negative_after_counter == _ELEMS_PARTICLES_AFTER);
}


void ASSERT_correlation_Equality(float* res_correlation, float* h_correlation, const int LEN) {

    bool all_equal = true;
    for (int i = 0; i < LEN; i++) {
        if (res_correlation[i] != h_correlation[i]) {
            all_equal = false;
            printf("index: %d --> %f, %f --> %s\n", i, res_correlation[i], h_correlation[i], (res_correlation[i] == h_correlation[i] ? "Equal" : ""));
        }
    }
    printf("\n--> Correlation All Equal: %s\n\n", all_equal ? "true" : "false");

}

void ASSERT_correlation_Equality(int* res_correlation, float* h_correlation, const int LEN) {

    bool all_equal = true;
    for (int i = 0; i < LEN; i++) {
        if (res_correlation[i] != h_correlation[i]) {
            all_equal = false;
            printf("index: %d --> %d, %f --> %s\n", i, res_correlation[i], h_correlation[i], (res_correlation[i] == h_correlation[i] ? "Equal" : ""));
        }
    }
    printf("\n--> Correlation All Equal: %s\n\n", all_equal ? "true" : "false");

}

void ASSERT_update_particle_weights(float* res_weights, float* h_weights, const int LEN, bool printVerbose) {

    for (int i = 0; i < LEN; i++) {
        float diff = abs(res_weights[i] - h_weights[i]);
        if(printVerbose == true) printf("%f <> %f, diff=%f\n", res_weights[i], h_weights[i], diff);
        assert(diff < 1e-4);
    }
    printf("\n--> Update Particle Weights Passed\n");
}


void ASSERT_resampling_indices(int* res_js, int* h_js, const int LEN, bool printVerbose) {

    for (int i = 0; i < LEN; i++) {
        if(printVerbose == true) printf("%d, %d | ", res_js[i], h_js[i]);
        assert(res_js[i] == h_js[i]);
    }
    printf("\n--> Resampling Indices All Passed\n\n");

}

void ASSERT_resampling_states(float* x, float* y, float* theta, float* x_updated, float* y_updated, float* theta_updated, int* res_js, const int LEN, bool printVerbose) {

    for (int i = 0; i < NUM_PARTICLES; i++) {
        int j = res_js[i];
        assert(x[j] == x_updated[i]);
        assert(y[j] == y_updated[i]);
        assert(theta[j] == theta_updated[i]);
        if(printVerbose == true) printf("x=%f <> %f, y=%f <> %f\n", x[j], x_updated[i], y[j], y_updated[i]);
    }
    printf("\n--> Resampling States All Passed\n\n");
}

void ASSERT_resampling_particles_index(int * h_particles_idx, int* res_particles_idx, const int LEN, bool printVerbose, int negative_particles) {

    if (negative_particles == 0) {
        for (int i = 0; i < LEN; i++) {

            if (printVerbose == true) printf("%d <> %d\n", h_particles_idx[i], res_particles_idx[i]);
            assert(h_particles_idx[i] == res_particles_idx[i]);
        }
        printf("\n--> Resampling Particles Index All Passed\n\n");
    }
    else {
        printf("\n--> Ignore Assert because of 'Negative Particles'\n\n");
    }

}



