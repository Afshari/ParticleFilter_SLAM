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

void ASSERT_processed_measurements(int* res_processed_measure_x, int* res_processed_measure_y, int* h_processed_measure_x,
    int* h_processed_measure_y, const int LEN) {

    int notEqualCounter = 0;
    for (int i = 0; i < LEN; i++) {
        if (res_processed_measure_x[i] != h_processed_measure_x[i]) {
            notEqualCounter += 1;
            printf("i=%d x --> %d <> %d\n", i, res_processed_measure_x[i], h_processed_measure_x[i]);
        }
        if (res_processed_measure_y[i] != h_processed_measure_y[i]) {
            notEqualCounter += 1;
            printf("i=%d y --> %d <> %d\n", i, res_processed_measure_y[i], h_processed_measure_y[i]);
        }
    }
    printf("\n--> Processed Measure Error Count: %d of Items: %d\n\n", notEqualCounter, (2 * LEN));
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


void ASSERT_rearrange_particles_states(int* res_particles_x, int* res_particles_y, float* res_states_x, float* res_states_y, float* res_states_theta,
    int* h_particles_x, int* h_particles_y, float* h_states_x, float* h_states_y, float* h_states_theta, const int PARTICLES_LEN, const int STATES_LEN) {

    for (int i = 0, j = 0; i < PARTICLES_LEN; i++) {
        if (h_particles_x[i] < 0 || h_particles_y[i] < 0)
            continue;

        if (res_particles_x[j] != h_particles_x[i])
            printf("i=%d --> %d, %d\n", i, res_particles_x[j], h_particles_x[i]);
        if (res_particles_y[j] != h_particles_y[i])
            printf("i=%d --> %d, %d\n", i, res_particles_y[j], h_particles_y[i]);
        assert(res_particles_x[j] == h_particles_x[i]);
        assert(res_particles_y[j] == h_particles_y[i]);

        j += 1;
    }

    for(int i = 0; i < STATES_LEN; i++) {
        assert(res_states_x[i] == h_states_x[i]);
        assert(res_states_y[i] == h_states_y[i]);
        assert(res_states_theta[i] == h_states_theta[i]);
    }
}

void ASSERT_log_odds(float* res_log_odds, float* pre_log_odds, float* post_log_odds, const int LEN) {

    int numError = 0;
    int numCorrect = 0;
    for (int i = 0; i < LEN; i++) {

        if (abs(res_log_odds[i] - post_log_odds[i]) > 0.01) {
            printf("%d: %f, %f, %f\n", i, res_log_odds[i], post_log_odds[i], pre_log_odds[i]);
            numError += 1;
        }
        else if (post_log_odds[i] != pre_log_odds[i]) {
            numCorrect += 1;
        }
    }
    printf("\n--> Log-Odds > Error: %d, Correct: %d\n", numError, numCorrect);
}

void ASSERT_log_odds_maps(int* res_grid_map, int* pre_grid_map, int* post_grid_map, const int LEN) {

    int numError = 0;
    int numCorrect = 0;
    for (int i = 0; i < LEN; i++) {

        if (abs(res_grid_map[i] - post_grid_map[i]) > 0.1) {
            printf("%d: %d, %d, %d\n", i, res_grid_map[i], pre_grid_map[i], post_grid_map[i]);
            numError += 1;
        }
        else {
            numCorrect += 1;
        }
    }
    printf("\n--> Log_Odds MAP --> Error: %d, Correct: %d\n", numError, numCorrect);
}

void ASSERT_particles_occupied(int* res_particles_x, int* res_particles_y, int* h_particles_x, int* h_particles_y, 
    const char* particle_type, const int LEN, bool printVerbose) {

    for (int i = 0; i < LEN; i++) {
        if(printVerbose == true)
            printf("%d <> %d,  %d <> %d\n", res_particles_x[i], h_particles_x[i], res_particles_y[i], h_particles_y[i]);

        assert(res_particles_x[i] == h_particles_x[i]);
        assert(res_particles_y[i] == h_particles_y[i]);
    }
    printf("\n--> All Unique %s Calculation Passed\n", particle_type);
}

void ASSERT_transition_world_lidar(float* res_transition_world_lidar, float* h_transition_world_lidar, const int LEN, bool printVerbose) {

    for (int i = 0; i < LEN; i++) {
        if(printVerbose == true) printf("%f (%f) ", res_transition_world_lidar[i], h_transition_world_lidar[i]);
        assert(abs(res_transition_world_lidar[i] - h_transition_world_lidar[i]) < 1e-4);
    }
    printf("\n--> Transition World Lidar All Correct\n\n");

}


void ASSERT_particles_wframe(float* res_particles_wframe_x, float* res_particles_wframe_y, float* h_particles_wframe_x, float* h_particles_wframe_y,
    const int LEN, bool printVerbose) {

    for (int i = 0; i < LEN; i++) {
        if(printVerbose == true) printf("i=%d --> %f <> %f, %f <> %f\n", i, res_particles_wframe_x[i], h_particles_wframe_x[i], res_particles_wframe_y[i], h_particles_wframe_y[i]);
        assert(abs(res_particles_wframe_x[i] - h_particles_wframe_x[i]) < 1e-3);
        assert(abs(res_particles_wframe_y[i] - h_particles_wframe_y[i]) < 1e-3);
    }
    printf("\n--> Particles World Frame All Correct\n");
}

void ASSERT_position_image_body(int* res_position_image_body, int* h_position_image_body) {

    printf("\n--> Position Image Body >> x: %d <> %d , y: %d <> %d\n", 
        res_position_image_body[0], h_position_image_body[0], res_position_image_body[1], h_position_image_body[1]);
    assert(res_position_image_body[0] == h_position_image_body[0]);
    assert(res_position_image_body[1] == h_position_image_body[1]);
}

void ASSERT_particles_free_index(int* res_particles_free_counter, int* h_particles_free_idx, int LEN, bool printVerbose) {

    for (int i = 0; i < LEN; i++) {
        assert(res_particles_free_counter[i] == h_particles_free_idx[i]);
        if(printVerbose == true) printf("i=%d --> %d <> %d\n", i, res_particles_free_counter[i], h_particles_free_idx[i]);
    }
    printf("\n--> Particles Free Index All Correct\n");
}

void ASSERT_particles_free_new_len(const int PARTICLES_NEW_LEN, const int PARTICLES_FREE_LEN) {

    printf("\nPARTICLES_NEW_LEN=%d <> PARTICLES_FREE_LEN=%d\n", PARTICLES_NEW_LEN, PARTICLES_FREE_LEN);
    assert(PARTICLES_NEW_LEN == PARTICLES_FREE_LEN);
    printf("--> Particles New Length All Correct\n\n");
}

void ASSERT_particles_free(int* res_particles_free_x, int* res_particles_free_y, int* h_particles_free_x, int* h_particles_free_y, const int LEN) {

    bool all_equal = true;
    int errors = 0;
    for (int i = 0; i < LEN; i++) {
        if (res_particles_free_x[i] != h_particles_free_x[i] || res_particles_free_y[i] != h_particles_free_y[i]) {
            all_equal = false;
            errors += 1;
            printf("%d -- %d, %d | %d, %d\n", i, res_particles_free_x[i], h_particles_free_x[i], res_particles_free_y[i], h_particles_free_y[i]);
        }
    }
    printf("-->Free Particles Calculation -> All Equal: %s\n", all_equal ? "true" : "false");
    printf("-->Errors: %d\n\n", errors);

}

