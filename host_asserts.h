#ifndef _HOST_ASSERT_H_
#define _HOST_ASSERT_H_

#include "headers.h"


void ASSERT_transition_frames(float* res_transition_world_body, float* res_transition_world_lidar, 							
    float* h_transition_world_body, float* h_transition_world_lidar, 
    const int LEN, bool printVerbose, bool start_new_line=false, bool end_new_line=false) {
    
    if (start_new_line == true) printf("\n");
    printf("--> Start Checking World Body Transition\n");
    for (int i = 0; i < 9 * LEN; i++) {
        if (printVerbose == true) printf("%f, %f | ", res_transition_world_body[i], h_transition_world_body[i]);
        assert(abs(res_transition_world_body[i] - h_transition_world_body[i]) < 1e-2);
    }
    printf("--> Start Checking World Lidar Transition\n");
    for (int i = 0; i < 9 * LEN; i++) {
        if (printVerbose == true) printf("%f, %f |  ", res_transition_world_lidar[i], h_transition_world_lidar[i]);
        assert(abs(res_transition_world_lidar[i] - h_transition_world_lidar[i]) < 1e-2);
    }
    printf("--> All Body Frame & World Frame Passed\n");
    if (end_new_line == true) printf("\n");
}


void ASSERT_processed_measurements(int* res_processed_measure_x, int* res_processed_measure_y, int* processed_measure, 
    const int LEN, const int LIDAR_COORDS_LEN, bool start_new_line = true, bool end_new_line = false) {

    if (start_new_line == true) printf("\n");
    printf("--> Measurement Check\n");
    int notEqualCounter = 0;
    int allItems = 0;
    for (int i = 0; i < LEN; i++) {
        int h_idx = 2 * i * LIDAR_COORDS_LEN;
        int res_idx = i * LIDAR_COORDS_LEN;
        for (int j = 0; j < LIDAR_COORDS_LEN; j++) {
            if (abs(res_processed_measure_x[j + res_idx] - processed_measure[j + h_idx]) > 1e-5) {
                printf("(x) index=%d, result=%d, expect=%d  |  ", (j + res_idx), res_processed_measure_x[j + res_idx], (int)processed_measure[j + h_idx]);
                notEqualCounter += 1;
                if (notEqualCounter > 50)
                    exit(-1);
            }
            allItems += 1;
        }
        h_idx += LIDAR_COORDS_LEN;
        for (int j = 0; j < LIDAR_COORDS_LEN; j++) {
            if (abs(res_processed_measure_y[j + res_idx] - processed_measure[j + h_idx]) > 1e-5) {
                printf("(y) index=%d, result=%d, expect=%d  |  ", (j + res_idx), res_processed_measure_y[j + res_idx], (int)processed_measure[j + h_idx]);
                notEqualCounter += 1;
                if (notEqualCounter > 50)
                    exit(-1);
            }
            allItems += 1;
        }
    }
    printf("--> Processed Measure Error Count: %d of Items: %d\n", notEqualCounter, allItems);
    if (end_new_line == true) printf("\n");
}

void ASSERT_processed_measurements(int* res_processed_measure_x, int* res_processed_measure_y,
    int* h_processed_measure_x, int* h_processed_measure_y, const int LEN, bool start_new_line = true, bool end_new_line = false) {

    if (start_new_line == true) printf("\n");
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
    printf("--> Processed Measure Error Count: %d of Items: %d\n", notEqualCounter, (2 * LEN));
    if (end_new_line == true) printf("\n");
}

void ASSERT_processed_measurements(int* res_processed_measure_x, int* res_processed_measure_y, int* res_processed_measure_idx, 
    int* h_processed_measure_x, int* h_processed_measure_y, const int LEN, const int LIDAR_COORDS_LEN, bool start_new_line = true, bool end_new_line = false) {

    if (start_new_line == true) printf("\n");
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
    printf("--> Processed Measure Error Count: %d of Items: %d\n", notEqualCounter, (2 * LEN));
    for (int i = 0; i < NUM_PARTICLES; i++) {
        int diff = (i == 0) ? 0 : (res_processed_measure_idx[i] - res_processed_measure_idx[i - 1]);
        // if (printVerbose == true) printf("index %d --> value: %d, diff: %d\n", i, res_processed_measure_idx[i], diff);
        if (i > 0) assert(diff == LIDAR_COORDS_LEN);
    }
    printf("--> Processed Measure Index Passed \n");
    if (end_new_line == true) printf("\n");
}

void ASSERT_create_2d_map_elements(uint8_t* res_map_2d, const int negative_before_counter, 
    const int GRID_WIDTH, const int GRID_HEIGHT, const int _NUM_PARTICLES, const int ELEMS_PARTICLES_START, bool start_new_line = false, bool end_new_line = false) {

    if (start_new_line == true) printf("\n");
    int nonZeroCounter = 0;
    for (int i = 0; i < GRID_WIDTH * GRID_HEIGHT * _NUM_PARTICLES; i++) {
        if (res_map_2d[i] == 1)
            nonZeroCounter += 1;
    }
    printf("~~$ Non Zero: \t\t\t%d\n", nonZeroCounter);
    printf("~~$ ELEMS_PARTICLES_START: \t%d\n", ELEMS_PARTICLES_START);
    printf("~~$ diff: \t\t\t%d\n", (ELEMS_PARTICLES_START - nonZeroCounter));
    printf("~~$ negative_before_counter: \t%d\n", negative_before_counter);
    assert(nonZeroCounter + negative_before_counter == ELEMS_PARTICLES_START);
    printf("--> 2D MAP Creation Passed\n");
    if (end_new_line == true) printf("\n");
}


void ASSERT_particles_pos_unique(int* res_particles_x, int* res_particles_y, int* h_particles_x_after_unique, int* h_particles_y_after_unique, 
    const int LEN, bool printVerbose = false, bool start_new_line = false, bool end_new_line = false) {

    if (start_new_line == true) printf("\n");
    for (int i = 0, j = 0; i < LEN; i++) {

        if (h_particles_x_after_unique[i] > 0 && h_particles_y_after_unique[i] > 0) {

            if (printVerbose == true) printf("h_particles_x_after_unique=%d, res_particles_x=%d\n", h_particles_x_after_unique[i], res_particles_x[j]);
            assert(h_particles_x_after_unique[i] == res_particles_x[j]);
            if (printVerbose == true) printf("h_particles_y_after_unique=%d, res_particles_y=%d\n", h_particles_y_after_unique[i], res_particles_y[j]);
            assert(h_particles_y_after_unique[i] == res_particles_y[j]);
            j += 1;
        }
    }
    printf("--> Particles Pose (x & y) are OK\n");
    if (end_new_line == true) printf("\n");
}

void ASSERT_particles_idx_unique(int* res_particles_idx, int* h_particles_idx_after_unique, int negative_count, 
    const int LEN, bool start_new_line = false, bool end_new_line = false) {

    if (start_new_line == true) printf("\n");
    if (negative_count == 0) {
        for (int i = 0; i < LEN; i++) {
            if (h_particles_idx_after_unique[i] != res_particles_idx[i])
                printf("res_particles_idx=%d <> h_particles_idx_after_unique=%d\n",res_particles_idx[i], h_particles_idx_after_unique[i]);
            assert(h_particles_idx_after_unique[i] == res_particles_idx[i]);
        }
        printf("--> Particles Unique Indices are Passed\n");
    }
    else {
        printf("--> Particles Unique Indices Ignored because of negative Particles\n");
    }
    if (end_new_line == true) printf("\n");
}

void ASSERT_new_len_calculation(const int NEW_LEN, const int _ELEMS_PARTICLES_AFTER, const int negative_after_counter) {

    printf("--> NEW_LEN: %d <> %d, diff=%d\n\n", (NEW_LEN + negative_after_counter), _ELEMS_PARTICLES_AFTER, abs(_ELEMS_PARTICLES_AFTER - NEW_LEN));
    assert((NEW_LEN + negative_after_counter) == _ELEMS_PARTICLES_AFTER);
}


void ASSERT_correlation_Equality(float* res_correlation, float* h_correlation, 
    const int LEN, bool start_new_line = false, bool end_new_line = false) {

    if (start_new_line == true) printf("\n");
    bool all_equal = true;
    for (int i = 0; i < LEN; i++) {
        if (res_correlation[i] != h_correlation[i]) {
            all_equal = false;
            printf("index: %d --> %f, %f --> %s\n", i, res_correlation[i], h_correlation[i], (res_correlation[i] == h_correlation[i] ? "Equal" : ""));
        }
    }
    printf("--> Correlation All Equal: %s\n", all_equal ? "true" : "false");
    if (end_new_line == true) printf("\n");
}

void ASSERT_correlation_Equality(int* res_correlation, float* h_correlation, 
    const int LEN, bool start_new_line = false, bool end_new_line = false) {

    if (start_new_line == true) printf("\n");
    bool all_equal = true;
    for (int i = 0; i < LEN; i++) {
        if (res_correlation[i] != h_correlation[i]) {
            all_equal = false;
            printf("index: %d --> %d, %f --> %s\n", i, res_correlation[i], h_correlation[i], (res_correlation[i] == h_correlation[i] ? "Equal" : ""));
        }
    }
    printf("--> Correlation All Equal: %s\n", all_equal ? "true" : "false");
    if (end_new_line == true) printf("\n");
}

void ASSERT_update_particle_weights(float* res_weights, float* h_weights, const int LEN, const char* particle_types, 
    bool printVerbose, bool start_new_line = false, bool end_new_line = false) {

    if (start_new_line == true) printf("\n");
    for (int i = 0; i < LEN; i++) {
        float diff = abs(res_weights[i] - h_weights[i]);
        if(printVerbose == true) printf("%f <> %f, diff=%f\n", res_weights[i], h_weights[i], diff);
        assert(diff < 1e-4);
    }
    printf("--> Update Particle Weights (%s) Passed\n", particle_types);
    if (end_new_line == true) printf("\n");
}


void ASSERT_resampling_indices(int* res_js, int* h_js, const int LEN, bool printVerbose, bool start_new_line = false, bool end_new_line = false) {

    if (start_new_line == true) printf("\n");
    for (int i = 0; i < LEN; i++) {
        if(printVerbose == true) printf("%d, %d | ", res_js[i], h_js[i]);
        assert(res_js[i] == h_js[i]);
    }
    printf("--> Resampling Indices All Passed\n");
    if (end_new_line == true) printf("\n");
}

void ASSERT_resampling_states(float* x, float* y, float* theta, float* x_updated, float* y_updated, float* theta_updated, int* res_js, const int LEN, 
    bool printVerbose, bool start_new_line = false, bool end_new_line = false) {

    if (start_new_line == true) printf("\n");
    for (int i = 0; i < NUM_PARTICLES; i++) {
        int j = res_js[i];
        assert(x[j] == x_updated[i]);
        assert(y[j] == y_updated[i]);
        assert(theta[j] == theta_updated[i]);
        if(printVerbose == true) printf("x=%f <> %f, y=%f <> %f\n", x[j], x_updated[i], y[j], y_updated[i]);
    }
    printf("--> Resampling States All Passed\n");
    if (end_new_line == true) printf("\n");
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

void ASSERT_log_odds(float* res_log_odds, float* pre_log_odds, float* post_log_odds, 
    const int LEN, bool start_new_line = true, bool end_new_line = false) {

    if (start_new_line == true) printf("\n");
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
    printf("--> Log-Odds --> Error: %d, Correct: %d\n", numError, numCorrect);
    if (end_new_line == true) printf("\n");
}

void ASSERT_log_odds_maps(int* res_grid_map, int* pre_grid_map, int* post_grid_map, 
    const int LEN, bool start_new_line = true, bool end_new_line = false) {

    if (start_new_line == true) printf("\n");
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
    printf("--> Log_Odds MAP --> Error: %d, Correct: %d\n", numError, numCorrect);
    if (end_new_line == true) printf("\n");
}

void ASSERT_particles_occupied(int* res_particles_x, int* res_particles_y, int* h_particles_x, int* h_particles_y, 
    const char* particle_type, const int LEN, bool printVerbose = false, bool start_new_line = true, bool end_new_line = false) {

    if (start_new_line == true) printf("\n");
    for (int i = 0; i < LEN; i++) {
        if(printVerbose == true)
            printf("%d <> %d,  %d <> %d\n", res_particles_x[i], h_particles_x[i], res_particles_y[i], h_particles_y[i]);

        assert(res_particles_x[i] == h_particles_x[i]);
        assert(res_particles_y[i] == h_particles_y[i]);
    }
    printf("--> All Unique %s Calculation Passed\n", particle_type);
    if (end_new_line == true) printf("\n");
}

void ASSERT_transition_world_lidar(float* res_transition_world_lidar, float* h_transition_world_lidar, 
    const int LEN, bool printVerbose, bool start_new_line = true, bool end_new_line = false) {

    if (start_new_line == true) printf("\n");
    for (int i = 0; i < LEN; i++) {
        if(printVerbose == true) printf("%f (%f) ", res_transition_world_lidar[i], h_transition_world_lidar[i]);
        assert(abs(res_transition_world_lidar[i] - h_transition_world_lidar[i]) < 1e-4);
    }
    printf("--> Transition World Lidar All Correct\n");
    if (end_new_line == true) printf("\n");
}


void ASSERT_particles_world_frame(float* res_particles_wframe_x, float* res_particles_wframe_y, float* h_particles_wframe_x, float* h_particles_wframe_y,
    const int LEN, bool printVerbose = false, bool start_new_line = true, bool end_new_line = false) {

    if (start_new_line == true) printf("\n");
    for (int i = 0; i < LEN; i++) {
        if(printVerbose == true) printf("i=%d --> %f <> %f, %f <> %f\n", i, res_particles_wframe_x[i], h_particles_wframe_x[i], res_particles_wframe_y[i], h_particles_wframe_y[i]);
        assert(abs(res_particles_wframe_x[i] - h_particles_wframe_x[i]) < 1e-3);
        assert(abs(res_particles_wframe_y[i] - h_particles_wframe_y[i]) < 1e-3);
    }
    printf("--> Particles World Frame All Correct\n");
    if (end_new_line == true) printf("\n");
}

void ASSERT_position_image_body(int* res_position_image_body, int* h_position_image_body, 
    bool start_new_line = true, bool end_new_line = false) {

    if (start_new_line == true) printf("\n");
    printf("--> Position Image Body >> x: %d <> %d , y: %d <> %d\n", 
        res_position_image_body[0], h_position_image_body[0], res_position_image_body[1], h_position_image_body[1]);
    assert(res_position_image_body[0] == h_position_image_body[0]);
    assert(res_position_image_body[1] == h_position_image_body[1]);
    if (end_new_line == true) printf("\n");
}

void ASSERT_particles_free_index(int* res_particles_free_counter, int* h_particles_free_idx, int LEN, 
    bool printVerbose = false, bool start_new_line = true, bool end_new_line = false) {

    if (start_new_line == true) printf("\n");
    for (int i = 0; i < LEN; i++) {
        if (printVerbose == true) printf("i=%d --> %d <> %d\n", i, res_particles_free_counter[i], h_particles_free_idx[i]);
        assert(res_particles_free_counter[i] == h_particles_free_idx[i]);
    }
    printf("--> Particles Free Index All Correct\n");
    if (end_new_line == true) printf("\n");
}

void ASSERT_particles_free_new_len(const int PARTICLES_NEW_LEN, const int PARTICLES_FREE_LEN, 
    bool start_new_line = true, bool end_new_line = false) {

    if (start_new_line == true) printf("\n");
    printf("~~$ PARTICLES_NEW_LEN=%d <> PARTICLES_FREE_LEN=%d\n", PARTICLES_NEW_LEN, PARTICLES_FREE_LEN);
    assert(PARTICLES_NEW_LEN == PARTICLES_FREE_LEN);
    printf("--> Particles New Length All Correct\n");
    if (end_new_line == true) printf("\n");
}

void ASSERT_particles_free(int* res_particles_free_x, int* res_particles_free_y, int* h_particles_free_x, int* h_particles_free_y, 
    const int LEN, bool start_new_line = true, bool end_new_line = false) {

    if (start_new_line == true) printf("\n");
    bool all_equal = true;
    int errors = 0;
    for (int i = 0; i < LEN; i++) {
        if (res_particles_free_x[i] != h_particles_free_x[i] || res_particles_free_y[i] != h_particles_free_y[i]) {
            all_equal = false;
            errors += 1;
            printf("%d -- %d, %d | %d, %d\n", i, res_particles_free_x[i], h_particles_free_x[i], res_particles_free_y[i], h_particles_free_y[i]);
        }
        if (errors > 50)
            break;
    }
    printf("--> Free Particles Calculation -> All Equal: %s\n", all_equal ? "true" : "false");
    printf("--> Errors: %d\n", errors);
    if (end_new_line == true) printf("\n");
}

#endif