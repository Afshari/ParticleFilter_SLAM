#ifndef _HOST_CHECK_H_
#define _HOST_CHECK_H_

#include "headers.h"

void CHECK_particles_pos_unique(int* res_particles_x, int* res_particles_y,
    int* h_particles_x_after_unique, int* h_particles_y_after_unique,
    const int LEN, bool print_verbose, bool start_new_line = true, bool end_new_line = false) {

    int errors = 0;
    if (start_new_line == true) printf("\n");
    for (int i = 0, j = 0; i < LEN; i++) {

        if (h_particles_x_after_unique[i] > 0 && h_particles_y_after_unique[i] > 0) {

            if (print_verbose == true) printf("h_particles_x_after_unique=%d, res_particles_x=%d\n", h_particles_x_after_unique[i], res_particles_x[j]);
            if (h_particles_x_after_unique[i] != res_particles_x[j])    errors += 1;
            if (print_verbose == true) printf("h_particles_y_after_unique=%d, res_particles_y=%d\n", h_particles_y_after_unique[i], res_particles_y[j]);
            if (h_particles_y_after_unique[i] != res_particles_y[j])    errors += 1;
            j += 1;
        }
    }
    printf("--> Particles Pose (x & y) Run with Erros: %d of: %d\n", errors, LEN);
    if (end_new_line == true) printf("\n");
}

void CHECK_particles_idx_unique(int* res_particles_idx, int* h_particles_idx_after_unique, int negative_count,
    const int LEN, bool print_verbose, bool start_new_line = true, bool end_new_line = false) {

    int errors = 0;
    if (start_new_line == true) printf("\n");
    if (negative_count == 0) {
        for (int i = 0; i < LEN; i++) {
            if (h_particles_idx_after_unique[i] != res_particles_idx[i]) {
                if(print_verbose == true)   printf("res_particles_idx=%d <> h_particles_idx_after_unique=%d\n", res_particles_idx[i], h_particles_idx_after_unique[i]);
                errors += 1;
            }
        }
        printf("--> Particles Unique Indices Run with Errors: %d of: %d\n", errors, LEN);
    }
    else {
        printf("--> Particles Unique Indices Ignored because of negative Particles\n");
    }
    if (end_new_line == true) printf("\n");
}

void CHECK_resampling_indices(int* res_js, int* h_js, const int LEN, bool print_verbose,
    bool start_new_line = true, bool end_new_line = false) {

    int errors = 0;
    if (start_new_line == true) printf("\n");
    for (int i = 0; i < LEN; i++) {
        if (print_verbose == true) printf("%d, %d | ", res_js[i], h_js[i]);
        if (res_js[i] != h_js[i])    errors += 1;
    }
    printf("--> Resampling Indices Run with Errors: %d of: %d\n", errors, LEN);
    if (end_new_line == true) printf("\n");
}

void CHECK_resampling_states(float* x, float* y, float* theta,
    float* x_updated, float* y_updated, float* theta_updated, int* res_js, const int LEN,
    bool print_verbose, bool start_new_line = true, bool end_new_line = false) {

    int errors = 0;
    if (start_new_line == true) printf("\n");
    for (int i = 0; i < NUM_PARTICLES; i++) {
        int j = res_js[i];
        if (x[j] != x_updated[i])    errors += 1;
        if (y[j] != y_updated[i])    errors += 1;
        if (theta[j] != theta_updated[i])    errors += 1;
        if (print_verbose == true) printf("x=%f <> %f, y=%f <> %f\n", x[j], x_updated[i], y[j], y_updated[i]);
    }
    printf("--> Resampling States Run with Errors: %d of: %d\n", errors, (3 * LEN));
    if (end_new_line == true) printf("\n");
}

void CHECK_new_len_calculation(const int NEW_LEN, const int _ELEMS_PARTICLES_AFTER, const int negative_after_counter) {

    printf("--> NEW_LEN: %d <> %d, diff=%d\n\n", (NEW_LEN + negative_after_counter), _ELEMS_PARTICLES_AFTER, abs(_ELEMS_PARTICLES_AFTER - NEW_LEN));
}

void CHECK_resampling_particles_index(int* h_particles_idx, int* res_particles_idx, const int LEN,
    bool print_verbose, int negative_particles) {

    int errors = 0;
    if (negative_particles == 0) {
        for (int i = 0; i < LEN; i++) {

            if (print_verbose == true) printf("%d <> %d\n", h_particles_idx[i], res_particles_idx[i]);
            if (h_particles_idx[i] != res_particles_idx[i]) errors += 1;
        }
        printf("\n--> Resampling Particles Index Run with Errors: %d of: %d\n\n", errors, LEN);
    }
    else {
        printf("\n--> Ignore Assert because of 'Negative Particles'\n\n");
    }
}

void CHECK_rearrange_particles_states(int* res_particles_x, int* res_particles_y, 
    float* res_states_x, float* res_states_y, float* res_states_theta,
    int* h_particles_x, int* h_particles_y, float* h_states_x, float* h_states_y, float* h_states_theta,
    const int PARTICLES_LEN, const int STATES_LEN, bool print_verbose) {

    int errors = 0;
    for (int i = 0, j = 0; i < PARTICLES_LEN; i++) {
        if (h_particles_x[i] < 0 || h_particles_y[i] < 0)
            continue;

        if (res_particles_x[j] != h_particles_x[i]) {
            if(print_verbose == true)   printf("i=%d --> %d, %d\n", i, res_particles_x[j], h_particles_x[i]);
            errors += 1;
        }
        if (res_particles_y[j] != h_particles_y[i]) {
            if (print_verbose == true)  printf("i=%d --> %d, %d\n", i, res_particles_y[j], h_particles_y[i]);
            errors += 1;
        }

        j += 1;
    }
    printf("Particles Check with Errors: %d of: %d\n\n", errors, (2 * PARTICLES_LEN));

    errors = 0;
    for (int i = 0; i < STATES_LEN; i++) {
        if (res_states_x[i] != h_states_x[i]) errors += 1;
        if (res_states_y[i] != h_states_y[i]) errors += 1;
        if (res_states_theta[i] != h_states_theta[i]) errors += 1;
    }
    printf("States Check with Errors: %d of: %d\n\n", errors, (3 * STATES_LEN));
}

#endif
