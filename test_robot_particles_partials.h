
#ifndef _TEST_ROBOT_PARTICLES_PARTIALS_H_
#define _TEST_ROBOT_PARTICLES_PARTIALS_H_

#include "headers.h"
#include "host_asserts.h"
#include "host_utils.h"
#include "kernels.cuh"


/**************************************************/
// #define UPDATE_STATE_EXEC
// #define UPDATE_PARTICLE_WEIGHTS_EXEC
// #define CORRELATION_EXEC
// #define RESAMPLING_EXEC
// #define UPDATE_PARTICLES_EXEC
// #define UPDATE_UNIQUE_EXEC
// #define UPDATE_LOOP_EXEC
#define UPDATE_FUNC_EXEC


/**************************************************/
#ifdef CORRELATION_EXEC
#include "data/map_correlation/4300.h"
#endif

#ifdef UPDATE_STATE_EXEC
#include "data/state_update/300.h"
#endif

#ifdef UPDATE_PARTICLE_WEIGHTS_EXEC
#include "data/particle_weights/1400.h"
#endif

#ifdef RESAMPLING_EXEC
#include "data/resampling/70.h"
#endif

#ifdef UPDATE_PARTICLES_EXEC
#include "data/update_particles/100.h"
// #include "data/update_loop/100.h"
#endif

#ifdef UPDATE_UNIQUE_EXEC
//#include "data/update_unique/200.h"
#include "data/update_unique/4800.h"
#endif

#ifdef UPDATE_LOOP_EXEC
#include "data/update_loop/4500.h"
#endif

#ifdef UPDATE_FUNC_EXEC
#include "data/update_func/4800.h"
#endif



void host_correlation();
void host_update_state();
void host_update_particle_weights();
void host_resampling();
void host_update_particles();
void host_update_unique();
void host_update_loop();
void host_update_func();


int test_robot_particles_partials_main() {

#ifdef CORRELATION_EXEC
    host_correlation();
#endif

#ifdef UPDATE_STATE_EXEC
    host_update_state();
#endif

#ifdef UPDATE_PARTICLE_WEIGHTS_EXEC
    host_update_particle_weights();
#endif

#ifdef RESAMPLING_EXEC
    host_resampling();
#endif

#ifdef UPDATE_PARTICLES_EXEC
    host_update_particles();
#endif

#ifdef UPDATE_UNIQUE_EXEC
    host_update_unique();
#endif

#ifdef UPDATE_LOOP_EXEC
    host_update_loop();
#endif

#ifdef UPDATE_FUNC_EXEC
    host_update_func();
#endif

    return 0;
}


#ifdef UPDATE_STATE_EXEC
void host_update_state() {

    thrust::device_vector<float> d_temp(xs, xs + NUM_PARTICLES);

    size_t sz_states_pos = NUM_PARTICLES * sizeof(float);

    float* d_states_x = NULL;
    float* d_states_y = NULL;
    float* d_states_theta = NULL;

    gpuErrchk(cudaMalloc((void**)&d_states_x, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&d_states_y, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&d_states_theta, sz_states_pos));


    gpuErrchk(cudaMemcpy(d_states_x, xs, sz_states_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_states_y, ys, sz_states_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_states_theta, thetas, sz_states_pos, cudaMemcpyHostToDevice));


    auto start_update_states = std::chrono::high_resolution_clock::now();

    thrust::device_vector<float> d_vec_states_x(d_states_x, d_states_x + NUM_PARTICLES);
    thrust::device_vector<float> d_vec_states_y(d_states_y, d_states_y + NUM_PARTICLES);
    thrust::device_vector<float> d_vec_states_theta(d_states_theta, d_states_theta + NUM_PARTICLES);


    thrust::host_vector<float> h_vec_states_x(d_vec_states_x.begin(), d_vec_states_x.end());
    thrust::host_vector<float> h_vec_states_y(d_vec_states_y.begin(), d_vec_states_y.end());
    thrust::host_vector<float> h_vec_states_theta(d_vec_states_theta.begin(), d_vec_states_theta.end());

    std::vector<float> std_vec_states_x(h_vec_states_x.begin(), h_vec_states_x.end());
    std::vector<float> std_vec_states_y(h_vec_states_y.begin(), h_vec_states_y.end());
    std::vector<float> std_vec_states_theta(h_vec_states_theta.begin(), h_vec_states_theta.end());


    std::map<std::tuple<float, float, float>, int> states;

    for (int i = 0; i < NUM_PARTICLES; i++) {

        if (states.find(std::make_tuple(std_vec_states_x[i], std_vec_states_y[i], std_vec_states_theta[i])) == states.end()) {
            states.insert({ std::make_tuple(std_vec_states_x[i], std_vec_states_y[i], std_vec_states_theta[i]), 1 });
        }
        else {
            states[std::make_tuple(std_vec_states_x[i], std_vec_states_y[i], std_vec_states_theta[i])] += 1;
        }
    }

    std::map<std::tuple<float, float, float>, int>::iterator best
        = std::max_element(states.begin(), states.end(), [](const std::pair<std::tuple<float, float, float>, int>& a,
            const std::pair<std::tuple<float, float, float>, int>& b)->bool { return a.second < b.second; });

    auto key = best->first;
    // std::cout << std::get<0>(key) << " " << std::get<1>(key) << " " << std::get<2>(key) << " " << best->second << "\n";

    float theta = std::get<2>(key);
    float _T_wb[] = { cos(theta), -sin(theta), std::get<0>(key),
                        sin(theta),  cos(theta), std::get<1>(key),
                        0, 0, 1 };

    auto stop_update_states = std::chrono::high_resolution_clock::now();
    auto duration_update_states = std::chrono::duration_cast<std::chrono::microseconds>(stop_update_states - start_update_states);

    std::cout << "Time taken by function (Update States): " << duration_update_states.count() << " microseconds" << std::endl;

    for (int i = 0; i < 9; i++) {
        printf("%f  ", _T_wb[i]);
        assert(T_wb[i] == _T_wb[i]);
    }
    printf("\n");
    printf("%f, %f, %f\n", std::get<0>(key), std::get<1>(key), std::get<2>(key));


}
#endif

#ifdef UPDATE_PARTICLE_WEIGHTS_EXEC
void host_update_particle_weights() {

    /********************************************************************/
    /************************ WEIGHTS VARIABLES *************************/
    /********************************************************************/
    size_t sz_weights = NUM_PARTICLES * sizeof(float);
    size_t sz_weights_max = sizeof(float);
    size_t sz_sum_exp = sizeof(double);

    float* d_weights = NULL;
    float* d_weights_max = NULL;
    double* d_sum_exp = NULL;


    float* res_weights = (float*)malloc(sz_weights);
    float* res_weights_max = (float*)malloc(sz_weights_max);
    double* res_sum_exp = (double*)malloc(sz_sum_exp);

    gpuErrchk(cudaMalloc((void**)&d_weights, sz_weights));
    gpuErrchk(cudaMalloc((void**)&d_weights_max, sz_weights_max));
    gpuErrchk(cudaMalloc((void**)&d_sum_exp, sz_sum_exp));


    gpuErrchk(cudaMemcpy(d_weights, pre_weights, sz_weights, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(d_weights_max, 0, sz_weights_max));
    gpuErrchk(cudaMemset(d_sum_exp, 0, sz_sum_exp));


    /********************************************************************/
    /********************** UPDATE WEIGHTS KERNEL ***********************/
    /********************************************************************/
    auto start_update_particle_weights = std::chrono::high_resolution_clock::now();

    int threadsPerBlock = 1;
    int blocksPerGrid = 1;

    kernel_arr_max << < blocksPerGrid, threadsPerBlock >> > (d_weights, d_weights_max, NUM_PARTICLES);
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(res_weights_max, d_weights_max, sz_weights_max, cudaMemcpyDeviceToHost));
    assert(res_weights_max[0] == ARR_MAX);

    float norm_value = -res_weights_max[0] + 50;

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_arr_increase << < blocksPerGrid, threadsPerBlock >> > (d_weights, norm_value, 0);
    cudaDeviceSynchronize();

    threadsPerBlock = 1;
    blocksPerGrid = 1;
    kernel_arr_sum_exp << < blocksPerGrid, threadsPerBlock >> > (d_weights, d_sum_exp, NUM_PARTICLES);
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(res_sum_exp, d_sum_exp, sz_sum_exp, cudaMemcpyDeviceToHost));

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_arr_normalize << < blocksPerGrid, threadsPerBlock >> > (d_weights, res_sum_exp[0]);
    cudaDeviceSynchronize();

    auto stop_update_particle_weights = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_weights, d_weights, sz_weights, cudaMemcpyDeviceToHost));

    ASSERT_update_particle_weights(res_weights, weights, NUM_PARTICLES, false);

    auto duration_update_particle_weights = std::chrono::duration_cast<std::chrono::microseconds>(stop_update_particle_weights - start_update_particle_weights);
    std::cout << "Time taken by function (Update Particle Weights): " << duration_update_particle_weights.count() << " microseconds" << std::endl;
}
#endif

#ifdef RESAMPLING_EXEC
void host_resampling() {

    /********************************************************************/
    /*********************** RESAMPLING VARIABLES ***********************/
    /********************************************************************/
    float* d_weights = NULL;
    int* d_js = NULL;
    float* d_rnd = NULL;

    size_t sz_weights = NUM_PARTICLES * sizeof(float);
    size_t sz_js = NUM_PARTICLES * sizeof(int);
    size_t sz_rnd = NUM_PARTICLES * sizeof(float);

    int* res_js = (int*)malloc(sz_js);

    gpuErrchk(cudaMalloc((void**)&d_weights, sz_weights));
    gpuErrchk(cudaMalloc((void**)&d_js, sz_js));
    gpuErrchk(cudaMalloc((void**)&d_rnd, sz_rnd));

    cudaMemcpy(d_weights, weights, sz_weights, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rnd, rnds, sz_rnd, cudaMemcpyHostToDevice);
    gpuErrchk(cudaMemset(d_js, 0, sz_js));

    /********************************************************************/
    /************************ RESAMPLING kerenel ************************/
    /********************************************************************/
    auto start_resampling = std::chrono::high_resolution_clock::now();

    int threadsPerBlock = NUM_PARTICLES;
    int blocksPerGrid = 1;

    kernel_resampling << <blocksPerGrid, threadsPerBlock >> > (d_weights, d_js, d_rnd, NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_resampling = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_js, d_js, sz_js, cudaMemcpyDeviceToHost));

    auto duration_resampling = std::chrono::duration_cast<std::chrono::microseconds>(stop_resampling - start_resampling);
    std::cout << "Time taken by function (Kernel Resampling): " << duration_resampling.count() << " microseconds" << std::endl;

    ASSERT_resampling(res_js, js, NUM_PARTICLES, true);

}
#endif

#ifdef UPDATE_PARTICLES_EXEC
void host_update_particles() {

    // ✓
    // [✓] - Change Execution Time to 'chrono time'
    // [ ] - Create 'd_measure_idx'
    // [ ] - Create a kernel for initializing 'd_measure_idx'

#ifndef lidar_coords_LEN
    const int lidar_coords_LEN = LIDAR_COORDS_LEN;

#else
    int* processed_measure = (int*)malloc(2 * NUM_PARTICLES * lidar_coords_LEN * sizeof(int));
    for (int i = 0; i < 2 * NUM_PARTICLES * lidar_coords_LEN; i++)
        processed_measure[i] = h_processed_measure_pos_float[i];
#endif

    printf("lidar_coords_LEN: %d \n", lidar_coords_LEN);

    //const int lidar_coords_LEN = LIDAR_COORDS_LEN;

    /********************************************************************/
    /************************** PRIOR VARIABLES *************************/
    /********************************************************************/
    size_t sz_states = NUM_PARTICLES * sizeof(float);
    size_t sz_lidar_coords = 2 * lidar_coords_LEN * sizeof(float);

    float* d_states_x = NULL;
    float* d_states_y = NULL;
    float* d_states_theta = NULL;
    float* d_lidar_coords = NULL;

    gpuErrchk(cudaMalloc((void**)&d_states_x, sz_states));
    gpuErrchk(cudaMalloc((void**)&d_states_y, sz_states));
    gpuErrchk(cudaMalloc((void**)&d_states_theta, sz_states));
    gpuErrchk(cudaMalloc((void**)&d_lidar_coords, sz_lidar_coords));

    cudaMemcpy(d_states_x, h_states_x, sz_states, cudaMemcpyHostToDevice);
    cudaMemcpy(d_states_y, h_states_y, sz_states, cudaMemcpyHostToDevice);
    cudaMemcpy(d_states_theta, h_states_theta, sz_states, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lidar_coords, lidar_coords, sz_lidar_coords, cudaMemcpyHostToDevice);


    /********************************************************************/
    /************************* MIDDLE VARIABLES *************************/
    /********************************************************************/
    size_t sz_transition_body_frame = 9 * NUM_PARTICLES * sizeof(float);
    size_t sz_transition_lidar_frame = 9 * sizeof(float);
    size_t sz_transition_world_frame = 9 * NUM_PARTICLES * sizeof(float);
    size_t sz_processed_measure_pos = NUM_PARTICLES * lidar_coords_LEN * sizeof(int);
    size_t sz_measure_idx = NUM_PARTICLES * lidar_coords_LEN * sizeof(int);

    float* d_transition_body_frame = NULL;
    float* d_transition_lidar_frame = NULL;
    float* d_transition_world_frame = NULL;
    int* d_processed_measure_x = NULL;
    int* d_processed_measure_y = NULL;
    int* d_measure_idx = NULL;

    gpuErrchk(cudaMalloc((void**)&d_transition_body_frame, sz_transition_body_frame));
    gpuErrchk(cudaMalloc((void**)&d_transition_lidar_frame, sz_transition_lidar_frame));
    gpuErrchk(cudaMalloc((void**)&d_transition_world_frame, sz_transition_world_frame));
    gpuErrchk(cudaMalloc((void**)&d_processed_measure_x, sz_processed_measure_pos));
    gpuErrchk(cudaMalloc((void**)&d_processed_measure_y, sz_processed_measure_pos));
    gpuErrchk(cudaMalloc((void**)&d_measure_idx, sz_measure_idx));


    /********************************************************************/
    /************************* HOST VARIABLES ***************************/
    /********************************************************************/
    float* res_transition_body_frame = (float*)malloc(sz_transition_body_frame);
    float* res_transition_world_frame = (float*)malloc(sz_transition_world_frame);
    int* res_processed_measure_x = (int*)malloc(sz_processed_measure_pos);
    int* res_processed_measure_y = (int*)malloc(sz_processed_measure_pos);
    int* res_measure_idx = (int*)malloc(sz_measure_idx);

    memset(res_transition_body_frame, 0, sz_transition_body_frame);
    memset(res_transition_world_frame, 0, sz_transition_world_frame);
    memset(res_processed_measure_x, 0, sz_processed_measure_pos);
    memset(res_processed_measure_y, 0, sz_processed_measure_pos);


    cudaMemcpy(d_transition_body_frame, res_transition_body_frame, sz_transition_body_frame, cudaMemcpyHostToDevice);
    cudaMemcpy(d_transition_world_frame, res_transition_world_frame, sz_transition_world_frame, cudaMemcpyHostToDevice);
    cudaMemcpy(d_processed_measure_x, res_processed_measure_x, sz_processed_measure_pos, cudaMemcpyHostToDevice);
    cudaMemcpy(d_processed_measure_y, res_processed_measure_y, sz_processed_measure_pos, cudaMemcpyHostToDevice);
    cudaMemcpy(d_transition_lidar_frame, h_transition_lidar_frame, sz_transition_lidar_frame, cudaMemcpyHostToDevice);


    /********************************************************************/
    /*************************** KERNEL EXEC ****************************/
    /********************************************************************/
    auto start_kernel = std::chrono::high_resolution_clock::now();

    int threadsPerBlock = NUM_PARTICLES;
    int blocksPerGrid = 1;

    kernel_update_particles_states << <blocksPerGrid, threadsPerBlock >> > (d_states_x, d_states_y, d_states_theta, d_transition_body_frame, d_transition_lidar_frame, d_transition_world_frame, NUM_PARTICLES);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = lidar_coords_LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (d_transition_world_frame, d_processed_measure_x, d_processed_measure_y, d_lidar_coords, res, xmin, ymax, lidar_coords_LEN, NUM_PARTICLES);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_index_init_const << < blocksPerGrid, threadsPerBlock >> > (d_measure_idx, lidar_coords_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = 1;
    blocksPerGrid = 1;
    kernel_update_unique_sum << < blocksPerGrid, threadsPerBlock >> > (d_measure_idx, NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_kernel = std::chrono::high_resolution_clock::now();
    auto duration_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_kernel - start_kernel);
    std::cout << "Time taken by function (Kernel): " << duration_kernel.count() << " microseconds" << std::endl;

    gpuErrchk(cudaMemcpy(res_transition_body_frame, d_transition_body_frame, sz_transition_body_frame, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_transition_world_frame, d_transition_world_frame, sz_transition_world_frame, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_processed_measure_x, d_processed_measure_x, sz_processed_measure_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_processed_measure_y, d_processed_measure_y, sz_processed_measure_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_measure_idx, d_measure_idx, sz_measure_idx, cudaMemcpyDeviceToHost));

    bool printVerbose = false;

    for (int i = 0; i < 9 * NUM_PARTICLES; i++) {
        if (printVerbose == true) printf("%f, %f | ", res_transition_body_frame[i], h_transition_body_frame[i]);
        assert(abs(res_transition_body_frame[i] - h_transition_body_frame[i]) < 1e-5);
    }
    for (int i = 0; i < 9 * NUM_PARTICLES; i++) {
        if (printVerbose == true) printf("%f, %f |  ", res_transition_world_frame[i], h_transition_world_frame[i]);
        assert(abs(res_transition_world_frame[i] - h_transition_world_frame[i]) < 1e-5);
    }

    //for (int i = 0; i < 2 * NUM_PARTICLES * lidar_coords_LEN; i++) {
    //    if ( abs(res_Y_wo[i] - Y_wo[i]) > 1e-5 ) {
    //        // printf("%f, %f, %f, %i  |  ", res_Y_wo[i], Y_wi[i], Y_wo[i], i);
    //        printf("%f, %f, %i  |  ", res_Y_wo[i], Y_wo[i], i);
    //    }
    //    // assert( res_Y_wo[i] == Y_wi[i] );
    //}


    int notEqualCounter = 0;
    for (int i = 0; i < NUM_PARTICLES; i++) {
        int h_idx = 2 * i * lidar_coords_LEN;
        int res_idx = i * lidar_coords_LEN;
        for (int j = 0; j < lidar_coords_LEN; j++) {
            if (abs(res_processed_measure_x[j + res_idx] - processed_measure[j + h_idx]) > 1e-5) {
                printf("%d, %d, %d  |  ", (i * lidar_coords_LEN + j), res_processed_measure_x[j + res_idx], processed_measure[j + h_idx]);
                notEqualCounter += 1;
                if (notEqualCounter > 50)
                    exit(-1);
            }
        }
        h_idx += lidar_coords_LEN;
        for (int j = 0; j < lidar_coords_LEN; j++) {
            if (abs(res_processed_measure_y[j + res_idx] - processed_measure[j + h_idx]) > 1e-5) {
                printf("%d, %d, %d  |  ", (i * lidar_coords_LEN + j), res_processed_measure_y[j + res_idx], processed_measure[j + h_idx]);
                notEqualCounter += 1;
                if (notEqualCounter > 50)
                    exit(-1);
            }
        }
    }
    printf("\nProcessed Measure Error Count: %d\n", notEqualCounter);

    for (int i = 0; i < NUM_PARTICLES; i++) {
        int diff = (i == 0) ? 0 : (res_measure_idx[i] - res_measure_idx[i - 1]);
        if (printVerbose == true) printf("index %d --> value: %d, diff: %d\n", i, res_measure_idx[i], diff);
    }
}
#endif

#ifdef UPDATE_UNIQUE_EXEC
void host_update_unique() {

    int negative_before_counter = getNegativeCounter(h_particles_x_prior, h_particles_y_prior, BEFORE_LEN);
    int count_bigger_than_height = getGreaterThanCounter(h_particles_y_prior, GRID_HEIGHT, BEFORE_LEN);
    int negative_after_counter = getNegativeCounter(h_particles_x_post, h_particles_y_post, AFTER_LEN);;


    printf("GRID_WIDTH: %d, GRID_HEIGHT: %d\n", GRID_WIDTH, GRID_HEIGHT);
    printf("MEASURE_LEN: %d\n", MEASURE_LEN);
    printf("negative_before_counter: %d\n", negative_before_counter);
    printf("negative_after_counter: %d\n", negative_after_counter);
    printf("count_bigger_than_height: %d\n", count_bigger_than_height);

    // [ ] - Create a kernel for cum_sum
    // [✓] - Change name of variables to (d_particles_x, d_particles_y, d_particles_idx)
    // [ ] - Write down function input & output --> Inputs: (d_particles_x, d_particles_y, d_particles_idx) & (d_measure_x, d_measure_y, d_measure_idx)   /   Outputs: ()
    // [ ] - Write down variables that created in the function: ()
    // [ ] - print 'measure_idx'

    const int UNIQUE_COUNTER_LEN = NUM_PARTICLES + 1;

    /********************************************************************/
    /************************** PRIOR VARIABLES *************************/
    /********************************************************************/
    int* d_particles_x = NULL;
    int* d_particles_y = NULL;
    int* d_particles_idx = NULL;
    size_t   sz_particles_pos = BEFORE_LEN * sizeof(int);
    size_t   sz_particles_idx = NUM_PARTICLES * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_particles_x, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_y, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_idx, sz_particles_idx));

    cudaMemcpy(d_particles_x, h_particles_x_prior, sz_particles_pos, cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles_y, h_particles_y_prior, sz_particles_pos, cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles_idx, h_particles_idx_prior, sz_particles_idx, cudaMemcpyHostToDevice);


    /********************************************************************/
    /**************************** MAP VARIABLES *************************/
    /********************************************************************/
    uint8_t* d_map_2d = NULL;
    int* d_unique_in_particle = NULL;
    int* d_unique_in_particle_col = NULL;

    size_t   sz_map_2d = GRID_WIDTH * GRID_HEIGHT * NUM_PARTICLES * sizeof(uint8_t);
    size_t   sz_unique_in_particle = UNIQUE_COUNTER_LEN * sizeof(int);
    size_t   sz_unique_in_particle_col = UNIQUE_COUNTER_LEN * GRID_WIDTH * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_map_2d, sz_map_2d));
    gpuErrchk(cudaMalloc((void**)&d_unique_in_particle, sz_unique_in_particle));
    gpuErrchk(cudaMalloc((void**)&d_unique_in_particle_col, sz_unique_in_particle_col));

    int* h_unique_in_particle = (int*)malloc(sz_unique_in_particle);
    int* h_unique_in_particle_col = (int*)malloc(sz_unique_in_particle_col);
    uint8_t* h_map_2d = (uint8_t*)malloc(sz_map_2d);

    cudaMemset(d_map_2d, 0, sz_map_2d);
    cudaMemset(d_unique_in_particle, 0, sz_unique_in_particle);
    cudaMemset(d_unique_in_particle_col, 0, sz_unique_in_particle_col);

    /********************************************************************/
    /*********************** MEASUREMENT VARIABLES **********************/
    /********************************************************************/
    int* d_measure_x = NULL;
    int* d_measure_y = NULL;
    int* d_measure_idx = NULL;
    size_t sz_measure_pos = MEASURE_LEN * sizeof(int);
    size_t sz_measure_idx = NUM_PARTICLES * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_measure_x, sz_measure_pos));
    gpuErrchk(cudaMalloc((void**)&d_measure_y, sz_measure_pos));
    gpuErrchk(cudaMalloc((void**)&d_measure_idx, sz_measure_idx));

    cudaMemcpy(d_measure_x, h_measure_x, sz_measure_pos, cudaMemcpyHostToDevice);
    cudaMemcpy(d_measure_y, h_measure_y, sz_measure_pos, cudaMemcpyHostToDevice);
    cudaMemcpy(d_measure_idx, h_measure_idx, sz_measure_idx, cudaMemcpyHostToDevice);


    /********************************************************************/
    /**************************** CREATE MAP ****************************/
    /********************************************************************/
    int threadsPerBlock = 100;
    int blocksPerGrid = NUM_PARTICLES; // NUM_ELEMS;

    auto start_create_map = std::chrono::high_resolution_clock::now();

    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (d_particles_x, d_particles_y, d_particles_idx, BEFORE_LEN, d_map_2d, d_unique_in_particle,
        d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES); // NUM_ELEMS);
    cudaDeviceSynchronize();

    auto stop_create_map = std::chrono::high_resolution_clock::now();


    /********************************************************************/
    /**************************** UPDATE MAP ****************************/
    /********************************************************************/
    auto start_update_map = std::chrono::high_resolution_clock::now();

    threadsPerBlock = NUM_PARTICLES; // NUM_ELEMS;
    blocksPerGrid = 1;

    kernel_update_2d_map_with_measure << <blocksPerGrid, threadsPerBlock >> > (d_measure_x, d_measure_y, d_measure_idx, MEASURE_LEN, d_map_2d, d_unique_in_particle,
        d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES); // NUM_ELEMS);
    cudaDeviceSynchronize();

    auto stop_update_map = std::chrono::high_resolution_clock::now();


    /********************************************************************/
    /************************* CUMULATIVE SUM ***************************/
    /********************************************************************/
    threadsPerBlock = UNIQUE_COUNTER_LEN;
    blocksPerGrid = 1;

    auto start_cumulative_sum = std::chrono::high_resolution_clock::now();

    kernel_update_unique_sum << <1, 1 >> > (d_unique_in_particle, UNIQUE_COUNTER_LEN);
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (d_unique_in_particle_col, GRID_WIDTH);
    cudaDeviceSynchronize();

    auto stop_cumulative_sum = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(h_map_2d, d_map_2d, sz_map_2d, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_unique_in_particle, d_unique_in_particle, sz_unique_in_particle, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_unique_in_particle_col, d_unique_in_particle_col, sz_unique_in_particle_col, cudaMemcpyDeviceToHost));

    int NEW_LEN = h_unique_in_particle[NUM_PARTICLES]; //[NUM_ELEMS - 1];
    gpuErrchk(cudaFree(d_particles_x));
    gpuErrchk(cudaFree(d_particles_y));

    sz_particles_pos = NEW_LEN * sizeof(int);
    gpuErrchk(cudaMalloc((void**)&d_particles_x, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_y, sz_particles_pos));
    int* res_particles_x = (int*)malloc(sz_particles_pos);
    int* res_particles_y = (int*)malloc(sz_particles_pos);
    int* res_particles_idx = (int*)malloc(sz_particles_idx);


    /********************************************************************/
    /************************ MAP RESTRUCTURE ***************************/
    /********************************************************************/
    threadsPerBlock = GRID_WIDTH;
    blocksPerGrid = NUM_PARTICLES;

    auto start_map_restructure = std::chrono::high_resolution_clock::now();

    cudaMemset(d_particles_idx, 0, sz_particles_idx);
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (d_map_2d, d_particles_x, d_particles_y, d_particles_idx,
        d_unique_in_particle, d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT);
    cudaDeviceSynchronize();

    kernel_update_unique_sum << <1, 1 >> > (d_particles_idx, NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_map_restructure = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_particles_x, d_particles_x, sz_particles_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_y, d_particles_y, sz_particles_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToHost));

    auto duration_create_map = std::chrono::duration_cast<std::chrono::milliseconds>(stop_create_map - start_create_map);
    auto duration_update_map = std::chrono::duration_cast<std::chrono::milliseconds>(stop_update_map - start_update_map);
    auto duration_cumulative_sum = std::chrono::duration_cast<std::chrono::microseconds>(stop_cumulative_sum - start_cumulative_sum);
    auto duration_map_restructure = std::chrono::duration_cast<std::chrono::milliseconds>(stop_map_restructure - start_map_restructure);

    std::cout << "Time taken by function (Create Map): " << duration_create_map.count() << " milliseconds" << std::endl;
    std::cout << "Time taken by function (Update Map): " << duration_update_map.count() << " milliseconds" << std::endl;
    std::cout << "Time taken by function (Cumulative Sum): " << duration_cumulative_sum.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Map Restructure): " << duration_map_restructure.count() << " milliseconds" << std::endl;

    printf("\nunique_in_particle: %d\n", NEW_LEN);
    printf("Measurement Length: %d\n", MEASURE_LEN);


    ASSERT_particles_pos_unique(res_particles_x, res_particles_y, h_particles_x_post, h_particles_y_post, NEW_LEN);

    //for (int i = 0; i < NUM_PARTICLES; i++) {
    //    int diff = (i == 0) ? 0 : (h_measure_idx[i] - h_measure_idx[i - 1]);
    //    printf("index %d --> value: %d, diff: %d\n", i, h_measure_idx[i], diff);
    //}

    //for (int i = 0; i < NUM_PARTICLES; i++) {
    //    printf("index %d: %d <> %d\n", i, res_particles_idx[i], h_particles_idx_post[i]);
    //}

    printf("All Passed\n");

}
#endif

#ifdef CORRELATION_EXEC
void host_correlation() {

    auto start_memory_copy = std::chrono::high_resolution_clock::now();

    /********************************************************************/
    /************************** PRIOR VARIABLES *************************/
    /********************************************************************/
    int* d_grid_map = NULL;
    int* d_particles_x = NULL;
    int* d_particles_y = NULL;
    int* d_particles_idx = NULL;
    int* d_extended_idx = NULL;

    const int num_elements_of_grid_map = GRID_WIDTH * GRID_HEIGHT;
    size_t sz_grid_map = num_elements_of_grid_map * sizeof(int);

    size_t sz_particles_pos = elems_particles * sizeof(int);
    size_t sz_particles_idx = NUM_PARTICLES * sizeof(int);
    size_t sz_extended_idx = elems_particles * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_grid_map, sz_grid_map));
    gpuErrchk(cudaMalloc((void**)&d_particles_x, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_y, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_extended_idx, sz_extended_idx));
    gpuErrchk(cudaMalloc((void**)&d_particles_idx, sz_particles_idx));

    cudaMemcpy(d_grid_map, grid_map, sz_grid_map, cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles_x, h_particles_x, sz_particles_pos, cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles_y, h_particles_y, sz_particles_pos, cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles_idx, h_particles_idx, sz_particles_idx, cudaMemcpyHostToDevice);

    size_t sz_weights = NUM_PARTICLES * sizeof(float);
    size_t sz_weights_raw = 25 * sz_weights;
    float* h_weights = (float*)malloc(sz_weights);
    int* h_extended_idx = (int*)malloc(sz_extended_idx);
    float* d_weights = NULL;
    float* d_weights_raw = NULL;
    memset(h_weights, 0, sz_weights);

    gpuErrchk(cudaMalloc((void**)&d_weights, sz_weights));
    gpuErrchk(cudaMalloc((void**)&d_weights_raw, sz_weights_raw));
    gpuErrchk(cudaMemset(d_weights_raw, 0, sz_weights_raw));
    // gpuErrchk(cudaMemcpy(d_all_correlation, h_correlation, sz_all_correlation, cudaMemcpyHostToDevice));

    auto stop_memory_copy = std::chrono::high_resolution_clock::now();


    /********************************************************************/
    /*************************** PRINT SUMMARY **************************/
    /********************************************************************/
    printf("Elements of particles_x: %d,  Size of particles_x: %d\n", (int)elems_particles, (int)sz_particles_pos);
    printf("Elements of particles_y: %d,  Size of particles_y: %d\n", (int)elems_particles, (int)sz_particles_pos);
    printf("Elements of particles_idx: %d,  Size of particles_idx: %d\n", (int)elems_particles, (int)sz_extended_idx);

    printf("Elements of Grid_Map: %d,  Size of Grid_Map: %d\n", (int)num_elements_of_grid_map, (int)sz_grid_map);

    /********************************************************************/
    /************************* INDEX EXPANSION **************************/
    /********************************************************************/
    auto start_index_expansion = std::chrono::high_resolution_clock::now();

    int threadsPerBlock = 100;
    int blocksPerGrid = NUM_PARTICLES;
    kernel_index_expansion << <blocksPerGrid, threadsPerBlock >> > (d_particles_idx, d_extended_idx, elems_particles);
    cudaDeviceSynchronize();

    auto stop_index_expansion = std::chrono::high_resolution_clock::now();

    /********************************************************************/
    /************************ KERNEL CORRELATION ************************/
    /********************************************************************/
    threadsPerBlock = 256;
    blocksPerGrid = (elems_particles + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads, All Threads: %d\n", blocksPerGrid, threadsPerBlock, blocksPerGrid * threadsPerBlock);

    auto start_kernel = std::chrono::high_resolution_clock::now();

    kernel_correlation << <blocksPerGrid, threadsPerBlock >> > (d_grid_map, d_particles_x, d_particles_y, d_extended_idx, d_weights_raw, GRID_WIDTH, GRID_HEIGHT, elems_particles);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_correlation_max << <blocksPerGrid, threadsPerBlock >> > (d_weights_raw, d_weights, NUM_PARTICLES);

    auto stop_kernel = std::chrono::high_resolution_clock::now();


    gpuErrchk(cudaMemcpy(h_weights, d_weights, sz_weights, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_extended_idx, d_extended_idx, sz_extended_idx, cudaMemcpyDeviceToHost));


    bool all_equal = true;
    for (int i = 0; i < NUM_PARTICLES; i++) {
        // printf("index: %d --> %d, %d\n", i, final_result[i], new_weights[i]); 
        if (h_weights[i] != new_weights[i])
            all_equal = false;
    }


    std::cout << std::endl << "Execution Time: " << std::endl;
    auto duration_kernel = std::chrono::duration_cast<std::chrono::milliseconds>(stop_kernel - start_kernel);
    auto duration_memory_copy = std::chrono::duration_cast<std::chrono::milliseconds>(stop_memory_copy - start_memory_copy);
    auto duration_index_expansion = std::chrono::duration_cast<std::chrono::microseconds>(stop_index_expansion - start_index_expansion);
    std::cout << "Time taken by function (Correlation): " << duration_kernel.count() << " milliseconds" << std::endl;
    // std::cout << "Time taken by function (Memory Copy): " << duration_memory_copy.count() << " milliseconds" << std::endl;
    std::cout << "Time taken by function (Index Expansion): " << duration_index_expansion.count() << " microseconds" << std::endl;

    printf("All Equal: %s\n", all_equal ? "true" : "false");

    printf("Program Finished\n");

    gpuErrchk(cudaFree(d_grid_map));
    gpuErrchk(cudaFree(d_particles_x));
    gpuErrchk(cudaFree(d_particles_y));
    gpuErrchk(cudaFree(d_particles_idx));
    gpuErrchk(cudaFree(d_extended_idx));
    gpuErrchk(cudaFree(d_weights_raw));
}
#endif

#ifdef UPDATE_LOOP_EXEC
void host_update_loop() {

    // ✓

    int negative_before_counter = getNegativeCounter(h_particles_x, h_particles_y, ELEMS_PARTICLES_START);
    int count_bigger_than_height = getGreaterThanCounter(h_particles_y, GRID_HEIGHT, ELEMS_PARTICLES_START);
    int negative_after_counter = getNegativeCounter(h_particles_x_after_unique, h_particles_y_after_unique, ELEMS_PARTICLES_AFTER);;

    printf("GRID_WIDTH: %d, GRID_HEIGHT: %d\n", GRID_WIDTH, GRID_HEIGHT);
    printf("negative_before_counter: %d\n", negative_before_counter);
    printf("negative_after_counter: %d\n", negative_after_counter);
    printf("count_bigger_than_height: %d\n", count_bigger_than_height);


    // const int NUM_ELEMS     = NUM_PARTICLES + 1;
    const int UNIQUE_COUNTER_LEN = NUM_PARTICLES + 1;
    const int MEASURE_LEN = NUM_PARTICLES * LIDAR_COORDS_LEN;

    printf("MEASURE_LEN: %d\n", MEASURE_LEN);

    /********************************************************************/
    /************************** PRIOR VARIABLES *************************/
    /********************************************************************/
    size_t sz_states_pos = NUM_PARTICLES * sizeof(float);
    size_t sz_lidar_coords = 2 * LIDAR_COORDS_LEN * sizeof(float);

    float* d_states_x = NULL;
    float* d_states_y = NULL;
    float* d_states_theta = NULL;
    float* d_lidar_coords = NULL;

    gpuErrchk(cudaMalloc((void**)&d_states_x, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&d_states_y, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&d_states_theta, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&d_lidar_coords, sz_lidar_coords));

    gpuErrchk(cudaMemcpy(d_states_x, h_states_x, sz_states_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_states_y, h_states_y, sz_states_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_states_theta, h_states_theta, sz_states_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_lidar_coords, lidar_coords, sz_lidar_coords, cudaMemcpyHostToDevice));

    /********************************************************************/
    /**************************** MAP VARIABLES *************************/
    /********************************************************************/
    size_t sz_particles_pos = ELEMS_PARTICLES_START * sizeof(int);
    size_t sz_particles_idx = NUM_PARTICLES * sizeof(int);
    size_t sz_extended_idx = ELEMS_PARTICLES_START * sizeof(int);
    size_t sz_grid_map = GRID_WIDTH * GRID_HEIGHT * sizeof(int);

    int* d_grid_map = NULL;
    int* d_particles_x = NULL;
    int* d_particles_y = NULL;
    int* d_particles_idx = NULL;
    int* d_extended_idx = NULL;

    int* res_particles_x = (int*)malloc(sz_particles_pos);
    int* res_particles_y = (int*)malloc(sz_particles_pos);
    int* res_particles_idx = (int*)malloc(sz_particles_idx);
    int* res_extended_idx = (int*)malloc(sz_extended_idx);

    gpuErrchk(cudaMalloc((void**)&d_grid_map, sz_grid_map));
    gpuErrchk(cudaMalloc((void**)&d_particles_x, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_y, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_idx, sz_particles_idx));
    gpuErrchk(cudaMalloc((void**)&d_extended_idx, sz_extended_idx));

    cudaMemcpy(d_grid_map, grid_map, sz_grid_map, cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles_x, h_particles_x, sz_particles_pos, cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles_y, h_particles_y, sz_particles_pos, cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles_idx, h_particles_idx, sz_particles_idx, cudaMemcpyHostToDevice);


    /********************************************************************/
    /********************** CORRELATION VARIABLES ***********************/
    /********************************************************************/
    size_t sz_weights = NUM_PARTICLES * sizeof(float);
    size_t sz_correlation_raw = 25 * sz_weights;

    float* h_weights = (float*)malloc(sz_weights);
    int* h_extended_idx = (int*)malloc(sz_extended_idx);
    float* res_weights = (float*)malloc(sz_weights);
    float* d_weights = NULL;
    float* d_weights_raw = NULL;
    memset(h_weights, 0, sz_weights);

    gpuErrchk(cudaMalloc((void**)&d_weights, sz_weights));
    gpuErrchk(cudaMalloc((void**)&d_weights_raw, sz_correlation_raw));
    gpuErrchk(cudaMemset(d_weights_raw, 0, sz_correlation_raw));


    /********************************************************************/
    /*********************** TRANSITION VARIABLES ***********************/
    /********************************************************************/
    size_t sz_transition_body_frame = 9 * NUM_PARTICLES * sizeof(float);
    size_t sz_transition_lidar_frame = 9 * sizeof(float);
    size_t sz_transition_world_frame = 9 * NUM_PARTICLES * sizeof(float);
    size_t sz_processed_measure_pos = NUM_PARTICLES * LIDAR_COORDS_LEN * sizeof(int);
    size_t sz_measure_idx = NUM_PARTICLES * LIDAR_COORDS_LEN * sizeof(int);

    float* d_transition_body_frame = NULL;
    float* d_transition_lidar_frame = NULL;
    float* d_transition_world_frame = NULL;
    int* d_processed_measure_x = NULL;
    int* d_processed_measure_y = NULL;
    int* d_measure_idx = NULL;

    float* res_transition_body_frame = (float*)malloc(sz_transition_body_frame);
    float* res_transition_world_frame = (float*)malloc(sz_transition_world_frame);
    int* res_processed_measure_x = (int*)malloc(sz_processed_measure_pos);
    int* res_processed_measure_y = (int*)malloc(sz_processed_measure_pos);


    gpuErrchk(cudaMalloc((void**)&d_transition_body_frame, sz_transition_body_frame));
    gpuErrchk(cudaMalloc((void**)&d_transition_lidar_frame, sz_transition_lidar_frame));
    gpuErrchk(cudaMalloc((void**)&d_transition_world_frame, sz_transition_world_frame));
    gpuErrchk(cudaMalloc((void**)&d_processed_measure_x, sz_processed_measure_pos));
    gpuErrchk(cudaMalloc((void**)&d_processed_measure_y, sz_processed_measure_pos));
    gpuErrchk(cudaMalloc((void**)&d_measure_idx, sz_measure_idx));


    gpuErrchk(cudaMemset(d_transition_body_frame, 0, sz_transition_body_frame));
    gpuErrchk(cudaMemset(d_transition_world_frame, 0, sz_transition_world_frame));
    gpuErrchk(cudaMemset(d_processed_measure_x, 0, sz_processed_measure_pos));
    gpuErrchk(cudaMemset(d_processed_measure_y, 0, sz_processed_measure_pos));
    gpuErrchk(cudaMemset(d_measure_idx, 0, sz_measure_idx));

    cudaMemcpy(d_transition_lidar_frame, h_transition_lidar_frame, sz_transition_lidar_frame, cudaMemcpyHostToDevice);

    /********************************************************************/
    /**************************** MAP VARIABLES *************************/
    /********************************************************************/
    size_t   sz_map_2d = GRID_WIDTH * GRID_HEIGHT * NUM_PARTICLES * sizeof(uint8_t);
    size_t   sz_unique_in_particle = UNIQUE_COUNTER_LEN * sizeof(int);
    size_t   sz_unique_in_particle_col = UNIQUE_COUNTER_LEN * GRID_WIDTH * sizeof(int);

    uint8_t* d_map_2d = NULL;
    int* d_unique_in_particle = NULL;
    int* d_unique_in_particle_col = NULL;

    uint8_t* res_map_2d = (uint8_t*)malloc(sz_map_2d);
    int* h_unique_in_particle = (int*)malloc(sz_unique_in_particle);

    gpuErrchk(cudaMalloc((void**)&d_map_2d, sz_map_2d));
    gpuErrchk(cudaMalloc((void**)&d_unique_in_particle, sz_unique_in_particle));
    gpuErrchk(cudaMalloc((void**)&d_unique_in_particle_col, sz_unique_in_particle_col));

    gpuErrchk(cudaMemset(d_map_2d, 0, sz_map_2d));
    gpuErrchk(cudaMemset(d_unique_in_particle, 0, sz_unique_in_particle));
    gpuErrchk(cudaMemset(d_unique_in_particle_col, 0, sz_unique_in_particle_col));

    /********************************************************************/
    /************************ TRANSITION KERNEL *************************/
    /********************************************************************/
    auto start_transition_kernel = std::chrono::high_resolution_clock::now();

    int threadsPerBlock = NUM_PARTICLES;
    int blocksPerGrid = 1;

    kernel_update_particles_states << <blocksPerGrid, threadsPerBlock >> > (d_states_x, d_states_y, d_states_theta, d_transition_body_frame, d_transition_lidar_frame, d_transition_world_frame, NUM_PARTICLES);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = LIDAR_COORDS_LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (d_transition_world_frame, d_processed_measure_x, d_processed_measure_y, d_lidar_coords, res, xmin, ymax, LIDAR_COORDS_LEN, NUM_PARTICLES);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_index_init_const << < blocksPerGrid, threadsPerBlock >> > (d_measure_idx, LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = 1;
    blocksPerGrid = 1;
    kernel_update_unique_sum << < blocksPerGrid, threadsPerBlock >> > (d_measure_idx, NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_transition_kernel = std::chrono::high_resolution_clock::now();


    gpuErrchk(cudaMemcpy(res_transition_body_frame, d_transition_body_frame, sz_transition_body_frame, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_transition_world_frame, d_transition_world_frame, sz_transition_world_frame, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_processed_measure_x, d_processed_measure_x, sz_processed_measure_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_processed_measure_y, d_processed_measure_y, sz_processed_measure_pos, cudaMemcpyDeviceToHost));

    ASSERT_transition_frames(res_transition_body_frame, res_transition_world_frame, h_transition_body_frame, h_transition_world_frame, NUM_PARTICLES, false);
    ASSERT_processed_measurements(res_processed_measure_x, res_processed_measure_y, processed_measure, NUM_PARTICLES, LIDAR_COORDS_LEN);

    /********************************************************************/
    /************************** CREATE 2D MAP ***************************/
    /********************************************************************/
    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;

    auto start_create_map = std::chrono::high_resolution_clock::now();

    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (d_particles_x, d_particles_y, d_particles_idx, ELEMS_PARTICLES_START, d_map_2d, d_unique_in_particle,
        d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_create_map = std::chrono::high_resolution_clock::now();

    //cudaError_t err = cudaPeekAtLastError();
    //printf("%s\n", cudaGetErrorString(err));

    gpuErrchk(cudaMemcpy(res_map_2d, d_map_2d, sz_map_2d, cudaMemcpyDeviceToHost));

    ASSERT_create_2d_map_elements(res_map_2d, negative_before_counter, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES, ELEMS_PARTICLES_START);

    /********************************************************************/
    /**************************** UPDATE MAP ****************************/
    /********************************************************************/
    auto start_update_map = std::chrono::high_resolution_clock::now();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;

    kernel_update_2d_map_with_measure << <blocksPerGrid, threadsPerBlock >> > (d_processed_measure_x, d_processed_measure_y, d_measure_idx,
        MEASURE_LEN, d_map_2d, d_unique_in_particle,
        d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_update_map = std::chrono::high_resolution_clock::now();

    /********************************************************************/
    /************************* CUMULATIVE SUM ***************************/
    /********************************************************************/
    threadsPerBlock = UNIQUE_COUNTER_LEN;
    blocksPerGrid = 1;

    auto start_cumulative_sum = std::chrono::high_resolution_clock::now();

    kernel_update_unique_sum << <1, 1 >> > (d_unique_in_particle, UNIQUE_COUNTER_LEN);
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (d_unique_in_particle_col, GRID_WIDTH);
    cudaDeviceSynchronize();

    auto stop_cumulative_sum = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(h_unique_in_particle, d_unique_in_particle, sz_unique_in_particle, cudaMemcpyDeviceToHost));

    int NEW_LEN = h_unique_in_particle[UNIQUE_COUNTER_LEN - 1];
    ASSERT_new_len_calculation(NEW_LEN, ELEMS_PARTICLES_AFTER, negative_after_counter);


    /********************************************************************/
    /******************* REINITIALIZE MAP VARIABLES *********************/
    /********************************************************************/
    gpuErrchk(cudaFree(d_particles_x));
    gpuErrchk(cudaFree(d_particles_y));
    gpuErrchk(cudaFree(d_extended_idx));
    free(res_particles_x);
    free(res_particles_y);

    sz_particles_pos = NEW_LEN * sizeof(int);
    sz_extended_idx = NEW_LEN * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_particles_x, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_y, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_extended_idx, sz_extended_idx));

    res_particles_x = (int*)malloc(sz_particles_pos);
    res_particles_y = (int*)malloc(sz_particles_pos);


    /********************************************************************/
    /************************ MAP RESTRUCTURE ***************************/
    /********************************************************************/
    threadsPerBlock = GRID_WIDTH;
    blocksPerGrid = NUM_PARTICLES;

    auto start_map_restructure = std::chrono::high_resolution_clock::now();

    cudaMemset(d_particles_idx, 0, sz_particles_idx);
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (d_map_2d, d_particles_x, d_particles_y, d_particles_idx,
        d_unique_in_particle, d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT);
    cudaDeviceSynchronize();

    kernel_update_unique_sum << <1, 1 >> > (d_particles_idx, NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_map_restructure = std::chrono::high_resolution_clock::now();

    auto start_copy_particles_pos = std::chrono::high_resolution_clock::now();
    gpuErrchk(cudaMemcpy(res_particles_x, d_particles_x, sz_particles_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_y, d_particles_y, sz_particles_pos, cudaMemcpyDeviceToHost));
    auto stop_copy_particles_pos = std::chrono::high_resolution_clock::now();

    ASSERT_particles_pos_unique(res_particles_x, res_particles_y, h_particles_x_after_unique, h_particles_y_after_unique, NEW_LEN);

    /********************************************************************/
    /************************* INDEX EXPANSION **************************/
    /********************************************************************/
    auto start_index_expansion = std::chrono::high_resolution_clock::now();

    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;

    kernel_index_expansion << <blocksPerGrid, threadsPerBlock >> > (d_particles_idx, d_extended_idx, NEW_LEN);
    cudaDeviceSynchronize();

    auto stop_index_expansion = std::chrono::high_resolution_clock::now();

    res_extended_idx = (int*)malloc(sz_extended_idx);
    gpuErrchk(cudaMemcpy(res_extended_idx, d_extended_idx, sz_extended_idx, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToHost));


    /********************************************************************/
    /************************ KERNEL CORRELATION ************************/
    /********************************************************************/
    threadsPerBlock = 256;
    blocksPerGrid = (NEW_LEN + threadsPerBlock - 1) / threadsPerBlock;
    printf("*** CUDA kernel launch with %d blocks of %d threads, All Threads: %d ***\n", blocksPerGrid, threadsPerBlock, blocksPerGrid * threadsPerBlock);

    auto start_correlation = std::chrono::high_resolution_clock::now();

    kernel_correlation << <blocksPerGrid, threadsPerBlock >> > (d_grid_map, d_particles_x, d_particles_y, d_extended_idx, d_weights_raw, GRID_WIDTH, GRID_HEIGHT, NEW_LEN);
    cudaDeviceSynchronize();


    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_correlation_max << <blocksPerGrid, threadsPerBlock >> > (d_weights_raw, d_weights, NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_correlation = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_weights, d_weights, sz_weights, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_extended_idx, d_extended_idx, sz_extended_idx, cudaMemcpyDeviceToHost));

    ASSERT_correlation_Equality(res_weights, new_weights, NUM_PARTICLES);


    auto duration_create_map = std::chrono::duration_cast<std::chrono::milliseconds>(stop_create_map - start_create_map);
    auto duration_update_map = std::chrono::duration_cast<std::chrono::milliseconds>(stop_update_map - start_update_map);
    auto duration_cumulative_sum = std::chrono::duration_cast<std::chrono::milliseconds>(stop_cumulative_sum - start_cumulative_sum);
    auto duration_map_restructure = std::chrono::duration_cast<std::chrono::milliseconds>(stop_map_restructure - start_map_restructure);
    auto duration_copy_particles_pos = std::chrono::duration_cast<std::chrono::milliseconds>(stop_copy_particles_pos - start_copy_particles_pos);
    auto duration_transition_kernel = std::chrono::duration_cast<std::chrono::milliseconds>(stop_transition_kernel - start_transition_kernel);
    auto duration_correlation = std::chrono::duration_cast<std::chrono::milliseconds>(stop_correlation - start_correlation);
    auto duration_sum = duration_create_map + duration_update_map + duration_cumulative_sum + duration_map_restructure + duration_copy_particles_pos +
        duration_transition_kernel + duration_correlation;

    std::cout << "Time taken by function (Create Map): " << duration_create_map.count() << " milliseconds" << std::endl;
    std::cout << "Time taken by function (Update Map): " << duration_update_map.count() << " milliseconds" << std::endl;
    std::cout << "Time taken by function (Cumulative Sum): " << duration_cumulative_sum.count() << " milliseconds" << std::endl;
    std::cout << "Time taken by function (Map Restructure): " << duration_map_restructure.count() << " milliseconds" << std::endl;
    std::cout << "Time taken by function (Copy Particles): " << duration_copy_particles_pos.count() << " milliseconds" << std::endl;
    std::cout << "Time taken by function (Transition Kernel): " << duration_transition_kernel.count() << " milliseconds" << std::endl;
    std::cout << "Time taken by function (Correlation Kernel): " << duration_correlation.count() << " milliseconds" << std::endl;
    std::cout << "Time taken by function (Sum): " << duration_sum.count() << " milliseconds" << std::endl;

    printf("\nFinished All\n");

}
#endif

#ifdef UPDATE_FUNC_EXEC
void host_update_func() {

    thrust::device_vector<float> d_temp(h_states_x, h_states_x + NUM_PARTICLES);

    int negative_before_counter = getNegativeCounter(h_particles_x, h_particles_y, ELEMS_PARTICLES_START);
    int count_bigger_than_height = getGreaterThanCounter(h_particles_y, GRID_HEIGHT, ELEMS_PARTICLES_START);
    int negative_after_counter = getNegativeCounter(h_particles_x_after_resampling, h_particles_y_after_resampling, ELEMS_PARTICLES_AFTER);;

    printf("GRID_WIDTH: %d, GRID_HEIGHT: %d\n", GRID_WIDTH, GRID_HEIGHT);
    printf("negative_before_counter: %d\n", negative_before_counter);
    printf("negative_after_counter: %d\n", negative_after_counter);
    printf("count_bigger_than_height: %d\n", count_bigger_than_height);

    const int UNIQUE_COUNTER_LEN = NUM_PARTICLES + 1;
    const int MEASURE_LEN = NUM_PARTICLES * LIDAR_COORDS_LEN;

    printf("MEASURE_LEN: %d\n", MEASURE_LEN);

    /**************************************************************************************************************************************************/
    /**************************************************************** VARIABLES SCOPE *****************************************************************/
    /**************************************************************************************************************************************************/

    /********************************************************************/
    /************************** STATES VARIABLES ************************/
    /********************************************************************/
    size_t sz_states_pos = NUM_PARTICLES * sizeof(float);
    size_t sz_lidar_coords = 2 * LIDAR_COORDS_LEN * sizeof(float);

    float* d_states_x = NULL;
    float* d_states_y = NULL;
    float* d_states_theta = NULL;
    float* d_lidar_coords = NULL;

    float* res_states_x = (float*)malloc(sz_states_pos);
    float* res_states_y = (float*)malloc(sz_states_pos);
    float* res_states_theta = (float*)malloc(sz_states_pos);

    gpuErrchk(cudaMalloc((void**)&d_states_x, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&d_states_y, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&d_states_theta, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&d_lidar_coords, sz_lidar_coords));

    gpuErrchk(cudaMemcpy(d_states_x, h_states_x, sz_states_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_states_y, h_states_y, sz_states_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_states_theta, h_states_theta, sz_states_pos, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_lidar_coords, lidar_coords, sz_lidar_coords, cudaMemcpyHostToDevice));

    /********************************************************************/
    /**************************** MAP VARIABLES *************************/
    /********************************************************************/
    size_t sz_particles_pos = ELEMS_PARTICLES_START * sizeof(int);
    size_t sz_particles_idx = NUM_PARTICLES * sizeof(int);
    size_t sz_particles_weight = NUM_PARTICLES * sizeof(float);
    size_t sz_extended_idx = ELEMS_PARTICLES_START * sizeof(int);
    size_t sz_grid_map = GRID_WIDTH * GRID_HEIGHT * sizeof(int);

    int* d_grid_map = NULL;
    int* d_particles_x = NULL;
    int* d_particles_y = NULL;
    int* d_particles_idx = NULL;
    float* d_particles_weight = NULL;
    int* d_extended_idx = NULL;

    int* res_particles_x = (int*)malloc(sz_particles_pos);
    int* res_particles_y = (int*)malloc(sz_particles_pos);
    int* res_particles_idx = (int*)malloc(sz_particles_idx);
    float* res_particles_weight = (float*)malloc(sz_particles_weight);
    int* res_extended_idx = (int*)malloc(sz_extended_idx);

    gpuErrchk(cudaMalloc((void**)&d_grid_map, sz_grid_map));
    gpuErrchk(cudaMalloc((void**)&d_particles_x, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_y, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_idx, sz_particles_idx));
    gpuErrchk(cudaMalloc((void**)&d_particles_weight, sz_particles_weight));
    gpuErrchk(cudaMalloc((void**)&d_extended_idx, sz_extended_idx));

    cudaMemcpy(d_grid_map, grid_map, sz_grid_map, cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles_x, h_particles_x, sz_particles_pos, cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles_y, h_particles_y, sz_particles_pos, cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles_idx, h_particles_idx, sz_particles_idx, cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles_weight, particles_weight_pre, sz_particles_weight, cudaMemcpyHostToDevice);


    /********************************************************************/
    /******************** PARTICLES COPY VARIABLES **********************/
    /********************************************************************/
    int* dc_particles_x = NULL;
    int* dc_particles_y = NULL;
    int* dc_particles_idx = NULL;

    float* dc_states_x = NULL;
    float* dc_states_y = NULL;
    float* dc_states_theta = NULL;

    gpuErrchk(cudaMalloc((void**)&dc_states_x, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&dc_states_y, sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&dc_states_theta, sz_states_pos));

    /********************************************************************/
    /********************** CORRELATION VARIABLES ***********************/
    /********************************************************************/
    size_t sz_weights = NUM_PARTICLES * sizeof(float);
    size_t sz_correlation_raw = 25 * sz_weights;

    float* h_weights = (float*)malloc(sz_weights);
    int* h_extended_idx = (int*)malloc(sz_extended_idx);
    float* res_weights = (float*)malloc(sz_weights);
    float* d_weights = NULL;
    float* d_weights_raw = NULL;
    memset(h_weights, 0, sz_weights);

    gpuErrchk(cudaMalloc((void**)&d_weights, sz_weights));
    gpuErrchk(cudaMalloc((void**)&d_weights_raw, sz_correlation_raw));
    gpuErrchk(cudaMemset(d_weights_raw, 0, sz_correlation_raw));


    /********************************************************************/
    /*********************** TRANSITION VARIABLES ***********************/
    /********************************************************************/
    size_t sz_transition_body_frame = 9 * NUM_PARTICLES * sizeof(float);
    size_t sz_transition_lidar_frame = 9 * sizeof(float);
    size_t sz_transition_world_frame = 9 * NUM_PARTICLES * sizeof(float);
    size_t sz_processed_measure_pos = NUM_PARTICLES * LIDAR_COORDS_LEN * sizeof(int);
    size_t sz_measure_idx = NUM_PARTICLES * LIDAR_COORDS_LEN * sizeof(int);

    float* d_transition_body_frame = NULL;
    float* d_transition_lidar_frame = NULL;
    float* d_transition_world_frame = NULL;
    int* d_processed_measure_x = NULL;
    int* d_processed_measure_y = NULL;
    int* d_measure_idx = NULL;

    float* res_transition_body_frame = (float*)malloc(sz_transition_body_frame);
    float* res_transition_world_frame = (float*)malloc(sz_transition_world_frame);
    int* res_processed_measure_x = (int*)malloc(sz_processed_measure_pos);
    int* res_processed_measure_y = (int*)malloc(sz_processed_measure_pos);


    gpuErrchk(cudaMalloc((void**)&d_transition_body_frame, sz_transition_body_frame));
    gpuErrchk(cudaMalloc((void**)&d_transition_lidar_frame, sz_transition_lidar_frame));
    gpuErrchk(cudaMalloc((void**)&d_transition_world_frame, sz_transition_world_frame));
    gpuErrchk(cudaMalloc((void**)&d_processed_measure_x, sz_processed_measure_pos));
    gpuErrchk(cudaMalloc((void**)&d_processed_measure_y, sz_processed_measure_pos));
    gpuErrchk(cudaMalloc((void**)&d_measure_idx, sz_measure_idx));


    gpuErrchk(cudaMemset(d_transition_body_frame, 0, sz_transition_body_frame));
    gpuErrchk(cudaMemset(d_transition_world_frame, 0, sz_transition_world_frame));
    gpuErrchk(cudaMemset(d_processed_measure_x, 0, sz_processed_measure_pos));
    gpuErrchk(cudaMemset(d_processed_measure_y, 0, sz_processed_measure_pos));
    gpuErrchk(cudaMemset(d_measure_idx, 0, sz_measure_idx));

    cudaMemcpy(d_transition_lidar_frame, h_transition_lidar_frame, sz_transition_lidar_frame, cudaMemcpyHostToDevice);

    /********************************************************************/
    /**************************** MAP VARIABLES *************************/
    /********************************************************************/
    size_t   sz_map_2d = GRID_WIDTH * GRID_HEIGHT * NUM_PARTICLES * sizeof(uint8_t);
    size_t   sz_unique_in_particle = UNIQUE_COUNTER_LEN * sizeof(int);
    size_t   sz_unique_in_particle_col = UNIQUE_COUNTER_LEN * GRID_WIDTH * sizeof(int);

    uint8_t* d_map_2d = NULL;
    int* d_unique_in_particle = NULL;
    int* d_unique_in_particle_col = NULL;

    uint8_t* res_map_2d = (uint8_t*)malloc(sz_map_2d);
    int* h_unique_in_particle = (int*)malloc(sz_unique_in_particle);

    gpuErrchk(cudaMalloc((void**)&d_map_2d, sz_map_2d));
    gpuErrchk(cudaMalloc((void**)&d_unique_in_particle, sz_unique_in_particle));
    gpuErrchk(cudaMalloc((void**)&d_unique_in_particle_col, sz_unique_in_particle_col));

    gpuErrchk(cudaMemset(d_map_2d, 0, sz_map_2d));
    gpuErrchk(cudaMemset(d_unique_in_particle, 0, sz_unique_in_particle));
    gpuErrchk(cudaMemset(d_unique_in_particle_col, 0, sz_unique_in_particle_col));

    /********************************************************************/
    /************************ WEIGHTS VARIABLES *************************/
    /********************************************************************/
    size_t sz_weights_max = sizeof(float);
    size_t sz_sum_exp = sizeof(double);

    float* d_weights_max = NULL;
    double* d_sum_exp = NULL;

    float* res_weights_max = (float*)malloc(sz_weights_max);
    double* res_sum_exp = (double*)malloc(sz_sum_exp);

    gpuErrchk(cudaMalloc((void**)&d_weights_max, sz_weights_max));
    gpuErrchk(cudaMalloc((void**)&d_sum_exp, sz_sum_exp));

    gpuErrchk(cudaMemset(d_weights_max, 0, sz_weights_max));
    gpuErrchk(cudaMemset(d_sum_exp, 0, sz_sum_exp));

    /********************************************************************/
    /*********************** RESAMPLING VARIABLES ***********************/
    /********************************************************************/
    int* d_js = NULL;
    float* d_rnd = NULL;

    size_t sz_js = NUM_PARTICLES * sizeof(int);
    size_t sz_rnd = NUM_PARTICLES * sizeof(float);

    int* res_js = (int*)malloc(sz_js);

    gpuErrchk(cudaMalloc((void**)&d_js, sz_js));
    gpuErrchk(cudaMalloc((void**)&d_rnd, sz_rnd));

    gpuErrchk(cudaMemcpy(d_rnd, rnds, sz_rnd, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(d_js, 0, sz_js));

    /********************************************************************/
    /********************* REARRANGEMENT VARIABLES **********************/
    /********************************************************************/
    std::vector<float> std_vec_states_x;
    std::vector<float> std_vec_states_y;
    std::vector<float> std_vec_states_theta;


    /**************************************************************************************************************************************************/
    /************************************************************* KERNEL EXECUTION SCOPE *************************************************************/
    /**************************************************************************************************************************************************/

    /********************************************************************/
    /************************ TRANSITION KERNEL *************************/
    /********************************************************************/
    auto start_transition_kernel = std::chrono::high_resolution_clock::now();

    int threadsPerBlock = NUM_PARTICLES;
    int blocksPerGrid = 1;

    kernel_update_particles_states << <blocksPerGrid, threadsPerBlock >> > (d_states_x, d_states_y, d_states_theta, d_transition_body_frame, d_transition_lidar_frame, d_transition_world_frame, NUM_PARTICLES);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = LIDAR_COORDS_LEN;
    kernel_update_particles_lidar << < blocksPerGrid, threadsPerBlock >> > (d_transition_world_frame, d_processed_measure_x, d_processed_measure_y, d_lidar_coords, res, xmin, ymax, LIDAR_COORDS_LEN, NUM_PARTICLES);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_index_init_const << < blocksPerGrid, threadsPerBlock >> > (d_measure_idx, LIDAR_COORDS_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = 1;
    blocksPerGrid = 1;
    kernel_update_unique_sum << < blocksPerGrid, threadsPerBlock >> > (d_measure_idx, NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_transition_kernel = std::chrono::high_resolution_clock::now();


    gpuErrchk(cudaMemcpy(res_transition_body_frame, d_transition_body_frame, sz_transition_body_frame, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_transition_world_frame, d_transition_world_frame, sz_transition_world_frame, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_processed_measure_x, d_processed_measure_x, sz_processed_measure_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_processed_measure_y, d_processed_measure_y, sz_processed_measure_pos, cudaMemcpyDeviceToHost));

    // ASSERT_transition_frames(res_transition_body_frame, res_transition_world_frame, h_transition_body_frame, h_transition_world_frame, NUM_PARTICLES, false);
    // ASSERT_processed_measurements(res_processed_measure_x, res_processed_measure_y, processed_measure, NUM_PARTICLES, LIDAR_COORDS_LEN);

    /********************************************************************/
    /************************** CREATE 2D MAP ***************************/
    /********************************************************************/
    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;

    auto start_create_map = std::chrono::high_resolution_clock::now();

    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (d_particles_x, d_particles_y, d_particles_idx, ELEMS_PARTICLES_START, d_map_2d, d_unique_in_particle,
        d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_create_map = std::chrono::high_resolution_clock::now();

    //cudaError_t err = cudaPeekAtLastError();
    //printf("%s\n", cudaGetErrorString(err));

    gpuErrchk(cudaMemcpy(res_map_2d, d_map_2d, sz_map_2d, cudaMemcpyDeviceToHost));

    ASSERT_create_2d_map_elements(res_map_2d, negative_before_counter, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES, ELEMS_PARTICLES_START);

    /********************************************************************/
    /**************************** UPDATE MAP ****************************/
    /********************************************************************/
    auto start_update_map = std::chrono::high_resolution_clock::now();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;

    kernel_update_2d_map_with_measure << <blocksPerGrid, threadsPerBlock >> > (d_processed_measure_x, d_processed_measure_y, d_measure_idx,
        MEASURE_LEN, d_map_2d, d_unique_in_particle,
        d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_update_map = std::chrono::high_resolution_clock::now();

    /********************************************************************/
    /************************* CUMULATIVE SUM ***************************/
    /********************************************************************/
    auto start_cumulative_sum = std::chrono::high_resolution_clock::now();

    threadsPerBlock = UNIQUE_COUNTER_LEN;
    blocksPerGrid = 1;
    kernel_update_unique_sum << <1, 1 >> > (d_unique_in_particle, UNIQUE_COUNTER_LEN);
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (d_unique_in_particle_col, GRID_WIDTH);
    cudaDeviceSynchronize();

    auto stop_cumulative_sum = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(h_unique_in_particle, d_unique_in_particle, sz_unique_in_particle, cudaMemcpyDeviceToHost));

    int NEW_LEN = h_unique_in_particle[UNIQUE_COUNTER_LEN - 1];
    int C_NEW_LEN = 0;
    // ASSERT_new_len_calculation(NEW_LEN, ELEMS_PARTICLES_AFTER, negative_after_counter);


    /*---------------------------------------------------------------------*/
    /*-------------------- REINITIALIZE MAP VARIABLES ---------------------*/
    /*---------------------------------------------------------------------*/
    /*---------------------------------------------------------------------*/
    gpuErrchk(cudaFree(d_particles_x));
    gpuErrchk(cudaFree(d_particles_y));
    gpuErrchk(cudaFree(d_extended_idx));
    free(res_particles_x);
    free(res_particles_y);

    sz_particles_pos = NEW_LEN * sizeof(int);
    sz_extended_idx = NEW_LEN * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_particles_x, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_y, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_extended_idx, sz_extended_idx));

    res_particles_x = (int*)malloc(sz_particles_pos);
    res_particles_y = (int*)malloc(sz_particles_pos);


    /********************************************************************/
    /************************ MAP RESTRUCTURE ***************************/
    /********************************************************************/
    threadsPerBlock = GRID_WIDTH;
    blocksPerGrid = NUM_PARTICLES;

    auto start_map_restructure = std::chrono::high_resolution_clock::now();

    cudaMemset(d_particles_idx, 0, sz_particles_idx);
    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (d_map_2d, d_particles_x, d_particles_y, d_particles_idx,
        d_unique_in_particle, d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT);
    cudaDeviceSynchronize();

    kernel_update_unique_sum << <1, 1 >> > (d_particles_idx, NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_map_restructure = std::chrono::high_resolution_clock::now();

    auto start_copy_particles_pos = std::chrono::high_resolution_clock::now();
    gpuErrchk(cudaMemcpy(res_particles_x, d_particles_x, sz_particles_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_y, d_particles_y, sz_particles_pos, cudaMemcpyDeviceToHost));
    auto stop_copy_particles_pos = std::chrono::high_resolution_clock::now();

    // ASSERT_particles_pos_unique(res_particles_x, res_particles_y, h_particles_x_after_unique, h_particles_y_after_unique, NEW_LEN);

    /********************************************************************/
    /************************* INDEX EXPANSION **************************/
    /********************************************************************/
    auto start_index_expansion = std::chrono::high_resolution_clock::now();

    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;
    kernel_index_expansion << <blocksPerGrid, threadsPerBlock >> > (d_particles_idx, d_extended_idx, NEW_LEN);
    cudaDeviceSynchronize();

    auto stop_index_expansion = std::chrono::high_resolution_clock::now();

    res_extended_idx = (int*)malloc(sz_extended_idx);
    gpuErrchk(cudaMemcpy(res_extended_idx, d_extended_idx, sz_extended_idx, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToHost));


    /********************************************************************/
    /************************ KERNEL CORRELATION ************************/
    /********************************************************************/
    threadsPerBlock = 256;
    blocksPerGrid = (NEW_LEN + threadsPerBlock - 1) / threadsPerBlock;
    printf("*** CUDA kernel launch with %d blocks of %d threads, All Threads: %d ***\n", blocksPerGrid, threadsPerBlock, blocksPerGrid * threadsPerBlock);

    auto start_correlation = std::chrono::high_resolution_clock::now();

    kernel_correlation << <blocksPerGrid, threadsPerBlock >> > (d_grid_map, d_particles_x, d_particles_y, d_extended_idx, d_weights_raw, GRID_WIDTH, GRID_HEIGHT, NEW_LEN);
    cudaDeviceSynchronize();


    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_correlation_max << <blocksPerGrid, threadsPerBlock >> > (d_weights_raw, d_weights, NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_correlation = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_weights, d_weights, sz_weights, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_extended_idx, d_extended_idx, sz_extended_idx, cudaMemcpyDeviceToHost));

    ASSERT_correlation_Equality(res_weights, pre_weights, NUM_PARTICLES);

    /********************************************************************/
    /********************** UPDATE WEIGHTS KERNEL ***********************/
    /********************************************************************/
    auto start_update_particle_weights = std::chrono::high_resolution_clock::now();

    threadsPerBlock = 1;
    blocksPerGrid = 1;

    kernel_arr_max << < blocksPerGrid, threadsPerBlock >> > (d_weights, d_weights_max, NUM_PARTICLES);
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(res_weights_max, d_weights_max, sz_weights_max, cudaMemcpyDeviceToHost));

    float norm_value = -res_weights_max[0] + 50;

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_arr_increase << < blocksPerGrid, threadsPerBlock >> > (d_weights, norm_value, 0);
    cudaDeviceSynchronize();

    threadsPerBlock = 1;
    blocksPerGrid = 1;
    kernel_arr_sum_exp << < blocksPerGrid, threadsPerBlock >> > (d_weights, d_sum_exp, NUM_PARTICLES);
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(res_sum_exp, d_sum_exp, sz_sum_exp, cudaMemcpyDeviceToHost));

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_arr_normalize << < blocksPerGrid, threadsPerBlock >> > (d_weights, res_sum_exp[0]);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_arr_mult << < blocksPerGrid, threadsPerBlock >> > (d_particles_weight, d_weights);
    cudaDeviceSynchronize();

    auto stop_update_particle_weights = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_weights, d_weights, sz_weights, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_weight, d_particles_weight, sz_particles_weight, cudaMemcpyDeviceToHost));

    ASSERT_update_particle_weights(res_weights, new_weights, NUM_PARTICLES, false);
    ASSERT_update_particle_weights(res_particles_weight, particles_weight_post, NUM_PARTICLES, false);


    /********************************************************************/
    /************************ RESAMPLING KERNEL *************************/
    /********************************************************************/
    auto start_resampling = std::chrono::high_resolution_clock::now();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_resampling << <blocksPerGrid, threadsPerBlock >> > (d_weights, d_js, d_rnd, NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_resampling = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_js, d_js, sz_js, cudaMemcpyDeviceToHost));

    ASSERT_resampling_indices(res_js, js, NUM_PARTICLES, false);

    ASSERT_resampling_states(h_states_x, h_states_y, h_states_theta, h_states_x_updated, h_states_y_updated, h_states_theta_updated, res_js, NUM_PARTICLES, false);


    /*---------------------------------------------------------------------*/
    /*----------------- REINITIALIZE PARTICLES VARIABLES ------------------*/
    /*---------------------------------------------------------------------*/
    /*---------------------------------------------------------------------*/
    size_t sz_last_len = sizeof(int);
    int* d_last_len = NULL;
    int* res_last_len = (int*)malloc(sizeof(int));

    gpuErrchk(cudaMalloc((void**)&d_last_len, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dc_particles_x, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&dc_particles_y, sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&dc_particles_idx, sz_particles_idx));

    auto start_clone_particles = std::chrono::high_resolution_clock::now();
    gpuErrchk(cudaMemcpy(dc_particles_x, d_particles_x, sz_particles_pos, cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(dc_particles_y, d_particles_y, sz_particles_pos, cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(dc_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToDevice));

    gpuErrchk(cudaMemcpy(dc_states_x, d_states_x, sz_states_pos, cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(dc_states_y, d_states_y, sz_states_pos, cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(dc_states_theta, d_states_theta, sz_states_pos, cudaMemcpyDeviceToDevice));
    auto stop_clone_particles = std::chrono::high_resolution_clock::now();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_rearrange_indecies << <blocksPerGrid, threadsPerBlock >> > (d_particles_idx, dc_particles_idx, d_js, d_last_len, NEW_LEN);
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(res_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_last_len, d_last_len, sz_last_len, cudaMemcpyDeviceToHost));

    /********************************************************************/
    /********************** REARRANGEMENT KERNEL ************************/
    /********************************************************************/
    auto start_rearrange_index = std::chrono::high_resolution_clock::now();
    kernel_update_unique_sum << <1, 1 >> > (d_particles_idx, NUM_PARTICLES);
    cudaDeviceSynchronize();
    auto stop_rearrange_index = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_particles_idx, d_particles_idx, sz_particles_idx, cudaMemcpyDeviceToHost));
    C_NEW_LEN = NEW_LEN;
    NEW_LEN = res_particles_idx[NUM_PARTICLES - 1] + res_last_len[0];
    printf("--> NEW_LEN=%d <> ELEMS_PARTICLES_AFTER=%d\n", NEW_LEN, ELEMS_PARTICLES_AFTER);
    assert(NEW_LEN + negative_after_counter == ELEMS_PARTICLES_AFTER);

    free(res_particles_x);
    free(res_particles_y);
    sz_particles_pos = NEW_LEN * sizeof(int);
    res_particles_x = (int*)malloc(sz_particles_pos);
    res_particles_y = (int*)malloc(sz_particles_pos);

    ASSERT_resampling_particles_index(h_particles_idx_after_resampling, res_particles_idx, NUM_PARTICLES, false, negative_after_counter);

    auto start_rearrange_particles_states = std::chrono::high_resolution_clock::now();
    threadsPerBlock = 100;
    blocksPerGrid = NUM_PARTICLES;
    kernel_rearrange_particles << <blocksPerGrid, threadsPerBlock >> > (d_particles_x, d_particles_y, d_particles_idx,
        dc_particles_x, dc_particles_y, dc_particles_idx, d_js,
        GRID_WIDTH, GRID_HEIGHT, NUM_PARTICLES, NEW_LEN, C_NEW_LEN);

    kernel_rearrange_states << <blocksPerGrid, threadsPerBlock >> > (d_states_x, d_states_y, d_states_theta,
        dc_states_x, dc_states_y, dc_states_theta, d_js);
    cudaDeviceSynchronize();
    auto stop_rearrange_particles_states = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_particles_x, d_particles_x, sz_particles_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_particles_y, d_particles_y, sz_particles_pos, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaMemcpy(res_states_x, d_states_x, sz_states_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_states_y, d_states_y, sz_states_pos, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_states_theta, d_states_theta, sz_states_pos, cudaMemcpyDeviceToHost));

    ASSERT_rearrange_particles_states(res_particles_x, res_particles_y, res_states_x, res_states_y, res_states_theta,
        h_particles_x_after_resampling, h_particles_y_after_resampling, h_states_x_updated, h_states_y_updated, h_states_theta_updated,
        NEW_LEN, NUM_PARTICLES);


    /********************************************************************/
    /********************** REARRANGEMENT KERNEL ************************/
    /********************************************************************/
    auto start_update_states = std::chrono::high_resolution_clock::now();

    thrust::device_vector<float> d_vec_states_x(d_states_x, d_states_x + NUM_PARTICLES);
    thrust::device_vector<float> d_vec_states_y(d_states_y, d_states_y + NUM_PARTICLES);
    thrust::device_vector<float> d_vec_states_theta(d_states_theta, d_states_theta + NUM_PARTICLES);

    thrust::host_vector<float> h_vec_states_x(d_vec_states_x.begin(), d_vec_states_x.end());
    thrust::host_vector<float> h_vec_states_y(d_vec_states_y.begin(), d_vec_states_y.end());
    thrust::host_vector<float> h_vec_states_theta(d_vec_states_theta.begin(), d_vec_states_theta.end());

    std_vec_states_x.clear();
    std_vec_states_y.clear();
    std_vec_states_theta.clear();
    std_vec_states_x.resize(h_vec_states_x.size());
    std_vec_states_y.resize(h_vec_states_y.size());
    std_vec_states_theta.resize(h_vec_states_theta.size());

    std::copy(h_vec_states_x.begin(), h_vec_states_x.end(), std_vec_states_x.begin());
    std::copy(h_vec_states_y.begin(), h_vec_states_y.end(), std_vec_states_y.begin());
    std::copy(h_vec_states_theta.begin(), h_vec_states_theta.end(), std_vec_states_theta.begin());

    std::map<std::tuple<float, float, float>, int> states;

    for (int i = 0; i < NUM_PARTICLES; i++) {
        if (states.find(std::make_tuple(std_vec_states_x[i], std_vec_states_y[i], std_vec_states_theta[i])) == states.end())
            states.insert({ std::make_tuple(std_vec_states_x[i], std_vec_states_y[i], std_vec_states_theta[i]), 1 });
        else
            states[std::make_tuple(std_vec_states_x[i], std_vec_states_y[i], std_vec_states_theta[i])] += 1;
    }

    std::map<std::tuple<float, float, float>, int>::iterator best
        = std::max_element(states.begin(), states.end(), [](const std::pair<std::tuple<float, float, float>, int>& a,
            const std::pair<std::tuple<float, float, float>, int>& b)->bool { return a.second < b.second; });

    auto key = best->first;

    float theta = std::get<2>(key);
    float n_res_transition_body_frame[] = { cos(theta), -sin(theta), std::get<0>(key),
                        sin(theta),  cos(theta), std::get<1>(key),
                        0, 0, 1 };
    auto stop_update_states = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 9; i++) {
        printf("%f  ", n_res_transition_body_frame[i]);
        // assert( abs(h_transition_body_frame[i] - n_res_transition_body_frame[i]) < 1e-2);
        if (abs(h_transition_body_frame[i] - n_res_transition_body_frame[i]) > 1e-4) {
            printf("(%f) ", h_transition_body_frame[i]);
        }
    }
    printf("\n");
    printf("%f, %f, %f\n", std::get<0>(key), std::get<1>(key), std::get<2>(key));


    /********************************************************************/
    /************************* EXECUTION TIMES **************************/
    /********************************************************************/
    auto duration_create_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_create_map - start_create_map);
    auto duration_update_map = std::chrono::duration_cast<std::chrono::microseconds>(stop_update_map - start_update_map);
    auto duration_cumulative_sum = std::chrono::duration_cast<std::chrono::microseconds>(stop_cumulative_sum - start_cumulative_sum);
    auto duration_map_restructure = std::chrono::duration_cast<std::chrono::microseconds>(stop_map_restructure - start_map_restructure);
    auto duration_copy_particles_pos = std::chrono::duration_cast<std::chrono::microseconds>(stop_copy_particles_pos - start_copy_particles_pos);
    auto duration_transition_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_transition_kernel - start_transition_kernel);
    auto duration_correlation = std::chrono::duration_cast<std::chrono::microseconds>(stop_correlation - start_correlation);
    auto duration_update_particle_weights = std::chrono::duration_cast<std::chrono::microseconds>(stop_update_particle_weights - start_update_particle_weights);
    auto duration_resampling = std::chrono::duration_cast<std::chrono::microseconds>(stop_resampling - start_resampling);
    auto duration_clone_particles = std::chrono::duration_cast<std::chrono::microseconds>(stop_clone_particles - start_clone_particles);
    auto duration_rearrange_particles_states = std::chrono::duration_cast<std::chrono::microseconds>(stop_rearrange_particles_states - start_rearrange_particles_states);
    auto duration_rearrange_index = std::chrono::duration_cast<std::chrono::microseconds>(stop_rearrange_index - start_rearrange_index);
    auto duration_update_states = std::chrono::duration_cast<std::chrono::microseconds>(stop_update_states - start_update_states);

    auto duration_sum = duration_create_map + duration_update_map + duration_cumulative_sum + duration_map_restructure + duration_copy_particles_pos +
        duration_transition_kernel + duration_correlation + duration_update_particle_weights + duration_resampling + duration_clone_particles +
        duration_rearrange_particles_states + duration_rearrange_index + duration_update_states;

    std::cout << "Time taken by function (Create Map): " << duration_create_map.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Update Map): " << duration_update_map.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Cumulative Sum): " << duration_cumulative_sum.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Map Restructure): " << duration_map_restructure.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Copy Particles): " << duration_copy_particles_pos.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Transition Kernel): " << duration_transition_kernel.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Correlation Kernel): " << duration_correlation.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Update Particle Weights): " << duration_update_particle_weights.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Kernel Resampling): " << duration_resampling.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Clone Particles): " << duration_clone_particles.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Rearrange Particles States): " << duration_rearrange_particles_states.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Rearrange Index): " << duration_rearrange_index.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Update States): " << duration_update_states.count() << " microseconds" << std::endl;
    std::cout << "Time taken by function (Sum): " << duration_sum.count() << " microseconds" << std::endl;

    printf("\nFinished All\n");

}
#endif



#endif
