
#include "headers.h"
#include "host_asserts.h"
#include "host_utils.h"
#include "device_utils.cuh"

// #define CORRELATION_EXEC
// #define BRESENHAM_EXEC
// #define UPDATE_MAP_EXEC
#define UPDATE_STATE_EXEC
// #define UPDATE_PARTICLE_WEIGHTS_EXEC
// #define RESAMPLING_EXEC
// #define UPDATE_PARTICLES_EXEC
// #define UPDATE_UNIQUE_EXEC
// #define UPDATE_LOOP_EXEC


#ifdef CORRELATION_EXEC
#include "data/map_correlation/4500.h"
#endif

#ifdef BRESENHAM_EXEC
#include "data/bresenham/500.h"
#endif

#ifdef UPDATE_MAP_EXEC
#include "data/log_odds/100.h"
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
#include "data/update_unique/200.h"
// #include "data/update_unique/4800.h"
#endif

#ifdef UPDATE_LOOP_EXEC
#include "data/update_loop/2500.h"
#endif

__global__ void kernel_bresenham(const int* arr_start_x, const int* arr_start_y, const int end_x, const int end_y,
    int* result_array_x, int* result_array_y, const int result_len, const int* index_array);

__global__ void kernel_index_init_const(int* indices, const int value);

__global__ void kernel_index_expansion(const int* idx, int* extended_idx, const int numElements);
__global__ void kernel_correlation_max(const int* correlation_raw, int* correlation, const int _NUM_PARTICLES);
__global__ void kernel_correlation(const int* d_grid_map, const int* d_Y_io_x, const int* d_Y_io_y,
                                    const int* d_Y_io_idx, int* result, const int _GRID_WIDTH, const int _GRID_HEIGHT, int numElements);


__global__ void kernel_update_log_odds(float *log_odds, int *f_x, int *f_y, const float _log_t,
                                        const int _GRID_WIDTH, const int _GRID_HEIGHT, const int numElements);

__global__ void kernel_update_map(int* grid_map, const float* log_odds, const float _LOG_ODD_PRIOR, const int _WALL, const int _FREE, const int numElements);

__global__ void kernel_resampling(const float* weights, int* js, const float* rnd, const int numElements);

__global__ void kernel_update_particles_states(const float* states_x, const float* states_y, const float* states_theta,
                                                float* transition_body_frame, const float* transition_lidar_frame, float* transition_world_frame, const int numElements);

__global__ void kernel_update_particles_lidar(float* transition_world_frame, int* processed_measure_x, int* processed_measure_y, const float* _lidar_coords, float _res, int _xmin, int _ymax,
                                                const int _lidar_coords_LEN, const int numElements);

__device__ void kernel_matrix_mul_3x3(const float* A, const float* B, float* C, int start_i);

__global__ void kernel_create_2d_map(const int* Y_io_x, const int* Y_io_y, const int* idx, const int IDX_LEN, uint8_t* whole_map,
    int* unique_in_particle, int* unique_in_particle_col, const int _GRID_WIDTH, const int _GRID_HEIGHT, const int _NUM_ELEMS);

__global__ void kernel_update_2d_map_with_measure(const int* Y_io_x, const int* Y_io_y, const int* idx, const int IDX_LEN, uint8_t* whole_map,
    int* unique_in_particle, int* unique_in_particle_col, const int _GRID_WIDTH, const int _GRID_HEIGHT, const int _NUM_ELEMS);

__global__ void kernel_update_unique_restructure(uint8_t* map_2d, int* particles_x, int* particles_y, int* particles_idx, int* unique_in_each_particle, int* unique_in_each_particle_col,
                                        const int _GRID_WIDTH, const int _GRID_HEIGHT);

__global__ void kernel_arr_increase(int* arr, const int increase_value, const int start_index);
__global__ void kernel_arr_increase(float* arr, const float increase_value, const int start_index);
__global__ void kernel_arr_max(float* arr, float* result, const int LEN);
__global__ void kernel_arr_sum_exp(float* arr, double* result, const int LEN);
__global__ void kernel_arr_normalize(float* arr, const double norm);
__global__ void kernel_update_unique_sum(int* unique_in_particle, const int _NUM_ELEMS);
__global__ void kernel_update_unique_sum_col(int* unique_in_particle_col, const int _GRID_WIDTH);


void host_correlation();
void host_bresenham();
void host_update_map();
void host_update_state();
void host_update_particle_weights();
void host_resampling();
void host_update_particles();
void host_update_unique();
void host_update_loop();

int main() {


#ifdef CORRELATION_EXEC
    host_correlation();
#endif

#ifdef BRESENHAM_EXEC
    host_bresenham();
#endif

#ifdef UPDATE_MAP_EXEC
    host_update_map();
#endif

#ifdef UPDATE_STATE_EXEC
    host_update_state();
#endif

//#ifdef __cplusplus
//    printf("C++\n");
//#endif

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



    return 0;
}


/*
* Host Functions
*/

#ifdef CORRELATION_EXEC
void host_correlation() {

    // [✓] - Create a kernel for expanding indices
    // [✓] - Create 'exp_idx' variable with len 'Y_Len'
    // [✓] - Change 'kernel_index_expansion' like 'kernel_create_2d_map'
    // [✓] - Change  threadsPerBlock = 100; blocksPerGrid = NUM_ELEMS for 'kernel_index_expansion'
    // [✓] - Create 'kernel_correlation_max'


    auto start_memory_copy = std::chrono::high_resolution_clock::now();

    /********************************************************************/
    /************************** PRIOR VARIABLES *************************/
    /********************************************************************/
    int* d_grid_map         = NULL;
    int* d_particles_x      = NULL;
    int* d_particles_y      = NULL;
    int* d_particles_idx    = NULL;
    int* d_extended_idx     = NULL;

    const int num_elements_of_grid_map  = GRID_WIDTH * GRID_HEIGHT;
    size_t sz_grid_map                  = num_elements_of_grid_map * sizeof(int);

    // const int num_elements_of_Y =  Y_Len; // Y_LENGTH;
    size_t sz_particles_pos     = elems_particles * sizeof(int);
    size_t sz_particles_idx     = NUM_PARTICLES   * sizeof(int);
    size_t sz_extended_idx      = elems_particles * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_grid_map,       sz_grid_map));
    gpuErrchk(cudaMalloc((void**)&d_particles_x,    sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_y,    sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_extended_idx,   sz_extended_idx));
    gpuErrchk(cudaMalloc((void**)&d_particles_idx,  sz_particles_idx));

    cudaMemcpy(d_grid_map,      grid_map,           sz_grid_map,        cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles_x,   h_particles_x,      sz_particles_pos,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles_y,   h_particles_y,      sz_particles_pos,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles_idx,      h_particles_idx,    sz_particles_idx,        cudaMemcpyHostToDevice);

    size_t sz_correlation           = NUM_PARTICLES * sizeof(int);
    size_t sz_correlation_raw       = 25 * sz_correlation;
    int* h_correlation              = (int*)malloc(sz_correlation);
    int* h_extended_idx             = (int*)malloc(sz_extended_idx);
    int* d_correlation              = NULL;
    int* d_correlation_raw          = NULL;
    memset(h_correlation, 0, sz_correlation);

    gpuErrchk(cudaMalloc((void**)&d_correlation,        sz_correlation));
    gpuErrchk(cudaMalloc((void**)&d_correlation_raw,    sz_correlation_raw));
    gpuErrchk(cudaMemset(d_correlation_raw, 0, sz_correlation_raw));
    // gpuErrchk(cudaMemcpy(d_all_correlation, h_correlation, sz_all_correlation, cudaMemcpyHostToDevice));

    auto stop_memory_copy = std::chrono::high_resolution_clock::now();


    /********************************************************************/
    /*************************** PRINT SUMMARY **************************/
    /********************************************************************/
    printf("Elements of Y_io_x: %d,  Size of Y_io_x: %d\n", (int)elems_particles, (int)sz_particles_pos);
    printf("Elements of Y_io_y: %d,  Size of Y_io_y: %d\n", (int)elems_particles, (int)sz_particles_pos);
    printf("Elements of Y_io_idx: %d,  Size of Y_io_idx: %d\n", (int)elems_particles, (int)sz_extended_idx);

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

    kernel_correlation << <blocksPerGrid, threadsPerBlock >> > (d_grid_map, d_particles_x, d_particles_y, d_extended_idx, d_correlation_raw, GRID_WIDTH, GRID_HEIGHT, elems_particles);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_correlation_max << <blocksPerGrid, threadsPerBlock >> > (d_correlation_raw, d_correlation, NUM_PARTICLES);

    auto stop_kernel = std::chrono::high_resolution_clock::now();


    gpuErrchk(cudaMemcpy(h_correlation, d_correlation, sz_correlation, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_extended_idx, d_extended_idx, sz_extended_idx, cudaMemcpyDeviceToHost));
    

    bool all_equal = true;
    for (int i = 0; i < NUM_PARTICLES; i++) {
        // printf("index: %d --> %d, %d\n", i, final_result[i], new_weights[i]); 
        if (h_correlation[i] != new_weights[i])
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
    gpuErrchk(cudaFree(d_correlation_raw));
}
#endif

#ifdef BRESENHAM_EXEC
void host_bresenham() {

    float time_total;
    cudaEvent_t start_total, stop_total;
    gpuErrchk(cudaEventCreate(&start_total));
    gpuErrchk(cudaEventCreate(&stop_total));

    size_t size_of_array_start = Y_io_shape * sizeof(int);
    size_t size_of_result = Y_if_shape * sizeof(int);

    int* d_start_x = NULL;
    int* d_start_y = NULL;
    int* d_index_array = NULL;

    int* d_result_array_x = NULL;
    int* d_result_array_y = NULL;

    gpuErrchk(cudaMalloc((void**)&d_start_x, size_of_array_start));
    gpuErrchk(cudaMalloc((void**)&d_start_y, size_of_array_start));
    gpuErrchk(cudaMalloc((void**)&d_index_array, size_of_array_start));

    gpuErrchk(cudaMalloc((void**)&d_result_array_x, size_of_result));
    gpuErrchk(cudaMalloc((void**)&d_result_array_y, size_of_result));

    int* result_x = (int*)malloc(size_of_result);
    int* result_y = (int*)malloc(size_of_result);
    memset(result_x, 0, size_of_result);
    memset(result_y, 0, size_of_result);


    cudaMemcpy(d_start_x, Y_io_x, size_of_array_start, cudaMemcpyHostToDevice);
    cudaMemcpy(d_start_y, Y_io_y, size_of_array_start, cudaMemcpyHostToDevice);
    cudaMemcpy(d_index_array, free_idx, size_of_array_start, cudaMemcpyHostToDevice);


    cudaMemcpy(d_result_array_x, result_x, size_of_result, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result_array_y, result_y, size_of_result, cudaMemcpyHostToDevice);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (Y_io_shape + threadsPerBlock - 1) / threadsPerBlock;
    // printf("CUDA kernel launch with %d blocks of %d threads, All Threads: %d\n", blocksPerGrid, threadsPerBlock, blocksPerGrid * threadsPerBlock);

    gpuErrchk(cudaEventRecord(start_total, 0));

    kernel_bresenham << <blocksPerGrid, threadsPerBlock >> > (d_start_x, d_start_y, p_ib[0], p_ib[1], d_result_array_x, d_result_array_y, Y_io_shape, d_index_array);
    cudaDeviceSynchronize();

    gpuErrchk(cudaEventRecord(stop_total, 0));
    gpuErrchk(cudaEventSynchronize(stop_total));
    gpuErrchk(cudaEventElapsedTime(&time_total, start_total, stop_total));

    printf("Total Time of Execution:  %3.1f ms\n", time_total);

    gpuErrchk(cudaMemcpy(result_x, d_result_array_x, size_of_result, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(result_y, d_result_array_y, size_of_result, cudaMemcpyDeviceToHost));

    bool all_equal = true;
    int errors = 0;
    printf("Start\n");
    for (int i = 0; i < Y_if_shape; i++) {
        if (result_x[i] != free_x[i] || result_y[i] != free_y[i]) {
            all_equal = false;
            errors += 1;
            printf("%d -- %d, %d | %d, %d\n", i, result_x[i], free_x[i], result_y[i], free_y[i]);
        }
    }

    printf("All Equal: %s\n", all_equal ? "true" : "false");
    printf("Errors: %d\n", errors);


    printf("Program Finished\n");

    gpuErrchk(cudaFree(d_start_x));
    gpuErrchk(cudaFree(d_start_y));
    gpuErrchk(cudaFree(d_index_array));
    gpuErrchk(cudaFree(d_result_array_x));
    gpuErrchk(cudaFree(d_result_array_y));
}
#endif

#ifdef UPDATE_MAP_EXEC
void host_update_map() {

    size_t size_of_io = Y_io_LEN * sizeof(int);
    size_t size_of_if = Y_if_LEN * sizeof(int);
    size_t size_of_log_odds = (GRID_WIDTH * GRID_HEIGHT) * sizeof(float);
    size_t size_of_map = (GRID_WIDTH * GRID_HEIGHT) * sizeof(int);

    float* result_log_odds = (float*)malloc(size_of_log_odds);
    memset(result_log_odds, 0, size_of_log_odds);

    int* result_grid_map = (int*)malloc(size_of_map);

    int* d_Y_io_x = NULL;
    int* d_Y_io_y = NULL;
    int* d_Y_if_x = NULL;
    int* d_Y_if_y = NULL;

    float* d_log_odds = NULL;
    int* d_grid_map = NULL;

    gpuErrchk(cudaMalloc((void**)&d_Y_io_x, size_of_io));
    gpuErrchk(cudaMalloc((void**)&d_Y_io_y, size_of_io));
    gpuErrchk(cudaMalloc((void**)&d_Y_if_x, size_of_if));
    gpuErrchk(cudaMalloc((void**)&d_Y_if_y, size_of_if));

    gpuErrchk(cudaMalloc((void**)&d_log_odds, size_of_log_odds));
    gpuErrchk(cudaMalloc((void**)&d_grid_map, size_of_map));

    cudaMemcpy(d_Y_io_x, Y_io_x, size_of_io, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y_io_y, Y_io_y, size_of_io, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y_if_x, Y_if_x, size_of_if, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y_if_y, Y_if_y, size_of_if, cudaMemcpyHostToDevice);

    cudaMemcpy(d_log_odds, pre_log_odds, size_of_log_odds, cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid_map, pre_grid_map, size_of_map, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (Y_io_LEN + threadsPerBlock - 1) / threadsPerBlock;

    kernel_update_log_odds << <blocksPerGrid, threadsPerBlock >> > (d_log_odds, d_Y_io_x, d_Y_io_y, 2 * log_t, GRID_WIDTH, GRID_HEIGHT, Y_io_LEN);
    cudaDeviceSynchronize();

    threadsPerBlock = 256;
    blocksPerGrid = (Y_if_LEN + threadsPerBlock - 1) / threadsPerBlock;

    kernel_update_log_odds << <blocksPerGrid, threadsPerBlock >> > (d_log_odds, d_Y_if_x, d_Y_if_y, (-1) * log_t, GRID_WIDTH, GRID_HEIGHT, Y_if_LEN);
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(result_log_odds, d_log_odds, size_of_log_odds, cudaMemcpyDeviceToHost));


    threadsPerBlock = 256;
    blocksPerGrid = ((GRID_WIDTH * GRID_HEIGHT) + threadsPerBlock - 1) / threadsPerBlock;

    // int* grid_map, const float* log_odds, const int _LOG_ODD_PRIOR, const int _WALL, const int _FREE, const int numElements
    kernel_update_map << <blocksPerGrid, threadsPerBlock >> > (d_grid_map, d_log_odds, LOG_ODD_PRIOR, WALL, FREE, GRID_WIDTH * GRID_HEIGHT);
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(result_grid_map, d_grid_map, size_of_map, cudaMemcpyDeviceToHost));


    int numError = 0;
    int numCorrect = 0;
    for (int i = 0; i < (GRID_WIDTH * GRID_HEIGHT); i++) {

        if (abs(result_log_odds[i] - post_log_odds[i]) > 0.1) {
            printf("%d: %f, %f, %f\n", i, result_log_odds[i], post_log_odds[i], pre_log_odds[i]);
            numError += 1;
        }
        else if (post_log_odds[i] != pre_log_odds[i]) {
            numCorrect += 1;
        }
    }
    printf("Error: %d, Correct: %d\n", numError, numCorrect);



    numError = 0;
    numCorrect = 0;
    for (int i = 0; i < (GRID_WIDTH * GRID_HEIGHT); i++) {

        if (abs(result_grid_map[i] - post_grid_map[i]) > 0.1) {
            printf("%d: %d, %d, %d\n", i, result_grid_map[i], pre_grid_map[i], post_grid_map[i]);
            numError += 1;
        }
        else {
            numCorrect += 1;
        }
    }

    printf("Error: %d, Correct: %d\n", numError, numCorrect);

}

#endif

#ifdef UPDATE_STATE_EXEC
void host_update_state() {

    thrust::device_vector<float> d_temp(xs, xs + NUM_PARTICLES);

    size_t sz_states_pos = NUM_PARTICLES * sizeof(float);

    float* d_states_x       = NULL;
    float* d_states_y       = NULL;
    float* d_states_theta   = NULL;

    gpuErrchk(cudaMalloc((void**)&d_states_x,       sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&d_states_y,       sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&d_states_theta,   sz_states_pos));


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
    size_t sz_weights       = NUM_PARTICLES * sizeof(float);
    size_t sz_weights_max   = sizeof(float);
    size_t sz_sum_exp       = sizeof(double);
    
    float*  d_weights        = NULL;
    float*  d_weights_max    = NULL;
    double* d_sum_exp        = NULL;


    float*  res_weights      = (float*)malloc(sz_weights);
    float*  res_weights_max  = (float*)malloc(sz_weights_max);
    double* res_sum_exp      = (double*)malloc(sz_sum_exp);

    gpuErrchk(cudaMalloc((void**)&d_weights,        sz_weights));
    gpuErrchk(cudaMalloc((void**)&d_weights_max,    sz_weights_max));
    gpuErrchk(cudaMalloc((void**)&d_sum_exp,        sz_sum_exp));


    gpuErrchk(cudaMemcpy(d_weights, pre_weights, sz_weights, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(d_weights_max, 0, sz_weights_max));
    gpuErrchk(cudaMemset(d_sum_exp,     0, sz_sum_exp));


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
    float*  d_weights   = NULL;
    int*    d_js        = NULL;
    float*  d_rnd       = NULL;

    size_t sz_weights   = NUM_PARTICLES * sizeof(float);
    size_t sz_js        = NUM_PARTICLES * sizeof(int);
    size_t sz_rnd       = NUM_PARTICLES * sizeof(float);

    int* res_js = (int*)malloc(sz_js);

    gpuErrchk(cudaMalloc((void**)&d_weights,    sz_weights));
    gpuErrchk(cudaMalloc((void**)&d_js,         sz_js));
    gpuErrchk(cudaMalloc((void**)&d_rnd,        sz_rnd));

    cudaMemcpy(d_weights,   weights,    sz_weights,     cudaMemcpyHostToDevice);
    cudaMemcpy(d_js,        res_js,     sz_js,          cudaMemcpyHostToDevice);
    cudaMemcpy(d_rnd,       rnds,       sz_rnd,         cudaMemcpyHostToDevice);

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
    size_t sz_states        = NUM_PARTICLES * sizeof(float);
    size_t sz_lidar_coords  = 2 * lidar_coords_LEN * sizeof(float);

    float* d_states_x     = NULL;
    float* d_states_y     = NULL;
    float* d_states_theta = NULL;
    float* d_lidar_coords = NULL;

    gpuErrchk(cudaMalloc((void**)&d_states_x,       sz_states));
    gpuErrchk(cudaMalloc((void**)&d_states_y,       sz_states));
    gpuErrchk(cudaMalloc((void**)&d_states_theta,   sz_states));
    gpuErrchk(cudaMalloc((void**)&d_lidar_coords,   sz_lidar_coords));

    cudaMemcpy(d_states_x,      h_states_x,     sz_states,          cudaMemcpyHostToDevice);
    cudaMemcpy(d_states_y,      h_states_y,     sz_states,          cudaMemcpyHostToDevice);
    cudaMemcpy(d_states_theta,  h_states_theta, sz_states,          cudaMemcpyHostToDevice);
    cudaMemcpy(d_lidar_coords,  lidar_coords,   sz_lidar_coords,    cudaMemcpyHostToDevice);


    /********************************************************************/
    /************************* MIDDLE VARIABLES *************************/
    /********************************************************************/
    size_t sz_transition_body_frame     = 9 * NUM_PARTICLES * sizeof(float);
    size_t sz_transition_lidar_frame    = 9 * sizeof(float);
    size_t sz_transition_world_frame    = 9 * NUM_PARTICLES * sizeof(float);
    size_t sz_processed_measure_pos     = NUM_PARTICLES * lidar_coords_LEN * sizeof(int);
    size_t sz_measure_idx               = NUM_PARTICLES * lidar_coords_LEN * sizeof(int);

    float* d_transition_body_frame  = NULL;
    float* d_transition_lidar_frame = NULL;
    float* d_transition_world_frame = NULL;
    int*   d_processed_measure_x    = NULL;
    int*   d_processed_measure_y    = NULL;
    int*   d_measure_idx            = NULL;

    gpuErrchk(cudaMalloc((void**)&d_transition_body_frame,  sz_transition_body_frame));
    gpuErrchk(cudaMalloc((void**)&d_transition_lidar_frame, sz_transition_lidar_frame));
    gpuErrchk(cudaMalloc((void**)&d_transition_world_frame, sz_transition_world_frame));
    gpuErrchk(cudaMalloc((void**)&d_processed_measure_x,    sz_processed_measure_pos));
    gpuErrchk(cudaMalloc((void**)&d_processed_measure_y,    sz_processed_measure_pos));
    gpuErrchk(cudaMalloc((void**)&d_measure_idx,            sz_measure_idx));


    /********************************************************************/
    /************************* HOST VARIABLES ***************************/
    /********************************************************************/
    float* res_transition_body_frame    = (float*)malloc(sz_transition_body_frame);
    float* res_transition_world_frame   = (float*)malloc(sz_transition_world_frame);
    int*   res_processed_measure_x      = (int*)malloc(sz_processed_measure_pos);
    int*   res_processed_measure_y      = (int*)malloc(sz_processed_measure_pos);
    int*   res_measure_idx              = (int*)malloc(sz_measure_idx);

    memset(res_transition_body_frame,   0, sz_transition_body_frame);
    memset(res_transition_world_frame,  0, sz_transition_world_frame);
    memset(res_processed_measure_x,     0, sz_processed_measure_pos);
    memset(res_processed_measure_y,     0, sz_processed_measure_pos);


    cudaMemcpy(d_transition_body_frame,     res_transition_body_frame,  sz_transition_body_frame,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_transition_world_frame,    res_transition_world_frame, sz_transition_world_frame,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_processed_measure_x,       res_processed_measure_x,    sz_processed_measure_pos,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_processed_measure_y,       res_processed_measure_y,    sz_processed_measure_pos,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_transition_lidar_frame,    h_transition_lidar_frame,   sz_transition_lidar_frame,  cudaMemcpyHostToDevice);


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

    gpuErrchk(cudaMemcpy(res_transition_body_frame,     d_transition_body_frame,    sz_transition_body_frame,   cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_transition_world_frame,    d_transition_world_frame,   sz_transition_world_frame,  cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_processed_measure_x,       d_processed_measure_x,      sz_processed_measure_pos,   cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_processed_measure_y,       d_processed_measure_y,      sz_processed_measure_pos,   cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_measure_idx,               d_measure_idx,              sz_measure_idx,             cudaMemcpyDeviceToHost));

    bool printVerbose = false;

    for (int i = 0; i < 9 * NUM_PARTICLES; i++) {
        if (printVerbose == true) printf("%f, %f | ", res_transition_body_frame[i], h_transition_body_frame[i]);
        assert(abs(res_transition_body_frame[i] - h_transition_body_frame[i]) < 1e-5);
    }
    for (int i = 0; i < 9 * NUM_PARTICLES; i++) {
        if(printVerbose == true) printf("%f, %f |  ", res_transition_world_frame[i], h_transition_world_frame[i]);
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
        if(printVerbose == true) printf("index %d --> value: %d, diff: %d\n", i, res_measure_idx[i], diff);
    }
}
#endif

#ifdef UPDATE_UNIQUE_EXEC
void host_update_unique() {

    int negative_before_counter = 0;
    int count_bigger_than_height = 0;
    for (int i = 0; i < BEFORE_LEN; i++) {
        if (h_particles_x_prior[i] < 0 || h_particles_y_prior[i] < 0)
            negative_before_counter += 1;

        if (h_particles_y_prior[i] >= GRID_HEIGHT)
            count_bigger_than_height += 1;
    }

    int negative_after_counter = 0;
    for (int i = 0; i < AFTER_LEN; i++) {
        if (h_particles_x_post[i] < 0 || h_particles_y_post[i] < 0)
            negative_after_counter += 1;
    }

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

    const int NUM_ELEMS = NUM_PARTICLES + 1;


    /********************************************************************/
    /************************** PRIOR VARIABLES *************************/
    /********************************************************************/
    int* d_particles_x      = NULL;
    int* d_particles_y      = NULL;
    int* d_particles_idx    = NULL;
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
    size_t   sz_unique_in_particle = NUM_ELEMS * sizeof(int);
    size_t   sz_unique_in_particle_col = NUM_ELEMS * GRID_WIDTH * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_map_2d,                 sz_map_2d));
    gpuErrchk(cudaMalloc((void**)&d_unique_in_particle,     sz_unique_in_particle));
    gpuErrchk(cudaMalloc((void**)&d_unique_in_particle_col, sz_unique_in_particle_col));

    int* h_unique_in_particle       = (int*)malloc(sz_unique_in_particle);
    int* h_unique_in_particle_col   = (int*)malloc(sz_unique_in_particle_col);
    uint8_t* h_map_2d               = (uint8_t*)malloc(sz_map_2d);

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
    int blocksPerGrid = NUM_ELEMS;

    auto start_create_map = std::chrono::high_resolution_clock::now();

    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (d_particles_x, d_particles_y, d_particles_idx, BEFORE_LEN, d_map_2d, d_unique_in_particle,
        d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT, NUM_ELEMS);
    cudaDeviceSynchronize();

    auto stop_create_map = std::chrono::high_resolution_clock::now();


    /********************************************************************/
    /**************************** UPDATE MAP ****************************/
    /********************************************************************/
    auto start_update_map = std::chrono::high_resolution_clock::now();

    threadsPerBlock = NUM_ELEMS;
    blocksPerGrid = 1;

    kernel_update_2d_map_with_measure << <blocksPerGrid, threadsPerBlock >> > (d_measure_x, d_measure_y, d_measure_idx, MEASURE_LEN, d_map_2d, d_unique_in_particle,
        d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT, NUM_ELEMS);
    cudaDeviceSynchronize();

    auto stop_update_map = std::chrono::high_resolution_clock::now();


    /********************************************************************/
    /************************* CUMULATIVE SUM ***************************/
    /********************************************************************/
    threadsPerBlock = NUM_ELEMS;
    blocksPerGrid = 1;

    auto start_cumulative_sum = std::chrono::high_resolution_clock::now();

    kernel_update_unique_sum << <1, 1 >> > (d_unique_in_particle, NUM_ELEMS);
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (d_unique_in_particle_col, GRID_WIDTH);
    cudaDeviceSynchronize();

    auto stop_cumulative_sum = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(h_map_2d, d_map_2d, sz_map_2d, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_unique_in_particle, d_unique_in_particle, sz_unique_in_particle, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_unique_in_particle_col, d_unique_in_particle_col, sz_unique_in_particle_col, cudaMemcpyDeviceToHost));

    int new_len = h_unique_in_particle[NUM_ELEMS - 1];
    gpuErrchk(cudaFree(d_particles_x));
    gpuErrchk(cudaFree(d_particles_y));

    sz_particles_pos = new_len * sizeof(int);
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

    printf("\nunique_in_particle: %d\n", h_unique_in_particle[NUM_ELEMS - 1]);
    printf("Measurement Length: %d\n", MEASURE_LEN);

    int num_fails = 0;
    for (int i = 0, j = 0; i < h_unique_in_particle[NUM_ELEMS - 1]; i++) {

        if (h_particles_x_post[i] > 0 && h_particles_y_post[i] > 0) {

            //if (Y_io_x_after[i] != res_Y_io_x[j] || Y_io_y_after[i] != res_Y_io_y[j]) {
            //    printf("%d: %d=%d, %d=%d\n", j, res_Y_io_x[j], Y_io_x_after[i], res_Y_io_y[j], Y_io_y_after[i]);
            //    num_fails += 1;
            //    if (num_fails > 70)
            //        exit(0);
            //}
            assert(h_particles_x_post[i] == res_particles_x[j]);
            assert(h_particles_y_post[i] == res_particles_y[j]);
            j += 1;
        }
    }

    //for (int i = 0; i < NUM_PARTICLES; i++) {
    //    int diff = (i == 0) ? 0 : (h_measure_idx[i] - h_measure_idx[i - 1]);
    //    printf("index %d --> value: %d, diff: %d\n", i, h_measure_idx[i], diff);
    //}

    //for (int i = 0; i < NUM_PARTICLES; i++)
    //    printf("index %d: %d\n", i, res_particles_idx[i]);

    printf("All Passed\n");

}
#endif

#ifdef UPDATE_LOOP_EXEC
void host_update_loop() {

    // ✓

    int negative_before_counter  = getNegativeCounter(h_particles_x, h_particles_y, ELEMS_PARTICLES_START);
    int count_bigger_than_height = getGreaterThanCounter(h_particles_y, GRID_HEIGHT, ELEMS_PARTICLES_START);
    int negative_after_counter   = getNegativeCounter(h_particles_x_after_unique, h_particles_y_after_unique, ELEMS_PARTICLES_AFTER);;

    printf("GRID_WIDTH: %d, GRID_HEIGHT: %d\n", GRID_WIDTH, GRID_HEIGHT);
    printf("negative_before_counter: %d\n", negative_before_counter);
    printf("negative_after_counter: %d\n", negative_after_counter);
    printf("count_bigger_than_height: %d\n", count_bigger_than_height);


    const int NUM_ELEMS     = NUM_PARTICLES + 1;
    const int MEASURE_LEN   = NUM_PARTICLES * LIDAR_COORDS_LEN;

    printf("MEASURE_LEN: %d\n", MEASURE_LEN);

    /********************************************************************/
    /************************** PRIOR VARIABLES *************************/
    /********************************************************************/
    size_t sz_states_pos    = NUM_PARTICLES * sizeof(float);
    size_t sz_lidar_coords  = 2 * LIDAR_COORDS_LEN * sizeof(float);

    float* d_states_x           = NULL;
    float* d_states_y           = NULL;
    float* d_states_theta       = NULL;
    float* d_lidar_coords       = NULL;

    gpuErrchk(cudaMalloc((void**)&d_states_x,       sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&d_states_y,       sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&d_states_theta,   sz_states_pos));
    gpuErrchk(cudaMalloc((void**)&d_lidar_coords,   sz_lidar_coords));

    gpuErrchk(cudaMemcpy(d_states_x,      h_states_x,     sz_states_pos,      cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_states_y,      h_states_y,     sz_states_pos,      cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_states_theta,  h_states_theta, sz_states_pos,      cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_lidar_coords,  lidar_coords,   sz_lidar_coords,    cudaMemcpyHostToDevice));

    /********************************************************************/
    /**************************** MAP VARIABLES *************************/
    /********************************************************************/
    size_t sz_particles_pos = ELEMS_PARTICLES_START * sizeof(int);
    size_t sz_particles_idx = NUM_PARTICLES * sizeof(int);
    size_t sz_extended_idx  = ELEMS_PARTICLES_START * sizeof(int);
    size_t sz_grid_map      = GRID_WIDTH * GRID_HEIGHT * sizeof(int);

    int* d_grid_map         = NULL;
    int* d_particles_x      = NULL;
    int* d_particles_y      = NULL;
    int* d_particles_idx    = NULL;
    int* d_extended_idx     = NULL;

    int* res_particles_x    = (int*)malloc(sz_particles_pos);
    int* res_particles_y    = (int*)malloc(sz_particles_pos);
    int* res_particles_idx  = (int*)malloc(sz_particles_idx);
    int* res_extended_idx   = (int*)malloc(sz_extended_idx);

    gpuErrchk(cudaMalloc((void**)&d_grid_map,       sz_grid_map));
    gpuErrchk(cudaMalloc((void**)&d_particles_x,    sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_y,    sz_particles_pos));
    gpuErrchk(cudaMalloc((void**)&d_particles_idx,  sz_particles_idx));
    gpuErrchk(cudaMalloc((void**)&d_extended_idx,   sz_extended_idx));
    
    cudaMemcpy(d_grid_map,      grid_map,         sz_grid_map,        cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles_x,   h_particles_x,    sz_particles_pos,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles_y,   h_particles_y,    sz_particles_pos,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles_idx, h_particles_idx,  sz_particles_idx,   cudaMemcpyHostToDevice);


    /********************************************************************/
    /********************** CORRELATION VARIABLES ***********************/
    /********************************************************************/
    size_t sz_correlation       = NUM_PARTICLES * sizeof(int);
    size_t sz_correlation_raw   = 25 * sz_correlation;

    int* h_correlation      = (int*)malloc(sz_correlation);
    int* h_extended_idx     = (int*)malloc(sz_extended_idx);
    int* res_correlation    = (int*)malloc(sz_correlation);
    int* d_correlation      = NULL;
    int* d_correlation_raw  = NULL;
    memset(h_correlation, 0, sz_correlation);

    gpuErrchk(cudaMalloc((void**)&d_correlation,        sz_correlation));
    gpuErrchk(cudaMalloc((void**)&d_correlation_raw,    sz_correlation_raw));
    gpuErrchk(cudaMemset(d_correlation_raw,     0,      sz_correlation_raw));


    /********************************************************************/
    /*********************** TRANSITION VARIABLES ***********************/
    /********************************************************************/
    size_t sz_transition_body_frame     = 9 * NUM_PARTICLES * sizeof(float);
    size_t sz_transition_lidar_frame    = 9 * sizeof(float);
    size_t sz_transition_world_frame    = 9 * NUM_PARTICLES * sizeof(float);
    size_t sz_processed_measure_pos     = NUM_PARTICLES * LIDAR_COORDS_LEN * sizeof(int);
    size_t sz_measure_idx               = NUM_PARTICLES * LIDAR_COORDS_LEN * sizeof(int);

    float* d_transition_body_frame  = NULL;
    float* d_transition_lidar_frame = NULL;
    float* d_transition_world_frame = NULL;
    int*   d_processed_measure_x    = NULL;
    int*   d_processed_measure_y    = NULL;
    int*   d_measure_idx            = NULL;

    float* res_transition_body_frame    = (float*)malloc(sz_transition_body_frame);
    float* res_transition_world_frame   = (float*)malloc(sz_transition_world_frame);
    int*   res_processed_measure_x      = (int*)malloc(sz_processed_measure_pos);
    int*   res_processed_measure_y      = (int*)malloc(sz_processed_measure_pos);


    gpuErrchk(cudaMalloc((void**)&d_transition_body_frame,  sz_transition_body_frame));
    gpuErrchk(cudaMalloc((void**)&d_transition_lidar_frame, sz_transition_lidar_frame));
    gpuErrchk(cudaMalloc((void**)&d_transition_world_frame, sz_transition_world_frame));
    gpuErrchk(cudaMalloc((void**)&d_processed_measure_x,    sz_processed_measure_pos));
    gpuErrchk(cudaMalloc((void**)&d_processed_measure_y,    sz_processed_measure_pos));
    gpuErrchk(cudaMalloc((void**)&d_measure_idx,            sz_measure_idx));


    gpuErrchk(cudaMemset(d_transition_body_frame,   0, sz_transition_body_frame));
    gpuErrchk(cudaMemset(d_transition_world_frame,  0, sz_transition_world_frame));
    gpuErrchk(cudaMemset(d_processed_measure_x,     0, sz_processed_measure_pos));
    gpuErrchk(cudaMemset(d_processed_measure_y,     0, sz_processed_measure_pos));
    gpuErrchk(cudaMemset(d_measure_idx,             0, sz_measure_idx));

    cudaMemcpy(d_transition_lidar_frame, h_transition_lidar_frame, sz_transition_lidar_frame, cudaMemcpyHostToDevice);

    /********************************************************************/
    /**************************** MAP VARIABLES *************************/
    /********************************************************************/
    size_t   sz_map_2d                  = GRID_WIDTH * GRID_HEIGHT * NUM_PARTICLES * sizeof(uint8_t);
    size_t   sz_unique_in_particle      = NUM_ELEMS * sizeof(int);
    size_t   sz_unique_in_particle_col  = NUM_ELEMS * GRID_WIDTH * sizeof(int);

    uint8_t* d_map_2d                   = NULL;
    int*     d_unique_in_particle       = NULL;
    int*     d_unique_in_particle_col   = NULL;

    uint8_t* res_map_2d             = (uint8_t*)malloc(sz_map_2d);
    int* h_unique_in_particle       = (int*)malloc(sz_unique_in_particle);

    gpuErrchk(cudaMalloc((void**)&d_map_2d,                 sz_map_2d));
    gpuErrchk(cudaMalloc((void**)&d_unique_in_particle,     sz_unique_in_particle));
    gpuErrchk(cudaMalloc((void**)&d_unique_in_particle_col, sz_unique_in_particle_col));
    
    gpuErrchk(cudaMemset(d_map_2d,                    0,  sz_map_2d));
    gpuErrchk(cudaMemset(d_unique_in_particle,        0,  sz_unique_in_particle));
    gpuErrchk(cudaMemset(d_unique_in_particle_col,    0,  sz_unique_in_particle_col));

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
    blocksPerGrid = NUM_ELEMS;

    auto start_create_map = std::chrono::high_resolution_clock::now();

    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (d_particles_x, d_particles_y, d_particles_idx, ELEMS_PARTICLES_START, d_map_2d, d_unique_in_particle,
                                                                    d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT, NUM_ELEMS);
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

    threadsPerBlock = NUM_ELEMS;
    blocksPerGrid = 1;
    
    kernel_update_2d_map_with_measure << <blocksPerGrid, threadsPerBlock >> > (d_processed_measure_x, d_processed_measure_y, d_measure_idx, 
        MEASURE_LEN, d_map_2d, d_unique_in_particle,
        d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT, NUM_ELEMS);
    cudaDeviceSynchronize();

    auto stop_update_map = std::chrono::high_resolution_clock::now();

    /********************************************************************/
    /************************* CUMULATIVE SUM ***************************/
    /********************************************************************/
    threadsPerBlock = NUM_ELEMS;
    blocksPerGrid = 1;

    auto start_cumulative_sum = std::chrono::high_resolution_clock::now();

    kernel_update_unique_sum << <1, 1 >> > (d_unique_in_particle, NUM_ELEMS);
    kernel_update_unique_sum_col << <blocksPerGrid, threadsPerBlock >> > (d_unique_in_particle_col, GRID_WIDTH);
    cudaDeviceSynchronize();

    auto stop_cumulative_sum = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(h_unique_in_particle, d_unique_in_particle, sz_unique_in_particle, cudaMemcpyDeviceToHost));

    int NEW_LEN = h_unique_in_particle[NUM_ELEMS - 1];
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
    sz_extended_idx  = NEW_LEN * sizeof(int);

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

    kernel_correlation << <blocksPerGrid, threadsPerBlock >> > (d_grid_map, d_particles_x, d_particles_y, d_extended_idx, d_correlation_raw, GRID_WIDTH, GRID_HEIGHT, NEW_LEN);
    cudaDeviceSynchronize();


    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_correlation_max << <blocksPerGrid, threadsPerBlock >> > (d_correlation_raw, d_correlation, NUM_PARTICLES);
    cudaDeviceSynchronize();

    auto stop_correlation = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_correlation,  d_correlation,  sz_correlation,  cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_extended_idx, d_extended_idx, sz_extended_idx, cudaMemcpyDeviceToHost));

    ASSERT_correlation_Equality(res_correlation, new_weights, NUM_PARTICLES);


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

/*
* Kernel Functions
*/

__global__ void kernel_bresenham(const int* arr_start_x, const int* arr_start_y, const int end_x, const int end_y,
                                    int* result_array_x, int* result_array_y, const int result_len, const int* index_array) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < result_len) {

        int x = arr_start_x[i];
        int y = arr_start_y[i];
        int x1 = x;
        int y1 = y;
        int x2 = end_x;
        int y2 = end_y;

        int dx = abs(x2 - x1);
        int dy = abs(y2 - y1);

        int start_index = index_array[i];

        if (dx == 0) {
        
            int sign = (y2 - y1) > 0 ? 1 : -1;
            result_array_x[start_index] = x;
            result_array_y[start_index] = y;

            for (int j = 1; j <= dy; j++) {
                result_array_x[start_index + j] = x;
                result_array_y[start_index + j] = y + sign * j;
            }
        }
        else {

            float gradient = dy / float(dx);
            bool should_reverse = false;

            if (gradient > 1) {

                swap(dx, dy);
                swap(x, y);
                swap(x1, y1);
                swap(x2, y2);
                should_reverse = true;
            }

            int p = 2 * dy - dx;
            if (should_reverse == false) {
                result_array_x[start_index] = x;
                result_array_y[start_index] = y;
            }
            else {
                result_array_x[start_index] = y;
                result_array_y[start_index] = x;
            }

            for (int j = 1; j <= dx; j++) {

                if (p > 0) {
                    y = (y < y2) ? y + 1 : y - 1;
                    p = p + 2 * (dy - dx);
                }
                else {
                    p = p + 2 * dy;
                }

                x = (x < x2) ? x + 1 : x - 1;

                if (should_reverse == false) {
                    result_array_x[start_index + j] = x;
                    result_array_y[start_index + j] = y;
                }
                else {
                    result_array_x[start_index + j] = y;
                    result_array_y[start_index + j] = x;
                }
            }
        }
    }
}

__global__ void kernel_index_init_const(int* indices, const int value) {
    
    int i = threadIdx.x;
    if (i > 0) {
        indices[i] = value;
    }
}

__global__ void kernel_index_expansion(const int *idx, int *extended_idx, const int numElements) {

    int i = blockIdx.x;
    int k = threadIdx.x;
    const int numThreads = blockDim.x;

    if (i < numThreads) {

        int first_idx = idx[i];
        int last_idx = (i < numThreads - 1) ? idx[i + 1] : numElements;
        int arr_len = last_idx - first_idx;

        int arr_end = last_idx;

        int start_idx = ((arr_len / blockDim.x) * k) + first_idx;
        int end_idx = ((arr_len / blockDim.x) * (k + 1)) + first_idx;
        end_idx = (k < blockDim.x - 1) ? end_idx : arr_end;

        for (int j = start_idx; j < end_idx; j++)
            extended_idx[j] = i;
    }
}

__global__ void kernel_correlation_max(const int* correlation_raw, int* correlation, const int _NUM_PARTICLES) {

    int i = threadIdx.x;

    int curr_max_value = correlation_raw[i];
    for (int j = 0; j < 25; j++) {
        int curr_value = correlation_raw[j * _NUM_PARTICLES + i];
        if (curr_value > curr_max_value) {
            curr_max_value = curr_value;
        }
    }
    correlation[i] = curr_max_value;
}

__global__ void kernel_correlation(const int* d_grid_map, const int* d_Y_io_x, const int* d_Y_io_y,
                                    const int* d_Y_io_idx, int* result, const int _GRID_WIDTH, const int _GRID_HEIGHT, int numElements) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {

        // int start_current_map_idx = i * _GRID_WIDTH * _GRID_HEIGHT;
        int loop_counter = 0;
        for (int x_offset = -2; x_offset <= 2; x_offset++) {

            for (int y_offset = -2; y_offset <= 2; y_offset++) {

                int idx = d_Y_io_idx[i];
                int x = d_Y_io_x[i] + x_offset;
                int y = d_Y_io_y[i] + y_offset;

                if (x >= 0 && y >= 0 && x < _GRID_WIDTH && y < _GRID_HEIGHT) {

                    int curr_idx = x * _GRID_HEIGHT + y;
                    // int curr_idx = start_current_map_idx + (x * _GRID_HEIGHT) + y;
                    int value = d_grid_map[curr_idx];

                    if (value != 0)
                        atomicAdd(&result[loop_counter * 100 + idx], value);
                }
                loop_counter++;
            }
        }
    }
}

__global__ void kernel_update_map(int* grid_map, const float* log_odds, const float _LOG_ODD_PRIOR, const int _WALL, const int _FREE, const int numElements) {


    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {

        if (log_odds[i] > 0)
            grid_map[i] = _WALL;

        if (log_odds[i] < _LOG_ODD_PRIOR)
            grid_map[i] = _FREE;
    }
}


__global__ void kernel_update_log_odds(float* log_odds, int* f_x, int* f_y, const float _log_t,
    const int _GRID_WIDTH, const int _GRID_HEIGHT, const int numElements) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {

        int x = f_x[i];
        int y = f_y[i];

        if (x >= 0 && y >= 0 && x < _GRID_WIDTH && y < _GRID_HEIGHT) {

            int grid_map_idx = x * _GRID_HEIGHT + y;

            log_odds[grid_map_idx] = log_odds[grid_map_idx] + _log_t;
        }
    }
}

__global__ void kernel_resampling(const float* weights, int* js, const float* rnd, const int numElements) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {

        float u = rnd[i] / numElements;
        int j = 0;
        float beta = u + float(i) / numElements;

        float accum = 0;
        for (int idx = 0; idx <= i; idx++) {
            accum += weights[idx];

            while (beta > accum) {
                j += 1;
                accum += weights[j];
            }
        }
        js[i] = j;
    }
}

__global__ void kernel_update_particles_states(const float* states_x, const float* states_y, const float* states_theta,
                                        float* transition_body_frame, const float* transition_lidar_frame, float* transition_world_frame, const int numElements) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {

        int T_idx = i * 9;

        float p_wb_0 = states_x[i];
        float p_wb_1 = states_y[i];

        float R_wb_0 =  cos(states_theta[i]);
        float R_wb_1 = -sin(states_theta[i]);
        float R_wb_2 =  sin(states_theta[i]);
        float R_wb_3 =  cos(states_theta[i]);

        transition_body_frame[T_idx + 0] = R_wb_0;   transition_body_frame[T_idx + 1] = R_wb_1;   transition_body_frame[T_idx + 2] = p_wb_0;
        transition_body_frame[T_idx + 3] = R_wb_2;   transition_body_frame[T_idx + 4] = R_wb_3;   transition_body_frame[T_idx + 5] = p_wb_1;
        transition_body_frame[T_idx + 6] = 0;        transition_body_frame[T_idx + 7] = 0;        transition_body_frame[T_idx + 8] = 1;

        kernel_matrix_mul_3x3(transition_body_frame, transition_lidar_frame, transition_world_frame, T_idx);
    }
}

__global__ void kernel_update_particles_lidar(float* transition_world_frame, int* processed_measure_x, int* processed_measure_y, const float* _lidar_coords, float _res, int _xmin, int _ymax,
                                                const int _lidar_coords_LEN, const int numElements) {

    int T_idx = threadIdx.x * 9;
    // int wo_idx = 2 * _lidar_coords_LEN * threadIdx.x;
    int wo_idx = _lidar_coords_LEN * threadIdx.x;
    int k = blockIdx.x;

    for (int j = 0; j < 2; j++) {

        double currVal = 0;
        currVal += transition_world_frame[T_idx + j * 3 + 0] * _lidar_coords[(0 * _lidar_coords_LEN) + k];
        currVal += transition_world_frame[T_idx + j * 3 + 1] * _lidar_coords[(1 * _lidar_coords_LEN) + k];
        currVal += transition_world_frame[T_idx + j * 3 + 2];

        // _Y_wo[wo_idx + (j * _lidar_coords_LEN) + k] = currVal; // ceil((currVal - _xmin) / _res);

        if (j == 0) {
            // processed_measure_y[wo_idx + (1 * _lidar_coords_LEN) + k] = ceil((currVal - _xmin) / _res);
            processed_measure_y[wo_idx + k] = (int) ceil((currVal - _xmin) / _res);
        }
        else {
            // processed_measure_x[wo_idx + (0 * _lidar_coords_LEN) + k] = ceil((_ymax - currVal) / _res);
            processed_measure_x[wo_idx + k] = (int) ceil((_ymax - currVal) / _res);
        }
    }
}

__device__ void kernel_matrix_mul_3x3(const float* A, const float* B, float* C, int start_i) {

    // A[i, j] --> A[i*3 + j]

    for (int i = 0; i < 3; i++) {

        for (int j = 0; j < 3; j++) {

            float currVal = 0;
            for (int k = 0; k < 3; k++) {
                currVal += A[start_i + (i * 3) + k] * B[k * 3 + j];
            }
            C[start_i + (i * 3) + j] = currVal;
        }
    }
}

__global__ void kernel_create_2d_map(const int *particles_x, const int *particles_y, const int *particles_idx, const int IDX_LEN, uint8_t *map_2d,
                                        int *unique_in_particle, int *unique_in_particle_col, const int _GRID_WIDTH, const int _GRID_HEIGHT, const int _NUM_ELEMS) {

    int i = blockIdx.x;
    int k = threadIdx.x;

    if (i < _NUM_ELEMS - 1) {

        int first_idx = particles_idx[i];
        int last_idx = (i < _NUM_ELEMS - 2) ? particles_idx[i + 1] : IDX_LEN;
        int arr_len = last_idx - first_idx;

        int arr_end = last_idx;

        int start_idx   = ((arr_len / blockDim.x) * k) + first_idx;
        int end_idx     = ((arr_len / blockDim.x) * (k + 1)) + first_idx;
        end_idx = (k < blockDim.x - 1) ? end_idx : arr_end;

        int start_of_current_map = i * _GRID_WIDTH * _GRID_HEIGHT;
        int start_of_col = i * _GRID_WIDTH;


        for (int j = start_idx; j < end_idx; j++) {

            int x = particles_x[j];
            int y = particles_y[j];

            int curr_idx = start_of_current_map + (x * _GRID_HEIGHT) + y;

            if (x >= 0 && y >= 0 && map_2d[curr_idx] == 0) {
                map_2d[curr_idx] = 1;
                atomicAdd(&unique_in_particle[i + 1], 1);
                atomicAdd(&unique_in_particle_col[start_of_col + x + 1], 1);
                // unique_in_particle_col[start_of_col + x + 1] = unique_in_particle_col[start_of_col + x + 1] + 1;
            }
        }
    }    
}

__global__ void kernel_update_2d_map_with_measure(const int* measure_x, const int* measure_y, const int* measure_idx, const int IDX_LEN, uint8_t* map_2d,
                    int* unique_in_particle, int* unique_in_particle_col, const int _GRID_WIDTH, const int _GRID_HEIGHT, const int _NUM_ELEMS) {

    int i = threadIdx.x;

    if (i < _NUM_ELEMS - 1) {

        int start_idx = measure_idx[i];
        int end_idx = (i < _NUM_ELEMS - 2) ? measure_idx[i + 1] : IDX_LEN;

        int start_of_current_map = i * _GRID_WIDTH * _GRID_HEIGHT;
        int start_of_col = i * _GRID_WIDTH;

        for (int j = start_idx; j < end_idx; j++) {

            int x = measure_x[j];
            int y = measure_y[j];

            int curr_idx = start_of_current_map + (x * _GRID_HEIGHT) + y;

            if (x >= 0 && y >= 0 && map_2d[curr_idx] == 0) {
                map_2d[curr_idx] = 1;
                atomicAdd(&unique_in_particle[i + 1], 1);
                atomicAdd(&unique_in_particle_col[start_of_col + x + 1], 1);
                // unique_in_particle_col[start_of_col + x + 1] = unique_in_particle_col[start_of_col + x + 1] + 1;
            }
        }
    }
}

__global__ void kernel_update_unique_restructure(uint8_t* map_2d, int* particles_x, int* particles_y, int* particles_idx, int* unique_in_particle, int* unique_in_particle_col,
                                                    const int _GRID_WIDTH, const int _GRID_HEIGHT) {

    int i = blockIdx.x;
    int l = threadIdx.x;

    int start_of_current_map    =  i * _GRID_WIDTH * _GRID_HEIGHT;
    int start_idx               = (i * _GRID_WIDTH * _GRID_HEIGHT) + (l * _GRID_HEIGHT);
    int end_idx                 = (i * _GRID_WIDTH * _GRID_HEIGHT) + ((l + 1) * _GRID_HEIGHT);
    int key                     = unique_in_particle_col[i * _GRID_WIDTH + l] + unique_in_particle[i];

    for (int j = start_idx; j < end_idx; j++) {


        if (map_2d[j] == 1) {

            int y = (j - start_of_current_map) % _GRID_HEIGHT;
            int x = (j - start_of_current_map) / _GRID_HEIGHT;

            particles_x[key] = x;
            particles_y[key] = y;
            key += 1;
            atomicAdd(&particles_idx[i + 1], 1);
        }
    }
}


__global__ void kernel_arr_increase(int* arr, const int increase_value, const int start_index) {

    int i = threadIdx.x;
    if (i >= start_index) {
        arr[i] += increase_value;
    }
}

__global__ void kernel_arr_increase(float* arr, const float increase_value, const int start_index) {

    int i = threadIdx.x;
    if (i >= start_index) {
        arr[i] += increase_value;
    }
}

__global__ void kernel_arr_max(float* arr, float* result, const int LEN) {

    float rs = arr[0];
    for (int i = 1; i < LEN; i++) {
        if (rs < arr[i])
            rs = arr[i];
    }
    result[0] = rs;
}

__global__ void kernel_arr_sum_exp(float* arr, double* result, const int LEN) {

    double s = 0;
    for (int i = 0; i < LEN; i++) {
        s += exp(arr[i]);
    }
    result[0] = s;
}

__global__ void kernel_arr_normalize(float* arr, const double norm) {

    int i = threadIdx.x;
    arr[i] = exp(arr[i]) / norm;
}

__global__ void kernel_update_unique_sum(int* unique_in_particle, const int _NUM_ELEMS) {

    for (int j = 1; j < _NUM_ELEMS; j++)
        unique_in_particle[j] = unique_in_particle[j] + unique_in_particle[j - 1];
}

__global__ void kernel_update_unique_sum_col(int * unique_in_particle_col, const int _GRID_WIDTH) {

    int i = threadIdx.x;

    for (int j = (i * _GRID_WIDTH) + 1; j < (i + 1) * _GRID_WIDTH; j++)
        unique_in_particle_col[j] = unique_in_particle_col[j] + unique_in_particle_col[j - 1];
}