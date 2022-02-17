
#include "headers.h"

// #define CORRELATION_EXEC
// #define BRESENHAM_EXEC
// #define UPDATE_MAP_EXEC
// #define UPDATE_STATE_EXEC
// #define UPDATE_PARTICLE_WEIGHTS_EXEC
// #define RESAMPLING_EXEC



#ifdef CORRELATION_EXEC
#include "data/correlation/combined_3423.h"
#endif

#ifdef BRESENHAM_EXEC
#include "data/bresenham/500.h"
#endif

#ifdef UPDATE_MAP_EXEC
#include "data/log_odds/100.h"
#endif

#ifdef UPDATE_STATE_EXEC
#include "data/state_update/100.h"
#endif

#ifdef UPDATE_PARTICLE_WEIGHTS_EXEC
#include "data/particle_weights/200.h"
#endif

#ifdef RESAMPLING_EXEC
#include "data/resampling/70.h"
#endif

#include "data/update_particles/10.h"


__global__ void kernel_correlation(const int* d_grid_map, const int* d_Y_io_x, const int* d_Y_io_y,
                                    const int* d_Y_io_idx, int* result, const int _GRID_WIDTH, const int _GRID_HEIGHT, int numElements);

__global__ void kernel_bresenham(const int* arr_start_x, const int* arr_start_y, const int end_x, const int end_y,
                                    int* result_array_x, int* result_array_y, const int result_len, const int* index_array);

__global__ void kernel_update_log_odds(float *log_odds, int *f_x, int *f_y, const float _log_t,
                                        const int _GRID_WIDTH, const int _GRID_HEIGHT, const int numElements);

__global__ void kernel_update_map(int* grid_map, const float* log_odds, const float _LOG_ODD_PRIOR, const int _WALL, const int _FREE, const int numElements);


__global__ void kernel_resampling(const float* weights, int* js, const float* rnd, const int numElements);


__global__ void kernel_update_particles(const float* xs, const float* ys, const float* thetas,
                                            float* T_wb, const float* T_bl, float* T_wl, int numElements);

__device__ void kernel_matrix_mul_3x3(const float* A, const float* B, float* C, int start_i);

void host_correlation();
void host_bresenham();
void host_update_map();
void host_update_state();
void host_update_particle_weights();
void host_resampling();



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

#ifdef __cplusplus 
    printf("C++\n");
#endif

#ifdef UPDATE_PARTICLE_WEIGHTS_EXEC
    host_update_particle_weights();
#endif

#ifdef RESAMPLING_EXEC
    host_resampling();
#endif

    // [✓] - Create a kernel with Input (xs, ys, thetas, numElements)
    // [✓] - Add Output to kernel(p_wb, R_wb) --> len of parameters (2x, 4x)
    // [✓] - Create device variables (d_xs, d_ys, d_thetas) & (d_p_wb, d_R_wb) & (res_p_wb, res_R_wb)
    // [✓] - Copy data from (xs, ys, thetas) -> (d_xs, d_ys, d_thetas)
    // [✓] - Initialize to zero (res_p_wb, res_R_wb) & (d_p_wb, d_R_wb)
    // [✓] - Change (p_wb, R_wb) to (T_wb)
    // [ ] - Add 'T_bl' to Input Parameters
    // [✓] - Create a custom MatrixMultiplication function
    // [ ] - Add 'lidar_coords' to function Input parameters
    // [ ] - Add Send length of 'lidar_coords' to the function
    // [ ] - Calculate Length of Result


    const int STATES_LEN = NUM_PARTICLES;

    size_t size_of_states = STATES_LEN * sizeof(float);

    float* d_xs       = NULL;
    float* d_ys       = NULL;
    float* d_thetas   = NULL;

    gpuErrchk(cudaMalloc((void**)&d_xs,     size_of_states));
    gpuErrchk(cudaMalloc((void**)&d_ys,     size_of_states));
    gpuErrchk(cudaMalloc((void**)&d_thetas, size_of_states));

    cudaMemcpy(d_xs, xs, size_of_states, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ys, ys, size_of_states, cudaMemcpyHostToDevice);
    cudaMemcpy(d_thetas, thetas, size_of_states, cudaMemcpyHostToDevice);


    //size_t size_of_p_wb = 2 * STATES_LEN * sizeof(float);
    //size_t size_of_R_wb = 4 * STATES_LEN * sizeof(float);
    //float* d_p_wb = NULL;
    //float* d_R_wb = NULL;
    //gpuErrchk(cudaMalloc((void**)&d_p_wb, size_of_p_wb));
    //gpuErrchk(cudaMalloc((void**)&d_R_wb, size_of_R_wb));

    size_t size_of_T_wb = 9 * STATES_LEN * sizeof(float);
    size_t size_of_T_bl = 9 * sizeof(float);
    size_t size_of_T_wl = 9 * NUM_PARTICLES * sizeof(float);
    size_t size_of_lidar_coords = 2 * lidar_coords_LEN * sizeof(float);
    float* d_T_wb = NULL;
    float* d_T_bl = NULL;
    float* d_T_wl = NULL;
    float* d_lidar_coords = NULL;
    gpuErrchk(cudaMalloc((void**)&d_T_wb, size_of_T_wb));
    gpuErrchk(cudaMalloc((void**)&d_T_bl, size_of_T_bl));
    gpuErrchk(cudaMalloc((void**)&d_T_wl, size_of_T_wl));
    gpuErrchk(cudaMalloc((void**)&d_lidar_coords, size_of_lidar_coords));

    //float* res_p_wb = (float*)malloc(size_of_p_wb);
    //float* res_R_wb = (float*)malloc(size_of_R_wb);
    //memset(res_p_wb, 0, size_of_p_wb);
    //memset(res_R_wb, 0, size_of_R_wb);
    //cudaMemcpy(d_p_wb, res_p_wb, size_of_p_wb, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_R_wb, res_R_wb, size_of_R_wb, cudaMemcpyHostToDevice);

    float* res_T_wb = (float*)malloc(size_of_T_wb);
    float* res_T_wl = (float*)malloc(size_of_T_wl);
    memset(res_T_wb, 0, size_of_T_wb);
    memset(res_T_wl, 0, size_of_T_wl);
    cudaMemcpy(d_T_wb, res_T_wb, size_of_T_wb, cudaMemcpyHostToDevice);
    cudaMemcpy(d_T_wl, res_T_wl, size_of_T_wl, cudaMemcpyHostToDevice);
    cudaMemcpy(d_T_bl, T_bl, size_of_T_bl, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lidar_coords, lidar_coords, size_of_lidar_coords, cudaMemcpyHostToDevice);

    int threadsPerBlock = NUM_PARTICLES;
    int blocksPerGrid = 1;

    // const float* xs, const float* ys, const float* thetas, float* T_wb, const float* T_bl, float* T_wl, int numElements
    kernel_update_particles << <blocksPerGrid, threadsPerBlock >> > (d_xs, d_ys, d_thetas, d_T_wb, d_T_bl, d_T_wl, NUM_PARTICLES);
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(res_T_wb, d_T_wb, size_of_T_wb, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_T_wl, d_T_wl, size_of_T_wl, cudaMemcpyDeviceToHost));

    for (int i = 0; i < 9 * NUM_PARTICLES; i++) {
        // printf("%f, %f | ", res_T_wb[i], T_wb[i]);
        assert( abs(res_T_wb[i] - T_wb[i]) < 1e-5 );
    }
    for (int i = 0; i < 9 * NUM_PARTICLES; i++) {
        printf("%f, %f |  ", res_T_wl[i], T_wl[i]);
        assert(abs(res_T_wl[i] - T_wl[i]) < 1e-5);
    }
    //printf("\n\n");
    //for (int i = 0; i < 4 * NUM_PARTICLES; i++) {
    //    printf("%e, %e  |  ", res_R_wb[i], R_wb[i]);
    //    assert( abs(res_R_wb[i] - R_wb[i]) < 1e-6 );
    //}

    return 0;
}


/*
* Host Functions
*/

#ifdef CORRELATION_EXEC
void host_correlation() {

    cudaError_t cudaStatus;
    float time_total, time_memory_copy, time_kernel, time_result_copy;
    cudaEvent_t start_total, stop_total, stop_memory_copy, start_kernel, stop_kernel, start_result_copy;
    gpuErrchk(cudaEventCreate(&start_total));
    gpuErrchk(cudaEventCreate(&stop_total));
    gpuErrchk(cudaEventCreate(&stop_memory_copy));
    gpuErrchk(cudaEventCreate(&start_kernel));
    gpuErrchk(cudaEventCreate(&stop_kernel));
    gpuErrchk(cudaEventCreate(&start_result_copy));


    const int num_elements_of_grid_map = GRID_WIDTH * GRID_HEIGHT;
    size_t size_of_grid_map = num_elements_of_grid_map * sizeof(int);

    printf("Elements of Grid_Map: %d,  Size of Grid_Map: %d\n", (int)num_elements_of_grid_map, (int)size_of_grid_map);

    const int num_elements_of_Y = Y_LENGTH;
    size_t size_of_Y_x_y = num_elements_of_Y * sizeof(int);
    size_t size_of_Y_idx = num_elements_of_Y * sizeof(int);

    printf("Elements of Y_io_x: %d,  Size of Y_io_x: %d\n", (int)num_elements_of_Y, (int)size_of_Y_x_y);
    printf("Elements of Y_io_y: %d,  Size of Y_io_y: %d\n", (int)num_elements_of_Y, (int)size_of_Y_x_y);
    printf("Elements of Y_io_idx: %d,  Size of Y_io_idx: %d\n", (int)num_elements_of_Y, (int)size_of_Y_idx);


    gpuErrchk(cudaEventRecord(start_total, 0));

    int* d_grid_map = NULL;
    int* d_Y_io_x = NULL;
    int* d_Y_io_y = NULL;
    int* d_Y_io_idx = NULL;


    gpuErrchk(cudaMalloc((void**)&d_grid_map, size_of_grid_map));
    gpuErrchk(cudaMalloc((void**)&d_Y_io_x, size_of_Y_x_y));
    gpuErrchk(cudaMalloc((void**)&d_Y_io_y, size_of_Y_x_y));
    gpuErrchk(cudaMalloc((void**)&d_Y_io_idx, size_of_Y_idx));


    cudaMemcpy(d_grid_map, grid_map, size_of_grid_map, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y_io_x, Y_io_x, size_of_Y_x_y, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y_io_y, Y_io_y, size_of_Y_x_y, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y_io_idx, Y_io_idx, size_of_Y_idx, cudaMemcpyHostToDevice);

    const int num_elements_of_particles = 100;
    size_t size_of_result = 25 * num_elements_of_particles * sizeof(int);
    int* result = (int*)malloc(size_of_result);
    memset(result, 0, size_of_result);
    int* d_result = NULL;

    gpuErrchk(cudaMalloc((void**)&d_result, size_of_result));
    gpuErrchk(cudaMemcpy(d_result, result, size_of_result, cudaMemcpyHostToDevice));


    gpuErrchk(cudaEventRecord(stop_memory_copy, 0));
    gpuErrchk(cudaEventSynchronize(stop_memory_copy));
    gpuErrchk(cudaEventElapsedTime(&time_memory_copy, start_total, stop_memory_copy));

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_elements_of_Y + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads, All Threads: %d\n", blocksPerGrid, threadsPerBlock, blocksPerGrid * threadsPerBlock);

    gpuErrchk(cudaEventRecord(start_kernel, 0));

    kernel_correlation << <blocksPerGrid, threadsPerBlock >> > (d_grid_map, d_Y_io_x, d_Y_io_y, d_Y_io_idx, d_result, GRID_WIDTH, GRID_HEIGHT, num_elements_of_Y);

    cudaDeviceSynchronize();

    gpuErrchk(cudaEventRecord(stop_kernel, 0));
    gpuErrchk(cudaEventSynchronize(stop_kernel));
    gpuErrchk(cudaEventElapsedTime(&time_kernel, start_kernel, stop_kernel));


    gpuErrchk(cudaEventRecord(start_result_copy, 0));

    gpuErrchk(cudaMemcpy(result, d_result, size_of_result, cudaMemcpyDeviceToHost));


    int final_result[num_elements_of_particles] = { 0 };

    for (int i = 0; i < num_elements_of_particles; i++) {
        int curr_max_value = result[i];
        for (int j = 0; j < 25; j++) {
            int curr_value = result[j * 100 + i];
            // printf("curr_value: %d\n", curr_value);
            if (curr_value > curr_max_value) {
                curr_max_value = curr_value;
            }
        }
        final_result[i] = curr_max_value;
    }

    bool all_equal = true;
    for (int i = 0; i < num_elements_of_particles; i++) {
        // printf("index: %d --> %d, %d\n", i, final_result[i], corr[i]);
        if (final_result[i] != corr[i])
            all_equal = false;
    }

    gpuErrchk(cudaEventRecord(stop_total, 0));
    gpuErrchk(cudaEventSynchronize(stop_total));
    gpuErrchk(cudaEventElapsedTime(&time_result_copy, start_result_copy, stop_total));
    gpuErrchk(cudaEventElapsedTime(&time_total, start_total, stop_total));


    printf("Memory Copy: %7.3f ms\t Kernel Execution: %7.3f ms\t Result Copy: %7.3f ms\n", time_memory_copy, time_kernel, time_result_copy);
    printf("Total Time of Execution:  %3.1f ms - Python Execution Time: %7.3f ms \n", time_total, EXEC_TIME * 1000);
    printf("All Equal: %s\n", all_equal ? "true" : "false");

    printf("Program Finished\n");

    gpuErrchk(cudaFree(d_grid_map));
    gpuErrchk(cudaFree(d_Y_io_x));
    gpuErrchk(cudaFree(d_Y_io_y));
    gpuErrchk(cudaFree(d_Y_io_idx));
    gpuErrchk(cudaFree(d_result));
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

    // [ ] - Create std::map with key:float and value:int --> states
    // [ ] - Create std::vector<float> for xs --> vec_xs
    // [ ] - Iterate over vec_xs:   If it is not in states add to it with value 1
    //                              If it is already in states then increase value by 1
    // [ ] - Change std::map keys to tuple<float, float, float>
    // [ ] - Find max value in std::map

    // [ ] - First create arrays in thrust and then copy them to the std::vec


    int N = 100;

    thrust::device_vector<float> d_xs(xs, xs + N);
    thrust::device_vector<float> d_ys(ys, ys + N);
    thrust::device_vector<float> d_thetas(thetas, thetas + N);

    float time_total;
    cudaEvent_t start_total, stop_total;
    gpuErrchk(cudaEventCreate(&start_total));
    gpuErrchk(cudaEventCreate(&stop_total));

    gpuErrchk(cudaEventRecord(start_total, 0));

    thrust::host_vector<float> h_xs(d_xs.begin(), d_xs.end());
    thrust::host_vector<float> h_ys(d_ys.begin(), d_ys.end());
    thrust::host_vector<float> h_thetas(d_thetas.begin(), d_thetas.end());

    std::vector<float> vec_xs(h_xs.begin(), h_xs.end());
    std::vector<float> vec_ys(h_ys.begin(), h_ys.end());
    std::vector<float> vec_thetas(h_thetas.begin(), h_thetas.end());


    std::map<std::tuple<float, float, float>, int> states;

    for (int i = 0; i < N; i++) {

        if (states.find(std::make_tuple(vec_xs[i], vec_ys[i], vec_thetas[i])) == states.end()) {
            states.insert({ std::make_tuple(vec_xs[i], vec_ys[i], vec_thetas[i]), 1 });
        }
        else {
            states[std::make_tuple(vec_xs[i], vec_ys[i], vec_thetas[i])] += 1;
        }
    }

    std::map<std::tuple<float, float, float>, int>::iterator best
        = std::max_element(states.begin(), states.end(), [](const std::pair<std::tuple<float, float, float>, int>& a,
            const std::pair<std::tuple<float, float, float>, int>& b)->bool { return a.second < b.second; });

    auto key = best->first;
    std::cout << std::get<0>(key) << " " << std::get<1>(key) << " " << std::get<2>(key) << " " << best->second << "\n";

    float theta = std::get<2>(key);
    float _T_wb[] = { cos(theta), -sin(theta), std::get<0>(key),
                        sin(theta),  cos(theta), std::get<1>(key),
                        0, 0, 1 };

    gpuErrchk(cudaEventRecord(stop_total, 0));
    gpuErrchk(cudaEventSynchronize(stop_total));
    gpuErrchk(cudaEventElapsedTime(&time_total, start_total, stop_total));

    printf("Total Time of Execution:  %3.1f ms\n", time_total);


    for (int i = 0; i < 9; i++) {
        printf("%f  ", _T_wb[i]);
    }
    printf("\n");

}
#endif

#ifdef UPDATE_PARTICLE_WEIGHTS_EXEC
void host_update_particle_weights() {

    int N = 100;
    thrust::device_vector<double> d_pre_weights(pre_weights, pre_weights + N);

    float time_total;
    cudaEvent_t start_total, stop_total;
    gpuErrchk(cudaEventCreate(&start_total));
    gpuErrchk(cudaEventCreate(&stop_total));

    gpuErrchk(cudaEventRecord(start_total, 0));


    thrust::host_vector<double> h_pre_weights(d_pre_weights.begin(), d_pre_weights.end());
    std::vector<double> vec_weights(h_pre_weights.begin(), h_pre_weights.end());
    double max_val = *max_element(vec_weights.begin(), vec_weights.end());

    thrust::for_each(d_pre_weights.begin(), d_pre_weights.end(), _1 -= max_val - 50);
    thrust::transform(d_pre_weights.begin(), d_pre_weights.end(), d_pre_weights.begin(), thrust_exp());

    h_pre_weights.assign(d_pre_weights.begin(), d_pre_weights.end());
    vec_weights.assign(h_pre_weights.begin(), h_pre_weights.end());
    auto sum = std::accumulate(vec_weights.begin(), vec_weights.end(), 0.0, std::plus<double>());

    thrust::transform(d_pre_weights.begin(), d_pre_weights.end(), d_pre_weights.begin(), thrust_div_sum(sum));

    gpuErrchk(cudaEventRecord(stop_total, 0));
    gpuErrchk(cudaEventSynchronize(stop_total));
    gpuErrchk(cudaEventElapsedTime(&time_total, start_total, stop_total));

    printf("Total Time of Execution:  %3.1f ms\n", time_total);

    for (int i = 0; i < N; i++) {
        printf("%.10e ", (double)d_pre_weights[i]);
    }
    printf("\n");

    printf("Max value: %f, %f\n", max_val, sum);
    // printf("Sum: %f\n", sum);
}
#endif

#ifdef RESAMPLING_EXEC
void host_resampling() {

    // [✓] - Create a new kernel 'kernel_resampling'
    // [✓] - Inputs to this kernel are --> (weights, 'j' as output, u)
    // [✓] - Must launch kernel with Grid: 1 & threadPerBlocks: 100
    // [✓] - Each thread has a for-loop. from 0 to 
    // [ ] - Try to Add New Particles with new Resampling

    float time_total;
    cudaEvent_t start_total, stop_total;
    gpuErrchk(cudaEventCreate(&start_total));
    gpuErrchk(cudaEventCreate(&stop_total));

    float* d_weights = NULL;
    int* d_js = NULL;
    float* d_rnd = NULL;

    size_t size_of_weights = NUM_PARTICLES * sizeof(float);
    size_t size_of_js = NUM_PARTICLES * sizeof(int);
    size_t size_of_rnd = NUM_PARTICLES * sizeof(float);

    int js_result[100] = { 0 };

    gpuErrchk(cudaMalloc((void**)&d_weights, size_of_weights));
    gpuErrchk(cudaMalloc((void**)&d_js, size_of_js));
    gpuErrchk(cudaMalloc((void**)&d_rnd, size_of_rnd));

    cudaMemcpy(d_weights, weights, size_of_weights, cudaMemcpyHostToDevice);
    cudaMemcpy(d_js, js_result, size_of_js, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rnd, rnds, size_of_rnd, cudaMemcpyHostToDevice);

    int threadsPerBlock = NUM_PARTICLES;
    int blocksPerGrid = 1;

    gpuErrchk(cudaEventRecord(start_total, 0));

    kernel_resampling << <blocksPerGrid, threadsPerBlock >> > (d_weights, d_js, d_rnd, NUM_PARTICLES);
    cudaDeviceSynchronize();

    gpuErrchk(cudaEventRecord(stop_total, 0));
    gpuErrchk(cudaEventSynchronize(stop_total));
    gpuErrchk(cudaEventElapsedTime(&time_total, start_total, stop_total));

    printf("Total Time of Execution:  %3.1f ms\n", time_total);

    gpuErrchk(cudaMemcpy(js_result, d_js, size_of_js, cudaMemcpyDeviceToHost));

    for (int i = 0; i < 100; i++) {
        printf("%d, %d | ", js_result[i], js[i]);
        assert(js_result[i] == js[i]);
    }
    printf("\n");
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


__global__ void kernel_correlation(const int* d_grid_map, const int* d_Y_io_x, const int* d_Y_io_y,
                                    const int* d_Y_io_idx, int* result, const int _GRID_WIDTH, const int _GRID_HEIGHT, int numElements) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {

        int loop_counter = 0;
        for (int x_offset = -2; x_offset <= 2; x_offset++) {

            for (int y_offset = -2; y_offset <= 2; y_offset++) {

                int idx = d_Y_io_idx[i];
                int x = d_Y_io_x[i] + x_offset;
                int y = d_Y_io_y[i] + y_offset;

                if (x >= 0 && y >= 0 && x < _GRID_WIDTH && y < _GRID_HEIGHT) {

                    int grid_map_idx = x * _GRID_HEIGHT + y;
                    int value = d_grid_map[grid_map_idx];

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


__global__ void kernel_update_particles(const float* xs, const float* ys, const float* thetas, 
                                        float* T_wb, const float *T_bl, float *T_wl, int numElements) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {

        int T_idx = i * 9;

        float p_wb_0 = xs[i];
        float p_wb_1 = ys[i];

        float R_wb_0 =  cos(thetas[i]);
        float R_wb_1 = -sin(thetas[i]);
        float R_wb_2 =  sin(thetas[i]);
        float R_wb_3 =  cos(thetas[i]);

        T_wb[T_idx + 0] = R_wb_0;   T_wb[T_idx + 1] = R_wb_1;   T_wb[T_idx + 2] = p_wb_0;
        T_wb[T_idx + 3] = R_wb_2;   T_wb[T_idx + 4] = R_wb_3;   T_wb[T_idx + 5] = p_wb_1;
        T_wb[T_idx + 6] = 0;        T_wb[T_idx + 7] = 0;        T_wb[T_idx + 8] = 1;

        kernel_matrix_mul_3x3(T_wb, T_bl, T_wl, T_idx);
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