
#include "headers.h"

#define CORRELATION_EXEC
// #define BRESENHAM_EXEC
// #define UPDATE_MAP_EXEC
// #define UPDATE_STATE_EXEC
// #define UPDATE_PARTICLE_WEIGHTS_EXEC
// #define RESAMPLING_EXEC
// #define UPDATE_PARTICLES_EXEC
// #define UPDATE_UNIQUE_EXEC


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
#include "data/state_update/100.h"
#endif

#ifdef UPDATE_PARTICLE_WEIGHTS_EXEC
#include "data/particle_weights/200.h"
#endif

#ifdef RESAMPLING_EXEC
#include "data/resampling/70.h"
#endif

#ifdef UPDATE_PARTICLES_EXEC
#include "data/update_particles/160.h"
#endif

#ifdef UPDATE_UNIQUE_EXEC
#include "data/update_unique/4800.h"
#endif

__global__ void kernel_bresenham(const int* arr_start_x, const int* arr_start_y, const int end_x, const int end_y,
    int* result_array_x, int* result_array_y, const int result_len, const int* index_array);


__global__ void kernel_index_expansion(const int* idx_pack, int* idx, const int numElements);
__global__ void kernel_correlation_max(const int* all_correlation, int* correlation, const int _NUM_PARTICLES);
__global__ void kernel_correlation(const int* d_grid_map, const int* d_Y_io_x, const int* d_Y_io_y,
                                    const int* d_Y_io_idx, int* result, const int _GRID_WIDTH, const int _GRID_HEIGHT, int numElements);


__global__ void kernel_update_log_odds(float *log_odds, int *f_x, int *f_y, const float _log_t,
                                        const int _GRID_WIDTH, const int _GRID_HEIGHT, const int numElements);

__global__ void kernel_update_map(int* grid_map, const float* log_odds, const float _LOG_ODD_PRIOR, const int _WALL, const int _FREE, const int numElements);

__global__ void kernel_resampling(const float* weights, int* js, const float* rnd, const int numElements);

__global__ void kernel_update_particles_states(const float* xs, const float* ys, const float* thetas,
                                            float* T_wb, const float* T_bl, float* T_wl, const int numElements);

__global__ void kernel_update_particles_lidar(float* T_wl, float* Y_wo, const float* _lidar_coords, float _res, int _xmin, int _ymax,
                                                const int _lidar_coords_LEN, const int numElements);

__device__ void kernel_matrix_mul_3x3(const float* A, const float* B, float* C, int start_i);

__global__ void kernel_create_2d_map(const int* Y_io_x, const int* Y_io_y, const int* idx, const int IDX_LEN, uint8_t* whole_map,
    int* unique_in_particle, int* unique_in_particle_col, const int _GRID_WIDTH, const int _GRID_HEIGHT, const int _NUM_ELEMS);

__global__ void kernel_update_2d_map_with_measure(const int* Y_io_x, const int* Y_io_y, const int* idx, const int IDX_LEN, uint8_t* whole_map,
    int* unique_in_particle, int* unique_in_particle_col, const int _GRID_WIDTH, const int _GRID_HEIGHT, const int _NUM_ELEMS);

__global__ void kernel_update_unique_restructure(uint8_t* whole_map, int* Y_io_x, int* Y_io_y, int* unique_in_each_particle, int* unique_in_each_particle_col,
                                        const int _GRID_WIDTH, const int _GRID_HEIGHT);

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

#ifdef UPDATE_PARTICLES_EXEC
    host_update_particles();
#endif

#ifdef UPDATE_UNIQUE_EXEC
    host_update_unique();
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
    int* d_grid_map = NULL;
    int* d_Y_io_x   = NULL;
    int* d_Y_io_y   = NULL;
    int* d_pack_idx = NULL;
    int* d_idx      = NULL;

    const int num_elements_of_grid_map  = GRID_WIDTH * GRID_HEIGHT;
    size_t sz_grid_map                  = num_elements_of_grid_map * sizeof(int);

    const int num_elements_of_Y =  Y_Len; // Y_LENGTH;
    size_t sz_Y_x_y             =  num_elements_of_Y * sizeof(int);
    size_t sz_pack_idx          =  NUM_PARTICLES * sizeof(int);
    size_t sz_idx               =  num_elements_of_Y * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_grid_map,   sz_grid_map));
    gpuErrchk(cudaMalloc((void**)&d_Y_io_x,     sz_Y_x_y));
    gpuErrchk(cudaMalloc((void**)&d_Y_io_y,     sz_Y_x_y));
    gpuErrchk(cudaMalloc((void**)&d_idx,        sz_idx));
    gpuErrchk(cudaMalloc((void**)&d_pack_idx,   sz_pack_idx));

    cudaMemcpy(d_grid_map,  grid_map,   sz_grid_map,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y_io_x,    Y_io_x,     sz_Y_x_y,       cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y_io_y,    Y_io_y,     sz_Y_x_y,       cudaMemcpyHostToDevice);
    cudaMemcpy(d_pack_idx,  Y_idx,      sz_pack_idx,    cudaMemcpyHostToDevice);

    // const int num_elements_of_particles = NUM_PARTICLES;
    size_t sz_correlation           = NUM_PARTICLES * sizeof(int);
    size_t sz_all_correlation       = 25 * sz_correlation;
    int* h_correlation              = (int*)malloc(sz_correlation);
    int* indices                    = (int*)malloc(sz_idx);
    int* d_correlation              = NULL;
    int* d_all_correlation          = NULL;
    memset(h_correlation, 0, sz_correlation);

    gpuErrchk(cudaMalloc((void**)&d_correlation,        sz_correlation));
    gpuErrchk(cudaMalloc((void**)&d_all_correlation,    sz_all_correlation));
    gpuErrchk(cudaMemset(d_all_correlation, 0, sz_all_correlation));
    // gpuErrchk(cudaMemcpy(d_all_correlation, h_correlation, sz_all_correlation, cudaMemcpyHostToDevice));

    auto stop_memory_copy = std::chrono::high_resolution_clock::now();


    /********************************************************************/
    /*************************** PRINT SUMMARY **************************/
    /********************************************************************/
    printf("Elements of Y_io_x: %d,  Size of Y_io_x: %d\n", (int)num_elements_of_Y, (int)sz_Y_x_y);
    printf("Elements of Y_io_y: %d,  Size of Y_io_y: %d\n", (int)num_elements_of_Y, (int)sz_Y_x_y);
    printf("Elements of Y_io_idx: %d,  Size of Y_io_idx: %d\n", (int)num_elements_of_Y, (int)sz_idx);

    printf("Elements of Grid_Map: %d,  Size of Grid_Map: %d\n", (int)num_elements_of_grid_map, (int)sz_grid_map);

    /********************************************************************/
    /************************* INDEX EXPANSION **************************/
    /********************************************************************/
    auto start_index_expansion = std::chrono::high_resolution_clock::now();

    int threadsPerBlock = 100;
    int blocksPerGrid = NUM_PARTICLES;

    kernel_index_expansion << <blocksPerGrid, threadsPerBlock >> > (d_pack_idx, d_idx, Y_Len);
    cudaDeviceSynchronize();

    auto stop__index_expansion = std::chrono::high_resolution_clock::now();

    /********************************************************************/
    /************************ KERNEL CORRELATION ************************/
    /********************************************************************/
    threadsPerBlock = 256;
    blocksPerGrid = (num_elements_of_Y + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads, All Threads: %d\n", blocksPerGrid, threadsPerBlock, blocksPerGrid * threadsPerBlock);

    auto start_kernel = std::chrono::high_resolution_clock::now();

    kernel_correlation << <blocksPerGrid, threadsPerBlock >> > (d_grid_map, d_Y_io_x, d_Y_io_y, d_idx, d_all_correlation, GRID_WIDTH, GRID_HEIGHT, num_elements_of_Y);
    cudaDeviceSynchronize();

    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = 1;
    kernel_correlation_max << <blocksPerGrid, threadsPerBlock >> > (d_all_correlation, d_correlation, NUM_PARTICLES);

    auto stop_kernel = std::chrono::high_resolution_clock::now();


    gpuErrchk(cudaMemcpy(h_correlation, d_correlation, sz_correlation, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(indices, d_idx, sz_idx, cudaMemcpyDeviceToHost));
    

    bool all_equal = true;
    for (int i = 0; i < NUM_PARTICLES; i++) {
        // printf("index: %d --> %d, %d\n", i, final_result[i], new_weights[i]); 
        if (h_correlation[i] != new_weights[i])
            all_equal = false;
    }


    auto duration_kernel = std::chrono::duration_cast<std::chrono::milliseconds>(stop_kernel - start_kernel);
    auto duration_memory_copy = std::chrono::duration_cast<std::chrono::milliseconds>(stop_memory_copy - start_memory_copy);
    auto duration_index_expansion = std::chrono::duration_cast<std::chrono::milliseconds>(stop__index_expansion - start_index_expansion);
    std::cout << "Time taken by function (Correlation): " << duration_kernel.count() << " milliseconds" << std::endl;
    // std::cout << "Time taken by function (Memory Copy): " << duration_memory_copy.count() << " milliseconds" << std::endl;
    std::cout << "Time taken by function (Index Expansion): " << duration_index_expansion.count() << " milliseconds" << std::endl;

    printf("All Equal: %s\n", all_equal ? "true" : "false");

    printf("Program Finished\n");

    gpuErrchk(cudaFree(d_grid_map));
    gpuErrchk(cudaFree(d_Y_io_x));
    gpuErrchk(cudaFree(d_Y_io_y));
    gpuErrchk(cudaFree(d_pack_idx));
    gpuErrchk(cudaFree(d_idx));
    gpuErrchk(cudaFree(d_all_correlation));
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

    auto start = std::chrono::high_resolution_clock::now();


    thrust::host_vector<double> h_pre_weights(d_pre_weights.begin(), d_pre_weights.end());
    std::vector<double> vec_weights(h_pre_weights.begin(), h_pre_weights.end());
    double max_val = *max_element(vec_weights.begin(), vec_weights.end());

    thrust::for_each(d_pre_weights.begin(), d_pre_weights.end(), _1 -= max_val - 50);
    thrust::transform(d_pre_weights.begin(), d_pre_weights.end(), d_pre_weights.begin(), thrust_exp());

    h_pre_weights.assign(d_pre_weights.begin(), d_pre_weights.end());
    vec_weights.assign(h_pre_weights.begin(), h_pre_weights.end());
    auto sum = std::accumulate(vec_weights.begin(), vec_weights.end(), 0.0, std::plus<double>());

    thrust::transform(d_pre_weights.begin(), d_pre_weights.end(), d_pre_weights.begin(), thrust_div_sum(sum));

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
    //printf("Total Time of Execution:  %3.1f ms\n", time_total);

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

#ifdef UPDATE_PARTICLES_EXEC
void host_update_particles() {

    // [✓] - Create a kernel with Input (xs, ys, thetas, numElements)
    // [✓] - Add Output to kernel(p_wb, R_wb) --> len of parameters (2x, 4x)
    // [✓] - Create device variables (d_xs, d_ys, d_thetas) & (d_p_wb, d_R_wb) & (res_p_wb, res_R_wb)
    // [✓] - Copy data from (xs, ys, thetas) -> (d_xs, d_ys, d_thetas)
    // [✓] - Initialize to zero (res_p_wb, res_R_wb) & (d_p_wb, d_R_wb)
    // [✓] - Change (p_wb, R_wb) to (T_wb)
    // [✓] - Add 'T_bl' to Input Parameters
    // [✓] - Create a custom MatrixMultiplication function
    // [✓] - Add 'lidar_coords' to function Input parameters
    // [✓] - Add Send length of 'lidar_coords' to the function
    // [✓] - Calculate Length of Result
    // [✓] - Calculate Execution Time of kernel
    // [✓] - Add res, xmin, ymax to the kernel
    // [✓] - Separate 'kernel_update_particles' into 'kernel_update_particles_states' & 'kernel_update_particles_lidar'
    // [✓] - Change blocksPerGrid   --> 1
    // [✓] - Change threadsPerBlock --> lidar_coords_LEN
    // [✓] - print blockId & threadId


    float time_total;
    cudaEvent_t start_total, stop_total;
    gpuErrchk(cudaEventCreate(&start_total));
    gpuErrchk(cudaEventCreate(&stop_total));

    const int STATES_LEN = NUM_PARTICLES;

    size_t size_of_states = STATES_LEN * sizeof(float);

    float* d_xs = NULL;
    float* d_ys = NULL;
    float* d_thetas = NULL;

    gpuErrchk(cudaMalloc((void**)&d_xs, size_of_states));
    gpuErrchk(cudaMalloc((void**)&d_ys, size_of_states));
    gpuErrchk(cudaMalloc((void**)&d_thetas, size_of_states));

    cudaMemcpy(d_xs, xs, size_of_states, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ys, ys, size_of_states, cudaMemcpyHostToDevice);
    cudaMemcpy(d_thetas, thetas, size_of_states, cudaMemcpyHostToDevice);

    size_t size_of_T_wb = 9 * STATES_LEN * sizeof(float);
    size_t size_of_T_bl = 9 * sizeof(float);
    size_t size_of_T_wl = 9 * NUM_PARTICLES * sizeof(float);
    size_t size_of_lidar_coords = 2 * lidar_coords_LEN * sizeof(float);
    size_t size_of_Y_wo = 2 * NUM_PARTICLES * lidar_coords_LEN * sizeof(float);
    float* d_T_wb = NULL;
    float* d_T_bl = NULL;
    float* d_T_wl = NULL;
    float* d_lidar_coords = NULL;
    float* d_Y_wo = NULL;
    gpuErrchk(cudaMalloc((void**)&d_T_wb, size_of_T_wb));
    gpuErrchk(cudaMalloc((void**)&d_T_bl, size_of_T_bl));
    gpuErrchk(cudaMalloc((void**)&d_T_wl, size_of_T_wl));
    gpuErrchk(cudaMalloc((void**)&d_lidar_coords, size_of_lidar_coords));
    gpuErrchk(cudaMalloc((void**)&d_Y_wo, size_of_Y_wo));


    float* res_T_wb = (float*)malloc(size_of_T_wb);
    float* res_T_wl = (float*)malloc(size_of_T_wl);
    float* res_Y_wo = (float*)malloc(size_of_Y_wo);
    memset(res_T_wb, 0, size_of_T_wb);
    memset(res_T_wl, 0, size_of_T_wl);
    memset(res_Y_wo, 0, size_of_Y_wo);
    cudaMemcpy(d_T_wb, res_T_wb, size_of_T_wb, cudaMemcpyHostToDevice);
    cudaMemcpy(d_T_wl, res_T_wl, size_of_T_wl, cudaMemcpyHostToDevice);
    cudaMemcpy(d_T_bl, T_bl, size_of_T_bl, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lidar_coords, lidar_coords, size_of_lidar_coords, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y_wo, res_Y_wo, size_of_Y_wo, cudaMemcpyHostToDevice);

    int threadsPerBlock = NUM_PARTICLES;
    int blocksPerGrid = 1;

    gpuErrchk(cudaEventRecord(start_total, 0));

    kernel_update_particles_states << <blocksPerGrid, threadsPerBlock >> > (d_xs, d_ys, d_thetas, d_T_wb, d_T_bl, d_T_wl, NUM_PARTICLES);
    cudaDeviceSynchronize();


    printf("%d \n", lidar_coords_LEN);
    threadsPerBlock = NUM_PARTICLES;
    blocksPerGrid = lidar_coords_LEN;
    kernel_update_particles_lidar << < blocksPerGrid, NUM_PARTICLES >> > (d_T_wl, d_Y_wo, d_lidar_coords, res, xmin, ymax, lidar_coords_LEN, NUM_PARTICLES);
    cudaDeviceSynchronize();

    gpuErrchk(cudaEventRecord(stop_total, 0));
    gpuErrchk(cudaEventSynchronize(stop_total));
    gpuErrchk(cudaEventElapsedTime(&time_total, start_total, stop_total));
    printf("Total Time of Execution:  %3.1f ms\n", time_total);

    gpuErrchk(cudaMemcpy(res_T_wb, d_T_wb, size_of_T_wb, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_T_wl, d_T_wl, size_of_T_wl, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_Y_wo, d_Y_wo, size_of_Y_wo, cudaMemcpyDeviceToHost));

    for (int i = 0; i < 9 * NUM_PARTICLES; i++) {
        // printf("%f, %f | ", res_T_wb[i], T_wb[i]);
        assert(abs(res_T_wb[i] - T_wb[i]) < 1e-5);
    }
    for (int i = 0; i < 9 * NUM_PARTICLES; i++) {
        // printf("%f, %f |  ", res_T_wl[i], T_wl[i]);
        assert(abs(res_T_wl[i] - T_wl[i]) < 1e-5);
    }

    //for (int i = 0; i < 2 * NUM_PARTICLES * lidar_coords_LEN; i++) {
    //    if ( abs(res_Y_wo[i] - Y_wo[i]) > 1e-5 ) {
    //        // printf("%f, %f, %f, %i  |  ", res_Y_wo[i], Y_wi[i], Y_wo[i], i);
    //        printf("%f, %f, %i  |  ", res_Y_wo[i], Y_wo[i], i);
    //    }
    //    // assert( res_Y_wo[i] == Y_wi[i] );
    //}

    // NUM_PARTICLES
    for (int i = 0; i < 2 * NUM_PARTICLES * lidar_coords_LEN; i++) {
        if (abs(res_Y_wo[i] - Y_wi[i]) > 1e-5) {
            // printf("%f, %f, %f, %i  |  ", res_Y_wo[i], Y_wi[i], Y_wo[i], i);
            printf("%f, %f, %i  |  ", res_Y_wo[i], Y_wi[i], i);
        }
        // assert( res_Y_wo[i] == Y_wi[i] );
    }
}
#endif

#ifdef UPDATE_UNIQUE_EXEC
void host_update_unique() {

    int negative_before_counter = 0;
    int count_bigger_than_height = 0;
    for (int i = 0; i < BEFORE_LEN; i++) {
        if (Y_io_x_before[i] < 0 || Y_io_y_before[i] < 0)
            negative_before_counter += 1;

        if (Y_io_y_before[i] >= GRID_HEIGHT)
            count_bigger_than_height += 1;
    }

    int negative_after_counter = 0;
    for (int i = 0; i < AFTER_LEN; i++) {
        if (Y_io_x_after[i] < 0 || Y_io_y_after[i] < 0)
            negative_after_counter += 1;
    }

    printf("GRID_WIDTH: %d, GRID_HEIGHT: %d\n", GRID_WIDTH, GRID_HEIGHT);
    printf("negative_before_counter: %d\n", negative_before_counter);
    printf("negative_after_counter: %d\n", negative_after_counter);
    printf("count_bigger_than_height: %d\n", count_bigger_than_height);

    printf("size of --> int16: %d, int32: %d, int: %d\n", sizeof(int16_t), sizeof(int32_t), sizeof(int));

    // [ ] - Create a kernel for cum_sum


    const int NUM_ELEMS = NUM_PARTICLES + 1;


    /********************************************************************/
    /************************** PRIOR VARIABLES *************************/
    /********************************************************************/
    int* d_Y_io_x = NULL;
    int* d_Y_io_y = NULL;
    int* d_idx = NULL;
    size_t   sz_Y_io = BEFORE_LEN * sizeof(int);
    size_t   sz_idx = NUM_PARTICLES * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_Y_io_x, sz_Y_io));
    gpuErrchk(cudaMalloc((void**)&d_Y_io_y, sz_Y_io));
    gpuErrchk(cudaMalloc((void**)&d_idx, sz_idx));

    cudaMemcpy(d_Y_io_x, Y_io_x_before, sz_Y_io, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y_io_y, Y_io_y_before, sz_Y_io, cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx, idx_before, sz_idx, cudaMemcpyHostToDevice);


    /********************************************************************/
    /**************************** MAP VARIABLES *************************/
    /********************************************************************/
    uint8_t* d_whole_map = NULL;
    int* d_unique_in_particle = NULL;
    int* d_unique_in_particle_col = NULL;

    size_t   sz_whole_map = GRID_WIDTH * GRID_HEIGHT * NUM_PARTICLES * sizeof(uint8_t);
    size_t   sz_unique_in_particle = NUM_ELEMS * sizeof(int);
    size_t   sz_unique_in_particle_col = NUM_ELEMS * GRID_WIDTH * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_whole_map, sz_whole_map));
    gpuErrchk(cudaMalloc((void**)&d_unique_in_particle, sz_unique_in_particle));
    gpuErrchk(cudaMalloc((void**)&d_unique_in_particle_col, sz_unique_in_particle_col));

    int* h_unique_in_particle = (int*)malloc(sz_unique_in_particle);
    int* h_unique_in_particle_col = (int*)malloc(sz_unique_in_particle_col);
    uint8_t* whole_map = (uint8_t*)malloc(sz_whole_map);

    cudaMemset(d_whole_map, 0, sz_whole_map);
    cudaMemset(d_unique_in_particle, 0, sz_unique_in_particle);
    cudaMemset(d_unique_in_particle_col, 0, sz_unique_in_particle_col);

    /********************************************************************/
    /*********************** MEASUREMENT VARIABLES **********************/
    /********************************************************************/
    int* d_measure_x = NULL;
    int* d_measure_y = NULL;
    int* d_measure_idx = NULL;
    size_t sz_measure = MEASURE_LEN * sizeof(int);
    size_t sz_measure_idx = NUM_PARTICLES * sizeof(int);

    gpuErrchk(cudaMalloc((void**)&d_measure_x, sz_measure));
    gpuErrchk(cudaMalloc((void**)&d_measure_y, sz_measure));
    gpuErrchk(cudaMalloc((void**)&d_measure_idx, sz_measure_idx));

    cudaMemcpy(d_measure_x, Y_io_x_measure, sz_measure, cudaMemcpyHostToDevice);
    cudaMemcpy(d_measure_y, Y_io_y_measure, sz_measure, cudaMemcpyHostToDevice);
    cudaMemcpy(d_measure_idx, idx_measure, sz_measure_idx, cudaMemcpyHostToDevice);


    /********************************************************************/
    /**************************** CREATE MAP ****************************/
    /********************************************************************/
    int threadsPerBlock = 100; // 1000;
    int blocksPerGrid = NUM_ELEMS;

    auto start_create_map = std::chrono::high_resolution_clock::now();

    kernel_create_2d_map << <blocksPerGrid, threadsPerBlock >> > (d_Y_io_x, d_Y_io_y, d_idx, BEFORE_LEN, d_whole_map, d_unique_in_particle,
        d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT, NUM_ELEMS);
    cudaDeviceSynchronize();

    auto stop_create_map = std::chrono::high_resolution_clock::now();


    /********************************************************************/
    /**************************** UPDATE MAP ****************************/
    /********************************************************************/
    auto start_update_map = std::chrono::high_resolution_clock::now();

    threadsPerBlock = NUM_ELEMS;
    blocksPerGrid = 1;

    kernel_update_2d_map_with_measure << <blocksPerGrid, threadsPerBlock >> > (d_measure_x, d_measure_y, d_measure_idx, MEASURE_LEN, d_whole_map, d_unique_in_particle,
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

    gpuErrchk(cudaMemcpy(whole_map, d_whole_map, sz_whole_map, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_unique_in_particle, d_unique_in_particle, sz_unique_in_particle, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_unique_in_particle_col, d_unique_in_particle_col, sz_unique_in_particle_col, cudaMemcpyDeviceToHost));

    int new_len = h_unique_in_particle[NUM_ELEMS - 1];
    gpuErrchk(cudaFree(d_Y_io_x));
    gpuErrchk(cudaFree(d_Y_io_y));

    sz_Y_io = new_len * sizeof(int);
    gpuErrchk(cudaMalloc((void**)&d_Y_io_x, sz_Y_io));
    gpuErrchk(cudaMalloc((void**)&d_Y_io_y, sz_Y_io));
    int* res_Y_io_x = (int*)malloc(sz_Y_io);
    int* res_Y_io_y = (int*)malloc(sz_Y_io);


    /********************************************************************/
    /************************ MAP RESTRUCTURE ***************************/
    /********************************************************************/
    threadsPerBlock = GRID_WIDTH;
    blocksPerGrid = NUM_PARTICLES;

    auto start_map_restructure = std::chrono::high_resolution_clock::now();

    kernel_update_unique_restructure << <blocksPerGrid, threadsPerBlock >> > (d_whole_map, d_Y_io_x, d_Y_io_y, d_unique_in_particle,
        d_unique_in_particle_col, GRID_WIDTH, GRID_HEIGHT);
    cudaDeviceSynchronize();

    auto stop_map_restructure = std::chrono::high_resolution_clock::now();

    gpuErrchk(cudaMemcpy(res_Y_io_x, d_Y_io_x, sz_Y_io, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_Y_io_y, d_Y_io_y, sz_Y_io, cudaMemcpyDeviceToHost));

    auto duration_create_map = std::chrono::duration_cast<std::chrono::milliseconds>(stop_create_map - start_create_map);
    auto duration_update_map = std::chrono::duration_cast<std::chrono::milliseconds>(stop_update_map - start_update_map);
    auto duration_cumulative_sum = std::chrono::duration_cast<std::chrono::milliseconds>(stop_cumulative_sum - start_cumulative_sum);
    auto duration_map_restructure = std::chrono::duration_cast<std::chrono::milliseconds>(stop_map_restructure - start_map_restructure);

    std::cout << "Time taken by function (Create Map): " << duration_create_map.count() << " milliseconds" << std::endl;
    std::cout << "Time taken by function (Update Map): " << duration_update_map.count() << " milliseconds" << std::endl;
    std::cout << "Time taken by function (Cumulative Sum): " << duration_cumulative_sum.count() << " milliseconds" << std::endl;
    std::cout << "Time taken by function (Map Restructure): " << duration_map_restructure.count() << " milliseconds" << std::endl;

    printf("\n%d\n", h_unique_in_particle[NUM_ELEMS - 1]);


    int num_fails = 0;
    for (int i = 0, j = 0; i < h_unique_in_particle[NUM_ELEMS - 1]; i++) {

        if (Y_io_x_after[i] > 0 && Y_io_y_after[i] > 0) {

            //if (Y_io_x_after[i] != res_Y_io_x[j] || Y_io_y_after[i] != res_Y_io_y[j]) {
            //    printf("%d: %d=%d, %d=%d\n", j, res_Y_io_x[j], Y_io_x_after[i], res_Y_io_y[j], Y_io_y_after[i]);
            //    num_fails += 1;
            //    if (num_fails > 70)
            //        exit(0);
            //}
            assert(Y_io_x_after[i] == res_Y_io_x[j]);
            assert(Y_io_y_after[i] == res_Y_io_y[j]);
            j += 1;
        }
    }
    printf("All Passed\n");

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

__global__ void kernel_index_expansion(const int *idx_pack, int *idx, const int numElements) {

    int i = blockIdx.x;
    int k = threadIdx.x;
    const int numThreads = blockDim.x;

    if (i < numThreads) {

        int first_idx = idx_pack[i];
        int last_idx = (i < numThreads - 1) ? idx_pack[i + 1] : numElements;
        int arr_len = last_idx - first_idx;

        int arr_end = last_idx;

        int start_idx = ((arr_len / blockDim.x) * k) + first_idx;
        int end_idx = ((arr_len / blockDim.x) * (k + 1)) + first_idx;
        end_idx = (k < blockDim.x - 1) ? end_idx : arr_end;

        for (int j = start_idx; j < end_idx; j++)
            idx[j] = i;
    }
}

__global__ void kernel_correlation_max(const int* all_correlation, int* correlation, const int _NUM_PARTICLES) {

    int i = threadIdx.x;

    int curr_max_value = all_correlation[i];
    for (int j = 0; j < 25; j++) {
        int curr_value = all_correlation[j * _NUM_PARTICLES + i];
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

        int start_current_map_idx = i * _GRID_WIDTH * _GRID_HEIGHT;
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


__global__ void kernel_update_particles_states(const float* xs, const float* ys, const float* thetas,
                                        float* T_wb, const float* T_bl, float* T_wl, const int numElements) {

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

__global__ void kernel_update_particles_lidar(float* _T_wl, float* _Y_wo, const float* _lidar_coords, float _res, int _xmin, int _ymax,
                                                const int _lidar_coords_LEN, const int numElements) {

    int T_idx = threadIdx.x * 9;
    int wo_idx = 2 * _lidar_coords_LEN * threadIdx.x;
    int k = blockIdx.x;

    for (int j = 0; j < 2; j++) {

        float currVal = 0;
        currVal += _T_wl[T_idx + j * 3 + 0] * _lidar_coords[(0 * _lidar_coords_LEN) + k];
        currVal += _T_wl[T_idx + j * 3 + 1] * _lidar_coords[(1 * _lidar_coords_LEN) + k];
        currVal += _T_wl[T_idx + j * 3 + 2];

        // _Y_wo[wo_idx + (j * _lidar_coords_LEN) + k] = currVal; // ceil((currVal - _xmin) / _res);

        if (j == 0) {
            _Y_wo[wo_idx + (1 * _lidar_coords_LEN) + k] = ceil((currVal - _xmin) / _res);
        }
        else {
            _Y_wo[wo_idx + (0 * _lidar_coords_LEN) + k] = ceil((_ymax - currVal) / _res);
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

__global__ void kernel_create_2d_map(const int *Y_io_x, const int *Y_io_y, const int *idx, const int IDX_LEN, uint8_t *whole_map,
                                        int *unique_in_particle, int *unique_in_particle_col, const int _GRID_WIDTH, const int _GRID_HEIGHT, const int _NUM_ELEMS) {

    int i = blockIdx.x;
    int k = threadIdx.x;

    if (i < _NUM_ELEMS - 1) {

        int first_idx = idx[i];
        int last_idx = (i < _NUM_ELEMS - 2) ? idx[i + 1] : IDX_LEN;
        int arr_len = last_idx - first_idx;

        int arr_end = last_idx;

        int start_idx   = ((arr_len / blockDim.x) * k) + first_idx;
        int end_idx     = ((arr_len / blockDim.x) * (k + 1)) + first_idx;
        end_idx = (k < blockDim.x - 1) ? end_idx : arr_end;

        int start_whole_map_idx = i * _GRID_WIDTH * _GRID_HEIGHT;
        int start_of_col = i * _GRID_WIDTH;

        for (int j = start_idx; j < end_idx; j++) {

            int x = Y_io_x[j];
            int y = Y_io_y[j];

            int curr_idx = start_whole_map_idx + (x * _GRID_HEIGHT) + y;

            if (whole_map[curr_idx] == 0 && x >= 0 && y >= 0) {
                whole_map[curr_idx] = 1;
                atomicAdd(&unique_in_particle[i + 1], 1);
                atomicAdd(&unique_in_particle_col[start_of_col + x + 1], 1);
                // unique_in_particle_col[start_of_col + x + 1] = unique_in_particle_col[start_of_col + x + 1] + 1;
            }
        }
    }    
}

__global__ void kernel_update_2d_map_with_measure(const int* measure_x, const int* measure_y, const int* idx, const int IDX_LEN, uint8_t* whole_map,
                    int* unique_in_particle, int* unique_in_particle_col, const int _GRID_WIDTH, const int _GRID_HEIGHT, const int _NUM_ELEMS) {

    int i = threadIdx.x;

    if (i < _NUM_ELEMS - 1) {

        int start_idx = idx[i];
        int end_idx = (i < _NUM_ELEMS - 2) ? idx[i + 1] : IDX_LEN;

        int start_whole_map_idx = i * _GRID_WIDTH * _GRID_HEIGHT;
        int start_of_col = i * _GRID_WIDTH;

        for (int j = start_idx; j < end_idx; j++) {

            int x = measure_x[j];
            int y = measure_y[j];

            int curr_idx = start_whole_map_idx + (x * _GRID_HEIGHT) + y;

            if (x >= 0 && y >= 0 && whole_map[curr_idx] == 0) {
                whole_map[curr_idx] = 1;
                atomicAdd(&unique_in_particle[i + 1], 1);
                atomicAdd(&unique_in_particle_col[start_of_col + x + 1], 1);
                // unique_in_particle_col[start_of_col + x + 1] = unique_in_particle_col[start_of_col + x + 1] + 1;
            }
        }
    }
}

__global__ void kernel_update_unique_restructure(uint8_t* whole_map, int* Y_io_x, int* Y_io_y, int* unique_in_particle, int* unique_in_particle_col,
                                                    const int _GRID_WIDTH, const int _GRID_HEIGHT) {

    int i = blockIdx.x;
    int l = threadIdx.x;

    int start_of_current_map = i * _GRID_WIDTH * _GRID_HEIGHT;
    int start_whole_map_idx = (i * _GRID_WIDTH * _GRID_HEIGHT) + (l * _GRID_HEIGHT);
    int end_whole_map_idx = (i * _GRID_WIDTH * _GRID_HEIGHT) + ((l + 1) * _GRID_HEIGHT);
    int key = unique_in_particle_col[i * _GRID_WIDTH + l] + unique_in_particle[i];

    for (int j = start_whole_map_idx; j < end_whole_map_idx; j++) {


        if (whole_map[j] == 1) {

            int y = (j - start_of_current_map) % _GRID_HEIGHT;
            int x = (j - start_of_current_map) / _GRID_HEIGHT;

            Y_io_x[key] = x;
            Y_io_y[key] = y;
            key += 1;
        }
    }
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