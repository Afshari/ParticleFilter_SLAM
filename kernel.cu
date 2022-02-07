
#include "headers.h"

#include "data/combined_3423.h"

// 1. Define a function kernel for calculating correlation
/// 1.1. The function has _ parameter as input 'grid_map', 'Y_io_x', 'Y_io_y' and 'Y_io_idx'
/// 1.2. Define 'grid_map', 'Y_io_x', 'Y_io_y' and 'Y_io_idx' as constant
//// 1.2.1. Define width of 'Y_io__' for further usage --> 'Y_LENGTH' 
/// 1.3. Define an array for result of correlation calculation --> 'corr_result'
/// 1.4. Calculation in each thread:
//// 1.4.1. Each thread should get its global index
//// 1.4.2. Each thread should get 'idx_x' & 'idx_y' and according to them get value from 'grid_map'
//// 1.4.3. If the corresponding value in the 'grid_map' is (-1 or 1) then add the value to 'corr_result'
//// 1.4.4. thread can get index of 'corr_result' from 'Y_io_idx'
//// 1.4.5. find formula to calculate 'grid_map' index from 'idx_x' & 'idx_y'

// 2. Find an example of 'Cuda Add' for big Data
// 3. Find an Execution Time Code for Cuda
/// 3.1. Find Execution Time of Memory Copy
/// 3.2. Find Execution Time of Kernel 
/// 3.3. Find Execution Time of Result Copy
/// 3.4. Find Execution Time of Total Execution

// *. Define a function for 'correlation host execution'
// 4. 



__global__ void calc_correlation(const int* d_grid_map, const int* d_Y_io_x, const int* d_Y_io_y,
    const int* d_Y_io_idx, int* result, int numElements);


void host_correlation();


int main() {

    host_correlation();

    return 0;
}


/*
* Host Functions
*/

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
    // printf("CUDA kernel launch with %d blocks of %d threads, All Threads: %d\n", blocksPerGrid, threadsPerBlock, blocksPerGrid * threadsPerBlock);

    gpuErrchk(cudaEventRecord(start_kernel, 0));

    calc_correlation << <blocksPerGrid, threadsPerBlock >> > (d_grid_map, d_Y_io_x, d_Y_io_y, d_Y_io_idx, d_result, num_elements_of_Y);

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


/*
* Kernel Functions
*/

__global__ void calc_correlation(const int* d_grid_map, const int* d_Y_io_x, const int* d_Y_io_y,
                                    const int* d_Y_io_idx, int* result, int numElements) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {

        int loop_counter = 0;
        for (int x_offset = -2; x_offset <= 2; x_offset++) {

            for (int y_offset = -2; y_offset <= 2; y_offset++) {

                int idx = d_Y_io_idx[i];
                int x = d_Y_io_x[i] + x_offset;
                int y = d_Y_io_y[i] + y_offset;

                if (x >= 0 && y >= 0 && x < GRID_WIDTH && y < GRID_HEIGHT) {

                    int grid_map_idx = x * GRID_HEIGHT + y;
                    int value = d_grid_map[grid_map_idx];

                    if (value != 0)
                        atomicAdd(&result[loop_counter * 100 + idx], value);
                }
                loop_counter++;
            }
        }
    }
}