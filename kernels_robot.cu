
#include "kernels_utils.cuh"
#include "kernels_robot.cuh"


__global__ void kernel_correlation_max(float* weights, const float* weights_raw, const int PARTICLES_LEN) {

    int i = threadIdx.x;

    float curr_max_value = weights_raw[i];
    for (int j = 0; j < 25; j++) {
        float curr_value = weights_raw[j * PARTICLES_LEN + i];
        if (curr_value > curr_max_value) {
            curr_max_value = curr_value;
        }
    }
    weights[i] = curr_max_value;
}

__global__ void kernel_correlation(float* weights, const int F_SEP, 
    const int* grid_map, const int* states_x, const int* states_y,
    const int* states_idx, const int GRID_WIDTH, const int GRID_HEIGHT, const int NUM_ELEMS) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < NUM_ELEMS) {

        int loop_counter = 0;
        for (int x_offset = -2; x_offset <= 2; x_offset++) {

            for (int y_offset = -2; y_offset <= 2; y_offset++) {

                int idx = states_idx[i];
                int x = states_x[i] + x_offset;
                int y = states_y[i] + y_offset;

                if (x >= 0 && y >= 0 && x < GRID_WIDTH && y < GRID_HEIGHT) {

                    int curr_idx = x * GRID_HEIGHT + y;
                    float value = grid_map[curr_idx];
                    value = (value == 2) ? 1 : -1;

                    if (value != 0)
                        atomicAdd(&weights[loop_counter * 100 + idx], value);
                }
                loop_counter++;
            }
        }
    }
}

__global__ void kernel_resampling(int* js, const float* weights, const float* rnd, const int NUM_ELEMS) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < NUM_ELEMS) {

        float u = rnd[i] / NUM_ELEMS;
        int j = 0;
        float beta = u + float(i) / NUM_ELEMS;

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

__global__ void kernel_create_2d_map(uint8_t* map_2d, int* unique_in_particle, int* unique_in_particle_col, const int F_SEP,
    const int* particles_x, const int* particles_y, const int* particles_idx, const int IDX_LEN, 
    const int GRID_WIDTH, const int GRID_HEIGHT, const int NUM_ELEMS) {

    int i = blockIdx.x;
    int k = threadIdx.x;

    if (i < NUM_ELEMS) {

        int first_idx = particles_idx[i];
        int last_idx = (i < NUM_ELEMS - 1) ? particles_idx[i + 1] : IDX_LEN;
        int arr_len = last_idx - first_idx;

        int arr_end = last_idx;

        int start_idx = ((arr_len / blockDim.x) * k) + first_idx;
        int end_idx = ((arr_len / blockDim.x) * (k + 1)) + first_idx;
        end_idx = (k < blockDim.x - 1) ? end_idx : arr_end;

        int start_of_current_map = i * GRID_WIDTH * GRID_HEIGHT;
        int start_of_col = i * GRID_WIDTH;


        for (int j = start_idx; j < end_idx; j++) {

            int x = particles_x[j];
            int y = particles_y[j];

            int curr_idx = start_of_current_map + (x * GRID_HEIGHT) + y;

            if (x >= 0 && y >= 0 && map_2d[curr_idx] == 0) {
                map_2d[curr_idx] = 1;
                atomicAdd(&unique_in_particle[i], 1);
                atomicAdd(&unique_in_particle_col[start_of_col + x + 1], 1);
            }
        }
    }
}

__global__ void kernel_update_2d_map_with_measure(uint8_t* map_2d, int* unique_in_particle, int* unique_in_particle_col, const int F_SEP,
    const int* measure_x, const int* measure_y, const int* measure_idx, const int IDX_LEN,
    const int GRID_WIDTH, const int GRID_HEIGHT, const int NUM_ELEMS) {

    int i = threadIdx.x;

    if (i < NUM_ELEMS) {

        int start_idx = measure_idx[i];
        int end_idx = (i < NUM_ELEMS - 1) ? measure_idx[i + 1] : IDX_LEN;

        int start_of_current_map = i * GRID_WIDTH * GRID_HEIGHT;
        int start_of_col = i * GRID_WIDTH;

        for (int j = start_idx; j < end_idx; j++) {

            int x = measure_x[j];
            int y = measure_y[j];

            int curr_idx = start_of_current_map + (x * GRID_HEIGHT) + y;

            if (x >= 0 && y >= 0 && map_2d[curr_idx] == 0) {
                map_2d[curr_idx] = 1;
                atomicAdd(&unique_in_particle[i], 1);
                atomicAdd(&unique_in_particle_col[start_of_col + x + 1], 1);
            }
        }
    }
}

/**
 * Kernel 'Update Particles States'
 *
 * @param[out] transition_world_body
 * @param[out] transition_world_lidar
 * @param[in]  F_SEP
 * @param[in]  states_x
 * @param[in]  states_y
 * @param[in]  states_theta
 * @param[in]  transition_body_lidar
 * @param[in]  NUM_ELEMS
 * @return None
 */
__global__ void kernel_update_particles_states(float* transition_world_body, float* transition_world_lidar, const int F_SEP,
    const float* states_x, const float* states_y, const float* states_theta,
    const float* transition_body_lidar, const int NUM_ELEMS) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < NUM_ELEMS) {

        int T_idx = i * 9;

        float p_wb_0 = states_x[i];
        float p_wb_1 = states_y[i];

        float R_wb_0 = cos(states_theta[i]);
        float R_wb_1 = -sin(states_theta[i]);
        float R_wb_2 = sin(states_theta[i]);
        float R_wb_3 = cos(states_theta[i]);

        transition_world_body[T_idx + 0] = R_wb_0;   transition_world_body[T_idx + 1] = R_wb_1;   transition_world_body[T_idx + 2] = p_wb_0;
        transition_world_body[T_idx + 3] = R_wb_2;   transition_world_body[T_idx + 4] = R_wb_3;   transition_world_body[T_idx + 5] = p_wb_1;
        transition_world_body[T_idx + 6] = 0;        transition_world_body[T_idx + 7] = 0;        transition_world_body[T_idx + 8] = 1;

        kernel_matrix_mul_3x3(transition_world_body, transition_body_lidar, transition_world_lidar, T_idx);
    }
}

/**
 * Kernel 'Update Particles Lidar'
 *
 * @param[out] processed_measure_x
 * @param[out] processed_measure_y
 * @param[in]  F_SEP
 * @param[in]  transition_world_lidar
 * @param[in]  lidar_coords
 * @param[in]  res
 * @param[in]  xmin
 * @param[in]  ymax
 * @param[in]  LIDAR_COORDS_LEN
 * @return None
 */
__global__ void kernel_update_particles_lidar(int* processed_measure_x, int* processed_measure_y, const int F_SEP, 
    const float* transition_world_lidar, const float* lidar_coords, 
    const float res, const int xmin, const int ymax, const int LIDAR_COORDS_LEN) {

    int T_idx = threadIdx.x * 9;
    int wo_idx = LIDAR_COORDS_LEN * threadIdx.x;
    int k = blockIdx.x;

    for (int j = 0; j < 2; j++) {

        double currVal = 0;
        currVal += transition_world_lidar[T_idx + j * 3 + 0] * lidar_coords[(0 * LIDAR_COORDS_LEN) + k];
        currVal += transition_world_lidar[T_idx + j * 3 + 1] * lidar_coords[(1 * LIDAR_COORDS_LEN) + k];
        currVal += transition_world_lidar[T_idx + j * 3 + 2];

        if (j == 0) {
            processed_measure_y[wo_idx + k] = (int)ceil((currVal - xmin) / res);
        }
        else {
            processed_measure_x[wo_idx + k] = (int)ceil((ymax - currVal) / res);
        }
    }
}

__global__ void kernel_update_unique_restructure(int* particles_x, int* particles_y, int* particles_idx, const int F_SEP,
    const uint8_t* map_2d, const int* unique_in_particle,
    const int* unique_in_particle_col, const int GRID_WIDTH, const int GRID_HEIGHT) {

    int i = blockIdx.x;
    int l = threadIdx.x;

    int start_of_current_map = i * GRID_WIDTH * GRID_HEIGHT;
    int start_idx = (i * GRID_WIDTH * GRID_HEIGHT) + (l * GRID_HEIGHT);
    int end_idx = (i * GRID_WIDTH * GRID_HEIGHT) + ((l + 1) * GRID_HEIGHT);
    int key = unique_in_particle_col[i * GRID_WIDTH + l] + unique_in_particle[i];

    for (int j = start_idx; j < end_idx; j++) {

        if (map_2d[j] == 1) {

            int y = (j - start_of_current_map) % GRID_HEIGHT;
            int x = (j - start_of_current_map) / GRID_HEIGHT;

            particles_x[key] = x;
            particles_y[key] = y;
            key += 1;
            atomicAdd(&particles_idx[i], 1);
        }
    }
}

__global__ void kernel_rearrange_particles(int* particles_x, int* particles_y, const int F_SEP, 
    const int* particles_idx, const int* c_particles_x, const int* c_particles_y, const int* c_particles_idx, const int* js,
    const int GRID_WIDTH, const int GRID_HEIGHT, const int NUM_ELEMS, const int IDX_LEN, const int C_IDX_LEN) {

    int i = blockIdx.x;
    int k = threadIdx.x;
    int m = js[i];

    if (i < NUM_ELEMS) {

        int first_idx = particles_idx[i];
        int last_idx = (i < NUM_ELEMS - 1) ? particles_idx[i + 1] : IDX_LEN;
        int arr_end = last_idx;
        int arr_len = last_idx - first_idx;

        int c_first_idx = c_particles_idx[m];
        int c_last_idx = (m < NUM_ELEMS - 1) ? c_particles_idx[m + 1] : C_IDX_LEN;
        int c_arr_end = c_last_idx;
        int c_arr_len = c_last_idx - c_first_idx;

        int start_idx = ((arr_len / blockDim.x) * k) + first_idx;
        int end_idx = ((arr_len / blockDim.x) * (k + 1)) + first_idx;
        end_idx = (k < blockDim.x - 1) ? end_idx : arr_end;

        int c_start_idx = ((c_arr_len / blockDim.x) * k) + c_first_idx;
        int c_end_idx = ((c_arr_len / blockDim.x) * (k + 1)) + c_first_idx;
        c_end_idx = (k < blockDim.x - 1) ? c_end_idx : c_arr_end;

        for (int j = start_idx, c_j = c_start_idx; j < end_idx && c_j < c_end_idx; j++, c_j++) {
            particles_x[j] = c_particles_x[c_j];
            particles_y[j] = c_particles_y[c_j];
        }
    }
}

__global__ void kernel_rearrange_states(float* states_x, float* states_y, float* states_theta, const int F_SEP,
    const float* c_states_x, const float* c_states_y, const float* c_states_theta, const int* js) {

    int i = threadIdx.x;
    int j = js[i];

    states_x[i] = c_states_x[j];
    states_y[i] = c_states_y[j];
    states_theta[i] = c_states_theta[j];
}

__global__ void kernel_rearrange_indecies(int* particles_idx, int* last_len, const int F_SEP, 
    const int* c_particles_idx, const int* js, const int ARR_LEN) {

    int i = threadIdx.x;
    int j = js[i];
    int idx_value = 0;

    if (j == blockDim.x - 1)
        idx_value = ARR_LEN - c_particles_idx[j];
    else
        idx_value = c_particles_idx[j + 1] - c_particles_idx[j];

    if (i == 0)
        particles_idx[i] = 0;

    if (i < blockDim.x - 1)
        particles_idx[i] = idx_value;
    else
        last_len[0] = idx_value;
}




