
#include "headers.h"
#include "kernels.cuh"

__global__ void kernel_bresenham_rearrange(int* particles_free_x, int* particles_free_y, int* particles_free_x_max, int* particles_free_y_max,
    int* particles_free_idx, const int MAX_DIST_IN_MAP, const int NUM_ELEMS) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < NUM_ELEMS) {

        int start_idx = particles_free_idx[i];
        int end_idx = particles_free_idx[i + 1];
        int curr_particles_len = end_idx - start_idx;
        int start_idx_max = i * MAX_DIST_IN_MAP;

        for (int j = 0; j < curr_particles_len; j++) {

            particles_free_x[start_idx + j] = particles_free_x_max[start_idx_max + j];
            particles_free_y[start_idx + j] = particles_free_y_max[start_idx_max + j];
        }
    }
}

__global__ void kernel_bresenham(const int* particles_occupied_x, const int* particles_occupied_y, const int* position_image_body,
    int* particles_free_x, int* particles_free_y, int* particles_free_counter, const int PARTICLES_LEN, const int MAX_DIST_IN_MAP) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < PARTICLES_LEN) {

        int pointsCounter = 0;
        int x = particles_occupied_x[i];
        int y = particles_occupied_y[i];
        int x1 = x;
        int y1 = y;
        int position_image_body_x = position_image_body[0];
        int position_image_body_y = position_image_body[1];
        int x2 = position_image_body_x;
        int y2 = position_image_body_y;

        int dx = abs(x2 - x1);
        int dy = abs(y2 - y1);

        int start_idx = i * MAX_DIST_IN_MAP;

        if (dx == 0) {

            int sign = (y2 - y1) > 0 ? 1 : -1;
            particles_free_x[start_idx] = x;
            particles_free_y[start_idx] = y;
            pointsCounter += 1;

            for (int j = 1; j <= dy; j++) {
                particles_free_x[start_idx + j] = x;
                particles_free_y[start_idx + j] = y + sign * j;
                pointsCounter += 1;
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
                particles_free_x[start_idx] = x;
                particles_free_y[start_idx] = y;
                pointsCounter += 1;
            }
            else {
                particles_free_x[start_idx] = y;
                particles_free_y[start_idx] = x;
                pointsCounter += 1;
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
                    particles_free_x[start_idx + j] = x;
                    particles_free_y[start_idx + j] = y;
                    pointsCounter += 1;
                }
                else {
                    particles_free_x[start_idx + j] = y;
                    particles_free_y[start_idx + j] = x;
                    pointsCounter += 1;
                }
            }
        }
        particles_free_counter[i] = pointsCounter;
    }
}


__global__ void kernel_bresenham(const int* particles_occupied_x, const int* particles_occupied_y,
    const int* position_image_body, int* particles_free_x, int* particles_free_y, const int* particles_free_idx, const int PARTICLES_LEN) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < PARTICLES_LEN) {

        int x = particles_occupied_x[i];
        int y = particles_occupied_y[i];
        int x1 = x;
        int y1 = y;
        int position_image_body_x = position_image_body[0];
        int position_image_body_y = position_image_body[1];
        int x2 = position_image_body_x;
        int y2 = position_image_body_y;

        int dx = abs(x2 - x1);
        int dy = abs(y2 - y1);

        int start_index = particles_free_idx[i];

        if (dx == 0) {

            int sign = (y2 - y1) > 0 ? 1 : -1;
            particles_free_x[start_index] = x;
            particles_free_y[start_index] = y;

            for (int j = 1; j <= dy; j++) {
                particles_free_x[start_index + j] = x;
                particles_free_y[start_index + j] = y + sign * j;
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
                particles_free_x[start_index] = x;
                particles_free_y[start_index] = y;
            }
            else {
                particles_free_x[start_index] = y;
                particles_free_y[start_index] = x;
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
                    particles_free_x[start_index + j] = x;
                    particles_free_y[start_index + j] = y;
                }
                else {
                    particles_free_x[start_index + j] = y;
                    particles_free_y[start_index + j] = x;
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

__global__ void kernel_index_expansion(const int* idx, int* extended_idx, const int numElements) {

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

__global__ void kernel_correlation_max(const float* weights_raw, float* weights, const int _NUM_PARTICLES) {

    int i = threadIdx.x;

    float curr_max_value = weights_raw[i];
    for (int j = 0; j < 25; j++) {
        float curr_value = weights_raw[j * _NUM_PARTICLES + i];
        if (curr_value > curr_max_value) {
            curr_max_value = curr_value;
        }
    }
    weights[i] = curr_max_value;
}

__global__ void kernel_correlation(const int* grid_map, const int* states_x, const int* states_y,
    const int* states_idx, float* weights, const int _GRID_WIDTH, const int _GRID_HEIGHT, int numElements) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {

        // int start_current_map_idx = i * _GRID_WIDTH * _GRID_HEIGHT;
        int loop_counter = 0;
        for (int x_offset = -2; x_offset <= 2; x_offset++) {

            for (int y_offset = -2; y_offset <= 2; y_offset++) {

                int idx = states_idx[i];
                int x = states_x[i] + x_offset;
                int y = states_y[i] + y_offset;

                if (x >= 0 && y >= 0 && x < _GRID_WIDTH && y < _GRID_HEIGHT) {

                    int curr_idx = x * _GRID_HEIGHT + y;
                    // int curr_idx = start_current_map_idx + (x * _GRID_HEIGHT) + y;
                    float value = grid_map[curr_idx];

                    if (value != 0)
                        atomicAdd(&weights[loop_counter * 100 + idx], value);
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

        float R_wb_0 = cos(states_theta[i]);
        float R_wb_1 = -sin(states_theta[i]);
        float R_wb_2 = sin(states_theta[i]);
        float R_wb_3 = cos(states_theta[i]);

        transition_body_frame[T_idx + 0] = R_wb_0;   transition_body_frame[T_idx + 1] = R_wb_1;   transition_body_frame[T_idx + 2] = p_wb_0;
        transition_body_frame[T_idx + 3] = R_wb_2;   transition_body_frame[T_idx + 4] = R_wb_3;   transition_body_frame[T_idx + 5] = p_wb_1;
        transition_body_frame[T_idx + 6] = 0;        transition_body_frame[T_idx + 7] = 0;        transition_body_frame[T_idx + 8] = 1;

        kernel_matrix_mul_3x3(transition_body_frame, transition_lidar_frame, transition_world_frame, T_idx);
    }
}

__global__ void kernel_update_particles_lidar(float* transition_world_frame, int* processed_measure_x, int* processed_measure_y,
    float* particles_wframe_x, float* particles_wframe_y, const float* _lidar_coords, float _res, int _xmin, int _ymax, const int _LIDAR_COORDS_LEN) {

    int k = blockIdx.x;

    for (int j = 0; j < 2; j++) {

        double currVal = 0;
        currVal += transition_world_frame[j * 3 + 0] * _lidar_coords[(0 * _LIDAR_COORDS_LEN) + k];
        currVal += transition_world_frame[j * 3 + 1] * _lidar_coords[(1 * _LIDAR_COORDS_LEN) + k];
        currVal += transition_world_frame[j * 3 + 2];

        if (j == 0) {
            particles_wframe_x[k] = currVal;
            processed_measure_y[k] = (int)ceil((currVal - _xmin) / _res);
        }
        else {
            particles_wframe_y[k] = currVal;
            processed_measure_x[k] = (int)ceil((_ymax - currVal) / _res);
        }
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
            processed_measure_y[wo_idx + k] = (int)ceil((currVal - _xmin) / _res);
        }
        else {
            // processed_measure_x[wo_idx + (0 * _lidar_coords_LEN) + k] = ceil((_ymax - currVal) / _res);
            processed_measure_x[wo_idx + k] = (int)ceil((_ymax - currVal) / _res);
        }
    }
}


__global__ void kernel_matrix_mul_3x3(const float* A, const float* B, float* C) {

    for (int i = 0; i < 3; i++) {

        for (int j = 0; j < 3; j++) {

            float currVal = 0;
            for (int k = 0; k < 3; k++) {
                currVal += A[(i * 3) + k] * B[k * 3 + j];
            }
            C[(i * 3) + j] = currVal;
        }
    }
}

__global__ void kernel_2d_map_counter(uint8_t* map_2d, int* unique_counter, int* unique_counter_col, const int _GRID_WIDHT, const int _GRID_HEIGHT) {

    int i = blockIdx.x;
    int k = threadIdx.x;

    int start_idx = k * _GRID_HEIGHT;
    int end_idx = (k + 1) * _GRID_HEIGHT;

    for (int j = start_idx; j < end_idx; j++) {

        if (map_2d[j] != 0) {
            atomicAdd(&unique_counter[i], 1);
            atomicAdd(&unique_counter_col[k + 1], 1);
        }
    }
}

__global__ void kernel_create_2d_map(const int* particles_x, const int* particles_y, const int* particles_idx, const int IDX_LEN, uint8_t* map_2d,
    const int _GRID_WIDTH, const int _GRID_HEIGHT, const int _NUM_ELEMS) {

    int i = blockIdx.x;
    int k = threadIdx.x;

    if (i < _NUM_ELEMS) {

        int first_idx = particles_idx[i];
        int last_idx = (i < _NUM_ELEMS - 1) ? particles_idx[i + 1] : IDX_LEN;
        int arr_len = last_idx - first_idx;

        int arr_end = last_idx;

        int start_idx = ((arr_len / blockDim.x) * k) + first_idx;
        int end_idx = ((arr_len / blockDim.x) * (k + 1)) + first_idx;
        end_idx = (k < blockDim.x - 1) ? end_idx : arr_end;

        int start_of_current_map = i * _GRID_WIDTH * _GRID_HEIGHT;

        for (int j = start_idx; j < end_idx; j++) {

            int x = particles_x[j];
            int y = particles_y[j];

            int curr_idx = start_of_current_map + (x * _GRID_HEIGHT) + y;

            if (x >= 0 && y >= 0 && map_2d[curr_idx] == 0) {
                map_2d[curr_idx] = 1;
            }
        }
    }
}

__global__ void kernel_create_2d_map(const int* particles_x, const int* particles_y, const int* particles_idx, const int IDX_LEN, uint8_t* map_2d,
    int* unique_in_particle, int* unique_in_particle_col, const int _GRID_WIDTH, const int _GRID_HEIGHT, const int _NUM_ELEMS) {

    int i = blockIdx.x;
    int k = threadIdx.x;

    if (i < _NUM_ELEMS) {

        int first_idx = particles_idx[i];
        int last_idx = (i < _NUM_ELEMS - 1) ? particles_idx[i + 1] : IDX_LEN;
        int arr_len = last_idx - first_idx;

        int arr_end = last_idx;

        int start_idx = ((arr_len / blockDim.x) * k) + first_idx;
        int end_idx = ((arr_len / blockDim.x) * (k + 1)) + first_idx;
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

    if (i < _NUM_ELEMS) {

        int start_idx = measure_idx[i];
        int end_idx = (i < _NUM_ELEMS - 1) ? measure_idx[i + 1] : IDX_LEN;

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

__global__ void kernel_update_unique_restructure(uint8_t* map_2d, int* particles_x, int* particles_y, int* unique_in_particle, int* unique_in_particle_col,
    const int _GRID_WIDTH, const int _GRID_HEIGHT) {

    int i = threadIdx.x;

    int start_idx = (i * _GRID_HEIGHT);
    int end_idx = ((i + 1) * _GRID_HEIGHT);
    int key = unique_in_particle_col[i];
    int first_key = key;

    for (int j = start_idx; j < end_idx; j++) {

        if (map_2d[j] == 1) {

            int y = j % _GRID_HEIGHT;
            int x = j / _GRID_HEIGHT;

            particles_x[key] = x;
            particles_y[key] = y;
            key += 1;
        }
    }
}

__global__ void kernel_update_unique_restructure(uint8_t* map_2d, int* particles_x, int* particles_y, int* particles_idx, int* unique_in_particle, int* unique_in_particle_col,
    const int _GRID_WIDTH, const int _GRID_HEIGHT) {

    int i = blockIdx.x;
    int l = threadIdx.x;

    int start_of_current_map = i * _GRID_WIDTH * _GRID_HEIGHT;
    int start_idx = (i * _GRID_WIDTH * _GRID_HEIGHT) + (l * _GRID_HEIGHT);
    int end_idx = (i * _GRID_WIDTH * _GRID_HEIGHT) + ((l + 1) * _GRID_HEIGHT);
    int key = unique_in_particle_col[i * _GRID_WIDTH + l] + unique_in_particle[i];

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

__global__ void kernel_rearrange_particles(int* particles_x, int* particles_y, const int* particles_idx,
    const int* c_particles_x, const int* c_particles_y, const int* c_particles_idx, const int* js,
    const int _GRID_WIDTH, const int _GRID_HEIGHT, const int _NUM_ELEMS, const int IDX_LEN, const int C_IDX_LEN) {

    int i = blockIdx.x;
    int k = threadIdx.x;
    int m = js[i];

    if (i < _NUM_ELEMS) {

        int first_idx = particles_idx[i];
        int last_idx = (i < _NUM_ELEMS - 1) ? particles_idx[i + 1] : IDX_LEN;
        int arr_end = last_idx;
        int arr_len = last_idx - first_idx;

        int c_first_idx = c_particles_idx[m];
        int c_last_idx = (m < _NUM_ELEMS - 1) ? c_particles_idx[m + 1] : C_IDX_LEN;
        int c_arr_end = c_last_idx;
        int c_arr_len = c_last_idx - c_first_idx;

        //if (arr_len != c_arr_len)
        //    printf("%d <> %d | i=%d, k=%d, m=%d\n", arr_len, c_arr_len, i, k, m);

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

__global__ void kernel_rearrange_states(float* states_x, float* states_y, float* states_theta,
    float* c_states_x, float* c_states_y, float* c_states_theta, int* js) {

    int i = threadIdx.x;
    int j = js[i];

    states_x[i] = c_states_x[j];
    states_y[i] = c_states_y[j];
    states_theta[i] = c_states_theta[j];
}

__global__ void kernel_position_to_image(int* position_image_body, float* transition_world_lidar, float _res, int _xmin, int _ymax) {

    float a = transition_world_lidar[2];
    float b = transition_world_lidar[5];

    position_image_body[0] = (int)ceil((_ymax - b) / _res);
    position_image_body[1] = (int)ceil((a - _xmin) / _res);
}

__global__ void kernel_rearrange_indecies(int* particles_idx, int* c_particles_idx, int* js, int* last_len, const int ARR_LEN) {

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
        particles_idx[i + 1] = idx_value;
    else
        last_len[0] = idx_value;
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

__global__ void kernel_arr_mult(float* arr, const float mult_value) {

    int i = threadIdx.x;
    arr[i] = arr[i] * mult_value;
}

__global__ void kernel_arr_mult(float* arr, float* mult_arr) {

    int i = threadIdx.x;
    arr[i] = arr[i] * mult_arr[i];
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

__global__ void kernel_update_unique_sum_col(int* unique_in_particle_col, const int _GRID_WIDTH) {

    int i = threadIdx.x;

    for (int j = (i * _GRID_WIDTH) + 1; j < (i + 1) * _GRID_WIDTH; j++)
        unique_in_particle_col[j] = unique_in_particle_col[j] + unique_in_particle_col[j - 1];
}