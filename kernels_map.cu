
#include "kernels_map.cuh"

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

__global__ void kernel_2d_map_counter(uint8_t* map_2d, int* unique_counter, int* unique_counter_col, const int GRID_WIDHT, const int GRID_HEIGHT) {

    int i = blockIdx.x;
    int k = threadIdx.x;

    int start_idx = k * GRID_HEIGHT;
    int end_idx = (k + 1) * GRID_HEIGHT;

    for (int j = start_idx; j < end_idx; j++) {

        if (map_2d[j] != 0) {
            atomicAdd(&unique_counter[i], 1);
            atomicAdd(&unique_counter_col[k + 1], 1);
        }
    }
}

__global__ void kernel_update_log_odds(float* log_odds, int* f_x, int* f_y, const float _log_t,
    const int GRID_WIDTH, const int GRID_HEIGHT, const int NUM_ELEMS) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < NUM_ELEMS) {

        int x = f_x[i];
        int y = f_y[i];

        if (x >= 0 && y >= 0 && x < GRID_WIDTH && y < GRID_HEIGHT) {

            int grid_map_idx = x * GRID_HEIGHT + y;

            log_odds[grid_map_idx] = log_odds[grid_map_idx] + _log_t;
        }
    }
}

__global__ void kernel_position_to_image(int* position_image_body,
    const float transition_world_lidar_x, const float transition_world_lidar_y,
    const float res, const int xmin, const int ymax) {

    position_image_body[0] = (int)ceil((ymax - transition_world_lidar_y) / res);
    position_image_body[1] = (int)ceil((transition_world_lidar_x - xmin) / res);
}

__global__ void kernel_position_to_image(int* position_image_body,
    const float transition_world_lidar_x, const float transition_world_lidar_y,
    const float res, const float xmin, const float ymax) {

    position_image_body[0] = (int)ceil((ymax - transition_world_lidar_y) / res);
    position_image_body[1] = (int)ceil((transition_world_lidar_x - xmin) / res);
}

__global__ void kernel_position_to_image(int* position_image_body, float* transition_world_lidar, float res, int xmin, int ymax) {

    float a = transition_world_lidar[2];
    float b = transition_world_lidar[5];

    position_image_body[0] = (int)ceil((ymax - b) / res);
    position_image_body[1] = (int)ceil((a - xmin) / res);
}

__global__ void kernel_update_map(int* grid_map, const float* log_odds, const float _LOG_ODD_PRIOR, const int _WALL, const int _FREE, const int NUM_ELEMS) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < NUM_ELEMS) {

        if (log_odds[i] > 0)
            grid_map[i] = _WALL;

        if (log_odds[i] < _LOG_ODD_PRIOR)
            grid_map[i] = _FREE;
    }
}

__global__ void kernel_update_particles_lidar(float* transition_world_lidar, int* processed_measure_x, int* processed_measure_y,
    float* particles_world_frame_x, float* particles_world_frame_y, const float* _lidar_coords, float _res, int _xmin, int _ymax, const int LIDAR_COORDS_LEN) {

    int k = blockIdx.x;

    for (int j = 0; j < 2; j++) {

        double currVal = 0;
        currVal += transition_world_lidar[j * 3 + 0] * _lidar_coords[(0 * LIDAR_COORDS_LEN) + k];
        currVal += transition_world_lidar[j * 3 + 1] * _lidar_coords[(1 * LIDAR_COORDS_LEN) + k];
        currVal += transition_world_lidar[j * 3 + 2];

        if (j == 0) {
            particles_world_frame_x[k] = currVal;
            processed_measure_y[k] = (int)ceil((currVal - _xmin) / _res);
        }
        else {
            particles_world_frame_y[k] = currVal;
            processed_measure_x[k] = (int)ceil((_ymax - currVal) / _res);
        }
    }
}

__global__ void kernel_create_2d_map(const int* particles_x, const int* particles_y, const int* particles_idx, const int IDX_LEN, uint8_t* map_2d,
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

        for (int j = start_idx; j < end_idx; j++) {

            int x = particles_x[j];
            int y = particles_y[j];

            int curr_idx = start_of_current_map + (x * GRID_HEIGHT) + y;

            if (x >= 0 && y >= 0 && map_2d[curr_idx] == 0) {
                map_2d[curr_idx] = 1;
            }
        }
    }
}

__global__ void kernel_update_unique_restructure(uint8_t* map_2d, int* particles_x, int* particles_y, int* unique_in_particle_col,
    const int GRID_WIDTH, const int GRID_HEIGHT) {

    int i = threadIdx.x;

    int start_idx = (i * GRID_HEIGHT);
    int end_idx = ((i + 1) * GRID_HEIGHT);
    int key = unique_in_particle_col[i];
    int first_key = key;

    for (int j = start_idx; j < end_idx; j++) {

        if (map_2d[j] == 1) {

            int y = j % GRID_HEIGHT;
            int x = j / GRID_HEIGHT;

            particles_x[key] = x;
            particles_y[key] = y;
            key += 1;
        }
    }
}


