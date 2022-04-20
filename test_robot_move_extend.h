#ifndef _TEST_ROBOT_MOVE_EXTEND_H_
#define _TEST_ROBOT_MOVE_EXTEND_H_

#include "headers.h"
#include "host_asserts.h"
#include "host_utils.h"
#include "kernels.cuh"
#include "kernels_robot.cuh"
#include "kernels_map.cuh"
#include "kernels_utils.cuh"

#include "data/robot_advance/300.h"
#include "data/robot_iteration/300.h"

/************************* STATES VARIABLES *************************/
size_t  sz_states_pos = 0;

float* d_states_x;
float* d_states_y;
float* d_states_theta;

float* res_states_x;
float* res_states_y;
float* res_states_theta;

float* d_rnds_encoder_counts;
float* d_rnds_yaws;

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

int NEW_GRID_WIDTH = 0;
int NEW_GRID_HEIGHT = 0;
int NEW_LIDAR_COORDS_LEN = 0;


float encoder_counts = 0;
float yaw = 0;
float dt = 0;

vector<int> vec_grid_map;
vector<float> vec_log_odds;
vector<float> vec_lidar_coords;

vector<float> vec_robot_transition_world_body;
vector<float> vec_robot_state;
vector<float> vec_particles_weight_post;
vector<float> vec_rnds;

vector<float> vec_rnds_encoder_counts;
vector<float> vec_rnds_yaws;
vector<float> vec_states_x;
vector<float> vec_states_y;
vector<float> vec_states_theta;

//int EXTRA_GRID_WIDTH = 0;
//int EXTRA_GRID_HEIGHT = 0;
//int EXTRA_PARTICLES_ITEMS_LEN = 0;

vector<int> extra_grid_map;
vector<int> extra_particles_x;
vector<int> extra_particles_y;
vector<int> extra_particles_idx;
vector<float> extra_states_x;
vector<float> extra_states_y;
vector<float> extra_states_theta;
vector<float> extra_new_weights;
vector<float> extra_particles_weight_pre;

//int extra_xmin = 0;
//int extra_xmax = 0;
//int extra_ymin = 0;
//int extra_ymax = 0;
//float extra_res = 0;
//float extra_log_t = 0;

//vector<float> extra_log_odds;
//vector<float> extra_transition_single_world_body;


void read_robot_move_data(int file_number, bool check_rnds_encoder_counts = false, bool check_rnds_yaws = false,
	bool check_states = false) {

	string file_name = std::to_string(file_number);

	const int SCALAR_VALUES = 1;
	const int RNDS_ENCODER_COUNTS_VALUES = 2;
	const int RNDS_YAWS_VALUES = 3;
	const int STATES_X_VALUES = 4;
	const int STATES_Y_VALUES = 5;
	const int STATES_THETA_VALUES = 6;
	const int SEPARATE_VALUES = 10;

	int curr_state = SCALAR_VALUES;
	string str_rnds_encoder_counts = "";
	string str_rnds_yaws = "";
	string str_states_x = "";
	string str_states_y = "";
	string str_states_theta = "";
	string segment;

	std::ifstream data("data/steps/robot_advance_" + file_name + ".txt");
	string line;

	while (getline(data, line)) {

		line = trim(line);

		if (curr_state == SCALAR_VALUES) {

			if (line == "encoder_counts") {
				getline(data, line);
				encoder_counts = std::stof(line);
			}
			else if (line == "yaw") {
				getline(data, line);
				yaw = std::stof(line);
			}
			else if (line == "dt") {
				getline(data, line);
				dt = std::stof(line);
			}
		}

		if (line == "rnds_encoder_counts") {
			curr_state = RNDS_ENCODER_COUNTS_VALUES;
			continue;
		}
		else if (line == "rnds_yaws") {
			curr_state = RNDS_YAWS_VALUES;
			continue;
		}
		else if (line == "states_x") {
			curr_state = STATES_X_VALUES;
			continue;
		}
		else if (line == "states_y") {
			curr_state = STATES_Y_VALUES;
			continue;
		}
		else if (line == "states_theta") {
			curr_state = STATES_THETA_VALUES;
			continue;
		}

		if (curr_state == RNDS_ENCODER_COUNTS_VALUES) {
			if (line == "SEPARATE") {
				curr_state = SEPARATE_VALUES;
			}
			else {
				str_rnds_encoder_counts += line;
			}
		}
		else if (curr_state == RNDS_YAWS_VALUES) {
			if (line == "SEPARATE") {
				curr_state = SEPARATE_VALUES;
			}
			else {
				str_rnds_yaws += line;
			}
		}
		else if (curr_state == STATES_X_VALUES) {
			if (line == "SEPARATE") {
				curr_state = SEPARATE_VALUES;
			}
			else {
				str_states_x += line;
			}
		}
		else if (curr_state == STATES_Y_VALUES) {
			if (line == "SEPARATE") {
				curr_state = SEPARATE_VALUES;
			}
			else {
				str_states_y += line;
			}
		}
		else if (curr_state == STATES_THETA_VALUES) {
			if (line == "SEPARATE") {
				curr_state = SEPARATE_VALUES;
			}
			else {
				str_states_theta += line;
			}
		}
	}

	stringstream stream_rnds_encoder_counts(str_rnds_encoder_counts);
	vec_rnds_encoder_counts.resize(NUM_PARTICLES);
	for (int i = 0; std::getline(stream_rnds_encoder_counts, segment, ','); i++) {
		vec_rnds_encoder_counts[i] = std::stof(segment);
	}
	stringstream stream_rnds_yaw(str_rnds_yaws);
	vec_rnds_yaws.resize(NUM_PARTICLES);
	for (int i = 0; std::getline(stream_rnds_yaw, segment, ','); i++) {
		vec_rnds_yaws[i] = std::stof(segment);
	}
	stringstream stream_states_x(str_states_x);
	vec_states_x.resize(NUM_PARTICLES);
	for (int i = 0; std::getline(stream_states_x, segment, ','); i++) {
		vec_states_x[i] = std::stof(segment);
	}
	stringstream stream_states_y(str_states_y);
	vec_states_y.resize(NUM_PARTICLES);
	for (int i = 0; std::getline(stream_states_y, segment, ','); i++) {
		vec_states_y[i] = std::stof(segment);
	}
	stringstream stream_states_theta(str_states_theta);
	vec_states_theta.resize(NUM_PARTICLES);
	for (int i = 0; std::getline(stream_states_theta, segment, ','); i++) {
		vec_states_theta[i] = std::stof(segment);
	}

	int num_equals = 0;

	if (check_rnds_encoder_counts == true) {
		num_equals = 0;
		for (int i = 0; i < vec_rnds_encoder_counts.size(); i++) {
			if (vec_rnds_encoder_counts[i] != h_rnds_encoder_counts[i])
				printf("%f <> %f\n", vec_rnds_encoder_counts[i], h_rnds_encoder_counts[i]);
			else
				num_equals += 1;
		}
		printf("Rnds Encoder Counts Num Equals=%d, Num Errors=%d\n\n", num_equals, int(vec_rnds_encoder_counts.size() - num_equals));
	}
	if (check_rnds_yaws == true) {
		num_equals = 0;
		for (int i = 0; i < vec_rnds_yaws.size(); i++) {
			if (vec_rnds_yaws[i] != h_rnds_yaws[i])
				printf("%f <> %f\n", vec_rnds_yaws[i], h_rnds_yaws[i]);
			else
				num_equals += 1;
		}
		printf("Rnds Yaws Num Equals=%d, Num Errors=%d\n\n", num_equals, int(vec_rnds_yaws.size() - num_equals));
	}
	if (check_states == true) {
		num_equals = 0;
		for (int i = 0; i < vec_states_x.size(); i++) {
			if (vec_states_x[i] != h_states_x[i])
				printf("%f <> %f\n", vec_states_x[i], h_states_x[i]);
			else
				num_equals += 1;
		}
		printf("States X Num Equals=%d, Num Errors=%d\n\n", num_equals, int(vec_states_x.size() - num_equals));

		num_equals = 0;
		for (int i = 0; i < vec_states_y.size(); i++) {
			if (vec_states_y[i] != h_states_y[i])
				printf("%f <> %f\n", vec_states_y[i], h_states_y[i]);
			else
				num_equals += 1;
		}
		printf("States Y Num Equals=%d, Num Errors=%d\n\n", num_equals, int(vec_states_y.size() - num_equals));

		num_equals = 0;
		for (int i = 0; i < vec_states_theta.size(); i++) {
			if (vec_states_theta[i] != h_states_theta[i])
				printf("%f <> %f\n", vec_states_theta[i], h_states_theta[i]);
			else
				num_equals += 1;
		}
		printf("States Theta Num Equals=%d, Num Errors=%d\n\n", num_equals, int(vec_states_theta.size() - num_equals));
	}
}

void read_robot_extra(int file_number, bool check_states = false) {

    string file_name = std::to_string(file_number);

    const int SCALAR_VALUES = 1;
    const int STATES_X_VALUES = 6;
    const int STATES_Y_VALUES = 7;
    const int STATES_THETA_VALUES = 8;
    const int SEPARATE_VALUES = 11;

    int curr_state = SCALAR_VALUES;
    string str_states_x = "";
    string str_states_y = "";
    string str_states_theta = "";
    string segment;

    std::ifstream data("data/extra/robot_" + file_name + ".txt");
    string line;

    while (getline(data, line)) {

        line = trim(line);
        if (line == "") continue;


        if (line == "states_x") {
            curr_state = STATES_X_VALUES;
            continue;
        }
        else if (line == "states_y") {
            curr_state = STATES_Y_VALUES;
            continue;
        }
        else if (line == "states_theta") {
            curr_state = STATES_THETA_VALUES;
            continue;
        }


		if (curr_state == STATES_X_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_states_x += line;
            }
        }
        else if (curr_state == STATES_Y_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_states_y += line;
            }
        }
        else if (curr_state == STATES_THETA_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_states_theta += line;
            }
        }
    }


    stringstream stream_states_x(str_states_x);
    extra_states_x.resize(NUM_PARTICLES);
    for (int i = 0; std::getline(stream_states_x, segment, ','); i++) {
        extra_states_x[i] = std::stof(segment);
    }
    stringstream stream_states_y(str_states_y);
    extra_states_y.resize(NUM_PARTICLES);
    for (int i = 0; std::getline(stream_states_y, segment, ','); i++) {
        extra_states_y[i] = std::stof(segment);
    }
    stringstream stream_states_theta(str_states_theta);
    extra_states_theta.resize(NUM_PARTICLES);
    for (int i = 0; std::getline(stream_states_theta, segment, ','); i++) {
        extra_states_theta[i] = std::stof(segment);
    }

    int num_equals = 0;

    if (check_states == true) {
        num_equals = 0;
        for (int i = 0; i < extra_states_x.size(); i++) {
            if (extra_states_x[i] != post_states_x[i])
                printf("%f <> %f\n", extra_states_x[i], post_states_x[i]);
            else
                num_equals += 1;
        }
        printf("States X Num Equals=%d, Num Errors=%d\n\n", num_equals, int(extra_states_x.size() - num_equals));

        num_equals = 0;
        for (int i = 0; i < extra_states_y.size(); i++) {
            if (extra_states_y[i] != post_states_y[i])
                printf("%f <> %f\n", extra_states_y[i], post_states_y[i]);
            else
                num_equals += 1;
        }
        printf("States Y Num Equals=%d, Num Errors=%d\n\n", num_equals, int(extra_states_y.size() - num_equals));

        num_equals = 0;
        for (int i = 0; i < extra_states_theta.size(); i++) {
            if (extra_states_theta[i] != post_states_theta[i])
                printf("%f <> %f\n", extra_states_theta[i], post_states_theta[i]);
            else
                num_equals += 1;
        }
        printf("States Theta Num Equals=%d, Num Errors=%d\n\n", num_equals, int(extra_states_theta.size() - num_equals));
    }
}



void alloc_init_state_vars(float* h_states_x, float* h_states_y, float* h_states_theta) {

	sz_states_pos = NUM_PARTICLES * sizeof(float);

	gpuErrchk(cudaMalloc((void**)&d_states_x, sz_states_pos));
	gpuErrchk(cudaMalloc((void**)&d_states_y, sz_states_pos));
	gpuErrchk(cudaMalloc((void**)&d_states_theta, sz_states_pos));

	gpuErrchk(cudaMemcpy(d_states_x, h_states_x, sz_states_pos, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_states_y, h_states_y, sz_states_pos, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_states_theta, h_states_theta, sz_states_pos, cudaMemcpyHostToDevice));
}

void alloc_init_movement_vars(float* h_rnds_encoder_counts, float* h_rnds_yaws) {

	gpuErrchk(cudaMalloc((void**)&d_rnds_encoder_counts, sz_states_pos));
	gpuErrchk(cudaMalloc((void**)&d_rnds_yaws, sz_states_pos));

	gpuErrchk(cudaMemcpy(d_rnds_encoder_counts, h_rnds_encoder_counts, sz_states_pos, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_rnds_yaws, h_rnds_yaws, sz_states_pos, cudaMemcpyHostToDevice));
}

void exec_robot_advance(float encoder_counts, float yaw, float dt) {

	int threadsPerBlock = NUM_PARTICLES;
	int blocksPerGrid = 1;
	kernel_robot_advance << <blocksPerGrid, threadsPerBlock >> > (d_states_x, d_states_y, d_states_theta, SEP,
		d_rnds_encoder_counts, d_rnds_yaws,
		encoder_counts, yaw, dt, ST_nv, ST_nw, NUM_PARTICLES);
	cudaDeviceSynchronize();
}

void assertRobotAdvanceResults(float* post_states_x, float* post_states_y, float* post_states_theta) {

	res_states_x = (float*)malloc(sz_states_pos);
	res_states_y = (float*)malloc(sz_states_pos);
	res_states_theta = (float*)malloc(sz_states_pos);

	gpuErrchk(cudaMemcpy(res_states_x, d_states_x, sz_states_pos, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(res_states_y, d_states_y, sz_states_pos, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(res_states_theta, d_states_theta, sz_states_pos, cudaMemcpyDeviceToHost));

	for (int i = 0; i < NUM_PARTICLES; i++) {

		if (abs(res_states_x[i] - post_states_x[i]) > 1e-4)
			printf("i=%d, x=%f, %f\n", i, res_states_x[i], post_states_x[i]);
		if (abs(res_states_y[i] - post_states_y[i]) > 1e-4)
			printf("i=%d, y=%f, %f\n", i, res_states_y[i], post_states_y[i]);
		if (abs(res_states_theta[i] - post_states_theta[i]) > 1e-4)
			printf("i=%d, theta=%f, %f\n", i, res_states_theta[i], post_states_theta[i]);
	}
}

void test_robot_move() {

	std::cout << "Start Robot Move" << std::endl;

	const int ST_FILE_NUMBER = 300;

	read_robot_move_data(ST_FILE_NUMBER, true, true, true);
	read_robot_extra(ST_FILE_NUMBER, true);

	for (int file_number = ST_FILE_NUMBER; file_number < ST_FILE_NUMBER + 20; file_number++) {

		read_robot_move_data(file_number);
		read_robot_extra(file_number);

		alloc_init_state_vars(vec_states_x.data(), vec_states_y.data(), vec_states_theta.data());
		alloc_init_movement_vars(vec_rnds_encoder_counts.data(), vec_rnds_yaws.data());

		auto start_robot_advance_kernel = std::chrono::high_resolution_clock::now();
		exec_robot_advance(encoder_counts, yaw, dt);
		auto stop_robot_advance_kernel = std::chrono::high_resolution_clock::now();

		assertRobotAdvanceResults(extra_states_x.data(), extra_states_y.data(), extra_states_theta.data());

		auto duration_robot_advance_total = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_advance_kernel - start_robot_advance_kernel);
		std::cout << std::endl;
		std::cout << "Time taken by function (Robot Advance Kernel): " << duration_robot_advance_total.count() << " microseconds" << std::endl;
	}

}




#endif
