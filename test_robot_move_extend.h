#ifndef _TEST_ROBOT_MOVE_EXTEND_H_
#define _TEST_ROBOT_MOVE_EXTEND_H_

#include "headers.h"
#include "host_asserts.h"
#include "host_utils.h"
#include "kernels.cuh"
#include "device_init_robot.h"
#include "device_exec_robot.h"
#include "device_assert_robot.h"


void test_robot_move_extend() {

	std::cout << "Start Robot Move" << std::endl;

	const int ST_FILE_NUMBER = 100;

	for (int file_number = ST_FILE_NUMBER; file_number < ST_FILE_NUMBER + 500; file_number += 100) {

		HostState h_state;
		HostState res_state;
		HostState post_state;

		DeviceState d_state;

		read_robot_move(file_number, h_state, post_state);

		alloc_init_state_vars(d_state, res_state, h_state);

		auto start_robot_advance_kernel = std::chrono::high_resolution_clock::now();
		exec_robot_advance(d_state, res_state);
		auto stop_robot_advance_kernel = std::chrono::high_resolution_clock::now();

		assert_robot_advance_results(d_state, res_state, post_state);

		auto duration_robot_advance_total = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_advance_kernel - start_robot_advance_kernel);
		std::cout << std::endl;
		std::cout << "Time taken by function (Robot Advance Kernel): " << duration_robot_advance_total.count() << " microseconds" << std::endl;
	}
}

#endif
