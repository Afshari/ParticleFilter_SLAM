#ifndef _TEST_ROBOT_MOVE_H_
#define _TEST_ROBOT_MOVE_H_

#include "headers.h"
#include "host_asserts.h"
#include "host_utils.h"
#include "kernels.cuh"
#include "device_init_robot.h"
#include "device_exec_robot.h"
#include "device_assert_robot.h"


void test_robot_move() {

	std::cout << "Start Robot Move" << std::endl;

	HostState pre_state; 
	HostState h_state;
	HostState post_state;
	HostRobotState h_robot_state;

	DeviceState d_state;
	DeviceState d_clone_state;

	read_robot_move(100, pre_state, post_state);

	alloc_init_state_vars(d_state, d_clone_state, h_state, h_robot_state, pre_state);

	auto start_robot_move_kernel = std::chrono::high_resolution_clock::now();
	exec_robot_move(d_state, h_state);
	auto stop_robot_move_kernel = std::chrono::high_resolution_clock::now();

	assert_robot_move_results(d_state, h_state, post_state);

	auto duration_robot_move_total = std::chrono::duration_cast<std::chrono::microseconds>(stop_robot_move_kernel - start_robot_move_kernel);
	std::cout << std::endl;
	std::cout << "Time taken by function (Robot Move Kernel): " << duration_robot_move_total.count() << " microseconds" << std::endl;
}

#endif
