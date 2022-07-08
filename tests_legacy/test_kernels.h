#ifndef _TEST_KERNELS_H_
#define _TEST_KERNELS_H_


#define TEST_FUNCTIONS

// TEST_ITERATIONS is Obsolete, it needs to change some of the functions
// #define TEST_ITERATIONS


#if defined(TEST_FUNCTIONS)
#include "test_map.h"
#include "test_map_iter.h"

#include "test_robot.h"
#include "test_robot_iter.h"
#include "test_robot_iter_simple.h"

#include "test_robot_move.h"
#include "test_robot_move_iter.h"
#endif

#if defined(TEST_ITERATIONS)
#include "test_iteration.h"
#include "test_iteration_multi.h"
#endif

void test_main() {

	if (fs::is_directory("data_test") == false) {

		std::cerr << "[Error] The 'data_test' directory is not Exists" << std::endl;
		std::cerr << "[Error] Make sure first to prepare the data test" << std::endl;
		exit(-1);
	}

#if defined(TEST_FUNCTIONS)
	test_map_main();
	test_map_iter();
	
	test_robot();
	test_robot_iter();

	//// test_robot_iter_simple();
	
	test_robot_move();
	test_robot_move_iter();
#endif

#if defined(TEST_ITERATIONS)
	test_iteration_single();
	test_iterations();
#endif

}


#endif
