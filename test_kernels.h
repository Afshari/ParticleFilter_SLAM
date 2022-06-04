#ifndef _TEST_KERNELS_H_
#define _TEST_KERNELS_H_

#define TEST_MAP
//#define TEST_ROBOT
//#define TEST_ROBOT_MOVE

//#define TEST_ITERATION_SINGLE
//#define TEST_ITERATION_MULTI


#if defined(TEST_MAP)
#include "test_map.h"
#include "test_map_extend.h"
#elif defined(TEST_ROBOT)
#include "test_robot.h"
#include "test_robot_extend.h"
#elif defined(TEST_ROBOT_MOVE)
#include "test_robot_move.h"
#include "test_robot_move_extend.h"
#elif defined(TEST_ITERATION_SINGLE)
#include "test_iteration.h"
#elif defined(TEST_ITERATION_MULTI)
#include "test_iteration_multi.h"
#endif


void test_main() {

#if defined(TEST_MAP)
	test_map_main();
	test_map_extend();
#elif defined(TEST_ROBOT)
	test_robot();
	//test_robot_extend();
#elif defined(TEST_ROBOT_MOVE)
	test_robot_move();
	test_robot_move_extend();
#elif defined(TEST_ITERATION_SINGLE)
	test_iteration_single();
#elif defined(TEST_ITERATION_MULTI)
	test_iterations();
#endif

}


#endif
