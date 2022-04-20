#ifndef _TEST_KERNELS_H_
#define _TEST_KERNELS_H_

//#define TEST_MAP
//#define TEST_MAP_EXTEND
//#define TEST_ROBOT
//#define TEST_ROBOT_EXTEND
//#define TEST_ROBOT_MOVE
//#define TEST_ROBOT_MOVE_EXTEND
//#define TEST_ITERATION_SINGLE
#define TEST_ITERATION_MULTI


#if defined(TEST_MAP)
#include "test_map.h"
#elif defined(TEST_MAP_EXTEND)
#include "test_map_extend.h"
#elif defined(TEST_ROBOT)
#include "test_robot.h"
#elif defined(TEST_ROBOT_EXTEND)
#include "test_robot_extend.h"
#elif defined(TEST_ROBOT_MOVE)
#include "test_robot_move.h"
#elif defined(TEST_ROBOT_MOVE_EXTEND)
#include "test_robot_move_extend.h"
#elif defined(TEST_ITERATION_SINGLE)
#include "test_iteration.h"
#elif defined(TEST_ITERATION_MULTI)
#include "test_iteration_multi.h"
#endif


void test_main() {


#if defined(TEST_MAP)
	test_map_main();
#elif defined(TEST_MAP_EXTEND)
	test_map_extend();
#elif defined(TEST_ROBOT)
	test_robot_particles_partials_main();
#elif defined(TEST_ROBOT_EXTEND)
	test_robot_extend();
#elif defined(TEST_ROBOT_MOVE)
	test_robot_move();
#elif defined(TEST_ROBOT_MOVE_EXTEND)
	test_robot_move();
#elif defined(TEST_ITERATION_SINGLE)
	test_iteration_single();
#elif defined(TEST_ITERATION_MULTI)
	test_iterations();
#endif

}


#endif
