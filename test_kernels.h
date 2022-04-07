#ifndef _TEST_KERNELS_H_
#define _TEST_KERNELS_H_

//#define TEST_MAP
//#define TEST_ROBOT
//#define TEST_ROBOT_ADVANCE
#define TEST_RUN

#if defined(TEST_ROBOT_ADVANCE)
#include "test_robot_advance.h"
#elif defined(TEST_MAP)
#include "test_map.h"
#elif defined(TEST_ROBOT)
#include "test_robot.h"
#elif defined(TEST_RUN)
#include "test_run.h"
#endif


void test_main() {

#if defined(TEST_ROBOT_ADVANCE)
	test_robot_advance_main();
#elif defined(TEST_MAP)
	test_map_main();
#elif defined(TEST_ROBOT)
	test_robot_particles_partials_main();
#elif defined(TEST_RUN)
	test_robot_advance_main();
	test_robot_particles_main();
	test_map_func();
#endif

}


#endif
