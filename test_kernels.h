#ifndef _TEST_KERNELS_H_
#define _TEST_KERNELS_H_

//#define TEST_MAP
//#define TEST_ROBOT_ADVANCE
#define TEST_ROBOT_PARTICLES
//#define TEST_RUN

#if defined(TEST_ROBOT_ADVANCE)
#include "test_robot_advance.h"
#elif defined(TEST_MAP)
#include "test_map.h"
#elif defined(TEST_ROBOT_PARTICLES)
#include "test_robot_particles.h"
#elif defined(TEST_RUN)
#include "test_run.h"
#endif


void test_main() {

#if defined(TEST_ROBOT_ADVANCE)
	test_robot_advance_main();
#elif defined(TEST_MAP)
	test_map_main();
#elif defined(TEST_ROBOT_PARTICLES)
	test_robot_particles_partials_main();
#elif defined(TEST_RUN)
	test_robot_particles_main();
#endif

}


#endif
