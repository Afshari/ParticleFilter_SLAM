#ifndef _TEST_KERNELS_H_
#define _TEST_KERNELS_H_

//#define TEST_ROBOT_ADVANCE
//#define TEST_MAP_EXTEND
#define TEST_MAP_PARTIALS
//#define TEST_ROBOT_PARTICLES


#if defined(TEST_ROBOT_ADVANCE)
#include "test_robot_advance.h"
#elif defined(TEST_MAP_EXTEND)
#include "test_map_extend.h"
#elif defined(TEST_MAP_PARTIALS)
#include "test_map_partials.h"
#elif defined(TEST_ROBOT_PARTICLES)
#include "test_robot_particles.h"
#endif


void test_main() {

#if defined(TEST_ROBOT_ADVANCE)
	test_robot_advance_main();
#elif defined(TEST_MAP_EXTEND)
	test_map_extend_main();
#elif defined(TEST_MAP_PARTIALS)
	test_map_partials_main();
#elif defined(TEST_ROBOT_PARTICLES)
	test_robot_particles_partials_main();
#endif

}


#endif
