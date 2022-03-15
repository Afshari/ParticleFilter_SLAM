#ifndef _TEST_KERNELS_H_
#define _TEST_KERNELS_H_

#define TEST_MAP
//#define TEST_MAP_PARTIALS
//#define TEST_ROBOT_PARTICLES
//#define TEST_ROBOT_PARTICLES_PARTIALS

#if defined(TEST_MAP)
#include "test_map.h"
#elif defined(TEST_MAP_PARTIALS)
#include "test_map_partials.h"
#elif defined(TEST_ROBOT_PARTICLES)
#include "test_robot_particles.h"
#elif defined(TEST_ROBOT_PARTICLES_PARTIALS)
#include "test_robot_particles_partials.h"
#endif


void test_main() {

#if defined(TEST_MAP)
	test_map_main();
#elif defined(TEST_MAP_PARTIALS)
	test_map_partials_main();
#elif defined(TEST_ROBOT_PARTICLES)
	test_robot_particles_main();
#elif defined(TEST_ROBOT_PARTICLES_PARTIALS)
	test_robot_particles_partials_main();
#endif

}


#endif
