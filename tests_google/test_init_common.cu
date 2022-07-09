
#include "gtest/gtest.h"
#include "../device_init_common.h"

TEST(DeviceInitCommon, AllocInitMeasurementVars) {

	DeviceMeasurements d_measurements;
	HostMeasurements h_measurements;
	HostMeasurements pre_measurements;
	pre_measurements.LEN = 100;

	alloc_init_measurement_vars(d_measurements, h_measurements, pre_measurements);

	EXPECT_EQ(pre_measurements.LEN, h_measurements.LEN);
	EXPECT_EQ(d_measurements.v_lidar_coords.size(), 2 * h_measurements.MAX_LEN);
	// EXPECT_TRUE(true);
}

TEST(DeviceInitCommon, ResetMeasurementVars) {

	DeviceMeasurements d_measurements; 
	HostMeasurements h_measurements; 
	host_vector<float> hvec_lidar_coords(100);

	reset_measurement_vars(d_measurements, h_measurements, hvec_lidar_coords);

	EXPECT_EQ(h_measurements.LEN, hvec_lidar_coords.size() / 2);
	EXPECT_EQ(d_measurements.v_lidar_coords.size() ,2 * h_measurements.LEN);
}

TEST(DeviceInitCommon, AllocInitMapVars) {

	DeviceMap d_map; 
	HostMap h_map; 
	HostMap pre_map;

	pre_map.GRID_WIDTH = 100;
	pre_map.GRID_HEIGHT = 100;
	pre_map.xmin = 1;
	pre_map.xmax = 1;
	pre_map.ymin = 2;
	pre_map.ymax = 2;
	pre_map.b_should_extend = true;
	pre_map.s_grid_map.resize(pre_map.GRID_WIDTH * pre_map.GRID_HEIGHT, 0);
	pre_map.s_log_odds.resize(pre_map.GRID_WIDTH * pre_map.GRID_HEIGHT, 0);

	alloc_init_map_vars(d_map, h_map, pre_map);

	EXPECT_EQ(d_map.s_grid_map.size(), pre_map.GRID_WIDTH * pre_map.GRID_HEIGHT);
	EXPECT_EQ(d_map.s_log_odds.size(), pre_map.GRID_WIDTH * pre_map.GRID_HEIGHT);
	EXPECT_EQ(d_map.c_should_extend.size(), 4);

	EXPECT_EQ(h_map.s_grid_map.size(), pre_map.GRID_WIDTH * pre_map.GRID_HEIGHT);
	EXPECT_EQ(h_map.s_log_odds.size(), pre_map.GRID_WIDTH * pre_map.GRID_HEIGHT);
	EXPECT_EQ(h_map.c_should_extend.size(), 4);
}

TEST(DeviceInitCommon, AllocInitBodyLidar) {

	DeviceTransition d_transition;
	alloc_init_body_lidar(d_transition);

	EXPECT_EQ(d_transition.c_body_lidar.size(), 9);
}

