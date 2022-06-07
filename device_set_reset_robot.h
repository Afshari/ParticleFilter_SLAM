#ifndef _DEVICE_SET_RESET_ROBOT_H_
#define _DEVICE_SET_RESET_ROBOT_H_

#include "headers.h"
#include "structures.h"


void reset_correlation(DeviceCorrelation& d_correlation) {

    thrust::fill(d_correlation.c_weight.begin(), d_correlation.c_weight.end(), 0);
    thrust::fill(d_correlation.c_raw.begin(), d_correlation.c_raw.end(), 0);
    thrust::fill(d_correlation.c_sum_exp.begin(), d_correlation.c_sum_exp.end(), 0);
    thrust::fill(d_correlation.c_max.begin(), d_correlation.c_max.end(), 0);
}


void reset_processed_measure(DeviceProcessedMeasure& d_processed_measure, HostMeasurements& h_measurements) {

    int MEASURE_LEN = NUM_PARTICLES * h_measurements.LEN;

    thrust::fill(d_processed_measure.v_x.begin(), d_processed_measure.v_x.begin() + MEASURE_LEN, 0);
    thrust::fill(d_processed_measure.v_y.begin(), d_processed_measure.v_y.begin() + MEASURE_LEN, 0);
    thrust::fill(d_processed_measure.c_idx.begin(), d_processed_measure.c_idx.end(), 0);
}

void set_resampling(DeviceResampling& d_resampling, HostResampling& pre_resampling) {

    d_resampling.c_rnds.assign(pre_resampling.c_rnds.begin(), pre_resampling.c_rnds.end());
    thrust::fill(d_resampling.c_js.begin(), d_resampling.c_js.end(), 0);
}

void set_resampling(DeviceResampling& d_resampling, std::vector<float>& c_rnds) {

    d_resampling.c_rnds.assign(c_rnds.begin(), c_rnds.end());
    thrust::fill(d_resampling.c_js.begin(), d_resampling.c_js.end(), 0);
}

void set_state(DeviceState& d_state, HostState& h_state, HostState& pre_state) {

    d_state.c_x.assign(pre_state.c_x.begin(), pre_state.c_x.end());
    d_state.c_y.assign(pre_state.c_y.begin(), pre_state.c_y.end());
    d_state.c_theta.assign(pre_state.c_theta.begin(), pre_state.c_theta.end());
    d_state.c_rnds_encoder_counts.assign(pre_state.c_rnds_encoder_counts.begin(), pre_state.c_rnds_encoder_counts.end());
    d_state.c_rnds_yaws.assign(pre_state.c_rnds_yaws.begin(), pre_state.c_rnds_yaws.end());

    h_state.c_x.assign(pre_state.c_x.begin(), pre_state.c_x.end());
    h_state.c_y.assign(pre_state.c_y.begin(), pre_state.c_y.end());
    h_state.c_theta.assign(pre_state.c_theta.begin(), pre_state.c_theta.end());
    h_state.c_rnds_encoder_counts.assign(pre_state.c_rnds_encoder_counts.begin(), pre_state.c_rnds_encoder_counts.end());
    h_state.c_rnds_yaws.assign(pre_state.c_rnds_yaws.begin(), pre_state.c_rnds_yaws.end());

    h_state.encoder_counts = pre_state.encoder_counts;
    h_state.yaw = pre_state.yaw;
    h_state.dt = pre_state.dt;
    h_state.nv = pre_state.nv;
    h_state.nw = pre_state.nw;
}

void set_state(DeviceState& d_state, HostState& h_state, HostState& pre_state, std::vector<float>& c_rnds_encoder_counts, 
    std::vector<float>& c_rnds_yaws, float encoder_counts, float yaw, float dt) {

    d_state.c_rnds_encoder_counts.assign(c_rnds_encoder_counts.begin(), c_rnds_encoder_counts.end());
    d_state.c_rnds_yaws.assign(c_rnds_yaws.begin(), c_rnds_yaws.end());

    h_state.c_rnds_encoder_counts.assign(c_rnds_encoder_counts.begin(), c_rnds_encoder_counts.end());
    h_state.c_rnds_yaws.assign(c_rnds_yaws.begin(), c_rnds_yaws.end());

    h_state.encoder_counts = encoder_counts;
    h_state.yaw = yaw;
    h_state.dt = dt;
    h_state.nv = pre_state.nv;
    h_state.nw = pre_state.nw;
}

void set_robot_particles(DeviceRobotParticles& d_robot_particles, HostRobotParticles& h_robot_particles,
    HostRobotParticles& pre_robot_particles) {

    h_robot_particles.LEN = pre_robot_particles.LEN;
    thrust::fill(d_robot_particles.f_extended_idx.begin(), d_robot_particles.f_extended_idx.begin() + h_robot_particles.LEN, 0);
    thrust::copy(pre_robot_particles.f_x.begin(), pre_robot_particles.f_x.end(), d_robot_particles.f_x.begin());
    thrust::copy(pre_robot_particles.f_y.begin(), pre_robot_particles.f_y.end(), d_robot_particles.f_y.begin());
    d_robot_particles.c_idx.assign(pre_robot_particles.c_idx.begin(), pre_robot_particles.c_idx.end());
    d_robot_particles.c_weight.assign(pre_robot_particles.c_weight.begin(), pre_robot_particles.c_weight.end());
}


#endif
