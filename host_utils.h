#ifndef _HOST_UTILS_H_
#define _HOST_UTILS_H_

#include "headers.h"
#include "structures.h"


int getNegativeCounter(int* x, int *y, const int LEN) {

    int negative_counter = 0;
    for (int i = 0; i < LEN; i++) {
        if (x[i] < 0 || y[i] < 0)
            negative_counter += 1;
    }
    return negative_counter;
}

int getGreaterThanCounter(int* x, const int VALUE, const int LEN) {
    
    int value_counter = 0;
    for (int i = 0; i < LEN; i++) {

        if (x[i] >= VALUE)
            value_counter += 1;
    }
    return value_counter;
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

void read_small_steps_vec_arr(string file_name, vector<vector<float>>& vec_arr, int max_lines = 500, string root = "") {

    string full_path = "data/" + root + "small_steps/" + file_name + ".txt";
    std::ifstream data(full_path);

    string line;
    string delimiter = ",";
    int line_counter = 0;

    while (getline(data, line)) {

        line = trim(line);

        size_t pos = 0;
        string token;
        vector<float> arr;

        while ((pos = line.find(delimiter)) != std::string::npos) {
            token = line.substr(0, pos);
            if (token != "")
                arr.push_back(std::stof(token));
            line.erase(0, pos + delimiter.length());
        }
        vec_arr.push_back(arr);

        line_counter += 1;
        if (line_counter > max_lines)
            break;
    }
}

void read_small_steps_vec(string file_name, vector<float>& vec, int max_lines = 500, string root = "") {

    string full_path = "data/" + root + "small_steps/" + file_name + ".txt";
    std::ifstream data(full_path);

    string line;
    string delimiter = ",";
    int line_counter = 0;

    while (getline(data, line)) {

        line = trim(line);

        size_t pos = line.find(delimiter);
        string token = line.substr(0, pos);
        if (token != "")
            vec.push_back(std::stof(token));

        line_counter += 1;
        if (line_counter > max_lines)
            break;
    }
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

void file_extractor(string file_name, string first_vec_title, 
    map<string, string>& scalar_values, map<string, string>& vec_values) {

    std::ifstream data(file_name);
    string line;
    string line_next;
    int line_index = 0;

    while (getline(data, line)) {

        line = trim(line);
        if (line.find(first_vec_title) != std::string::npos)
            break;
        getline(data, line_next);
        line_next = trim(line_next);
        scalar_values[line] = line_next;
    }

    while (true) {

        string curr_title = trim(line);
        string curr_value = "";

        while (getline(data, line) && line.find("SEPARATE") == std::string::npos) {
            curr_value += trim(line);
        }
        vec_values[curr_title] = curr_value; // curr_value.substr(0, curr_value.length() - 1);

        if (!getline(data, line))
            break;
    }
}

//template <class T>
//void string_extractor(string data, vector<T>& vec) {
//
//    string delimiter = ",";
//
//    size_t pos = 0;
//    string token;
//
//    while ((pos = data.find(delimiter)) != std::string::npos) {
//        token = data.substr(0, pos);
//        if (token != "") {
//            if (std::is_same<T, float>::value) vec.push_back(std::stof(token));
//            else if (std::is_same<T, int>::value) vec.push_back(std::stoi(token));
//        }
//        data.erase(0, pos + delimiter.length());
//    }
//}

template <class T>
void string_extractor(string data, vector<T>& vec) {

    std::stringstream ss(data);
    ss.imbue(std::locale(std::locale(), new tokens()));
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    std::vector<std::string> vstrings(begin, end);
    
    vec.resize(vstrings.size());
    if (std::is_same<T, float>::value) {
        std::transform(vstrings.begin(), vstrings.end(), vec.begin(), [](const std::string& val) {
            return std::stof(val);
            }
        );
    }
    else if (std::is_same<T, int>::value) {
        std::transform(vstrings.begin(), vstrings.end(), vec.begin(), [](const std::string& val) {
            return std::stoi(val);
            }
        );
    }
}

template <class T>
void string_extractor(string data, host_vector<T>& vec) {

    std::stringstream ss(data);
    ss.imbue(std::locale(std::locale(), new tokens()));
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    std::vector<std::string> vstrings(begin, end);

    vec.resize(vstrings.size());
    if (std::is_same<T, float>::value) {
        std::transform(vstrings.begin(), vstrings.end(), vec.begin(), [](const std::string& val) {
            return std::stof(val);
            }
        );
    }
    else if (std::is_same<T, int>::value) {
        std::transform(vstrings.begin(), vstrings.end(), vec.begin(), [](const std::string& val) {
            return std::stoi(val);
            }
        );
    }
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////


void read_update_map(int file_number, HostMap& pre_map, HostMap& post_bg_map,
    HostMap& post_map, GeneralInfo& general_info, HostMeasurements& pre_measurements,
    HostParticles& pre_particles, HostParticles& post_particles, HostPosition& post_position, 
    HostTransition& pre_transition, HostTransition& post_transition, string root = "") {

    string file_name = "data/" + root + "map/" + std::to_string(file_number) + ".txt";
    string first_vec_title = "h_lidar_coords";
    map<string, string> scalar_values; 
    map<string, string> vec_values;

    file_extractor(file_name, first_vec_title, scalar_values, vec_values);

    general_info.res = std::stof(scalar_values["ST_res"]);
    general_info.log_t = std::stof(scalar_values["ST_log_t"]);

    pre_map.GRID_WIDTH = std::stoi(scalar_values["ST_GRID_WIDTH"]);
    pre_map.GRID_HEIGHT = std::stoi(scalar_values["ST_GRID_HEIGHT"]);
    pre_map.xmin = std::stoi(scalar_values["ST_xmin"]);
    pre_map.xmax = std::stoi(scalar_values["ST_xmax"]);
    pre_map.ymin = std::stoi(scalar_values["ST_ymin"]);
    pre_map.ymax = std::stoi(scalar_values["ST_ymax"]);
    pre_map.b_should_extend = (scalar_values["ST_EXTEND"] == "true") ? true : false;
    string_extractor<int>(vec_values["h_grid_map"], pre_map.s_grid_map);
    string_extractor<float>(vec_values["h_log_odds"], pre_map.s_log_odds);


    post_bg_map.GRID_WIDTH = std::stoi(scalar_values["AF_GRID_WIDTH"]);
    post_bg_map.GRID_HEIGHT = std::stoi(scalar_values["AF_GRID_HEIGHT"]);
    string_extractor<int>(vec_values["h_bg_grid_map"], post_bg_map.s_grid_map);
    string_extractor<float>(vec_values["h_bg_log_odds"], post_bg_map.s_log_odds);


    post_map.GRID_WIDTH = std::stoi(scalar_values["AF_GRID_WIDTH"]);
    post_map.GRID_HEIGHT = std::stoi(scalar_values["AF_GRID_HEIGHT"]);
    post_map.xmin = std::stoi(scalar_values["AF_xmin"]);
    post_map.xmax = std::stoi(scalar_values["AF_xmax"]);
    post_map.ymin = std::stoi(scalar_values["AF_ymin"]);
    post_map.ymax = std::stoi(scalar_values["AF_ymax"]);
    string_extractor<int>(vec_values["h_post_grid_map"], post_map.s_grid_map);
    string_extractor<float>(vec_values["h_post_log_odds"], post_map.s_log_odds);

    pre_measurements.LEN = std::stoi(scalar_values["ST_LIDAR_COORDS_LEN"]);
    string_extractor<float>(vec_values["h_lidar_coords"], pre_measurements.v_lidar_coords);
    string_extractor<int>(vec_values["h_coord"], pre_measurements.c_coord);

    pre_particles.OCCUPIED_LEN = std::stoi(scalar_values["ST_PARTICLES_OCCUPIED_LEN"]);

    post_particles.FREE_LEN = std::stoi(scalar_values["ST_PARTICLES_FREE_LEN"]);
    post_particles.OCCUPIED_UNIQUE_LEN = std::stoi(scalar_values["ST_PARTICLES_OCCUPIED_UNIQUE_LEN"]);
    post_particles.FREE_UNIQUE_LEN = std::stoi(scalar_values["ST_PARTICLES_FREE_UNIQUE_LEN"]);

    string_extractor<int>(vec_values["h_particles_occupied_x"], post_particles.v_occupied_x);
    string_extractor<int>(vec_values["h_particles_occupied_y"], post_particles.v_occupied_y);
    string_extractor<int>(vec_values["h_particles_occupied_unique_x"], post_particles.f_occupied_unique_x);
    string_extractor<int>(vec_values["h_particles_occupied_unique_y"], post_particles.f_occupied_unique_y);
    string_extractor<float>(vec_values["h_particles_world_x"], post_particles.v_world_x);
    string_extractor<float>(vec_values["h_particles_world_y"], post_particles.v_world_y);
    string_extractor<int>(vec_values["h_particles_free_x"], post_particles.f_free_x);
    string_extractor<int>(vec_values["h_particles_free_y"], post_particles.f_free_y);
    string_extractor<int>(vec_values["h_particles_free_idx"], post_particles.v_free_idx);
    string_extractor<int>(vec_values["h_particles_free_unique_x"], post_particles.f_free_unique_x);
    string_extractor<int>(vec_values["h_particles_free_unique_y"], post_particles.f_free_unique_y);

    string_extractor<float>(vec_values["h_transition_world_body"], pre_transition.c_world_body);

    string_extractor<int>(vec_values["h_position_image_body"], post_position.c_image_body);
    string_extractor<float>(vec_values["h_position_world_body"], post_position.world_body);
    string_extractor<float>(vec_values["h_transition_world_lidar"], post_transition.c_world_lidar);
}

void read_robot_move(int file_number, HostState& pre_state, HostState& post_state, string root = "") {

    string file_name = "data/" + root + "robot_move/" + std::to_string(file_number) + ".txt";
    string first_vec_title = "h_states_x";
    map<string, string> scalar_values;
    map<string, string> vec_values;

    file_extractor(file_name, first_vec_title, scalar_values, vec_values);

    pre_state.encoder_counts = std::stof(scalar_values["encoder_counts"]);
    pre_state.yaw = std::stof(scalar_values["yaw"]);
    pre_state.dt = std::stof(scalar_values["dt"]);
    pre_state.nv = std::stof(scalar_values["nv"]);
    pre_state.nw = std::stof(scalar_values["nw"]);

    string_extractor<float>(vec_values["h_states_x"], pre_state.c_x);
    string_extractor<float>(vec_values["h_states_y"], pre_state.c_y);
    string_extractor<float>(vec_values["h_states_theta"], pre_state.c_theta);
    string_extractor<float>(vec_values["h_rnds_encoder_counts"], pre_state.c_rnds_encoder_counts);
    string_extractor<float>(vec_values["h_rnds_yaws"], pre_state.c_rnds_yaws);

    string_extractor<float>(vec_values["post_states_x"], post_state.c_x);
    string_extractor<float>(vec_values["post_states_y"], post_state.c_y);
    string_extractor<float>(vec_values["post_states_theta"], post_state.c_theta);
}

void read_update_robot(int file_number, HostMap& pre_map, HostMeasurements& pre_measurements, 
    HostRobotParticles& pre_robot_particles,
    HostRobotParticles& pre_resampling_robot_particles, HostRobotParticles& post_resampling_robot_particles,
    HostRobotParticles& post_unique_robot_particles, HostProcessedMeasure& post_processed_measure, HostState& pre_state, HostState& post_state,
    HostResampling& pre_resampling, HostRobotState& post_robot_state, HostParticlesTransition& post_particles_transition,
    host_vector<float>& pre_weights, host_vector<float>& post_loop_weights, host_vector<float>& post_weights,
    GeneralInfo& general_info, string root = "") {

    string file_name = "data/" + root + "robot/" + std::to_string(file_number) + ".txt";
    string first_vec_title = "h_lidar_coords";
    map<string, string> scalar_values;
    map<string, string> vec_values;

    file_extractor(file_name, first_vec_title, scalar_values, vec_values);

    pre_map.GRID_WIDTH = std::stoi(scalar_values["GRID_WIDTH"]);
    pre_map.GRID_HEIGHT = std::stoi(scalar_values["GRID_HEIGHT"]);
    pre_map.xmin = std::stoi(scalar_values["xmin"]);
    pre_map.ymax = std::stoi(scalar_values["ymax"]);
    string_extractor<int>(vec_values["h_grid_map"], pre_map.s_grid_map);

    pre_measurements.LEN = std::stoi(scalar_values["LIDAR_COORDS_LEN"]);
    string_extractor<float>(vec_values["h_lidar_coords"], pre_measurements.v_lidar_coords);

    pre_robot_particles.LEN = std::stoi(scalar_values["PARTICLES_ITEMS_LEN"]);
    string_extractor<int>(vec_values["h_particles_x"], pre_robot_particles.f_x);
    string_extractor<int>(vec_values["h_particles_y"], pre_robot_particles.f_y);
    string_extractor<int>(vec_values["h_particles_idx"], pre_robot_particles.c_idx);
    string_extractor<float>(vec_values["h_particles_weight_pre"], pre_robot_particles.c_weight);
    assert(pre_robot_particles.LEN == pre_robot_particles.f_x.size());

    post_unique_robot_particles.LEN = std::stoi(scalar_values["PARTICLES_ITEMS_LEN_UNIQUE"]);
    string_extractor<int>(vec_values["h_particles_x_after_unique"], post_unique_robot_particles.f_x);
    string_extractor<int>(vec_values["h_particles_y_after_unique"], post_unique_robot_particles.f_y);
    string_extractor<int>(vec_values["h_particles_idx_after_unique"], post_unique_robot_particles.c_idx);
    string_extractor<float>(vec_values["h_particles_weight_post"], post_unique_robot_particles.c_weight);
    assert(post_unique_robot_particles.LEN == post_unique_robot_particles.f_x.size());

    string_extractor<int>(vec_values["h_particles_idx_before_resampling"], pre_resampling_robot_particles.c_idx);

    post_resampling_robot_particles.LEN = std::stoi(scalar_values["PARTICLES_ITEMS_LEN_RESAMPLING"]);
    string_extractor<int>(vec_values["h_particles_x_after_resampling"], post_resampling_robot_particles.f_x);
    string_extractor<int>(vec_values["h_particles_y_after_resampling"], post_resampling_robot_particles.f_y);
    string_extractor<int>(vec_values["h_particles_idx_after_resampling"], post_resampling_robot_particles.c_idx);
    assert(post_resampling_robot_particles.LEN == post_resampling_robot_particles.f_x.size());

    string_extractor<int>(vec_values["h_processed_measure_x"], post_processed_measure.v_x);
    string_extractor<int>(vec_values["h_processed_measure_y"], post_processed_measure.v_y);
    string_extractor<int>(vec_values["h_processed_measure_idx"], post_processed_measure.c_idx);

    string_extractor<float>(vec_values["h_states_x"], pre_state.c_x);
    string_extractor<float>(vec_values["h_states_y"], pre_state.c_y);
    string_extractor<float>(vec_values["h_states_theta"], pre_state.c_theta);

    string_extractor<float>(vec_values["h_states_x_updated"], post_state.c_x);
    string_extractor<float>(vec_values["h_states_y_updated"], post_state.c_y);
    string_extractor<float>(vec_values["h_states_theta_updated"], post_state.c_theta);

    //string_extractor<float>(vec_values["h_position_world_body"], h_particles_position.world_body);
    //string_extractor<float>(vec_values["h_rotation_world_body"], h_particles_rotation.world_body);
    
    //string_extractor<float>(vec_values["h_transition_world_body"], h_position_transition.transition_world_body);
    //string_extractor<float>(vec_values["h_transition_world_lidar"], h_position_transition.transition_world_lidar);

    //string_extractor<float>(vec_values["h_particles_world"], post_particles_transition.world);
    //string_extractor<float>(vec_values["h_particles_world_homo"], post_particles_transition.world_homo);
    string_extractor<float>(vec_values["h_transition_world_body"], post_particles_transition.c_world_body);
    string_extractor<float>(vec_values["h_transition_world_lidar"], post_particles_transition.c_world_lidar);


    string_extractor<float>(vec_values["h_rnds"], pre_resampling.c_rnds);
    string_extractor<int>(vec_values["h_js"], pre_resampling.c_js);

    string_extractor<float>(vec_values["h_robot_state"], post_robot_state.state);
    string_extractor<float>(vec_values["h_robot_transition_world_body"], post_robot_state.transition_world_body);

    string_extractor<float>(vec_values["h_pre_weights"], pre_weights);
    string_extractor<float>(vec_values["h_new_weights"], post_loop_weights);
    string_extractor<float>(vec_values["h_updated_weights"], post_weights);

    general_info.res = std::stof(scalar_values["res"]);
}

void read_iteration(int file_number, HostState& pre_state, HostState& post_robot_move_state, HostState& post_state,
    HostRobotParticles& pre_robot_particles, HostRobotParticles& post_unique_robot_particles,
    HostRobotParticles& pre_resampling_robot_particles, HostRobotParticles& post_resampling_robot_particles,
    HostProcessedMeasure& post_processed_measure, HostParticlesTransition& post_particles_transition,
    HostResampling& pre_resampling, HostRobotState& post_robot_state,
    HostMap& pre_map, HostMap& post_bg_map, HostMap& post_map, HostMeasurements& pre_measurements,
    HostPosition& post_position, HostTransition& pre_transition, HostTransition& post_transition,
    HostParticles& pre_particles, HostParticles& post_particles, GeneralInfo& general_info, 
    host_vector<float>& pre_weights, host_vector<float>& post_loop_weights, host_vector<float>& post_weights, string root="") {

    string file_name = "data/" + root + "robot_move/" + std::to_string(file_number) + ".txt";
    string first_vec_title_robot_move = "h_states_x";
    map<string, string> scalar_values_robot_move;
    map<string, string> vec_values_robot_move;
    file_extractor(file_name, first_vec_title_robot_move, scalar_values_robot_move, vec_values_robot_move);

    // file_extractor(file_name, first_vec_title_robot_move, scalar_values_robot_move, vec_values_robot_move);

    pre_state.encoder_counts = std::stof(scalar_values_robot_move["encoder_counts"]);
    pre_state.yaw = std::stof(scalar_values_robot_move["yaw"]);
    pre_state.dt = std::stof(scalar_values_robot_move["dt"]);
    pre_state.nv = std::stof(scalar_values_robot_move["nv"]);
    pre_state.nw = std::stof(scalar_values_robot_move["nw"]);

    string_extractor<float>(vec_values_robot_move["h_states_x"], pre_state.c_x);
    string_extractor<float>(vec_values_robot_move["h_states_y"], pre_state.c_y);
    string_extractor<float>(vec_values_robot_move["h_states_theta"], pre_state.c_theta);
    string_extractor<float>(vec_values_robot_move["h_rnds_encoder_counts"], pre_state.c_rnds_encoder_counts);
    string_extractor<float>(vec_values_robot_move["h_rnds_yaws"], pre_state.c_rnds_yaws);

    string_extractor<float>(vec_values_robot_move["post_states_x"], post_robot_move_state.c_x);
    string_extractor<float>(vec_values_robot_move["post_states_y"], post_robot_move_state.c_y);
    string_extractor<float>(vec_values_robot_move["post_states_theta"], post_robot_move_state.c_theta);

    //////////////////////////////////////////////////////////////////////////////////////////////////

    file_name = "data/" + root + "robot/" + std::to_string(file_number) + ".txt";
    string first_vec_title_robot = "h_lidar_coords";
    map<string, string> scalar_values_robot;
    map<string, string> vec_values_robot;
    file_extractor(file_name, first_vec_title_robot, scalar_values_robot, vec_values_robot);

    //pre_map.GRID_WIDTH = std::stoi(scalar_values_robot["GRID_WIDTH"]);
    //pre_map.GRID_HEIGHT = std::stoi(scalar_values_robot["GRID_HEIGHT"]);
    //pre_map.xmin = std::stoi(scalar_values_robot["xmin"]);
    //pre_map.ymax = std::stoi(scalar_values_robot["ymax"]);
    //string_extractor<int>(vec_values_robot["h_grid_map"], pre_map.grid_map);

    //pre_measurements.LEN = std::stoi(scalar_values_robot["LIDAR_COORDS_LEN"]);
    //string_extractor<float>(vec_values_robot["h_lidar_coords"], pre_measurements.v_lidar_coords);

    pre_robot_particles.LEN = std::stoi(scalar_values_robot["PARTICLES_ITEMS_LEN"]);
    string_extractor<int>(vec_values_robot["h_particles_x"], pre_robot_particles.f_x);
    string_extractor<int>(vec_values_robot["h_particles_y"], pre_robot_particles.f_y);
    string_extractor<int>(vec_values_robot["h_particles_idx"], pre_robot_particles.c_idx);
    string_extractor<float>(vec_values_robot["h_particles_weight_pre"], pre_robot_particles.c_weight);
    assert(pre_robot_particles.LEN == pre_robot_particles.f_x.size());

    post_unique_robot_particles.LEN = std::stoi(scalar_values_robot["PARTICLES_ITEMS_LEN_UNIQUE"]);
    string_extractor<int>(vec_values_robot["h_particles_x_after_unique"], post_unique_robot_particles.f_x);
    string_extractor<int>(vec_values_robot["h_particles_y_after_unique"], post_unique_robot_particles.f_y);
    string_extractor<int>(vec_values_robot["h_particles_idx_after_unique"], post_unique_robot_particles.c_idx);
    string_extractor<float>(vec_values_robot["h_particles_weight_post"], post_unique_robot_particles.c_weight);
    assert(post_unique_robot_particles.LEN == post_unique_robot_particles.f_x.size());

    string_extractor<int>(vec_values_robot["h_particles_idx_before_resampling"], pre_resampling_robot_particles.c_idx);

    post_resampling_robot_particles.LEN = std::stoi(scalar_values_robot["PARTICLES_ITEMS_LEN_RESAMPLING"]);
    string_extractor<int>(vec_values_robot["h_particles_x_after_resampling"], post_resampling_robot_particles.f_x);
    string_extractor<int>(vec_values_robot["h_particles_y_after_resampling"], post_resampling_robot_particles.f_y);
    string_extractor<int>(vec_values_robot["h_particles_idx_after_resampling"], post_resampling_robot_particles.c_idx);
    assert(post_resampling_robot_particles.LEN == post_resampling_robot_particles.f_x.size());

    string_extractor<int>(vec_values_robot["h_processed_measure_x"], post_processed_measure.v_x);
    string_extractor<int>(vec_values_robot["h_processed_measure_y"], post_processed_measure.v_y);
    string_extractor<int>(vec_values_robot["h_processed_measure_idx"], post_processed_measure.c_idx);

    //string_extractor<float>(vec_values_robot["h_states_x"], pre_state.c_x);
    //string_extractor<float>(vec_values_robot["h_states_y"], pre_state.c_y);
    //string_extractor<float>(vec_values_robot["h_states_theta"], pre_state.c_theta);

    string_extractor<float>(vec_values_robot["h_states_x_updated"], post_state.c_x);
    string_extractor<float>(vec_values_robot["h_states_y_updated"], post_state.c_y);
    string_extractor<float>(vec_values_robot["h_states_theta_updated"], post_state.c_theta);

    //string_extractor<float>(vec_values_robot["h_particles_world"], post_particles_transition.world);
    //string_extractor<float>(vec_values_robot["h_particles_world_homo"], post_particles_transition.world_homo);
    string_extractor<float>(vec_values_robot["h_transition_world_body"], post_particles_transition.c_world_body);
    string_extractor<float>(vec_values_robot["h_transition_world_lidar"], post_particles_transition.c_world_lidar);

    string_extractor<float>(vec_values_robot["h_rnds"], pre_resampling.c_rnds);
    string_extractor<int>(vec_values_robot["h_js"], pre_resampling.c_js);

    string_extractor<float>(vec_values_robot["h_robot_state"], post_robot_state.state);
    string_extractor<float>(vec_values_robot["h_robot_transition_world_body"], post_robot_state.transition_world_body);

    string_extractor<float>(vec_values_robot["h_pre_weights"], pre_weights);
    string_extractor<float>(vec_values_robot["h_new_weights"], post_loop_weights);
    string_extractor<float>(vec_values_robot["h_updated_weights"], post_weights);

    //general_info.res = std::stof(scalar_values_robot["res"]);

    //////////////////////////////////////////////////////////////////////////////////////////////////

    file_name = "data/" + root + "map/" + std::to_string(file_number) + ".txt";
    string first_vec_title_map = "h_lidar_coords";
    map<string, string> scalar_values_map;
    map<string, string> vec_values_map;
    file_extractor(file_name, first_vec_title_map, scalar_values_map, vec_values_map);

    general_info.res = std::stof(scalar_values_map["ST_res"]);
    general_info.log_t = std::stof(scalar_values_map["ST_log_t"]);

    pre_map.GRID_WIDTH = std::stoi(scalar_values_map["ST_GRID_WIDTH"]);
    pre_map.GRID_HEIGHT = std::stoi(scalar_values_map["ST_GRID_HEIGHT"]);
    pre_map.xmin = std::stoi(scalar_values_map["ST_xmin"]);
    pre_map.xmax = std::stoi(scalar_values_map["ST_xmax"]);
    pre_map.ymin = std::stoi(scalar_values_map["ST_ymin"]);
    pre_map.ymax = std::stoi(scalar_values_map["ST_ymax"]);
    pre_map.b_should_extend = (scalar_values_map["ST_EXTEND"] == "true") ? true : false;
    string_extractor<int>(vec_values_map["h_grid_map"], pre_map.s_grid_map);
    string_extractor<float>(vec_values_map["h_log_odds"], pre_map.s_log_odds);


    post_bg_map.GRID_WIDTH = std::stoi(scalar_values_map["AF_GRID_WIDTH"]);
    post_bg_map.GRID_HEIGHT = std::stoi(scalar_values_map["AF_GRID_HEIGHT"]);
    string_extractor<int>(vec_values_map["h_bg_grid_map"], post_bg_map.s_grid_map);
    string_extractor<float>(vec_values_map["h_bg_log_odds"], post_bg_map.s_log_odds);


    post_map.GRID_WIDTH = std::stoi(scalar_values_map["AF_GRID_WIDTH"]);
    post_map.GRID_HEIGHT = std::stoi(scalar_values_map["AF_GRID_HEIGHT"]);
    post_map.xmin = std::stoi(scalar_values_map["AF_xmin"]);
    post_map.xmax = std::stoi(scalar_values_map["AF_xmax"]);
    post_map.ymin = std::stoi(scalar_values_map["AF_ymin"]);
    post_map.ymax = std::stoi(scalar_values_map["AF_ymax"]);
    string_extractor<int>(vec_values_map["h_post_grid_map"], post_map.s_grid_map);
    string_extractor<float>(vec_values_map["h_post_log_odds"], post_map.s_log_odds);

    pre_measurements.LEN = std::stoi(scalar_values_map["ST_LIDAR_COORDS_LEN"]);
    string_extractor<float>(vec_values_map["h_lidar_coords"], pre_measurements.v_lidar_coords);
    string_extractor<int>(vec_values_map["h_coord"], pre_measurements.c_coord);

    pre_particles.OCCUPIED_LEN = std::stoi(scalar_values_map["ST_PARTICLES_OCCUPIED_LEN"]);

    post_particles.FREE_LEN = std::stoi(scalar_values_map["ST_PARTICLES_FREE_LEN"]);
    post_particles.OCCUPIED_UNIQUE_LEN = std::stoi(scalar_values_map["ST_PARTICLES_OCCUPIED_UNIQUE_LEN"]);
    post_particles.FREE_UNIQUE_LEN = std::stoi(scalar_values_map["ST_PARTICLES_FREE_UNIQUE_LEN"]);

    string_extractor<int>(vec_values_map["h_particles_occupied_x"], post_particles.v_occupied_x);
    string_extractor<int>(vec_values_map["h_particles_occupied_y"], post_particles.v_occupied_y);
    string_extractor<int>(vec_values_map["h_particles_occupied_unique_x"], post_particles.f_occupied_unique_x);
    string_extractor<int>(vec_values_map["h_particles_occupied_unique_y"], post_particles.f_occupied_unique_y);
    string_extractor<float>(vec_values_map["h_particles_world_x"], post_particles.v_world_x);
    string_extractor<float>(vec_values_map["h_particles_world_y"], post_particles.v_world_y);
    string_extractor<int>(vec_values_map["h_particles_free_x"], post_particles.f_free_x);
    string_extractor<int>(vec_values_map["h_particles_free_y"], post_particles.f_free_y);
    string_extractor<int>(vec_values_map["h_particles_free_idx"], post_particles.v_free_idx);
    string_extractor<int>(vec_values_map["h_particles_free_unique_x"], post_particles.f_free_unique_x);
    string_extractor<int>(vec_values_map["h_particles_free_unique_y"], post_particles.f_free_unique_y);

    string_extractor<float>(vec_values_map["h_transition_world_body"], pre_transition.c_world_body);

    string_extractor<int>(vec_values_map["h_position_image_body"], post_position.c_image_body);
    string_extractor<float>(vec_values_map["h_position_world_body"], post_position.world_body);
    string_extractor<float>(vec_values_map["h_transition_world_lidar"], post_transition.c_world_lidar);
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

void read_robot_simple(int file_number, HostMap& pre_map, HostMeasurements& pre_measurements,
    HostRobotParticles& pre_robot_particles, HostState& pre_state, HostResampling& pre_resampling, 
    GeneralInfo& general_info, string root = "") {

    string file_name = "data/" + root + "robot_simple/" + std::to_string(file_number) + ".txt";
    string first_vec_title = "h_lidar_coords";
    map<string, string> scalar_values;
    map<string, string> vec_values;

    file_extractor(file_name, first_vec_title, scalar_values, vec_values);

    pre_map.GRID_WIDTH = std::stoi(scalar_values["GRID_WIDTH"]);
    pre_map.GRID_HEIGHT = std::stoi(scalar_values["GRID_HEIGHT"]);
    pre_map.xmin = std::stoi(scalar_values["xmin"]);
    pre_map.ymax = std::stoi(scalar_values["ymax"]);
    string_extractor<int>(vec_values["h_grid_map"], pre_map.s_grid_map);

    pre_measurements.LEN = std::stoi(scalar_values["LIDAR_COORDS_LEN"]);
    string_extractor<float>(vec_values["h_lidar_coords"], pre_measurements.v_lidar_coords);

    pre_robot_particles.LEN = std::stoi(scalar_values["PARTICLES_ITEMS_LEN"]);
    string_extractor<int>(vec_values["h_particles_x"], pre_robot_particles.f_x);
    string_extractor<int>(vec_values["h_particles_y"], pre_robot_particles.f_y);
    string_extractor<int>(vec_values["h_particles_idx"], pre_robot_particles.c_idx);
    string_extractor<float>(vec_values["h_particles_weight_pre"], pre_robot_particles.c_weight);
    assert(pre_robot_particles.LEN == pre_robot_particles.f_x.size());

    string_extractor<float>(vec_values["h_states_x"], pre_state.c_x);
    string_extractor<float>(vec_values["h_states_y"], pre_state.c_y);
    string_extractor<float>(vec_values["h_states_theta"], pre_state.c_theta);

    string_extractor<float>(vec_values["h_rnds"], pre_resampling.c_rnds);
    string_extractor<int>(vec_values["h_js"], pre_resampling.c_js);

    general_info.res = std::stof(scalar_values["res"]);
}


///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

bool compareNat(const std::string& a, const std::string& b) {

    if (a.empty())
        return true;
    if (b.empty())
        return false;
    if (std::isdigit(a[0]) && !std::isdigit(b[0]))
        return true;
    if (!std::isdigit(a[0]) && std::isdigit(b[0]))
        return false;
    if (!std::isdigit(a[0]) && !std::isdigit(b[0]))
    {
        if (std::toupper(a[0]) == std::toupper(b[0]))
            return compareNat(a.substr(1), b.substr(1));
        return (std::toupper(a[0]) < std::toupper(b[0]));
    }

    // Both strings begin with digit --> parse both numbers
    std::istringstream issa(a);
    std::istringstream issb(b);
    int ia, ib;
    issa >> ia;
    issb >> ib;
    if (ia != ib)
        return ia < ib;

    // Numbers are the same --> remove numbers and recurse
    std::string anew, bnew;
    std::getline(issa, anew);
    std::getline(issb, bnew);
    return (compareNat(anew, bnew));
}


void getFiles(string dir, vector<int>& files) {

    for (const auto& file : fs::directory_iterator(dir)) {
        files.push_back(std::stoi(file.path().filename().replace_extension().string()));
    }

    std::sort(files.begin(), files.end());
}

//void print_world_body(device_vector<float>& dvec_world_body) {
//
//    printf("Calculated: ");
//    host_vector<float> hvec_world_body;
//    hvec_world_body.resize(dvec_world_body.size());
//    hvec_world_body.assign(dvec_world_body.begin(), dvec_world_body.end());
//    for (int i = 0; i < hvec_world_body.size(); i++)
//        printf("%f, ", hvec_world_body[i]);
//    printf("\n");
//}
//void print_world_body(std::vector<float>& hvec_world_body) {
//
//    printf("Python    : ");
//    for (int i = 0; i < hvec_world_body.size(); i++)
//        printf("%f, ", hvec_world_body[i]);
//    printf("\n");
//}
//
//void check_NaN(host_vector<float>& values, int idx) {
//
//    for (int i = 0; i < values.size(); i++) {
//        if (isnan(values[i]) == true) {
//            printf("idx: %d, i: %d, Nan Found\n", idx, i);
//        }
//    }
//}
//
//void check_NaN(device_vector<float>& dvalues, int idx) {
//
//    host_vector<float> hvalues;
//    hvalues.resize(dvalues.size());
//    hvalues.assign(dvalues.begin(), dvalues.end());
//
//    for (int i = 0; i < hvalues.size(); i++) {
//        if (isnan(hvalues[i]) == true) {
//            printf("idx: %d, i: %d, Nan Found\n", idx, i);
//        }
//    }
//}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

std::random_device rnd_device;
std::mt19937 mersenne_engine{ 1 };
std::uniform_real_distribution<> dist_uniform{ 0, 1 };
std::normal_distribution<> dist_normal;

auto gen_uniform = []() { return dist_uniform(mersenne_engine); };
auto gen_normal = []() { return dist_normal(mersenne_engine); };

void gen_uniform_numbers(vector<float>& vec, const int LEN = NUM_PARTICLES) {

    if (vec.size() == 0)
        vec.resize(LEN);
    generate(begin(vec), end(vec), gen_uniform);
}

void gen_normal_numbers(vector<float>& vec, const int LEN = NUM_PARTICLES) {

    if (vec.size() == 0)
        vec.resize(LEN);
    generate(begin(vec), end(vec), gen_normal);
}

#endif