#ifndef _HOST_UTILS_H_
#define _HOST_UTILS_H_

#include "headers.h"
#include "structures.h"

#ifdef ADD_HEADER_DATA
#include "data/robot_advance/300.h"
#include "data/robot_iteration/300.h"
#include "data/map_iteration/300.h"
#endif

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

void read_small_steps_vec_arr(string file_name, vector<vector<float>>& vec_arr, int max_lines = 500) {

    std::ifstream data("data/small_steps/" + file_name + ".txt");

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

void read_small_steps_vec(string file_name, vector<float>& vec, int max_lines = 500) {

    std::ifstream data("data/small_steps/" + file_name + ".txt");

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

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////


void read_update_map(int file_number, MapData& map_data, MapData& map_data_bg,
        MapData& map_data_post, GeneralInfo& general_info, Measurements& measurements,
        ParticlesData& particles_data, PositionTransition& position_transition) {

    string file_name = "data/map/" + std::to_string(file_number) + ".txt";
    string first_vec_title = "h_lidar_coords";
    map<string, string> scalar_values; 
    map<string, string> vec_values;

    file_extractor(file_name, first_vec_title, scalar_values, vec_values);

    general_info.res = std::stof(scalar_values["ST_res"]);
    general_info.log_t = std::stof(scalar_values["ST_log_t"]);

    map_data.GRID_WIDTH = std::stoi(scalar_values["ST_GRID_WIDTH"]);
    map_data.GRID_HEIGHT = std::stoi(scalar_values["ST_GRID_HEIGHT"]);
    map_data.xmin = std::stoi(scalar_values["ST_xmin"]);
    map_data.xmax = std::stoi(scalar_values["ST_xmax"]);
    map_data.ymin = std::stoi(scalar_values["ST_ymin"]);
    map_data.ymax = std::stoi(scalar_values["ST_ymax"]);
    map_data.should_extend = (scalar_values["ST_EXTEND"] == "true") ? true : false;
    string_extractor<int>(vec_values["h_grid_map"], map_data.grid_map);
    string_extractor<float>(vec_values["h_log_odds"], map_data.log_odds);


    map_data_bg.GRID_WIDTH = std::stoi(scalar_values["AF_GRID_WIDTH"]);
    map_data_bg.GRID_HEIGHT = std::stoi(scalar_values["AF_GRID_HEIGHT"]);
    string_extractor<int>(vec_values["h_bg_grid_map"], map_data_bg.grid_map);
    string_extractor<float>(vec_values["h_bg_log_odds"], map_data_bg.log_odds);


    map_data_post.GRID_WIDTH = std::stoi(scalar_values["AF_GRID_WIDTH"]);
    map_data_post.GRID_HEIGHT = std::stoi(scalar_values["AF_GRID_HEIGHT"]);
    map_data_post.xmin = std::stoi(scalar_values["AF_xmin"]);
    map_data_post.xmax = std::stoi(scalar_values["AF_xmax"]);
    map_data_post.ymin = std::stoi(scalar_values["AF_ymin"]);
    map_data_post.ymax = std::stoi(scalar_values["AF_ymax"]);
    string_extractor<int>(vec_values["h_post_grid_map"], map_data_post.grid_map);
    string_extractor<float>(vec_values["h_post_log_odds"], map_data_post.log_odds);

    measurements.LIDAR_COORDS_LEN = std::stoi(scalar_values["ST_LIDAR_COORDS_LEN"]);
    string_extractor<float>(vec_values["h_lidar_coords"], measurements.lidar_coords);
    string_extractor<float>(vec_values["h_coord"], measurements.coord);

    particles_data.PARTICLES_OCCUPIED_LEN = std::stoi(scalar_values["ST_PARTICLES_OCCUPIED_LEN"]);
    particles_data.PARTICLES_OCCUPIED_UNIQUE_LEN = std::stoi(scalar_values["ST_PARTICLES_OCCUPIED_UNIQUE_LEN"]);
    particles_data.PARTICLES_FREE_LEN = std::stoi(scalar_values["ST_PARTICLES_FREE_LEN"]);
    particles_data.PARTICLES_FREE_UNIQUE_LEN = std::stoi(scalar_values["ST_PARTICLES_FREE_UNIQUE_LEN"]);
    string_extractor<int>(vec_values["h_particles_occupied_x"], particles_data.particles_occupied_x);
    string_extractor<int>(vec_values["h_particles_occupied_y"], particles_data.particles_occupied_y);
    string_extractor<int>(vec_values["h_particles_occupied_unique_x"], particles_data.particles_occupied_unique_x);
    string_extractor<int>(vec_values["h_particles_occupied_unique_y"], particles_data.particles_occupied_unique_y);
    string_extractor<float>(vec_values["h_particles_world_x"], particles_data.particles_world_x);
    string_extractor<float>(vec_values["h_particles_world_y"], particles_data.particles_world_y);
    string_extractor<int>(vec_values["h_particles_free_x"], particles_data.particles_free_x);
    string_extractor<int>(vec_values["h_particles_free_y"], particles_data.particles_free_y);
    string_extractor<int>(vec_values["h_particles_free_idx"], particles_data.particles_free_idx);
    string_extractor<int>(vec_values["h_particles_free_unique_x"], particles_data.particles_free_unique_x);
    string_extractor<int>(vec_values["h_particles_free_unique_y"], particles_data.particles_free_unique_y);

    string_extractor<int>(vec_values["h_position_image_body"], position_transition.position_image_body);
    string_extractor<float>(vec_values["h_position_world_body"], position_transition.position_world_body);
    string_extractor<float>(vec_values["h_transition_world_body"], position_transition.transition_world_body);
    string_extractor<float>(vec_values["h_transition_world_lidar"], position_transition.transition_world_lidar);
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

void read_robot_move_data(int file_number,
    vector<float>& vec_rnds_encoder_counts, vector<float>& vec_rnds_yaws,
    vector<float>& vec_states_x, vector<float>& vec_states_y, vector<float>& vec_states_theta,
    float& encoder_counts, float& yaw, float& dt,
    bool check_rnds_encoder_counts = true, bool check_rnds_yaws = true,
    bool check_states = true) {

    string file_name = std::to_string(file_number);

    const int SCALAR_VALUES = 1;
    const int RNDS_ENCODER_COUNTS_VALUES = 2;
    const int RNDS_YAWS_VALUES = 3;
    const int STATES_X_VALUES = 4;
    const int STATES_Y_VALUES = 5;
    const int STATES_THETA_VALUES = 6;
    const int SEPARATE_VALUES = 10;

    int curr_state = SCALAR_VALUES;
    string str_rnds_encoder_counts = "";
    string str_rnds_yaws = "";
    string str_states_x = "";
    string str_states_y = "";
    string str_states_theta = "";
    string segment;

    std::ifstream data("data/steps/robot_advance_" + file_name + ".txt");
    string line;

    while (getline(data, line)) {

        line = trim(line);

        if (curr_state == SCALAR_VALUES) {

            if (line == "encoder_counts") {
                getline(data, line);
                encoder_counts = std::stof(line);
            }
            else if (line == "yaw") {
                getline(data, line);
                yaw = std::stof(line);
            }
            else if (line == "dt") {
                getline(data, line);
                dt = std::stof(line);
            }
        }

        if (line == "rnds_encoder_counts") {
            curr_state = RNDS_ENCODER_COUNTS_VALUES;
            continue;
        }
        else if (line == "rnds_yaws") {
            curr_state = RNDS_YAWS_VALUES;
            continue;
        }
        else if (line == "states_x") {
            curr_state = STATES_X_VALUES;
            continue;
        }
        else if (line == "states_y") {
            curr_state = STATES_Y_VALUES;
            continue;
        }
        else if (line == "states_theta") {
            curr_state = STATES_THETA_VALUES;
            continue;
        }

        if (curr_state == RNDS_ENCODER_COUNTS_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_rnds_encoder_counts += line;
            }
        }
        else if (curr_state == RNDS_YAWS_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_rnds_yaws += line;
            }
        }
        else if (curr_state == STATES_X_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_states_x += line;
            }
        }
        else if (curr_state == STATES_Y_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_states_y += line;
            }
        }
        else if (curr_state == STATES_THETA_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_states_theta += line;
            }
        }
    }

    stringstream stream_rnds_encoder_counts(str_rnds_encoder_counts);
    vec_rnds_encoder_counts.resize(NUM_PARTICLES);
    for (int i = 0; std::getline(stream_rnds_encoder_counts, segment, ','); i++) {
        vec_rnds_encoder_counts[i] = std::stof(segment);
    }
    stringstream stream_rnds_yaw(str_rnds_yaws);
    vec_rnds_yaws.resize(NUM_PARTICLES);
    for (int i = 0; std::getline(stream_rnds_yaw, segment, ','); i++) {
        vec_rnds_yaws[i] = std::stof(segment);
    }
    stringstream stream_states_x(str_states_x);
    vec_states_x.resize(NUM_PARTICLES);
    for (int i = 0; std::getline(stream_states_x, segment, ','); i++) {
        vec_states_x[i] = std::stof(segment);
    }
    stringstream stream_states_y(str_states_y);
    vec_states_y.resize(NUM_PARTICLES);
    for (int i = 0; std::getline(stream_states_y, segment, ','); i++) {
        vec_states_y[i] = std::stof(segment);
    }
    stringstream stream_states_theta(str_states_theta);
    vec_states_theta.resize(NUM_PARTICLES);
    for (int i = 0; std::getline(stream_states_theta, segment, ','); i++) {
        vec_states_theta[i] = std::stof(segment);
    }

#ifdef ADD_HEADER_DATA
    int num_equals = 0;

    if (check_rnds_encoder_counts == true) {
        num_equals = 0;
        for (int i = 0; i < vec_rnds_encoder_counts.size(); i++) {
            if (vec_rnds_encoder_counts[i] != h_rnds_encoder_counts[i])
                printf("%f <> %f\n", vec_rnds_encoder_counts[i], h_rnds_encoder_counts[i]);
            else
                num_equals += 1;
        }
        printf("Rnds Encoder Counts Num Equals=%d, Num Errors=%d\n\n", num_equals, int(vec_rnds_encoder_counts.size() - num_equals));
    }
    if (check_rnds_yaws == true) {
        num_equals = 0;
        for (int i = 0; i < vec_rnds_yaws.size(); i++) {
            if (vec_rnds_yaws[i] != h_rnds_yaws[i])
                printf("%f <> %f\n", vec_rnds_yaws[i], h_rnds_yaws[i]);
            else
                num_equals += 1;
        }
        printf("Rnds Yaws Num Equals=%d, Num Errors=%d\n\n", num_equals, int(vec_rnds_yaws.size() - num_equals));
    }
    if (check_states == true) {
        num_equals = 0;
        for (int i = 0; i < vec_states_x.size(); i++) {
            if (vec_states_x[i] != h_states_x[i])
                printf("%f <> %f\n", vec_states_x[i], h_states_x[i]);
            else
                num_equals += 1;
        }
        printf("States X Num Equals=%d, Num Errors=%d\n\n", num_equals, int(vec_states_x.size() - num_equals));

        num_equals = 0;
        for (int i = 0; i < vec_states_y.size(); i++) {
            if (vec_states_y[i] != h_states_y[i])
                printf("%f <> %f\n", vec_states_y[i], h_states_y[i]);
            else
                num_equals += 1;
        }
        printf("States Y Num Equals=%d, Num Errors=%d\n\n", num_equals, int(vec_states_y.size() - num_equals));

        num_equals = 0;
        for (int i = 0; i < vec_states_theta.size(); i++) {
            if (vec_states_theta[i] != h_states_theta[i])
                printf("%f <> %f\n", vec_states_theta[i], h_states_theta[i]);
            else
                num_equals += 1;
        }
        printf("States Theta Num Equals=%d, Num Errors=%d\n\n", num_equals, int(vec_states_theta.size() - num_equals));
    }
#endif
}

void read_robot_data(int file_number,
    vector<float>& vec_robot_transition_world_body, vector<float>& vec_robot_state,
    vector<float>& vec_particles_weight_post, vector<float>& vec_rnds,
    bool check_robot_transition = true, bool check_state = true,
    bool check_particles_weight = true, bool check_rnds = true) {

    string file_name = std::to_string(file_number);

    const int ROBOT_TRANSITION_WORLD_BODY_VALUES = 2;
    const int ROBOT_STATE_VALUES = 3;
    const int PARTICLES_WEIGHT_POST_VALUES = 4;
    const int RNDS_VALUES = 5;
    const int SEPARATE_VALUES = 10;

    int curr_state = SEPARATE_VALUES;
    string str_robot_transition_world_body = "";
    string str_robot_state = "";
    string str_particles_weight_post = "";
    string str_rnds = "";
    string segment;

    std::ifstream data("data/steps/robot_" + file_name + ".txt");
    string line;

    while (getline(data, line)) {

        line = trim(line);

        if (line == "robot_transition_world_body") {
            curr_state = ROBOT_TRANSITION_WORLD_BODY_VALUES;
            continue;
        }
        else if (line == "robot_state") {
            curr_state = ROBOT_STATE_VALUES;
            continue;
        }
        else if (line == "particles_weight_post") {
            curr_state = PARTICLES_WEIGHT_POST_VALUES;
            continue;
        }
        else if (line == "rnds") {
            curr_state = RNDS_VALUES;
            continue;
        }

        if (curr_state == ROBOT_TRANSITION_WORLD_BODY_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_robot_transition_world_body += line;
            }
        }
        else if (curr_state == ROBOT_STATE_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_robot_state += line;
            }
        }
        else if (curr_state == PARTICLES_WEIGHT_POST_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_particles_weight_post += line;
            }
        }
        else if (curr_state == RNDS_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_rnds += line;
            }
        }
    }

    stringstream stream_robot_transition_world_body(str_robot_transition_world_body);
    vec_robot_transition_world_body.resize(9);
    for (int i = 0; std::getline(stream_robot_transition_world_body, segment, ','); i++) {
        vec_robot_transition_world_body[i] = std::stof(segment);
    }
    stringstream stream_robot_state(str_robot_state);
    vec_robot_state.resize(3);
    for (int i = 0; std::getline(stream_robot_state, segment, ','); i++) {
        vec_robot_state[i] = std::stof(segment);
    }
    stringstream stream_particles_weight_post(str_particles_weight_post);
    vec_particles_weight_post.resize(NUM_PARTICLES);
    for (int i = 0; std::getline(stream_particles_weight_post, segment, ','); i++) {
        vec_particles_weight_post[i] = std::stof(segment);
    }
    stringstream stream_rnds(str_rnds);
    vec_rnds.resize(NUM_PARTICLES);
    for (int i = 0; std::getline(stream_rnds, segment, ','); i++) {
        vec_rnds[i] = std::stof(segment);
    }

#ifdef ADD_HEADER_DATA
    int num_equals = 0;

    if (check_robot_transition == true) {
        num_equals = 0;
        for (int i = 0; i < vec_robot_transition_world_body.size(); i++) {
            if (vec_robot_transition_world_body[i] != h_robot_transition_world_body[i])
                printf("%f <> %f\n", vec_robot_transition_world_body[i], h_robot_transition_world_body[i]);
            else
                num_equals += 1;
        }
        printf("Robot Transition World Body Num Equals=%d, Num Errors=%d\n\n", num_equals, int(vec_robot_transition_world_body.size() - num_equals));
    }

    if (check_state == true) {
        num_equals = 0;
        for (int i = 0; i < vec_robot_state.size(); i++) {
            if (vec_robot_state[i] != h_robot_state[i])
                printf("%f <> %f\n", vec_robot_state[i], h_robot_state[i]);
            else
                num_equals += 1;
        }
        printf("Robot State Num Equals=%d, Num Errors=%d\n\n", num_equals, int(vec_robot_state.size() - num_equals));
    }

    if (check_particles_weight == true) {
        num_equals = 0;
        for (int i = 0; i < vec_particles_weight_post.size(); i++) {
            if (vec_particles_weight_post[i] != h_particles_weight_post[i])
                printf("%f <> %f\n", vec_particles_weight_post[i], h_particles_weight_post[i]);
            else
                num_equals += 1;
        }
        printf("Particles Weights Num Equals=%d, Num Errors=%d\n\n", num_equals, int(vec_particles_weight_post.size() - num_equals));
    }

    if (check_rnds == true) {
        num_equals = 0;
        for (int i = 0; i < vec_rnds.size(); i++) {
            if (vec_rnds[i] != h_rnds[i])
                printf("%f <> %f\n", vec_rnds[i], h_rnds[i]);
            else
                num_equals += 1;
        }
        printf("Rnds Num Equals=%d, Num Errors=%d\n\n", num_equals, int(vec_rnds.size() - num_equals));
    }
#endif
}

void read_robot_extra(int file_number,
    vector<int>& extra_grid_map, vector<int>& extra_particles_x, vector<int>& extra_particles_y, vector<int>& extra_particles_idx,
    vector<float>& extra_states_x, vector<float>& extra_states_y, vector<float>& extra_states_theta,
    vector<float>& extra_new_weights, vector<float>& extra_particles_weight_pre,
    int& EXTRA_GRID_WIDTH, int& EXTRA_GRID_HEIGHT, int& EXTRA_PARTICLES_ITEMS_LEN,
    bool check_grid_map = true, bool check_particles = true,
    bool check_states = true, bool check_weights = true) {

    string file_name = std::to_string(file_number);

    const int SCALAR_VALUES = 1;
    const int GRID_MAP_VALUES = 2;
    const int PARTICLES_X_VALUES = 3;
    const int PARTICLES_Y_VALUES = 4;
    const int PARTICLES_IDX_VALUES = 5;
    const int STATES_X_VALUES = 6;
    const int STATES_Y_VALUES = 7;
    const int STATES_THETA_VALUES = 8;
    const int NEW_WEIGHTS_VALUES = 9;
    const int PARTICLES_WEIGHT_VALUES = 10;
    const int SEPARATE_VALUES = 11;

    int curr_state = SCALAR_VALUES;
    string str_grid_map = "";
    string str_particles_x = "";
    string str_particles_y = "";
    string str_particles_idx = "";
    string str_states_x = "";
    string str_states_y = "";
    string str_states_theta = "";
    string str_new_weights = "";
    string str_particles_weight = "";
    string segment;

    std::ifstream data("data/extra/robot_" + file_name + ".txt");
    string line;

    while (getline(data, line)) {

        line = trim(line);
        if (line == "") continue;

        if (curr_state == SCALAR_VALUES) {

            if (line == "GRID_WIDTH") {
                getline(data, line);
                EXTRA_GRID_WIDTH = std::stoi(line);
            }
            else if (line == "GRID_HEIGHT") {
                getline(data, line);
                EXTRA_GRID_HEIGHT = std::stoi(line);
            }
            else if (line == "PARTICLES_ITEMS_LEN") {
                getline(data, line);
                EXTRA_PARTICLES_ITEMS_LEN = std::stoi(line);
            }
        }

        if (line == "grid_map") {
            curr_state = GRID_MAP_VALUES;
            continue;
        }
        else if (line == "particles_x") {
            curr_state = PARTICLES_X_VALUES;
            continue;
        }
        else if (line == "particles_y") {
            curr_state = PARTICLES_Y_VALUES;
            continue;
        }
        else if (line == "particles_idx") {
            curr_state = PARTICLES_IDX_VALUES;
            continue;
        }
        else if (line == "states_x") {
            curr_state = STATES_X_VALUES;
            continue;
        }
        else if (line == "states_y") {
            curr_state = STATES_Y_VALUES;
            continue;
        }
        else if (line == "states_theta") {
            curr_state = STATES_THETA_VALUES;
            continue;
        }
        else if (line == "new_weights") {
            curr_state = NEW_WEIGHTS_VALUES;
            continue;
        }
        else if (line == "particles_weight_pre") {
            curr_state = PARTICLES_WEIGHT_VALUES;
            continue;
        }

        if (curr_state == GRID_MAP_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_grid_map += line;
            }
        }
        else if (curr_state == PARTICLES_X_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_particles_x += line;
            }
        }
        else if (curr_state == PARTICLES_Y_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_particles_y += line;
            }
        }
        else if (curr_state == PARTICLES_IDX_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_particles_idx += line;
            }
        }
        else if (curr_state == STATES_X_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_states_x += line;
            }
        }
        else if (curr_state == STATES_Y_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_states_y += line;
            }
        }
        else if (curr_state == STATES_THETA_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_states_theta += line;
            }
        }
        else if (curr_state == NEW_WEIGHTS_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_new_weights += line;
            }
        }
        else if (curr_state == PARTICLES_WEIGHT_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_particles_weight += line;
            }
        }

    }

    int GRID_SIZE = EXTRA_GRID_WIDTH * EXTRA_GRID_HEIGHT;

    stringstream stream_grid_map(str_grid_map);
    extra_grid_map.resize(GRID_SIZE);
    for (int i = 0; std::getline(stream_grid_map, segment, ','); i++) {
        extra_grid_map[i] = std::stoi(segment);
    }
    stringstream stream_particles_x(str_particles_x);
    extra_particles_x.resize(EXTRA_PARTICLES_ITEMS_LEN);
    for (int i = 0; std::getline(stream_particles_x, segment, ','); i++) {
        extra_particles_x[i] = std::stoi(segment);
    }
    stringstream stream_particles_y(str_particles_y);
    extra_particles_y.resize(EXTRA_PARTICLES_ITEMS_LEN);
    for (int i = 0; std::getline(stream_particles_y, segment, ','); i++) {
        extra_particles_y[i] = std::stoi(segment);
    }
    stringstream stream_particles_idx(str_particles_idx);
    extra_particles_idx.resize(NUM_PARTICLES);
    for (int i = 0; std::getline(stream_particles_idx, segment, ','); i++) {
        extra_particles_idx[i] = std::stoi(segment);
    }
    stringstream stream_states_x(str_states_x);
    extra_states_x.resize(NUM_PARTICLES);
    for (int i = 0; std::getline(stream_states_x, segment, ','); i++) {
        extra_states_x[i] = std::stof(segment);
    }
    stringstream stream_states_y(str_states_y);
    extra_states_y.resize(NUM_PARTICLES);
    for (int i = 0; std::getline(stream_states_y, segment, ','); i++) {
        extra_states_y[i] = std::stof(segment);
    }
    stringstream stream_states_theta(str_states_theta);
    extra_states_theta.resize(NUM_PARTICLES);
    for (int i = 0; std::getline(stream_states_theta, segment, ','); i++) {
        extra_states_theta[i] = std::stof(segment);
    }
    stringstream stream_new_weights(str_new_weights);
    extra_new_weights.resize(NUM_PARTICLES);
    for (int i = 0; std::getline(stream_new_weights, segment, ','); i++) {
        extra_new_weights[i] = std::stof(segment);
    }
    stringstream stream_particles_weight_pre(str_particles_weight);
    extra_particles_weight_pre.resize(NUM_PARTICLES);
    for (int i = 0; std::getline(stream_particles_weight_pre, segment, ','); i++) {
        extra_particles_weight_pre[i] = std::stof(segment);
    }

#ifdef ADD_HEADER_DATA
    int num_equals = 0;

    if (check_grid_map == true) {
        num_equals = 0;
        for (int i = 0; i < extra_grid_map.size(); i++) {
            if (extra_grid_map[i] != h_grid_map[i])
                printf("%d <> %d\n", extra_grid_map[i], h_grid_map[i]);
            else
                num_equals += 1;
        }
        printf("Grid Map Num Equals=%d, Num Errors=%d\n\n", num_equals, int(extra_grid_map.size() - num_equals));
    }

    if (check_particles == true) {
        num_equals = 0;
        for (int i = 0; i < extra_particles_x.size(); i++) {
            if (extra_particles_x[i] != h_particles_x[i])
                printf("%d <> %d\n", extra_particles_x[i], h_particles_x[i]);
            else
                num_equals += 1;
        }
        printf("Particles X Num Equals=%d, Num Errors=%d\n\n", num_equals, int(extra_particles_x.size() - num_equals));

        num_equals = 0;
        for (int i = 0; i < extra_particles_y.size(); i++) {
            if (extra_particles_y[i] != h_particles_y[i])
                printf("%d <> %d\n", extra_particles_x[i], h_particles_y[i]);
            else
                num_equals += 1;
        }
        printf("Particles Y Num Equals=%d, Num Errors=%d\n\n", num_equals, int(extra_particles_y.size() - num_equals));

        num_equals = 0;
        for (int i = 0; i < extra_particles_idx.size(); i++) {
            if (extra_particles_idx[i] != h_particles_idx[i])
                printf("%d <> %d\n", extra_particles_idx[i], h_particles_idx[i]);
            else
                num_equals += 1;
        }
        printf("Particles Idx Num Equals=%d, Num Errors=%d\n\n", num_equals, int(extra_particles_idx.size() - num_equals));
    }

    if (check_states == true) {
        num_equals = 0;
        for (int i = 0; i < extra_states_x.size(); i++) {
            if (extra_states_x[i] != post_states_x[i])
                printf("%f <> %f\n", extra_states_x[i], post_states_x[i]);
            else
                num_equals += 1;
        }
        printf("States X Num Equals=%d, Num Errors=%d\n\n", num_equals, int(extra_states_x.size() - num_equals));

        num_equals = 0;
        for (int i = 0; i < extra_states_y.size(); i++) {
            if (extra_states_y[i] != post_states_y[i])
                printf("%f <> %f\n", extra_states_y[i], post_states_y[i]);
            else
                num_equals += 1;
        }
        printf("States Y Num Equals=%d, Num Errors=%d\n\n", num_equals, int(extra_states_y.size() - num_equals));

        num_equals = 0;
        for (int i = 0; i < extra_states_theta.size(); i++) {
            if (extra_states_theta[i] != post_states_theta[i])
                printf("%f <> %f\n", extra_states_theta[i], post_states_theta[i]);
            else
                num_equals += 1;
        }
        printf("States Theta Num Equals=%d, Num Errors=%d\n\n", num_equals, int(extra_states_theta.size() - num_equals));
    }

    if (check_weights == true) {
        num_equals = 0;
        for (int i = 0; i < extra_new_weights.size(); i++) {
            if (extra_new_weights[i] != h_new_weights[i])
                printf("%f <> %f\n", extra_new_weights[i], h_new_weights[i]);
            else
                num_equals += 1;
        }
        printf("New Weights Num Equals=%d, Num Errors=%d\n\n", num_equals, int(extra_new_weights.size() - num_equals));

        num_equals = 0;
        for (int i = 0; i < extra_particles_weight_pre.size(); i++) {
            if (extra_particles_weight_pre[i] != h_particles_weight_pre[i])
                printf("%f <> %f\n", extra_particles_weight_pre[i], h_particles_weight_pre[i]);
            else
                num_equals += 1;
        }
        printf("Particles Weight Num Equals=%d, Num Errors=%d\n\n", num_equals, int(extra_particles_weight_pre.size() - num_equals));
    }
#endif
}

void read_map_data(int file_number,
    vector<int>& vec_grid_map, vector<float>& vec_log_odds, vector<float>& vec_lidar_coords,
    int& NEW_GRID_WIDTH, int& NEW_GRID_HEIGHT, int& NEW_LIDAR_COORDS_LEN,
    bool check_grid_map = true, bool check_log_odds = true, bool check_lidar_coords = true) {

    const int SCALAR_VALUES = 1;
    const int GRID_MAP_VALUES = 2;
    const int LOG_ODDS_VALUES = 3;
    const int LIDAR_COORDS_VALUES = 4;
    const int SEPARATE_VALUES = 10;

    int curr_state = SCALAR_VALUES;
    string str_grid_map = "";
    string str_log_odds = "";
    string str_lidar_coords = "";
    string segment;

    string file_name = std::to_string(file_number);

    std::ifstream data("data/steps/map_" + file_name + ".txt");
    string line;

    while (getline(data, line)) {

        line = trim(line);

        if (curr_state == SCALAR_VALUES) {

            if (line == "GRID_WIDTH") {
                getline(data, line);
                NEW_GRID_WIDTH = std::stoi(line);
            }
            else if (line == "GRID_HEIGHT") {
                getline(data, line);
                NEW_GRID_HEIGHT = std::stoi(line);
            }
            else if (line == "LIDAR_COORDS_LEN") {
                getline(data, line);
                NEW_LIDAR_COORDS_LEN = std::stoi(line);
            }
        }

        if (line == "grid_map") {
            curr_state = GRID_MAP_VALUES;
            continue;
        }
        else if (line == "log_odds") {
            curr_state = LOG_ODDS_VALUES;
            continue;
        }
        else if (line == "lidar_coords") {
            curr_state = LIDAR_COORDS_VALUES;
            continue;
        }

        if (curr_state == GRID_MAP_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_grid_map += line;
            }
        }
        else if (curr_state == LOG_ODDS_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_log_odds += line;
            }
        }
        else if (curr_state == LIDAR_COORDS_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_lidar_coords += line;
            }
        }
    }

    int GRID_SIZE = NEW_GRID_WIDTH * NEW_GRID_HEIGHT;

    stringstream stream_grid_map(str_grid_map);
    vec_grid_map.resize(GRID_SIZE);
    for (int i = 0; std::getline(stream_grid_map, segment, ','); i++) {
        vec_grid_map[i] = std::stoi(segment);
    }
    stringstream stream_log_odds(str_log_odds);
    vec_log_odds.resize(GRID_SIZE);
    for (int i = 0; std::getline(stream_log_odds, segment, ','); i++) {
        vec_log_odds[i] = std::stof(segment);
    }
    stringstream stream_lidar_coords(str_lidar_coords);
    vec_lidar_coords.resize(2 * NEW_LIDAR_COORDS_LEN);
    for (int i = 0; std::getline(stream_lidar_coords, segment, ','); i++) {
        vec_lidar_coords[i] = std::stof(segment);
    }

    int num_equals = 0;

#ifdef ADD_HEADER_DATA
    if (check_grid_map == true) {
        for (int i = 0; i < vec_grid_map.size(); i++) {
            if (vec_grid_map[i] != h_post_grid_map[i])
                printf("%d <> %d\n", vec_grid_map[i], h_post_grid_map[i]);
            else
                num_equals += 1;
        }
        printf("Grid Map Num Equals=%d\n\n", num_equals);
    }

    if (check_log_odds == true) {
        num_equals = 0;
        for (int i = 0; i < vec_log_odds.size(); i++) {
            if (vec_log_odds[i] != h_post_log_odds[i])
                printf("%f <> %f\n", vec_log_odds[i], h_post_log_odds[i]);
            else
                num_equals += 1;
        }
        printf("Log Odds Num Equals=%d, Num Errors=%d\n\n", num_equals, int(vec_log_odds.size() - num_equals));
    }
    if (check_lidar_coords == true) {
        num_equals = 0;
        for (int i = 0; i < 2 * NEW_LIDAR_COORDS_LEN; i++) {
            if (vec_lidar_coords[i] != h_lidar_coords[i])
                printf("%f <> %f\n", vec_lidar_coords[i], h_lidar_coords[i]);
            else
                num_equals += 1;
        }
        printf("LIDAR Coords Num Equals=%d, Num Errors=%d\n\n", num_equals, int(vec_lidar_coords.size() - num_equals));
    }
#endif
}

void read_map_extra(int file_number,
    vector<int>& extra_grid_map, vector<float>& extra_log_odds, vector<float>& extra_transition_single_world_body,
    int& extra_xmin, int& extra_xmax, int& extra_ymin, int& extra_ymax, float& extra_res, float& extra_log_t,
    int& EXTRA_GRID_WIDTH, int& EXTRA_GRID_HEIGHT,
    const int NEW_GRID_WIDTH, const int NEW_GRID_HEIGHT,
    bool check_grid_map = true, bool check_log_odds = true, bool check_transition = true) {

    const int SCALAR_VALUES = 1;
    const int GRID_MAP_VALUES = 2;
    const int LOG_ODDS_VALUES = 3;
    const int TRANSITION_VALUES = 4;
    const int SEPARATE_VALUES = 10;

    int curr_state = SCALAR_VALUES;
    string str_grid_map = "";
    string str_log_odds = "";
    string str_transition = "";
    string segment;

    string file_name = std::to_string(file_number);

    std::ifstream data("data/extra/map_" + file_name + ".txt");
    string line;

    while (getline(data, line)) {

        line = trim(line);

        if (curr_state == SCALAR_VALUES) {

            if (line == "GRID_WIDTH") {
                getline(data, line);
                EXTRA_GRID_WIDTH = std::stoi(line);
            }
            else if (line == "GRID_HEIGHT") {
                getline(data, line);
                EXTRA_GRID_HEIGHT = std::stoi(line);
            }
            else if (line == "xmin") {
                getline(data, line);
                extra_xmin = std::stoi(line);
            }
            else if (line == "xmax") {
                getline(data, line);
                extra_xmax = std::stoi(line);
            }
            else if (line == "ymin") {
                getline(data, line);
                extra_ymin = std::stoi(line);
            }
            else if (line == "ymax") {
                getline(data, line);
                extra_ymax = std::stoi(line);
            }
            else if (line == "res") {
                getline(data, line);
                extra_res = std::stof(line);
            }
            else if (line == "log_t") {
                getline(data, line);
                extra_log_t = std::stof(line);
            }
        }

        if (line == "grid_map") {
            curr_state = GRID_MAP_VALUES;
            continue;
        }
        else if (line == "log_odds") {
            curr_state = LOG_ODDS_VALUES;
            continue;
        }
        else if (line == "transition_single_world_body") {
            curr_state = TRANSITION_VALUES;
            continue;
        }

        if (curr_state == GRID_MAP_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_grid_map += line;
            }
        }
        else if (curr_state == LOG_ODDS_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_log_odds += line;
            }
        }
        else if (curr_state == TRANSITION_VALUES) {
            if (line == "SEPARATE") {
                curr_state = SEPARATE_VALUES;
            }
            else {
                str_transition += line;
            }
        }
    }

    int GRID_SIZE = NEW_GRID_WIDTH * NEW_GRID_HEIGHT;

    stringstream stream_grid_map(str_grid_map);
    extra_grid_map.resize(GRID_SIZE);
    for (int i = 0; std::getline(stream_grid_map, segment, ','); i++) {
        extra_grid_map[i] = std::stoi(segment);
    }
    stringstream stream_log_odds(str_log_odds);
    extra_log_odds.resize(GRID_SIZE);
    for (int i = 0; std::getline(stream_log_odds, segment, ','); i++) {
        extra_log_odds[i] = std::stof(segment);
    }
    stringstream stream_transition(str_transition);
    extra_transition_single_world_body.resize(9);
    for (int i = 0; std::getline(stream_transition, segment, ','); i++) {
        extra_transition_single_world_body[i] = std::stof(segment);
    }

    int num_equals = 0;

#ifdef ADD_HEADER_DATA
    if (check_grid_map == true) {
        for (int i = 0; i < extra_grid_map.size(); i++) {
            if (extra_grid_map[i] != h_grid_map[i])
                printf("%d <> %d\n", extra_grid_map[i], h_grid_map[i]);
            else
                num_equals += 1;
        }
        printf("Extra Grid Map Num Equals=%d\n\n", num_equals);
    }

    if (check_log_odds == true) {
        num_equals = 0;
        for (int i = 0; i < extra_log_odds.size(); i++) {
            if (extra_log_odds[i] != h_log_odds[i])
                printf("%f <> %f\n", extra_log_odds[i], h_log_odds[i]);
            else
                num_equals += 1;
        }
        printf("Extra Log Odds Num Equals=%d, Num Errors=%d\n\n", num_equals, int(extra_log_odds.size() - num_equals));
    }

    if (check_transition == true) {
        num_equals = 0;
        for (int i = 0; i < extra_transition_single_world_body.size(); i++) {
            if (extra_transition_single_world_body[i] != h_transition_single_world_body[i])
                printf("%f <> %f\n", extra_transition_single_world_body[i], h_transition_single_world_body[i]);
            else
                num_equals += 1;
        }
        printf("Transition World Body Num Equals=%d, Num Errors=%d\n\n", num_equals, int(extra_transition_single_world_body.size() - num_equals));
    }
#endif


}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

#endif