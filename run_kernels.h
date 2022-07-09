#ifndef _RUN_KERNELS_H_
#define _RUN_KERNELS_H_

#include "headers.h"
#include "host_utils.h"
#include "kernels.cuh"
#include "kernels_robot.cuh"
#include "kernels_map.cuh"
#include "kernels_utils.cuh"
#include "gl_draw_utils.h"
#include "device_init_common.h"
#include "device_init_robot.h"
#include "device_init_map.h"
#include "device_exec_robot.h"
#include "device_exec_map.h"
#include "device_assert_robot.h"
#include "device_assert_map.h"
#include "device_set_reset_map.h"
#include "device_set_reset_robot.h"


Window mainWindow;
vector<Shader> shader_list;
Camera camera;

vector<Mesh*> freeList;
vector<Mesh*> wallList;

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

void run_robot(DeviceRobot& d_robot, DeviceMapping& d_mapping, DeviceLidar& d_lidar,
    HostRobot& h_robot, HostMapping& h_mapping, HostLidar& h_lidar, GeneralInfo& general_info) {

    const int MEASURE_LEN = NUM_PARTICLES * h_lidar.measurements.LEN;
    int* h_last_len = (int*)malloc(sizeof(int));

    exec_robot_move(d_robot.state, h_robot.state);

    exec_calc_transition(d_robot.particles_transition, d_robot.state, d_robot.transition, h_robot.particles_transition);
    exec_process_measurements(d_lidar.processed_measure, d_robot.particles_transition, d_lidar.measurements, h_mapping.map, h_lidar.measurements, general_info);

    exec_create_2d_map(d_robot._2d_unique, d_robot.robot_particles, h_mapping.map, h_robot.robot_particles);

    exec_update_map(d_robot._2d_unique, d_lidar.processed_measure, h_mapping.map, MEASURE_LEN);
    exec_particle_unique_cum_sum(d_robot._2d_unique, h_mapping.map, h_robot._2d_unique, h_robot.robot_particles);

    reinit_map_vars(d_robot.robot_particles, h_robot.robot_particles);
    exec_map_restructure(d_robot.robot_particles, d_robot._2d_unique, h_mapping.map);

    exec_index_expansion(d_robot.robot_particles, h_robot.robot_particles);
    exec_correlation(d_mapping.map, d_robot.robot_particles, d_robot.correlation, h_mapping.map, h_robot.robot_particles);

    exec_update_weights(d_robot.robot_particles, d_robot.correlation, h_robot.robot_particles, h_robot.correlation);

    exec_resampling(d_robot.correlation, d_robot.resampling);
    reinit_particles_vars(d_robot.state, d_robot.robot_particles, d_robot.resampling, d_robot.clone_robot_particles, d_robot.clone_state, h_robot.robot_particles, h_robot.state, h_last_len);

    exec_rearrangement(d_robot.robot_particles, d_robot.state, d_robot.resampling, d_robot.clone_robot_particles, d_robot.clone_state, h_mapping.map,
        h_robot.robot_particles, h_robot.clone_robot_particles, h_last_len);
    exec_update_states(d_robot.state, d_robot.transition, h_robot.state, h_robot.robot_state);
}

void run_map(DeviceRobot& d_robot, DeviceMapping& d_mapping, DeviceLidar& d_lidar,
    HostRobot& h_robot, HostMapping& h_mapping, HostLidar& h_lidar, GeneralInfo& general_info) {

    exec_world_to_image_transform_step_1(d_mapping.position, d_robot.transition, d_mapping.particles, d_lidar.measurements, h_lidar.measurements);

    bool EXTEND = false;
    exec_map_extend(d_mapping.map, d_lidar.measurements, d_mapping.particles, d_mapping.unique_occupied, d_mapping.unique_free,
        h_mapping.map, h_lidar.measurements, h_mapping.unique_occupied, h_mapping.unique_free, general_info, EXTEND);

    exec_world_to_image_transform_step_2(d_lidar.measurements, d_mapping.particles, d_mapping.position, d_robot.transition,
        h_mapping.map, h_lidar.measurements, general_info);

    int MAX_DIST_IN_MAP = sqrt(pow(h_mapping.map.GRID_WIDTH, 2) + pow(h_mapping.map.GRID_HEIGHT, 2));

    exec_bresenham(d_mapping.particles, d_mapping.position, d_robot.transition, h_mapping.particles, MAX_DIST_IN_MAP);

    reinit_map_idx_vars(d_mapping.unique_free, h_mapping.particles, h_mapping.unique_free);
    exec_create_map(d_mapping.particles, d_mapping.unique_occupied, d_mapping.unique_free, h_mapping.map, h_mapping.particles);

    reinit_map_vars(d_mapping.particles, d_mapping.unique_occupied, d_mapping.unique_free, h_mapping.particles, h_mapping.unique_occupied, h_mapping.unique_free);
    exec_map_restructure(d_mapping.particles, d_mapping.unique_occupied, d_mapping.unique_free, h_mapping.map);

    exec_log_odds(d_mapping.map, d_mapping.particles, h_mapping.map, h_mapping.particles, general_info);
}


void run_init_general_info(GeneralInfo& general_info) {

    general_info.log_t = 1.3862943611198908;
    general_info.res = 0.1;
}

void run_init_pre_map(HostMap& pre_map, const GeneralInfo& general_info) {

    pre_map.xmin = -10;
    pre_map.xmax = 15;
    pre_map.ymin = -10;
    pre_map.ymax = 15;
    pre_map.b_should_extend = false;
    pre_map.GRID_WIDTH = ceil((pre_map.ymax - pre_map.ymin) / general_info.res + 1);
    pre_map.GRID_HEIGHT = ceil((pre_map.xmax - pre_map.xmin) / general_info.res + 1);
    pre_map.s_grid_map.resize(pre_map.GRID_WIDTH * pre_map.GRID_HEIGHT, 0);
    pre_map.s_log_odds.resize(pre_map.GRID_WIDTH * pre_map.GRID_HEIGHT, LOG_ODD_PRIOR);
}

void run_init_pre_measurements(HostMeasurements& pre_measurements, const vector<double>& vec_lidar_coords) {

    pre_measurements.LEN = vec_lidar_coords.size() / 2;
    pre_measurements.v_lidar_coords.resize(2 * pre_measurements.LEN, 0);
    thrust::copy(vec_lidar_coords.begin(), vec_lidar_coords.end(), pre_measurements.v_lidar_coords.begin());
    vector<int> init_coord = { 0, 0 };
    pre_measurements.c_coord.resize(init_coord.size());
    pre_measurements.c_coord.assign(init_coord.begin(), init_coord.end());
}

void run_init_pre_transition(HostTransition& pre_transition) {

    vector<float> init_world_body = { 0.99989295,-0.014631824,0.0118173845,0.014631824,0.99989295,0.0001729284,0.0,0.0,1.0 };
    pre_transition.c_world_body.resize(init_world_body.size());
    pre_transition.c_world_body.assign(init_world_body.begin(), init_world_body.end());
}

void run_init_pre_state(HostState& pre_state, const double& encoder_counts, const double& yaws, const double& dt) {

    pre_state.encoder_counts = encoder_counts;
    pre_state.yaw = yaws;
    pre_state.dt = dt;
    pre_state.nv = ST_nv;
    pre_state.nw = ST_nw;

    host_vector<float> vec_init(NUM_PARTICLES, 0);
    pre_state.c_x.assign(vec_init.begin(), vec_init.end());
    pre_state.c_y.assign(vec_init.begin(), vec_init.end());
    pre_state.c_theta.assign(vec_init.begin(), vec_init.end());
}

void run_init_pre_robot_particles(HostRobotParticles& pre_robot_particles) {

    pre_robot_particles.LEN = 0;
    pre_robot_particles.c_weight.resize(NUM_PARTICLES, 1.0f);
    pre_robot_particles.c_idx.resize(NUM_PARTICLES, 0);
}

void run_init(DeviceRobot& d_robot, DeviceMapping& d_mapping, DeviceLidar& d_lidar,
    HostRobot& h_robot, HostMapping& h_mapping, HostLidar& h_lidar, GeneralInfo& general_info,
    vector<double>& vec_lidar_coords, double& encoder_counts, double& yaws, double& dt,
    host_vector<int>& hvec_occupied_map_idx, host_vector<int>& hvec_free_map_idx, const int LIDAR_COORDS_LEN) {

    HostState pre_state;
    HostRobotParticles pre_robot_particles;
    HostMap pre_map;
    HostMeasurements pre_measurements;
    HostTransition pre_transition;
    HostParticles pre_particles;

    DeviceParticlesPosition d_particles_position;
    DeviceParticlesRotation d_particles_rotation;
    HostParticlesPosition h_particles_position;
    HostParticlesRotation h_particles_rotation;

    run_init_general_info(general_info);
    run_init_pre_map(pre_map, general_info);
    pre_particles.OCCUPIED_LEN = LIDAR_COORDS_LEN;
    run_init_pre_measurements(pre_measurements, vec_lidar_coords);
    run_init_pre_transition(pre_transition);
    run_init_pre_state(pre_state, encoder_counts, yaws, dt);
    run_init_pre_robot_particles(pre_robot_particles);

    alloc_init_state_vars(d_robot.state, d_robot.clone_state, h_robot.state, h_robot.robot_state, pre_state);
    alloc_init_measurement_vars(d_lidar.measurements, h_lidar.measurements, pre_measurements);
    alloc_init_map_vars(d_mapping.map, h_mapping.map, pre_map);
    alloc_init_robot_particles_vars(d_robot.robot_particles, d_robot.clone_robot_particles, h_robot.robot_particles, pre_robot_particles);
    alloc_correlation_vars(d_robot.correlation, h_robot.correlation);
    alloc_particles_transition_vars(d_robot.particles_transition, d_particles_position, d_particles_rotation,
        h_robot.particles_transition, h_particles_position, h_particles_rotation);
    alloc_init_body_lidar(d_robot.transition);
    alloc_init_processed_measurement_vars(d_lidar.processed_measure, h_lidar.processed_measure, h_lidar.measurements);
    alloc_map_2d_var(d_robot._2d_unique, h_robot._2d_unique, h_mapping.map);
    alloc_resampling_vars(d_robot.resampling, h_robot.resampling);

    hvec_occupied_map_idx.resize(2, 0);
    hvec_free_map_idx.resize(2, 0);

    alloc_init_transition_vars(d_mapping.position, d_robot.transition, h_mapping.position, h_mapping.transition, pre_transition);
    int MAX_DIST_IN_MAP = sqrt(pow(pre_map.GRID_WIDTH, 2) + pow(pre_map.GRID_HEIGHT, 2));
    alloc_init_particles_vars(d_mapping.particles, h_mapping.particles, h_lidar.measurements, pre_particles, MAX_DIST_IN_MAP);
    hvec_occupied_map_idx[1] = h_mapping.particles.OCCUPIED_LEN;
    hvec_free_map_idx[1] = 0;
    alloc_init_unique_map_vars(d_mapping.unique_occupied, h_mapping.unique_occupied, h_mapping.map, hvec_occupied_map_idx);
    alloc_init_unique_map_vars(d_mapping.unique_free, h_mapping.unique_free, h_mapping.map, hvec_free_map_idx);
}

int THR_GRID_WIDTH = 0;
int THR_GRID_HEIGHT = 0;
HostMap thr_map;

timed_mutex timed_mutex_draw;

void thread_draw() {

    GLfloat delta_time = 0.0f;
    GLfloat last_time = 0.0f;

    Light main_light;

    int CURR_GRID_WIDTH = 0;
    int CURR_GRID_HEIGHT = 0;

    while (timed_mutex_draw.try_lock_until(std::chrono::steady_clock::now() + std::chrono::seconds(1)) == false);
    printf("Draw Thread Started ...\n");

    CURR_GRID_WIDTH = THR_GRID_WIDTH;
    CURR_GRID_HEIGHT = THR_GRID_HEIGHT;
    printf("CURR_GRID_WIDTH: %d, CURR_GRID_HEIGHT: %d\n", CURR_GRID_WIDTH, CURR_GRID_HEIGHT);

    // Vertex Shader
    static const char* vShader = "Shaders/shader.vert";

    // Fragment Shader
    static const char* fShader = "Shaders/shader.frag";

    mainWindow.initialize();

    //gpuErrchk(cudaMemcpy(res_grid_map, d_grid_map, sz_grid_map, cudaMemcpyDeviceToHost));

    CreateObjects(freeList, thr_map.s_grid_map.data(), CURR_GRID_WIDTH * CURR_GRID_HEIGHT, 1, 0.1f, CURR_GRID_HEIGHT);
    CreateObjects(wallList, thr_map.s_grid_map.data(), CURR_GRID_WIDTH * CURR_GRID_HEIGHT, 2, 0.5f, CURR_GRID_HEIGHT);
    CreateShaders(shader_list, vShader, fShader);

    camera = Camera(glm::vec3(-3.0f, 12.0f, 23.0f), glm::vec3(0.0f, 1.0f, 0.0f), -53.0f, -42.0f, 5.0f, 0.1f);

    main_light = Light(1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    GLuint uniformModel = 0, uniformProjection = 0, uniformView = 0, uniformColor = 0,
        uniformAmbientIntensity = 0, uniformAmbientColor = 0;
    glm::mat4 projection = glm::perspective(45.0f,
        (GLfloat)mainWindow.getBufferWidth() / (GLfloat)mainWindow.getBufferHeight(), 0.1f, 90.0f);

    // Loop until windows closed
    while (!mainWindow.getShouldClose()) {

        if (timed_mutex_draw.try_lock_until(std::chrono::steady_clock::now() + std::chrono::milliseconds(10)) == true) {
            
            CURR_GRID_WIDTH = THR_GRID_WIDTH;
            CURR_GRID_HEIGHT = THR_GRID_HEIGHT;
            // printf("CURR_GRID_WIDTH: %d, CURR_GRID_HEIGHT: %d\n", CURR_GRID_WIDTH, CURR_GRID_HEIGHT);
            // camera.printInfo();

            freeList.clear();
            wallList.clear();

            CreateObjects(freeList, thr_map.s_grid_map.data(), CURR_GRID_WIDTH * CURR_GRID_HEIGHT, 1, 0.1f, CURR_GRID_HEIGHT);
            CreateObjects(wallList, thr_map.s_grid_map.data(), CURR_GRID_WIDTH * CURR_GRID_HEIGHT, 2, 0.5f, CURR_GRID_HEIGHT);
        }

        GLfloat now = glfwGetTime();
        delta_time = now - last_time;
        last_time = now;

        // Get+Handle user inputs
        glfwPollEvents();

        camera.keyControl(mainWindow.getskeys(), delta_time);
        camera.mouseControl(mainWindow.getXChange(), mainWindow.getYChange(), delta_time);

        // Clear window
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        shader_list[0].UseShader();
        uniformModel = shader_list[0].GetModelLocation();
        uniformProjection = shader_list[0].GetProjectionLocation();
        uniformView = shader_list[0].GetViewLocation();
        uniformColor = shader_list[0].GetColorLocation();
        uniformAmbientColor = shader_list[0].GetAmbientColorLocation();
        uniformAmbientIntensity = shader_list[0].GetAmbientIntensityLocation();

        main_light.UseLight(uniformAmbientIntensity, uniformAmbientColor, 0.0f, 0.0f);

        glm::mat4 model = glm::identity<glm::mat4>();

        model = glm::translate(model, glm::vec3(0.0f, 0.0f, -5.0f));
        //model = glm::scale(model, glm::vec3(0.4f, 0.4f, 0.4f));
        glUniformMatrix4fv(uniformModel, 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(uniformProjection, 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(uniformView, 1, GL_FALSE, glm::value_ptr(camera.calculateViewMatrix()));

        glUniform4f(uniformColor, 0.8f, 0.8f, 0.8f, 1.0f);
        for (int i = 0; i < freeList.size(); i++) {
            freeList[i]->RenderMesh();
        }

        glUniform4f(uniformColor, 0.0f, 0.2f, 0.9f, 1.0f);
        for (int i = 0; i < wallList.size(); i++) {
            wallList[i]->RenderMesh();
        }

        glUseProgram(0);

        mainWindow.swapBuffers();
    }
}

void run_reset_middle_variables(DeviceCorrelation& d_correlation, DeviceProcessedMeasure& d_processed_measure, DeviceResampling& d_resampling,
    Device2DUniqueFinder& d_2d_unique, DeviceRobotParticles& d_robot_particles,
    HostMap& h_map, HostMeasurements& h_measurements, HostProcessedMeasure& h_processed_measure) {

    int num_items = NUM_PARTICLES * h_measurements.LEN;
    reset_processed_measure(d_processed_measure, h_measurements);

    thrust::fill(d_2d_unique.s_map.begin(), d_2d_unique.s_map.end(), 0);
    thrust::fill(d_2d_unique.c_in_map.begin(), d_2d_unique.c_in_map.end(), 0);
    thrust::fill(d_2d_unique.s_in_col.begin(), d_2d_unique.s_in_col.end(), 0);

    reset_correlation(d_correlation);

    thrust::fill(d_resampling.c_js.begin(), d_resampling.c_js.end(), 0);
    thrust::fill(d_robot_particles.c_weight.begin(), d_robot_particles.c_weight.end(), 0);
}

void run_main() {

    std::cout << "Run Application" << std::endl;

    vector<double> vec_rnds_encoder_counts;
    vector<double> vec_rnds_yaws;
    vector<double> vec_rnds;

    printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
    printf("Reading Data Files\n");
    printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");


    GeneralInfo general_info;

    DeviceRobot d_robot;
    DeviceLidar d_lidar;
    DeviceMapping d_mapping;

    HostRobot h_robot;
    HostLidar h_lidar;
    HostMapping h_mapping;

    host_vector<int> hvec_occupied_map_idx;
    host_vector<int> hvec_free_map_idx;

    const int INFO_STEP = 1000;
    const int DRAW_STEP = 100;
    int PRE_GRID_SIZE = 0;

    std::thread t(thread_draw);
    lock_guard<timed_mutex> l(timed_mutex_draw);

    const int LIDAR_COORDS_MAX_LEN = 2200;
    int lidar_idx = 0;
    auto start_read_file = std::chrono::high_resolution_clock::now();

    string lidar_path = "data_meas/meas_lidar.npy";
    if (fs::is_regular_file(lidar_path) == false) {

        std::cerr << "[Error] The Measurement File does not Exists." << std::endl;
        std::cerr << "[Error] Make sure to prepare Measurement File with this path: 'data_meas/meas_lidar.npy'" << std::endl;
        exit(-1);
    }
    cnpy::NpyArray arr_lidar_coords = cnpy::npy_load(lidar_path);
    vector<double> lidar_coords = arr_lidar_coords.as_vec<double>();
    vector<double> curr_lidar_coords;
    
    string extra_path = "data_meas/meas_extra.npz";
    if (fs::is_regular_file(extra_path) == false) {

        std::cerr << "[Error] The Measurement File does not Exists." << std::endl;
        std::cerr << "[Error] Make sure to prepare Measurement File with this path: 'data_meas/meas_extra.npz'" << std::endl;
        exit(-1);
    }
    cnpy::npz_t extra = cnpy::npz_load(extra_path);
    cnpy::NpyArray arr_dt = extra["dt"];
    vector<double> dt = arr_dt.as_vec<double>();
    cnpy::NpyArray arr_imu_w = extra["imu_w"];
    vector<double> imu_w = arr_imu_w.as_vec<double>();
    cnpy::NpyArray arr_encoder_v = extra["encoder_v"];
    vector<double> encoder_v = arr_encoder_v.as_vec<double>();
    cnpy::NpyArray arr_lidar_len = extra["lidar_len"];
    vector<int> lidar_len = arr_lidar_len.as_vec<int>();

    auto stop_read_file = std::chrono::high_resolution_clock::now();
    auto duration_read_file = std::chrono::duration_cast<std::chrono::milliseconds>(stop_read_file - start_read_file);
    std::cout << std::endl;
    std::cout << "Time taken by function (Read Data Files): " << duration_read_file.count() << " milliseconds" << std::endl;
    std::cout << std::endl;


    auto start_total_time = std::chrono::high_resolution_clock::now();
    for (int idx = 0; idx < arr_lidar_len.shape[0]; idx += 1) {

        if (idx == 0) {

            printf("Iteration: %d\n", idx);

            curr_lidar_coords.clear();
            curr_lidar_coords.resize(lidar_len[idx], 0);
            lidar_idx = (idx * LIDAR_COORDS_MAX_LEN);
            std::copy(lidar_coords.begin() + lidar_idx, lidar_coords.begin() + lidar_idx + lidar_len[idx], curr_lidar_coords.begin());
            run_init(d_robot, d_mapping, d_lidar, h_robot, h_mapping, h_lidar, general_info,
                curr_lidar_coords, encoder_v[idx], imu_w[idx], dt[idx], hvec_occupied_map_idx, hvec_free_map_idx, lidar_len[idx] / 2);
        }
        else {

            if (idx % INFO_STEP == 0) {
                printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
                printf("Iteration: %d\n", idx);
            }

            auto start_alloc_init_step = std::chrono::high_resolution_clock::now();

            h_lidar.measurements.LEN = lidar_len[idx] / 2;
            int MEASURE_LEN = NUM_PARTICLES * h_lidar.measurements.LEN;
            h_mapping.particles.OCCUPIED_LEN = h_lidar.measurements.LEN;
            int PARTICLE_UNIQUE_COUNTER = h_mapping.particles.OCCUPIED_LEN + 1;

            curr_lidar_coords.clear();
            curr_lidar_coords.resize(lidar_len[idx], 0);
            lidar_idx = (idx * LIDAR_COORDS_MAX_LEN);
            std::copy(lidar_coords.begin() + lidar_idx, lidar_coords.begin() + lidar_idx + lidar_len[idx], curr_lidar_coords.begin());
            set_measurement_vars(d_lidar.measurements, h_lidar.measurements, curr_lidar_coords, lidar_len[idx] / 2);
            reset_processed_measure(d_lidar.processed_measure, h_lidar.measurements);
            reset_correlation(d_robot.correlation);
            thrust::fill(d_robot.robot_particles.c_weight.begin(), d_robot.robot_particles.c_weight.end(), 0);
            gen_normal_numbers(vec_rnds_encoder_counts);
            gen_normal_numbers(vec_rnds_yaws);
            
            set_state(d_robot.state, h_robot.state, vec_rnds_encoder_counts, vec_rnds_yaws, encoder_v[idx], imu_w[idx], dt[idx]);
            gen_uniform_numbers(vec_rnds);
            set_resampling(d_robot.resampling, vec_rnds);
            
            int curr_grid_size = h_mapping.map.GRID_WIDTH * h_mapping.map.GRID_HEIGHT;
            if (curr_grid_size != PRE_GRID_SIZE) {

                int MAX_DIST_IN_MAP = sqrt(pow(h_mapping.map.GRID_WIDTH, 2) + pow(h_mapping.map.GRID_HEIGHT, 2));
                resize_particles_vars(d_mapping.particles, h_lidar.measurements, MAX_DIST_IN_MAP);
                resize_unique_map_vars(d_mapping.unique_occupied, h_mapping.unique_occupied, h_mapping.map);
                resize_unique_map_vars(d_mapping.unique_free, h_mapping.unique_free, h_mapping.map);

                PRE_GRID_SIZE = curr_grid_size;
            }
            
            thrust::fill(d_mapping.map.c_should_extend.begin(), d_mapping.map.c_should_extend.end(), 0);

            hvec_occupied_map_idx[1] = h_mapping.particles.OCCUPIED_LEN;
            hvec_free_map_idx[1] = 0;
            reset_unique_map_vars(d_mapping.unique_occupied, hvec_occupied_map_idx);
            reset_unique_map_vars(d_mapping.unique_free, hvec_free_map_idx);
            alloc_map_2d_var(d_robot._2d_unique, h_robot._2d_unique, h_mapping.map);

            if (idx % INFO_STEP == 0) {

                auto stop_alloc_init_step = std::chrono::high_resolution_clock::now();
                auto duration_alloc_init_step = std::chrono::duration_cast<std::chrono::microseconds>(stop_alloc_init_step - start_alloc_init_step);
                std::cout << std::endl;
                std::cout << "Time taken by function (Alloc & Init Step): " << duration_alloc_init_step.count() << " microseconds" << std::endl;
                std::cout << std::endl;
            }
        }
        
        auto start_run_step = std::chrono::high_resolution_clock::now();
        run_robot(d_robot, d_mapping, d_lidar, h_robot, h_mapping, h_lidar, general_info);
        run_map(d_robot, d_mapping, d_lidar, h_robot, h_mapping, h_lidar, general_info);

        PRE_GRID_SIZE = h_mapping.map.GRID_WIDTH * h_mapping.map.GRID_HEIGHT;

        if (idx % INFO_STEP == 0) {

            auto stop_run_step = std::chrono::high_resolution_clock::now();

            auto duration_run_step = std::chrono::duration_cast<std::chrono::microseconds>(stop_run_step - start_run_step);
            std::cout << std::endl;
            std::cout << "Time taken by function (Run Step): " << duration_run_step.count() << " microseconds" << std::endl;
            std::cout << std::endl;
        }

        if(idx % DRAW_STEP == 0) {

            THR_GRID_WIDTH = h_mapping.map.GRID_WIDTH;
            THR_GRID_HEIGHT = h_mapping.map.GRID_HEIGHT;
            thr_map.s_grid_map.clear();
            thr_map.s_grid_map.resize(THR_GRID_WIDTH* THR_GRID_HEIGHT, 0);
            thr_map.s_grid_map.assign(d_mapping.map.s_grid_map.begin(), d_mapping.map.s_grid_map.end());
            timed_mutex_draw.unlock();
        }
    }

    auto stop_total_time = std::chrono::high_resolution_clock::now();
    auto duration_total_time = std::chrono::duration_cast<std::chrono::seconds>(stop_total_time - start_total_time);
    std::cout << std::endl;
    std::cout << "Time taken by function (Execution Time): " << duration_total_time.count() << " seconds" << std::endl;
    std::cout << std::endl;

    printf("Execution Finished\n\n");

    t.join();
}

#endif

