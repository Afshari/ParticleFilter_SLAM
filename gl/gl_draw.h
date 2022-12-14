
#include "headers.h"
#include "host_utils.h"
#include "gl_draw_utils.h"


Window mainWindow;
vector<Shader> shader_list;
Camera camera;

vector<Mesh*> freeList;
vector<Mesh*> wallList;

GLfloat delta_time = 0.0f;
GLfloat last_time   = 0.0f;

Light main_light;

// Vertex Shader
static const char* vShader = "Shaders/shader.vert";

// Fragment Shader
static const char* fShader = "Shaders/shader.frag";



int draw_main()
{
	mainWindow.initialize();

	int file_number = 2800;
	string file_name = "data/map/" + std::to_string(file_number) + ".txt";
	string first_vec_title = "h_lidar_coords";
	map<string, string> scalar_values;
	map<string, string> vec_values;

	file_extractor(file_name, first_vec_title, scalar_values, vec_values);

	host_vector<int> h_grid_map;
	string_extractor<int>(vec_values["h_grid_map"], h_grid_map);
	int GRID_WIDTH = std::stoi(scalar_values["ST_GRID_WIDTH"]);
	int GRID_HEIGHT = std::stoi(scalar_values["ST_GRID_HEIGHT"]);
	std::cout << "GRID_WIDTH: " << GRID_WIDTH << std::endl;
	std::cout << "GRID_HEIGHT: " << GRID_HEIGHT << std::endl;

	CreateObjects(freeList, h_grid_map.data(), GRID_WIDTH * GRID_HEIGHT, 1, 0.1f, GRID_HEIGHT);
	CreateObjects(wallList, h_grid_map.data(), GRID_WIDTH * GRID_HEIGHT, 2, 0.5f, GRID_HEIGHT);
	CreateShaders(shader_list, vShader, fShader);

	camera = Camera(glm::vec3(-2.0f, 4.0f, 12.0f), glm::vec3(0.0f, 1.0f, 0.0f), -90.0f, -45.0f, 5.0f, 0.1f);

	main_light = Light(1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f);

	GLuint uniformModel = 0, uniformProjection = 0, uniformView = 0, uniformColor = 0,
		uniformAmbientIntensity = 0, uniformAmbientColor = 0;
	glm::mat4 projection = glm::perspective(45.0f, 
		(GLfloat)mainWindow.getBufferWidth() / (GLfloat)mainWindow.getBufferHeight(), 0.1f, 90.0f);


	// Loop until windows closed
	while (!mainWindow.getShouldClose()) {

		GLfloat now = glfwGetTime();
		delta_time = now - last_time;
		last_time = now;

		// Get+Handle user inputs
		glfwPollEvents();

		camera.keyControl(mainWindow.getskeys(), delta_time);
		camera.mouseControl(mainWindow.getXChange(), mainWindow.getYChange(), delta_time);

		// Clear window
		glClearColor(0.0f, 0.0f,0.0f, 1.0f);
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


	return 0;
}