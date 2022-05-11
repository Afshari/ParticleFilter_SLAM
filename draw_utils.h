#ifndef _DRAW_UTILS_H_
#define _DRAW_UTILS_H_

#include "headers.h"

void CreateObjects(vector<Mesh*>& meshList, int* h_grid_map, int num_of_rectangles, int array_value, const float height,
	const int GRID_WIDTH) {

	int num_lines = 0;
	std::vector<unsigned int> indices;
	std::vector<GLfloat> vertices;
	const float width = 0.05f;
	const int NUM_VERTICES = 8;

	for (int i = 0; i < num_of_rectangles; i++) {

		if (h_grid_map[i] == array_value) {

			// FRONT
			indices.push_back((num_lines * NUM_VERTICES) + 0);
			indices.push_back((num_lines * NUM_VERTICES) + 1);
			indices.push_back((num_lines * NUM_VERTICES) + 2);

			indices.push_back((num_lines * NUM_VERTICES) + 1);
			indices.push_back((num_lines * NUM_VERTICES) + 2);
			indices.push_back((num_lines * NUM_VERTICES) + 3);

			// TOP
			indices.push_back((num_lines * NUM_VERTICES) + 1);
			indices.push_back((num_lines * NUM_VERTICES) + 4);
			indices.push_back((num_lines * NUM_VERTICES) + 5);

			indices.push_back((num_lines * NUM_VERTICES) + 1);
			indices.push_back((num_lines * NUM_VERTICES) + 3);
			indices.push_back((num_lines * NUM_VERTICES) + 5);

			// RIGHT
			indices.push_back((num_lines * NUM_VERTICES) + 2);
			indices.push_back((num_lines * NUM_VERTICES) + 3);
			indices.push_back((num_lines * NUM_VERTICES) + 5);

			indices.push_back((num_lines * NUM_VERTICES) + 2);
			indices.push_back((num_lines * NUM_VERTICES) + 5);
			indices.push_back((num_lines * NUM_VERTICES) + 6);

			// LEFT
			indices.push_back((num_lines * NUM_VERTICES) + 0);
			indices.push_back((num_lines * NUM_VERTICES) + 1);
			indices.push_back((num_lines * NUM_VERTICES) + 7);

			indices.push_back((num_lines * NUM_VERTICES) + 1);
			indices.push_back((num_lines * NUM_VERTICES) + 4);
			indices.push_back((num_lines * NUM_VERTICES) + 7);

			// BOTTOM
			indices.push_back((num_lines * NUM_VERTICES) + 0);
			indices.push_back((num_lines * NUM_VERTICES) + 2);
			indices.push_back((num_lines * NUM_VERTICES) + 7);

			indices.push_back((num_lines * NUM_VERTICES) + 2);
			indices.push_back((num_lines * NUM_VERTICES) + 6);
			indices.push_back((num_lines * NUM_VERTICES) + 7);

			// BACK
			indices.push_back((num_lines * NUM_VERTICES) + 4);
			indices.push_back((num_lines * NUM_VERTICES) + 5);
			indices.push_back((num_lines * NUM_VERTICES) + 6);

			indices.push_back((num_lines * NUM_VERTICES) + 4);
			indices.push_back((num_lines * NUM_VERTICES) + 6);
			indices.push_back((num_lines * NUM_VERTICES) + 7);


			num_lines += 1;

			int x = i % GRID_WIDTH;
			float z = (1) * (float(i) / GRID_WIDTH);

			// 0
			vertices.push_back(x * width);				// x
			vertices.push_back(0);						// 0
			vertices.push_back(z * width);				// z

			// 1
			vertices.push_back(x * width);				// x
			vertices.push_back(height);					// height
			vertices.push_back(z * width);				// z

			// 2
			vertices.push_back((x + 1) * width);		// x + 1
			vertices.push_back(0);						// 0
			vertices.push_back(z * width);				// z

			// 3
			vertices.push_back((x + 1) * width);		// x + 1
			vertices.push_back(height);					// height
			vertices.push_back(z * width);				// z

			// 4
			vertices.push_back(x * width);				// x
			vertices.push_back(height);					// height
			vertices.push_back((z + 1) * width);		// z + 1

			// 5
			vertices.push_back((x + 1) * width);		// x + 1
			vertices.push_back(height);					// height
			vertices.push_back((z + 1) * width);		// z + 1

			// 6
			vertices.push_back((x + 1) * width);		// x + 1
			vertices.push_back(0);						// 0
			vertices.push_back((z + 1) * width);		// z + 1

			// 7
			vertices.push_back(x * width);				// x
			vertices.push_back(0);						// 0
			vertices.push_back((z + 1) * width);		// z + 1


			if (num_lines > 5000) {

				Mesh* obj = new Mesh();
				obj->CreateMesh(vertices.data(), indices.data(), vertices.size(), indices.size());
				meshList.push_back(obj);

				//printf("Reach 5000 --> %d, %d\n", vertices.size(), indices.size());
				num_lines = 0;
				vertices.clear();
				indices.clear();
			}
		}
	}

	Mesh* obj = new Mesh();
	obj->CreateMesh(vertices.data(), indices.data(), vertices.size(), indices.size());
	meshList.push_back(obj);
}

void CreateShaders(vector<Shader>& shader_list, const char* vShader, const char* fShader) {
	Shader* shader1 = new Shader();
	shader1->CreateFromFiles(vShader, fShader);
	shader_list.push_back(*shader1);
}


#endif

