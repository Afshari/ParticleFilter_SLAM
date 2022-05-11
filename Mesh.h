#include <GL/glew.h>

#pragma once
class Mesh
{
public:
	Mesh();
	void CreateMesh(GLfloat* vertices, unsigned int* indices, unsigned int numOfVertices, unsigned int numOfIndices);
	void RenderMesh();
	~Mesh();
private:
	GLuint VAO, VBO, IBO, indexCount;
};