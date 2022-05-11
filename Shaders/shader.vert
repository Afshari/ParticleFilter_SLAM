#version 330

layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 norm;

uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;

out vec3 Normal;

void main() {
	gl_Position = projection * view * model * vec4(pos, 1.0);

	Normal = mat3(transpose(inverse(model))) * norm;
}