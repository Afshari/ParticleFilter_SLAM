#include "gl_camera.h"

Camera::Camera() {

}

Camera::Camera(glm::vec3 start_position, glm::vec3 start_up, GLfloat start_yaw, GLfloat start_pitch,
	GLfloat start_move_speed, GLfloat start_turn_speed) {

	position = start_position;
	world_up = start_up;
	yaw = start_yaw;
	pitch = start_pitch;
	front = glm::vec3(0.0f, 0.0f, -1.0f);

	move_speed = start_move_speed;
	turn_speed = start_turn_speed;

	update();
}

void Camera::keyControl(bool* keys, GLfloat delta_time) {

	GLfloat velocity = move_speed * delta_time;

	if (keys[GLFW_KEY_W]) {
		position += front * velocity;
	}
	if (keys[GLFW_KEY_S]) {
		position -= front * velocity;
	}
	if (keys[GLFW_KEY_A]) {
		position -= right * velocity;
	}
	if (keys[GLFW_KEY_D]) {
		position += right * velocity;
	}
	if (keys[GLFW_KEY_U]) {
		position += up * velocity;
	}
}

void Camera::mouseControl(GLfloat xChange, GLfloat yChange, GLfloat delta_time) {

	xChange *= turn_speed; // *delta_time;
	yChange *= turn_speed; // *delta_time;

	yaw += xChange;
	pitch += yChange;

	if (pitch > 89.0f)
		pitch = 89.0f;
	if (pitch < -89.0f)
		pitch = -89.0f;

	update();
}

glm::mat4 Camera::calculateViewMatrix() {

	return glm::lookAt(position, position + front, up);
}

void Camera::update() {

	front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
	front.y = sin(glm::radians(pitch));
	front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
	front = glm::normalize(front);

	right = glm::normalize(glm::cross(front, world_up));
	up = glm::normalize(glm::cross(right, front));
}

Camera::~Camera() {

}
