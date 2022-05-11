#version 330

uniform vec4 vCol;

in  vec3 Normal;
out vec4 color;

struct DirectionalLight {
	vec3  color;
	float ambientIntensity;
	vec3  direction;
	float diffuseIntensity;
};

uniform DirectionalLight directionalLight;

void main()	 {
	vec4 ambientColor = vec4(directionalLight.color, 1.0f) * directionalLight.ambientIntensity;

	float diffuseFactor = max(dot(normalize(Normal), normalize(directionalLight.direction)), 0.0f);
	vec4 diffuseColor = vec4(directionalLight.color, 1.0f) * directionalLight.diffuseIntensity * diffuseFactor;

	color = vCol * (ambientColor + diffuseColor);
}