#include "gl_shader.h"

Shader::Shader() {
	shader = 0, uniformModel = 0, uniformProjection = 0;
}

void Shader::CreateFromFiles(const char* vShader, const char* fShader) {
	std::string vShaderCode = readShaderCodeFromFile(vShader);
	std::string fShaderCode = readShaderCodeFromFile(fShader);

	compileShaders(vShaderCode.c_str(), fShaderCode.c_str());
}


Shader::~Shader() { }

std::string Shader::readShaderCodeFromFile(const char* shaderPath) {

	std::string code;
	std::ifstream shaderFile;
	shaderFile.exceptions(std::ifstream::badbit);
	try
	{
		// Открываем файлы
		shaderFile.open(shaderPath);
		std::stringstream shaderStream;
		// Считываем данные в потоки
		shaderStream << shaderFile.rdbuf();
		// Закрываем файлы
		shaderFile.close();
		// Преобразовываем потоки в массив GLchar
		code = shaderStream.str();
	}
	catch (std::ifstream::failure e)
	{
		std::cout << "Shader file " << shaderPath << " cannot be read" << std::endl;
	}

	return code;
}

void Shader::addShader(GLuint theProgram, const char* shaderCode, GLenum shaderType)
{
	GLuint theShader = glCreateShader(shaderType);

	const GLchar* theCode[1];
	theCode[0] = shaderCode;

	GLint codeLength[1];
	codeLength[0] = strlen(shaderCode);

	glShaderSource(theShader, 1, theCode, codeLength);
	glCompileShader(theShader);

	GLint result = 0;
	GLchar errLog[1024] = { 0 };

	glGetShaderiv(theShader, GL_COMPILE_STATUS, &result);

	if (!result)
	{
		glGetShaderInfoLog(theShader, sizeof(errLog), NULL, errLog);
		std::cerr << "Error compiling the " << shaderType << " shader: '" << errLog << "'\n";
		return;
	}

	glAttachShader(theProgram, theShader);
}

void Shader::compileShaders(const char* vShaderCode, const char* fShaderCode)
{
	shader = glCreateProgram();

	if (!shader) {
		std::cerr << "Error creating shader program\n";
		return;
	}

	addShader(shader, vShaderCode, GL_VERTEX_SHADER);
	addShader(shader, fShaderCode, GL_FRAGMENT_SHADER);

	GLint result = 0;
	GLchar errLog[1024] = { 0 };

	glLinkProgram(shader);
	glGetProgramiv(shader, GL_LINK_STATUS, &result);

	if (!result) {
		glGetProgramInfoLog(shader, sizeof(errLog), NULL, errLog);
		std::cerr << "Error linking program: '" << errLog << "'\n";
		return;
	}

	glValidateProgram(shader);
	glGetProgramiv(shader, GL_VALIDATE_STATUS, &result);

	if (!result) {
		glGetProgramInfoLog(shader, sizeof(errLog), NULL, errLog);
		std::cerr << "Error validating program: '" << errLog << "'\n";
		return;
	}

	uniformProjection = glGetUniformLocation(shader, "projection");
	uniformModel = glGetUniformLocation(shader, "model");
	uniformView = glGetUniformLocation(shader, "view");
	uniformColor = glGetUniformLocation(shader, "vCol");
	uniformAmbientColor = glGetUniformLocation(shader, "directionalLight.color");
	uniformAmbientIntensity = glGetUniformLocation(shader, "directionalLight.ambientIntensity");
	uniformDirection = glGetUniformLocation(shader, "directionalLight.direction");
	uniformDiffuseIntensity = glGetUniformLocation(shader, "directionalLight.diffuseIntensity");
}