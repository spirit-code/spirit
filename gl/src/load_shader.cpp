#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>

#include "load_shader.h"

GLuint load_shader(const char *shader_source_ptr, GLenum shader_type) {
    // Create the shader
    GLuint shader_id = glCreateShader(shader_type);

    // Compile shader
    glShaderSource(shader_id, 1, &shader_source_ptr , NULL);
    glCompileShader(shader_id);

    // Check shader
    GLint result = GL_FALSE;
    int info_log_length;

    glGetShaderiv(shader_id, GL_COMPILE_STATUS, &result);
    glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &info_log_length);
    if (info_log_length > 0){
        std::vector<char> shader_error_message(info_log_length + 1);
        glGetShaderInfoLog(shader_id, info_log_length, NULL, &shader_error_message[0]);
        std::cerr << &shader_error_message[0] << std::endl;
    }
    if(!result) {
        return 0;
    }

    return shader_id;
}

GLuint load_program(const char *vertex_source_ptr, const char *fragment_source_ptr) {
    GLuint vertex_shader_id = load_shader(vertex_source_ptr, GL_VERTEX_SHADER);
    GLuint fragment_shader_id = load_shader(fragment_source_ptr, GL_FRAGMENT_SHADER);

    // Check shader load for errors
    if(!(vertex_shader_id && fragment_shader_id)) {
        return 0;
    }

    // Link the program
    GLuint program_id = glCreateProgram();
    glAttachShader(program_id, vertex_shader_id);
    glAttachShader(program_id, fragment_shader_id);
    glLinkProgram(program_id);

    // Check the program
    GLint result = GL_FALSE;
    int info_log_length;

    glGetProgramiv(program_id, GL_LINK_STATUS, &result);
    glGetProgramiv(program_id, GL_INFO_LOG_LENGTH, &info_log_length);
    if (info_log_length > 0){
        std::vector<char> program_error_message(info_log_length+1);
        glGetProgramInfoLog(program_id, info_log_length, NULL, &program_error_message[0]);
        std::cerr << &program_error_message[0] << std::endl;
    }
    if(!result) {
        return 0;
    }

    glDetachShader(program_id, vertex_shader_id);
    glDetachShader(program_id, fragment_shader_id);

    glDeleteShader(vertex_shader_id);
    glDeleteShader(fragment_shader_id);

    return program_id;
}

//GLuint load_shader(const char *shader_file_path, GLenum shader_type) {
//	// Create the shader
//	GLuint shader_id = glCreateShader(shader_type);
//
//	// Read the shader code from the file
//	std::string shader_code;
//	std::ifstream shader_stream(shader_file_path, std::ios::in);
//	if (shader_stream.is_open()) {
//		std::string line;
//		while (getline(shader_stream, line))
//			shader_code += "\n" + line;
//		shader_stream.close();
//	}
//	else {
//		std::cerr << "Could not load shader " << shader_file_path << std::endl;
//		return 0;
//	}
//
//	// Compile shader
//	char const * shader_source_ptr = shader_code.c_str();
//	glShaderSource(shader_id, 1, &shader_source_ptr, NULL);
//	glCompileShader(shader_id);
//
//	// Check shader
//	GLint result = GL_FALSE;
//	int info_log_length;
//
//	glGetShaderiv(shader_id, GL_COMPILE_STATUS, &result);
//	glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &info_log_length);
//	if (info_log_length > 0) {
//		std::vector<char> shader_error_message(info_log_length + 1);
//		glGetShaderInfoLog(shader_id, info_log_length, NULL, &shader_error_message[0]);
//		std::cerr << &shader_error_message[0] << std::endl;
//	}
//	if (!result) {
//		return 0;
//	}
//
//	return shader_id;
//}
//
//GLuint load_program(const char *vertex_file_path, const char *fragment_file_path) {
//	GLuint vertex_shader_id = load_shader(vertex_file_path, GL_VERTEX_SHADER);
//	GLuint fragment_shader_id = load_shader(fragment_file_path, GL_FRAGMENT_SHADER);
//
//	// Check shader load for errors
//	if (!(vertex_shader_id && fragment_shader_id)) {
//		return 0;
//	}
//
//	// Link the program
//	GLuint program_id = glCreateProgram();
//	glAttachShader(program_id, vertex_shader_id);
//	glAttachShader(program_id, fragment_shader_id);
//	glLinkProgram(program_id);
//
//	// Check the program
//	GLint result = GL_FALSE;
//	int info_log_length;
//
//	glGetProgramiv(program_id, GL_LINK_STATUS, &result);
//	glGetProgramiv(program_id, GL_INFO_LOG_LENGTH, &info_log_length);
//	if (info_log_length > 0) {
//		std::vector<char> program_error_message(info_log_length + 1);
//		glGetProgramInfoLog(program_id, info_log_length, NULL, &program_error_message[0]);
//		std::cerr << &program_error_message[0] << std::endl;
//	}
//	if (!result) {
//		return 0;
//	}
//
//	glDetachShader(program_id, vertex_shader_id);
//	glDetachShader(program_id, fragment_shader_id);
//
//	glDeleteShader(vertex_shader_id);
//	glDeleteShader(fragment_shader_id);
//
//	return program_id;
//}