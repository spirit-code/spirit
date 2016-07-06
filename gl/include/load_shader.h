#ifndef LOAD_SHADER_H
#define LOAD_SHADER_H

#include <glad/glad.h>

GLuint load_shader(const char *shader_file_path, GLenum shader_type);
GLuint load_program(const char *vertex_file_path, const char *fragment_file_path);

#endif
