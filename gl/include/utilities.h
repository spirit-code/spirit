#ifndef UTILITIES_H
#define UTILITIES_H

#include <vector>
#ifndef __gl_h_
#include <glad/glad.h>
#endif

GLuint createProgram(const std::string& vertexShaderSource,
                     const std::string& fragmentShaderSource,
                     const std::vector<std::string>& attributes);

std::string getColormapImplementation(const std::string& colormapName);

#endif
