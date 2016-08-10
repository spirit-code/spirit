#include <iostream>
#include "utilities.h"

GLuint createProgram(const std::string& vertexShaderSource,
                     const std::string& fragmentShaderSource,
                     const std::vector<std::string>& attributes) {
  CHECK_GL_ERROR;
  GLint status = 0;
  
  const char *vertexShaderSourceCStr = vertexShaderSource.c_str();
  GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertexShader, 1, &vertexShaderSourceCStr, nullptr);
  glCompileShader(vertexShader);
  glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &status);
  if (!status) {
    GLsizei length = 0;
    glGetShaderiv(vertexShader, GL_INFO_LOG_LENGTH, &length);
    if (length > 0) {
      char *infoLog = new char[length];
      glGetShaderInfoLog(vertexShader, length, nullptr, infoLog);
      std::cerr << "vertex shader info log:\n" << infoLog << std::endl;
      delete[] infoLog;
    }
    return 0;
  }
  
  const char *fragmentShaderSourceCStr = fragmentShaderSource.c_str();
  GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentShader, 1, &fragmentShaderSourceCStr, nullptr);
  glCompileShader(fragmentShader);
  glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &status);
  if (!status) {
    GLsizei length = 0;
    glGetShaderiv(fragmentShader, GL_INFO_LOG_LENGTH, &length);
    if (length > 0) {
      char *infoLog = new char[length];
      glGetShaderInfoLog(fragmentShader, length, nullptr, infoLog);
      std::cerr << "fragment shader info log:\n" << infoLog << std::endl;
      delete[] infoLog;
    }
    return 0;
  }
  
  GLuint program = glCreateProgram();
  glAttachShader(program, vertexShader);
  glAttachShader(program, fragmentShader);
  for (std::vector<std::string>::size_type i = 0; i < attributes.size(); i++) {
    glBindAttribLocation(program, i, attributes[i].c_str());
  }
  glBindFragDataLocation(program, 0, "fo_FragColor");
  glLinkProgram(program);
  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);
  
  glGetProgramiv(program, GL_LINK_STATUS, &status);
  if (!status) {
    GLsizei length = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
    if (length > 0) {
      char *infoLog = new char[length];
      glGetProgramInfoLog(program, length, nullptr, infoLog);
      std::cerr << "program info log:\n" << infoLog << std::endl;
      delete[] infoLog;
    }
    return 0;
  }
  return program;
  CHECK_GL_ERROR;
}

std::string getColormapImplementation(const std::string& colormapName) {
  if (colormapName == "hsv") {
    return
#include "colormap.hsv.txt"
    ;
  } else if (colormapName == "redblue") {
    return
#include "colormap.redblue.txt"
    ;
  } else {
    return "vec3 colormap(vec3 direction) {return vec3(1.0, 1.0, 1.0);}";
  }
}

void FPSCounter::tick() {
  if (_previous_frame_time_point != std::chrono::steady_clock::time_point()) {
    auto previous_duration = std::chrono::steady_clock::now() - _previous_frame_time_point;
    _n_frame_duration += previous_duration;
    _frame_durations.push(previous_duration);
    while (_frame_durations.size() > (unsigned int)_max_n) {
      _n_frame_duration -= _frame_durations.front();
      _frame_durations.pop();
    }
  }
  _previous_frame_time_point = std::chrono::steady_clock::now();
}

float FPSCounter::getFramerate() const {
  return _frame_durations.size() / _n_frame_duration.count();
}
