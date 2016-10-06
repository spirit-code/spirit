#ifndef UTILITIES_H
#define UTILITIES_H

#include <iostream>
#include <cassert>
#include <vector>
#include <chrono>
#include <queue>
#ifndef __gl_h_
#include <glad/glad.h>
#endif

GLuint createProgram(const std::string& vertexShaderSource,
                     const std::string& fragmentShaderSource,
                     const std::vector<std::string>& attributes);

std::string getColormapImplementation(const std::string& colormapName);

class FPSCounter {
public:
  void tick();
  float getFramerate() const;
private:
  int _max_n = 60;
  std::chrono::duration<float> _n_frame_duration = std::chrono::duration<float>::zero();
  std::chrono::steady_clock::time_point _previous_frame_time_point;
  std::queue<std::chrono::duration<float>> _frame_durations;
};


#define CHECK_GL_ERROR do {\
  GLuint error = glGetError();\
  if (error) {\
    std::cerr << "OpenGL error (" << error << ") detected in " << __FILE__ << " (l. " << __LINE__ << ")" << std::endl;\
  }\
  assert(!error);\
} while(0)

#endif
