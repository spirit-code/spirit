
#include <iostream>

#ifndef __gl_h_
#include <glad/glad.h>
#endif
#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "SurfaceSpinRenderer.h"
#include "utilities.h"

SurfaceSpinRenderer::SurfaceSpinRenderer() {
  // TODO: initGL if possible
  // TODO: updateSpins if possible
}

SurfaceSpinRenderer::~SurfaceSpinRenderer() {
  glDeleteVertexArrays(1, &_vao);
  glDeleteBuffers(1, &_ibo);
  glDeleteBuffers(1, &_instancePositionVbo);
  glDeleteBuffers(1, &_instanceDirectionVbo);
  glDeleteProgram(_program);
  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
}

void SurfaceSpinRenderer::optionsHaveChanged(const std::vector<int>& changedOptions) {
  bool updateShader = false;
  bool updateIndices = false;
  for (auto it = changedOptions.cbegin(); it != changedOptions.cend(); it++) {
    if (*it == ISpinRendererOptions::COLORMAP_IMPLEMENTATION) {
      updateShader = true;
    } else if (*it == SurfaceSpinRendererOptions::SURFACE_INDICES) {
      updateIndices = true;
    }
  }
  if (updateShader) {
    _updateShaderProgram();
  }
  if (updateIndices) {
    _updateSurfaceIndices();
  }
}

void SurfaceSpinRenderer::initGL() {
  glGenVertexArrays(1, &_vao);
  glBindVertexArray(_vao);
  
  glGenBuffers(1, &_ibo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ibo);
  _numIndices = 0;
  
  glGenBuffers(1, &_instancePositionVbo);
  glBindBuffer(GL_ARRAY_BUFFER, _instancePositionVbo);
  glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, nullptr);
  glEnableVertexAttribArray(0);
  
  glGenBuffers(1, &_instanceDirectionVbo);
  glBindBuffer(GL_ARRAY_BUFFER, _instanceDirectionVbo);
  glVertexAttribPointer(1, 3, GL_FLOAT, false, 0, nullptr);
  glEnableVertexAttribArray(1);
  
  _updateShaderProgram();
  _updateSurfaceIndices();
}

void SurfaceSpinRenderer::updateSpins(const std::vector<glm::vec3>& positions,
                                    const std::vector<glm::vec3>& directions) {
  glBindVertexArray(_vao);
  glBindBuffer(GL_ARRAY_BUFFER, _instancePositionVbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * positions.size(), positions.data(), GL_STREAM_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, _instanceDirectionVbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * directions.size(), directions.data(), GL_STREAM_DRAW);
}

void SurfaceSpinRenderer::draw(double aspectRatio) const {
  if (_numIndices <= 0) {
    return;
  }
  glBindVertexArray(_vao);
  glUseProgram(_program);
  
  glm::vec2 zRange = _options.get<ISpinRendererOptions::Z_RANGE>();
  double verticalFieldOfView = _options.get<ISpinRendererOptions::VERTICAL_FIELD_OF_VIEW>();
  glm::vec3 cameraPosition = _options.get<ISpinRendererOptions::CAMERA_POSITION>();
  glm::vec3 centerPosition = _options.get<ISpinRendererOptions::CENTER_POSITION>();
  glm::vec3 upVector = _options.get<ISpinRendererOptions::UP_VECTOR>();
  
  glm::mat4 projectionMatrix = glm::perspective(verticalFieldOfView, aspectRatio, 0.1, 10000.0);
  glm::mat4 modelviewMatrix = glm::lookAt(cameraPosition, centerPosition, upVector);
  glm::vec4 lightPosition = modelviewMatrix * glm::vec4(cameraPosition, 1.0);
  
  glUniformMatrix4fv(glGetUniformLocation(_program, "uProjectionMatrix"), 1, false, glm::value_ptr(projectionMatrix));
  glUniformMatrix4fv(glGetUniformLocation(_program, "uModelviewMatrix"), 1, false, glm::value_ptr(modelviewMatrix));
  glUniform3f(glGetUniformLocation(_program, "uLightPosition"), lightPosition[0], lightPosition[1], lightPosition[2]);
  glUniform2f(glGetUniformLocation(_program, "uZRange"), zRange[0], zRange[1]);
  
  glDisable(GL_CULL_FACE);
  glDrawElements(GL_TRIANGLES, _numIndices, GL_UNSIGNED_INT, nullptr);
  assert(!glGetError());
  glEnable(GL_CULL_FACE);
}

void SurfaceSpinRenderer::_updateShaderProgram() {
  if (_program) {
    glDeleteProgram(_program);
  }
  std::string vertexShaderSource =
#include "surface.vert.txt"
  ;
  vertexShaderSource += _options.get<ISpinRendererOptions::COLORMAP_IMPLEMENTATION>();
  std::string fragmentShaderSource =
#include "surface.frag.txt"
  ;
  fragmentShaderSource += _options.get<ISpinRendererOptions::COLORMAP_IMPLEMENTATION>();
  GLuint program = createProgram(vertexShaderSource, fragmentShaderSource, {"ivPosition", "ivDirection"});
  if (program) {
    _program = program;
  }
}

void SurfaceSpinRenderer::_updateSurfaceIndices() {
  const std::vector<GLuint>& surfaceIndices = _options.get<SurfaceSpinRendererOptions::SURFACE_INDICES>();
  
  // Enforce valid range
  if (surfaceIndices.size() < 3) {
    _numIndices = 0;
    return;
  }
  glBindVertexArray(_vao);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ibo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * surfaceIndices.size(), surfaceIndices.data(), GL_STREAM_DRAW);
  _numIndices = surfaceIndices.size();
}

std::vector<unsigned int> SurfaceSpinRenderer::generateCartesianSurfaceIndices(int nx, int ny) {
  std::vector<unsigned int> surfaceIndices;
  for (int i = 0; i < ny-1; i++) {
    for (int j = 0; j < nx-1; j++) {
      surfaceIndices.push_back(i*nx + j);
      surfaceIndices.push_back(i*nx + j + 1);
      surfaceIndices.push_back((i+1)*nx + j);
      surfaceIndices.push_back((i+1)*nx + j);
      surfaceIndices.push_back(i*nx + j + 1);
      surfaceIndices.push_back((i+1)*nx + j + 1);
    }
  }
  return surfaceIndices;
};
