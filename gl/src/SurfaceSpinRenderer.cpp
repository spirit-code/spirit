
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
  CHECK_GL_ERROR;
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
  CHECK_GL_ERROR;
}

SurfaceSpinRenderer::~SurfaceSpinRenderer() {
  CHECK_GL_ERROR;
  glDeleteVertexArrays(1, &_vao);
  glDeleteBuffers(1, &_ibo);
  glDeleteBuffers(1, &_instancePositionVbo);
  glDeleteBuffers(1, &_instanceDirectionVbo);
  glDeleteProgram(_program);
  CHECK_GL_ERROR;
}

void SurfaceSpinRenderer::optionsHaveChanged(const std::vector<int>& changedOptions) {
  CHECK_GL_ERROR;
  bool updateShader = false;
  bool updateIndices = false;
  for (auto it = changedOptions.cbegin(); it != changedOptions.cend(); it++) {
    if (*it == ISpinRenderer::Option::COLORMAP_IMPLEMENTATION) {
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
  CHECK_GL_ERROR;
}

void SurfaceSpinRenderer::updateSpins(const std::vector<glm::vec3>& positions,
                                      const std::vector<glm::vec3>& directions) {
  CHECK_GL_ERROR;
  glBindVertexArray(_vao);
  glBindBuffer(GL_ARRAY_BUFFER, _instancePositionVbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * positions.size(), positions.data(), GL_STREAM_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, _instanceDirectionVbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * directions.size(), directions.data(), GL_STREAM_DRAW);
  CHECK_GL_ERROR;
}

void SurfaceSpinRenderer::draw(float aspectRatio) const {
  CHECK_GL_ERROR;
  if (_numIndices <= 0) {
    return;
  }
  glBindVertexArray(_vao);
  glUseProgram(_program);
  
  glm::vec2 zRange = _options.get<ISpinRenderer::Option::Z_RANGE>();
  if (zRange.x <= -1) {
    zRange.x = -2;
  }
  if (zRange.y >= 1) {
    zRange.y = 2;
  }
  float verticalFieldOfView = _options.get<ISpinRenderer::Option::VERTICAL_FIELD_OF_VIEW>();
  glm::vec3 cameraPosition = _options.get<ISpinRenderer::Option::CAMERA_POSITION>();
  glm::vec3 centerPosition = _options.get<ISpinRenderer::Option::CENTER_POSITION>();
  glm::vec3 upVector = _options.get<ISpinRenderer::Option::UP_VECTOR>();
  
  glm::mat4 projectionMatrix;
  if (verticalFieldOfView > 0) {
    projectionMatrix = glm::perspective(verticalFieldOfView, aspectRatio, 0.1f, 10000.0f);
  } else {
    float camera_distance = glm::length(cameraPosition-centerPosition);
    float leftRight = camera_distance * aspectRatio;
    float bottomTop = camera_distance;
    projectionMatrix = glm::ortho(-leftRight, leftRight, -bottomTop, bottomTop, -10000.0f, 10000.0f);
  }
  glm::mat4 modelviewMatrix = glm::lookAt(cameraPosition, centerPosition, upVector);
  glm::vec4 lightPosition = modelviewMatrix * glm::vec4(cameraPosition, 1.0f);
  
  glUniformMatrix4fv(glGetUniformLocation(_program, "uProjectionMatrix"), 1, false, glm::value_ptr(projectionMatrix));
  glUniformMatrix4fv(glGetUniformLocation(_program, "uModelviewMatrix"), 1, false, glm::value_ptr(modelviewMatrix));
  glUniform3f(glGetUniformLocation(_program, "uLightPosition"), lightPosition[0], lightPosition[1], lightPosition[2]);
  glUniform2f(glGetUniformLocation(_program, "uZRange"), zRange[0], zRange[1]);
  
  glDisable(GL_CULL_FACE);
  glDrawElements(GL_TRIANGLES, _numIndices, GL_UNSIGNED_INT, nullptr);
  glEnable(GL_CULL_FACE);
  CHECK_GL_ERROR;
}

void SurfaceSpinRenderer::_updateShaderProgram() {
  CHECK_GL_ERROR;
  if (_program) {
    glDeleteProgram(_program);
  }
  std::string vertexShaderSource =
#include "surface.vert.txt"
  ;
  vertexShaderSource += _options.get<ISpinRenderer::Option::COLORMAP_IMPLEMENTATION>();
  std::string fragmentShaderSource =
#include "surface.frag.txt"
  ;
  fragmentShaderSource += _options.get<ISpinRenderer::Option::COLORMAP_IMPLEMENTATION>();
  GLuint program = createProgram(vertexShaderSource, fragmentShaderSource, {"ivPosition", "ivDirection"});
  if (program) {
    _program = program;
  }
  CHECK_GL_ERROR;
}

void SurfaceSpinRenderer::_updateSurfaceIndices() {
  CHECK_GL_ERROR;
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
  CHECK_GL_ERROR;
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
