
#include <iostream>

#ifndef __gl_h_
#include <glad/glad.h>
#endif
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "IsosurfaceSpinRenderer.hpp"
#include "VectorfieldIsosurface.hpp"
#include "GLSpins.hpp"
#include "utilities.hpp"

IsosurfaceSpinRenderer::IsosurfaceSpinRenderer() {
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
  _updateIsosurfaceIndices();
  CHECK_GL_ERROR;
}

IsosurfaceSpinRenderer::~IsosurfaceSpinRenderer() {
  CHECK_GL_ERROR;
  glDeleteVertexArrays(1, &_vao);
  glDeleteBuffers(1, &_ibo);
  glDeleteBuffers(1, &_instancePositionVbo);
  glDeleteBuffers(1, &_instanceDirectionVbo);
  glDeleteProgram(_program);
  CHECK_GL_ERROR;
}

void IsosurfaceSpinRenderer::optionsHaveChanged(const std::vector<int>& changedOptions) {
  CHECK_GL_ERROR;
  bool updateShader = false;
  bool updateIndices = false;
  for (auto it = changedOptions.cbegin(); it != changedOptions.cend(); it++) {
    if (*it == ISpinRenderer::Option::COLORMAP_IMPLEMENTATION) {
      updateShader = true;
    } else if (*it == IsosurfaceSpinRendererOptions::ISOVALUE) {
      updateIndices = true;
    } else if (*it == GLSpins::Option::TETRAHEDRA_INDICES) {
      updateIndices = true;
    }
  }
  if (updateShader) {
    _updateShaderProgram();
  }
  if (updateIndices) {
    _updateIsosurfaceIndices();
  }
  CHECK_GL_ERROR;
}

void IsosurfaceSpinRenderer::updateSpins(const std::vector<glm::vec3>& positions,
                                      const std::vector<glm::vec3>& directions) {
  _positions = positions;
  _directions = directions;
  _updateIsosurfaceIndices();
}

void IsosurfaceSpinRenderer::draw(float aspectRatio) const {
  CHECK_GL_ERROR;
  if (_numIndices <= 0) {
    return;
  }
  glBindVertexArray(_vao);
  glUseProgram(_program);
  
  // Disable z-Filtering, that's what the isosurface is for, after all.
  glm::vec2 zRange = {-2, 2};

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

void IsosurfaceSpinRenderer::_updateShaderProgram() {
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

void IsosurfaceSpinRenderer::_updateIsosurfaceIndices() {
  float isovalue = _options.get<IsosurfaceSpinRendererOptions::ISOVALUE>();

  const std::vector<std::array<int, 4>>& tetrahedraIndices = _options.get<GLSpins::Option::TETRAHEDRA_INDICES>();
  if (tetrahedraIndices.size() == 0) {
    _numIndices = 0;
    return;
  } else if (_positions.size() < 4) {
    _numIndices = 0;
    return;
  }
  
  std::vector<float> values;
  for (auto& direction : _directions) {
    values.push_back(direction.z);
  }
  
  VectorfieldIsosurface isosurface(VectorfieldIsosurface::calculate(_positions, _directions, values, isovalue, tetrahedraIndices));
  
  const std::vector<GLuint> surfaceIndices(isosurface.triangle_indices.begin(), isosurface.triangle_indices.end());
  
  CHECK_GL_ERROR;
  glBindVertexArray(_vao);
  glBindBuffer(GL_ARRAY_BUFFER, _instancePositionVbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * isosurface.positions.size(), isosurface.positions.data(), GL_STREAM_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, _instanceDirectionVbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * isosurface.directions.size(), isosurface.directions.data(), GL_STREAM_DRAW);
  CHECK_GL_ERROR;
  
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
