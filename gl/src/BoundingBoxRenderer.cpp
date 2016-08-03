
#include <iostream>

#ifndef __gl_h_
#include <glad/glad.h>
#endif
#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "GLSpins.h"
#include "BoundingBoxRenderer.h"
#include "utilities.h"


BoundingBoxRenderer::BoundingBoxRenderer() {
  // TODO: initGL if possible
  // TODO: updateSpins if possible
}

BoundingBoxRenderer::~BoundingBoxRenderer() {
  glDeleteVertexArrays(1, &_vao);
  glDeleteBuffers(1, &_vbo);
  glDeleteProgram(_program);
  glDisableVertexAttribArray(0);
}

void BoundingBoxRenderer::optionsHaveChanged(const std::vector<int>& changedOptions) {
  bool updateVertices = false;
  for (auto it = changedOptions.cbegin(); it != changedOptions.cend(); it++) {
    if (*it == GLSpins::Option::BOUNDING_BOX_MIN) {
      updateVertices = true;
    } else if (*it == GLSpins::Option::BOUNDING_BOX_MAX) {
      updateVertices = true;
    }
  }
  if (updateVertices) {
    _updateVertexData();
  }
}

void BoundingBoxRenderer::initGL() {
  glGenVertexArrays(1, &_vao);
  glBindVertexArray(_vao);
  glGenBuffers(1, &_vbo);
  glBindBuffer(GL_ARRAY_BUFFER, _vbo);
  glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, nullptr);
  glEnableVertexAttribArray(0);

  _updateVertexData();

  std::string vertexShaderSource =
#include "boundingbox.vert.txt"
  ;
  std::string fragmentShaderSource =
#include "boundingbox.frag.txt"
  ;
  GLuint program = createProgram(vertexShaderSource, fragmentShaderSource, {"ivPosition"});
  if (program) {
    _program = program;
  }
}

void BoundingBoxRenderer::updateSpins(const std::vector<glm::vec3>& positions,
                                     const std::vector<glm::vec3>& directions) {
}

void BoundingBoxRenderer::draw(double aspectRatio) const {
  glUseProgram(_program);
  glBindVertexArray(_vao);

  double verticalFieldOfView = _options.get<ISpinRenderer::Option::VERTICAL_FIELD_OF_VIEW>();
  glm::vec3 cameraPosition = _options.get<ISpinRenderer::Option::CAMERA_POSITION>();
  glm::vec3 centerPosition = _options.get<ISpinRenderer::Option::CENTER_POSITION>();
  glm::vec3 upVector = _options.get<ISpinRenderer::Option::UP_VECTOR>();
  auto color = _options.get<BoundingBoxRendererOptions::COLOR>();
  
  glm::mat4 projectionMatrix;
  if (verticalFieldOfView > 0) {
    projectionMatrix = glm::perspective(verticalFieldOfView, aspectRatio, 0.1, 10000.0);
  } else {
    float camera_distance = glm::length(cameraPosition-centerPosition);
    float leftRight = camera_distance * aspectRatio;
    float bottomTop = camera_distance;
    projectionMatrix = glm::ortho(-leftRight, leftRight, -bottomTop, bottomTop, 0.1f, 10000.0f);
  }
  glm::mat4 modelviewMatrix = glm::lookAt(cameraPosition, centerPosition, upVector);

  glUniformMatrix4fv(glGetUniformLocation(_program, "uProjectionMatrix"), 1, false, glm::value_ptr(projectionMatrix));
  glUniformMatrix4fv(glGetUniformLocation(_program, "uModelviewMatrix"), 1, false, glm::value_ptr(modelviewMatrix));
  glUniform3f(glGetUniformLocation(_program, "uColor"), color[0], color[1], color[2]);

  glDisable(GL_CULL_FACE);
  glDrawArrays(GL_LINES, 0, 24);
  glEnable(GL_CULL_FACE);
}

void BoundingBoxRenderer::_updateVertexData() {
  glBindBuffer(GL_ARRAY_BUFFER, _vbo);
  auto color = _options.get<BoundingBoxRendererOptions::COLOR>();
  auto min = _options.get<GLSpins::Option::BOUNDING_BOX_MIN>();
  auto max = _options.get<GLSpins::Option::BOUNDING_BOX_MAX>();
  std::vector<GLfloat> vertices = {
    min[0], min[1], min[2],
    max[0], min[1], min[2],
    max[0], min[1], min[2],
    max[0], max[1], min[2],
    max[0], max[1], min[2],
    min[0], max[1], min[2],
    min[0], max[1], min[2],
    min[0], min[1], min[2],
    min[0], min[1], max[2],
    max[0], min[1], max[2],
    max[0], min[1], max[2],
    max[0], max[1], max[2],
    max[0], max[1], max[2],
    min[0], max[1], max[2],
    min[0], max[1], max[2],
    min[0], min[1], max[2],
    min[0], min[1], min[2],
    min[0], min[1], max[2],
    max[0], min[1], min[2],
    max[0], min[1], max[2],
    max[0], max[1], min[2],
    max[0], max[1], max[2],
    min[0], max[1], min[2],
    min[0], max[1], max[2]};
  glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*vertices.size(), vertices.data(), GL_STATIC_DRAW);
}
