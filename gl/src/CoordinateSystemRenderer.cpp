
#include <iostream>

#ifndef __gl_h_
#include <glad/glad.h>
#endif
#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "CoordinateSystemRenderer.h"
#include "utilities.h"

CoordinateSystemRenderer::CoordinateSystemRenderer() {
  // TODO: initGL if possible
  // TODO: updateSpins if possible
}

CoordinateSystemRenderer::~CoordinateSystemRenderer() {
  glDeleteVertexArrays(1, &_vao);
  glDeleteBuffers(1, &_vbo);
  glDeleteProgram(_program);
  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
}

void CoordinateSystemRenderer::optionsHaveChanged(const std::vector<int>& changedOptions) {
  bool updateShader = false;
  for (auto it = changedOptions.cbegin(); it != changedOptions.cend(); it++) {
    if (*it == ISpinRenderer::Option::COLORMAP_IMPLEMENTATION) {
      updateShader = true;
    }
  }
  if (updateShader) {
    _updateShaderProgram();
  }
}

void CoordinateSystemRenderer::initGL() {
  glGenVertexArrays(1, &_vao);
  glBindVertexArray(_vao);
  glGenBuffers(1, &_vbo);
  glBindBuffer(GL_ARRAY_BUFFER, _vbo);
  std::vector<GLfloat> vertices = {
    0, 0, 0, 1, 0, 0,
    1, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 1, 0,
    0, 1, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 1,
    0, 0, 1, 0, 0, 1,
  };
  glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*vertices.size(), vertices.data(), GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, false, 4*3*2, nullptr);
  glVertexAttribPointer(1, 3, GL_FLOAT, false, 4*3*2,  (void *)(4*3));
  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);

  _updateShaderProgram();
}

void CoordinateSystemRenderer::updateSpins(const std::vector<glm::vec3>& positions,
                                      const std::vector<glm::vec3>& directions) {
}

void CoordinateSystemRenderer::draw(double aspectRatio) const {
  glUseProgram(_program);
  glBindVertexArray(_vao);

  double verticalFieldOfView = _options.get<ISpinRenderer::Option::VERTICAL_FIELD_OF_VIEW>();
  glm::vec3 cameraPosition = _options.get<ISpinRenderer::Option::CAMERA_POSITION>();
  glm::vec3 centerPosition = _options.get<ISpinRenderer::Option::CENTER_POSITION>();
  glm::vec3 upVector = _options.get<ISpinRenderer::Option::UP_VECTOR>();
  auto origin = _options.get<CoordinateSystemRendererOptions::ORIGIN>();
  auto axis_length = glm::normalize(_options.get<CoordinateSystemRendererOptions::AXIS_LENGTH>());

  glm::mat4 projectionMatrix;
  if (verticalFieldOfView > 0) {
    projectionMatrix = glm::perspective(verticalFieldOfView, aspectRatio, 0.1, 10000.0);
  } else {
    float camera_distance = 1;//glm::length(cameraPosition-centerPosition);
    float leftRight = camera_distance * aspectRatio;
    float bottomTop = camera_distance;
    projectionMatrix = glm::ortho(-leftRight, leftRight, -bottomTop, bottomTop, -10000.0f, 10000.0f);
  }
  glm::mat4 modelviewMatrix = glm::lookAt(glm::normalize(cameraPosition-centerPosition), glm::vec3(0.0, 0.0, 0.0), upVector);

  glUniformMatrix4fv(glGetUniformLocation(_program, "uProjectionMatrix"), 1, false, glm::value_ptr(projectionMatrix));
  glUniformMatrix4fv(glGetUniformLocation(_program, "uModelviewMatrix"), 1, false, glm::value_ptr(modelviewMatrix));
  glUniform3f(glGetUniformLocation(_program, "uOrigin"), 0, 0, 0);
  glUniform3f(glGetUniformLocation(_program, "uAxisLength"), 0.5, 0.5, 0.5);

  glDisable(GL_CULL_FACE);
  glDrawArrays(GL_LINES, 0, 6);
  glEnable(GL_CULL_FACE);
}

void CoordinateSystemRenderer::_updateShaderProgram() {
  if (_program) {
    glDeleteProgram(_program);
  }

  std::string vertexShaderSource =
#include "coordinatesystem.vert.txt"
  ;
  vertexShaderSource += _options.get<ISpinRenderer::Option::COLORMAP_IMPLEMENTATION>();
  std::string fragmentShaderSource =
#include "coordinatesystem.frag.txt"
  ;
  GLuint program = createProgram(vertexShaderSource, fragmentShaderSource, {"ivPosition", "ivDirection"});
  if (program) {
    _program = program;
  }
}
