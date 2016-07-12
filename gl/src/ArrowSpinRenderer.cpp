
#include <iostream>

#ifndef __gl_h_
#include <glad/glad.h>
#endif
#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "ArrowSpinRenderer.h"
#include "utilities.h"

ArrowSpinRenderer::ArrowSpinRenderer() {
  // TODO: initGL if possible
  // TODO: updateSpins if possible
}

ArrowSpinRenderer::~ArrowSpinRenderer() {
  glDeleteVertexArrays(1, &_vao);
  glDeleteBuffers(1, &_vbo);
  glDeleteBuffers(1, &_ibo);
  glDeleteBuffers(1, &_instancePositionVbo);
  glDeleteBuffers(1, &_instanceDirectionVbo);
  glDeleteProgram(_program);
  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  glDisableVertexAttribArray(2);
  glDisableVertexAttribArray(3);
}

void ArrowSpinRenderer::updateOptions() {
  // TODO: implement updateOptions
}

void ArrowSpinRenderer::initGL() {
  glGenVertexArrays(1, &_vao);
  glBindVertexArray(_vao);
  glGenBuffers(1, &_vbo);
  glBindBuffer(GL_ARRAY_BUFFER, _vbo);
  glVertexAttribPointer(0, 3, GL_FLOAT, false, 4*3*2, nullptr);
  glEnableVertexAttribArray(0);
  glVertexAttribDivisor(0, 0);
  glVertexAttribPointer(1, 3, GL_FLOAT, false, 4*3*2, (void *)(4*3));
  glEnableVertexAttribArray(1);
  glVertexAttribDivisor(1, 0);
  
  glGenBuffers(1, &_ibo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ibo);
  _numIndices = 0;
  
  glGenBuffers(1, &_instancePositionVbo);
  glBindBuffer(GL_ARRAY_BUFFER, _instancePositionVbo);
  glVertexAttribPointer(2, 3, GL_FLOAT, false, 0, nullptr);
  glEnableVertexAttribArray(2);
  glVertexAttribDivisor(2, 1);
  
  glGenBuffers(1, &_instanceDirectionVbo);
  glBindBuffer(GL_ARRAY_BUFFER, _instanceDirectionVbo);
  glVertexAttribPointer(3, 3, GL_FLOAT, false, 0, nullptr);
  glEnableVertexAttribArray(3);
  glVertexAttribDivisor(3, 1);
  _numInstances = 0;
  
  _updateShaderProgram();
  _updateVertexData();
}

void ArrowSpinRenderer::updateSpins(const std::vector<glm::vec3>& positions,
                                    const std::vector<glm::vec3>& directions) {
  glBindVertexArray(_vao);
  _numInstances = positions.size();
  glBindBuffer(GL_ARRAY_BUFFER, _instancePositionVbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * positions.size(), positions.data(), GL_STREAM_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, _instanceDirectionVbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * directions.size(), directions.data(), GL_STREAM_DRAW);
}

void ArrowSpinRenderer::draw(double aspectRatio) const {
  if (_numInstances <= 0) {
    return;
  }
  glBindVertexArray(_vao);
  glUseProgram(_program);
  
  glm::vec2 zRange = {-1, 1};
  double verticalFieldOfView = 45;
  glm::vec3 cameraPosition = {14.5, 14.5, 30};
  glm::vec3 centerPosition {14.5, 14.5, 0};
  glm::vec3 upVector = {0, 1, 0};
  
  glm::mat4 projectionMatrix = glm::perspective(verticalFieldOfView, aspectRatio, 0.1, 10000.0);
  glm::mat4 modelviewMatrix = glm::lookAt(cameraPosition, centerPosition, upVector);
  glm::vec4 lightPosition = modelviewMatrix * glm::vec4(cameraPosition, 1.0);
  
  glUniformMatrix4fv(glGetUniformLocation(_program, "uProjectionMatrix"), 1, false, glm::value_ptr(projectionMatrix));
  glUniformMatrix4fv(glGetUniformLocation(_program, "uModelviewMatrix"), 1, false, glm::value_ptr(modelviewMatrix));
  glUniform3f(glGetUniformLocation(_program, "uLightPosition"), lightPosition[0], lightPosition[1], lightPosition[2]);
  glUniform2f(glGetUniformLocation(_program, "uZRange"), zRange[0], zRange[1]);
  
  glDrawElementsInstanced(GL_TRIANGLES, _numIndices, GL_UNSIGNED_SHORT, nullptr, _numInstances);
}

void ArrowSpinRenderer::_updateShaderProgram() {
  if (_program) {
    glDeleteProgram(_program);
  }
  std::string vertexShaderSource =
#include "vertex.txt"
  + getColormapImplementation("hsv");
  std::string fragmentShaderSource =
#include "fragment.txt"
  ;
  GLuint program = createProgram(vertexShaderSource, fragmentShaderSource, {"ivPosition", "ivNormal", "ivInstanceOffset", "ivInstanceDirection"});
  if (program) {
    _program = program;
  }
}

void ArrowSpinRenderer::_updateVertexData() {
  unsigned int levelOfDetail = 20;
  double coneHeight = 0.6;
  double coneRadius = 0.25;
  double cylinderHeight = 0.7;
  double cylinderRadius = 0.125;
  
  // Enforce valid range
  if (levelOfDetail < 3) {
    levelOfDetail = 3;
  }
  if (coneHeight < 0) {
    coneHeight = 0;
  }
  if (coneRadius < 0) {
    coneRadius = 0;
  }
  if (cylinderHeight < 0) {
    cylinderHeight = 0;
  }
  if (cylinderRadius < 0) {
    cylinderRadius = 0;
  }
  unsigned int i;
  glm::vec3 baseNormal = {0, 0, -1};
  double zOffset = (cylinderHeight-coneHeight)/2;
  double l = sqrt(coneRadius*coneRadius+coneHeight*coneHeight);
  double f1 = coneRadius/l;
  double f2 = coneHeight/l;
  std::vector<glm::vec3> vertexData;
  vertexData.reserve(levelOfDetail*5*2);
  // The tip has no normal to prevent a discontinuity.
  vertexData.push_back({0, 0, zOffset+coneHeight});
  vertexData.push_back({0, 0, 0});
  for (i = 0; i < levelOfDetail; i++) {
    double alpha = 2*M_PI*i/levelOfDetail;
    vertexData.push_back({coneRadius*cos(alpha), coneRadius*sin(alpha), zOffset});
    vertexData.push_back({f2*cos(alpha), f2*sin(alpha), f1});
  }
  for (i = 0; i < levelOfDetail; i++) {
    double alpha = 2*M_PI*i/levelOfDetail;
    vertexData.push_back({coneRadius*cos(alpha), coneRadius*sin(alpha), zOffset});
    vertexData.push_back(baseNormal);
  }
  for (i = 0; i < levelOfDetail; i++) {
    double alpha = 2*M_PI*i/levelOfDetail;
    vertexData.push_back({cylinderRadius*cos(alpha), cylinderRadius*sin(alpha), zOffset-cylinderHeight});
    vertexData.push_back(baseNormal);
  }
  for (i = 0; i < levelOfDetail; i++) {
    double alpha = 2*M_PI*i/levelOfDetail;
    vertexData.push_back({cylinderRadius*cos(alpha), cylinderRadius*sin(alpha), zOffset-cylinderHeight});
    vertexData.push_back({cos(alpha), sin(alpha), 0});
  }
  for (i = 0; i < levelOfDetail; i++) {
    double alpha = 2*M_PI*i/levelOfDetail;
    vertexData.push_back({cylinderRadius*cos(alpha), cylinderRadius*sin(alpha), zOffset});
    vertexData.push_back({cos(alpha), sin(alpha), 0});
  }
  std::vector<GLushort> indices;
  indices.reserve(levelOfDetail*15);
  for (i = 0; i < levelOfDetail; i++) {
    indices.push_back(1+i);
    indices.push_back(1+(i+1)%levelOfDetail);
    indices.push_back(0);
  }
  for (i = 0; i < levelOfDetail; i++) {
    indices.push_back(levelOfDetail+1);
    indices.push_back(levelOfDetail+1+(i+1)%levelOfDetail);
    indices.push_back(levelOfDetail+1+i);
  }
  for (i = 0; i < levelOfDetail; i++) {
    indices.push_back(levelOfDetail*2+1);
    indices.push_back(levelOfDetail*2+1+(i+1)%levelOfDetail);
    indices.push_back(levelOfDetail*2+1+i);
  }
  for (i = 0; i < levelOfDetail; i++) {
    indices.push_back(levelOfDetail*3+1+i);
    indices.push_back(levelOfDetail*3+1+(i+1)%levelOfDetail);
    indices.push_back(levelOfDetail*4+1+i);
    indices.push_back(levelOfDetail*4+1+i);
    indices.push_back(levelOfDetail*3+1+(i+1)%levelOfDetail);
    indices.push_back(levelOfDetail*4+1+(i+1)%levelOfDetail);
  }
  glBindBuffer(GL_ARRAY_BUFFER, _vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * vertexData.size(), vertexData.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ibo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLushort) * indices.size(), indices.data(), GL_STATIC_DRAW);
  _numIndices = indices.size();
}