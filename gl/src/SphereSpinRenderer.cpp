
#include <iostream>

#ifndef __gl_h_
#include <glad/glad.h>
#endif
#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "SphereSpinRenderer.h"
#include "utilities.h"

SphereSpinRenderer::SphereSpinRenderer() {
  // TODO: initGL if possible
  // TODO: updateSpins if possible
}

SphereSpinRenderer::~SphereSpinRenderer() {
  glDeleteVertexArrays(1, &_vao1);
  glDeleteVertexArrays(1, &_vao2);
  glDeleteBuffers(1, &_fakeSphereVbo);
  glDeleteBuffers(1, &_instanceDirectionVbo);
  glDeleteProgram(_program1);
  glDeleteProgram(_program2);
  glDisableVertexAttribArray(0);
}

void SphereSpinRenderer::optionsHaveChanged(const std::vector<int>& changedOptions) {
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

void SphereSpinRenderer::initGL() {
  glGenVertexArrays(1, &_vao1);
  glBindVertexArray(_vao1);
  glGenBuffers(1, &_instanceDirectionVbo);
  glBindBuffer(GL_ARRAY_BUFFER, _instanceDirectionVbo);
  glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, nullptr);
  glEnableVertexAttribArray(0);
  
  glGenVertexArrays(1, &_vao2);
  glBindVertexArray(_vao2);
  glGenBuffers(1, &_fakeSphereVbo);
  glBindBuffer(GL_ARRAY_BUFFER, _fakeSphereVbo);
  glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, nullptr);
  std::vector<GLfloat> vertices = {
    -1, -1, 0,
    1, -1, 0,
    -1, 1, 0,
    -1, 1, 0,
    1, -1, 0,
    1, 1, 0};
  glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*vertices.size(), vertices.data(), GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  
  _updateShaderProgram();
}

void SphereSpinRenderer::updateSpins(const std::vector<glm::vec3>& positions,
                                      const std::vector<glm::vec3>& directions) {
  glBindVertexArray(_vao2);
  glBindBuffer(GL_ARRAY_BUFFER, _instanceDirectionVbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * directions.size(), directions.data(), GL_STREAM_DRAW);
  _numInstances = directions.size();
}

void SphereSpinRenderer::draw(double aspectRatio) const {
  double innerSphereRadius = _options.get<SphereSpinRendererOptions::INNER_SPHERE_RADIUS>();
  if (innerSphereRadius > 0.0) {
    if (innerSphereRadius > 0.99) {
      innerSphereRadius = 0.99;
    }
    glUseProgram(_program2);
    
    glBindVertexArray(_vao2);
    glUniform1f(glGetUniformLocation(_program2, "uAspectRatio"), aspectRatio);
    glUniform1f(glGetUniformLocation(_program2, "uInnerSphereRadius"), innerSphereRadius);
    glDisable(GL_CULL_FACE);
    glDepthMask(false);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glDepthMask(true);
    glEnable(GL_CULL_FACE);
  }
  
  if (_numInstances <= 0) {
    return;
  }
  
  glUseProgram(_program1);
  glBindVertexArray(_vao1);
  
  
  glm::vec2 zRange = _options.get<ISpinRenderer::Option::Z_RANGE>();
  if (zRange.x <= -1) {
    zRange.x = -2;
  }
  if (zRange.y >= 1) {
    zRange.y = 2;
  }
  glm::vec3 cameraPosition = _options.get<ISpinRenderer::Option::CAMERA_POSITION>();
  glm::vec3 centerPosition = _options.get<ISpinRenderer::Option::CENTER_POSITION>();
  glm::vec3 upVector = _options.get<ISpinRenderer::Option::UP_VECTOR>();
  glm::vec2 pointSizeRange = _options.get<SphereSpinRendererOptions::POINT_SIZE_RANGE>();
  
  glm::mat4 projectionMatrix = glm::ortho(-aspectRatio, aspectRatio, -1.0, 1.0, 2.0, 0.0);
  glm::mat4 modelviewMatrix = glm::lookAt(glm::normalize(cameraPosition-centerPosition), {0, 0, 0}, upVector);
  glm::vec4 lightPosition = modelviewMatrix * glm::vec4(cameraPosition, 1.0);
  
  glUniformMatrix4fv(glGetUniformLocation(_program1, "uProjectionMatrix"), 1, false, glm::value_ptr(projectionMatrix));
  glUniformMatrix4fv(glGetUniformLocation(_program1, "uModelviewMatrix"), 1, false, glm::value_ptr(modelviewMatrix));
  glUniform3f(glGetUniformLocation(_program1, "uLightPosition"), lightPosition[0], lightPosition[1], lightPosition[2]);
  glUniform2f(glGetUniformLocation(_program1, "uZRange"), zRange[0], zRange[1]);
  glUniform2f(glGetUniformLocation(_program1, "uPointSizeRange"), pointSizeRange[0], pointSizeRange[1]);
  
  glUniform1f(glGetUniformLocation(_program1, "uAspectRatio"), aspectRatio);
  glUniform1f(glGetUniformLocation(_program1, "uInnerSphereRadius"), innerSphereRadius);
  bool useSphereFakePerspective = _options.get<SphereSpinRendererOptions::USE_SPHERE_FAKE_PERSPECTIVE>();
  if (useSphereFakePerspective) {
    glUniform1f(glGetUniformLocation(_program1, "uUseFakePerspective"), 1.0);
  } else {
    glUniform1f(glGetUniformLocation(_program1, "uUseFakePerspective"), 0.0);
  }
  
  glEnable(GL_PROGRAM_POINT_SIZE);
  glDisable(GL_CULL_FACE);
  glDrawArrays(GL_POINTS, 0, _numInstances);
  glEnable(GL_CULL_FACE);
  glDisable(GL_PROGRAM_POINT_SIZE);
}

void SphereSpinRenderer::_updateShaderProgram() {
  {
    if (_program1) {
      glDeleteProgram(_program1);
    }
    std::string vertexShaderSource =
  #include "sphere_points.vert.txt"
    ;
    vertexShaderSource += _options.get<ISpinRenderer::Option::COLORMAP_IMPLEMENTATION>();
    std::string fragmentShaderSource =
  #include "sphere_points.frag.txt"
    ;
    fragmentShaderSource += _options.get<ISpinRenderer::Option::COLORMAP_IMPLEMENTATION>();
    GLuint program1 = createProgram(vertexShaderSource, fragmentShaderSource, {"ivDirection"});
    if (program1) {
      _program1 = program1;
    }
  }
  {
    if (_program2) {
      glDeleteProgram(_program2);
    }
    std::string vertexShaderSource =
  #include "sphere_background.vert.txt"
    ;
    std::string fragmentShaderSource =
  #include "sphere_background.frag.txt"
    ;
    GLuint program2 = createProgram(vertexShaderSource, fragmentShaderSource, {"ivPosition"});
    if (program2) {
      _program2 = program2;
    }
  }
}
