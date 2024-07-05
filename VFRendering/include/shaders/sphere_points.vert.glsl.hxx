#ifndef SPHERE_POINTS_VERT_GLSL_HXX
#define SPHERE_POINTS_VERT_GLSL_HXX

#include "shader_header.hxx"

static const std::string SPHERE_POINTS_VERT_GLSL = VERT_SHADER_HEADER + R"LITERAL(

uniform mat4 uProjectionMatrix;
uniform mat4 uModelviewMatrix;
uniform vec2 uPointSizeRange;
uniform float uAspectRatio;
uniform float uInnerSphereRadius;
uniform float uUseFakePerspective;
in vec3 ivPosition;
in vec3 ivDirection;
out vec3 vfDirection;
out vec3 vfPosition;

void main(void) {
  vfPosition = ivPosition;
  vfDirection = normalize(ivDirection);
  gl_Position = uProjectionMatrix * uModelviewMatrix * vec4(vfDirection*0.99, 1.0);
  vec2 clipPosition;
  if (uAspectRatio > 1.0) {
    clipPosition = gl_Position.xy * vec2(uAspectRatio, 1.0);
  } else {
    clipPosition = gl_Position.xy * vec2(1.0, 1.0/uAspectRatio);
  }
  float clipRadius = length(clipPosition);
  float rotatedDirectionZ = dot(vec3(uModelviewMatrix[0][2], uModelviewMatrix[1][2], uModelviewMatrix[2][2]), vfDirection);
  if ((clipRadius <= uInnerSphereRadius) && (rotatedDirectionZ < 0.0)) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
  }
  gl_PointSize = uPointSizeRange.x + (uPointSizeRange.y-uPointSizeRange.x) * sqrt(max(0.0, 1.0-clipRadius*clipRadius)) * (5.0-uUseFakePerspective*gl_Position.z) / 5.0;
}
)LITERAL";

#endif

