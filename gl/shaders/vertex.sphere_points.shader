#version 330

uniform mat4 uProjectionMatrix;
uniform mat4 uModelviewMatrix;
uniform vec2 uPointSizeRange;
uniform float uAspectRatio;
uniform float uInnerSphereRadius;
uniform float uUseFakePerspective;
in vec3 ivDirection;
out vec3 vfDirection;

void main(void) {
  vfDirection = normalize(ivDirection);
  gl_Position = uProjectionMatrix * uModelviewMatrix * vec4(vfDirection*0.99, 1.0);
  vec2 clipPosition = vec2(gl_Position.x * uAspectRatio, gl_Position.y);
  float clipRadius = length(clipPosition);
  float rotatedDirectionZ = dot(vec3(uModelviewMatrix[0][2], uModelviewMatrix[1][2], uModelviewMatrix[2][2]), vfDirection);
  if ((clipRadius <= uInnerSphereRadius) && (rotatedDirectionZ < 0.0)) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
  }
  gl_PointSize = uPointSizeRange.x + (uPointSizeRange.y-uPointSizeRange.x) * sqrt(max(0.0, 1.0-clipRadius*clipRadius)) * (5.0-uUseFakePerspective*gl_Position.z) / 5.0;
}
