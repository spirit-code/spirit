#version 330

uniform mat4 uProjectionMatrix;
uniform mat4 uModelviewMatrix;
in vec3 ivPosition;
in vec3 ivDirection;
out vec3 vfDirection;

void main(void) {
  vfDirection = normalize(ivDirection);
  gl_Position = uProjectionMatrix * (uModelviewMatrix * vec4(ivPosition, 1.0));
}
