#version 330

uniform float uAspectRatio;
uniform float uInnerSphereRadius;
in vec3 ivPosition;
out vec3 vfPosition;

void main(void) {
  vfPosition = ivPosition;
  gl_Position = vec4(vfPosition.xy*vec2(uInnerSphereRadius/uAspectRatio, uInnerSphereRadius), 0.0, 1.0);
}
