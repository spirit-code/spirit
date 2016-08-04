#version 330

uniform vec3 uColor;
out vec4 fo_FragColor;

void main(void) {
  fo_FragColor = vec4(uColor, 1.0);
}
