#version 330

in vec3 vfPosition;
out vec4 fo_FragColor;

void main(void) {
  float l = length(vfPosition);
  if (l > 1.0) {
    discard;
  } else {
    vec3 color = 0.2+0.4*sqrt(1.0-l*l)*vec3(1.0, 1.0, 1.0);
    fo_FragColor = vec4(color, 1.0);
  }
}
