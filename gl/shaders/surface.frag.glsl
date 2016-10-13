#version 330

uniform vec2 uZRange;
in vec3 vfDirection;
out vec4 fo_FragColor;

vec3 colormap(vec3 direction);

void main(void) {
  vec3 direction = normalize(vfDirection);
  if (direction.z >= uZRange.x && direction.z <= uZRange.y) {
    vec3 color = colormap(direction);
    fo_FragColor = vec4(color, 1.0);
  } else {
    discard;
  }
}
