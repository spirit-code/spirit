#ifndef SURFACE_FRAG_GLSL_HXX
#define SURFACE_FRAG_GLSL_HXX

static const std::string SURFACE_FRAG_GLSL = R"LITERAL(
#version 330

in vec3 vfPosition;
in vec3 vfDirection;
out vec4 fo_FragColor;

vec3 colormap(vec3 direction);
bool is_visible(vec3 position, vec3 direction);

void main(void) {
  if (is_visible(vfPosition, vfDirection)) {
    vec3 color = colormap(normalize(vfDirection));
    fo_FragColor = vec4(color, 1.0);
  } else {
    discard;
  }
}
)LITERAL";

#endif

