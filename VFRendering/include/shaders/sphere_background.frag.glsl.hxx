#ifndef SPHERE_BACKGROUND_FRAG_GLSL_HXX
#define SPHERE_BACKGROUND_FRAG_GLSL_HXX

#include "shader_header.hxx"

static const std::string SPHERE_BACKGROUND_FRAG_GLSL = FRAG_SHADER_HEADER + R"LITERAL(

in vec3 vfPosition;

void main(void) {
  float l = length(vfPosition);
  if (l > 1.0) {
    discard;
  } else {
    vec3 color = 0.2+0.4*sqrt(1.0-l*l)*vec3(1.0, 1.0, 1.0);
    fo_FragColor = vec4(color, 1.0);
  }
}
)LITERAL";

#endif

