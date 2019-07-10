#ifndef ISOSURFACE_FRAG_GLSL_HXX
#define ISOSURFACE_FRAG_GLSL_HXX

#include "shader_header.hxx"

static const std::string ISOSURFACE_FRAG_GLSL = FRAG_SHADER_HEADER + R"LITERAL(

in vec3 vfPosition;
in vec3 vfDirection;
in vec3 vfNormal;

vec3 colormap(vec3 direction);
bool is_visible(vec3 position, vec3 direction);
float lighting(vec3 position, vec3 normal);

void main(void) {
  if (is_visible(vfPosition, vfDirection)) {
    vec3 color = colormap(normalize(vfDirection));
    fo_FragColor = vec4(color * lighting(vfPosition, normalize(vfNormal)), 1.0);
  } else {
    discard;
  }
}

)LITERAL";

#endif

