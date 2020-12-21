#ifndef COORDINATESYSTEM_FRAG_GLSL_HXX
#define COORDINATESYSTEM_FRAG_GLSL_HXX

#include "shader_header.hxx"

static const std::string COORDINATESYSTEM_FRAG_GLSL = FRAG_SHADER_HEADER + R"LITERAL(

in vec3 vfColor;
in vec3 vfNormal;

void main(void) {
  fo_FragColor = vec4(vfColor*abs(normalize(vfNormal).z), 1.0);
}
)LITERAL";

#endif

