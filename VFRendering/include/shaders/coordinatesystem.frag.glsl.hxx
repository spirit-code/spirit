#ifndef COORDINATESYSTEM_FRAG_GLSL_HXX
#define COORDINATESYSTEM_FRAG_GLSL_HXX

static const std::string COORDINATESYSTEM_FRAG_GLSL = R"LITERAL(
#version 330

in vec3 vfColor;
out vec4 fo_FragColor;

void main(void) {
  fo_FragColor = vec4(vfColor, 1.0);
}
)LITERAL";

#endif

