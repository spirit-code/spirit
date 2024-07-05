#ifndef DOTS_SQUARE_FRAG_GLSL_HXX
#define DOTS_SQUARE_FRAG_GLSL_HXX
#include "shader_header.hxx"

static const std::string DOT_SQUARE_FRAG_GLSL = FRAG_SHADER_HEADER + R"LITERAL(

in vec3 vfColor;

void main(void) {
   fo_FragColor = vec4( vfColor, 1.0 ); 
}
)LITERAL";

#endif

