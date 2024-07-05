#ifndef BOUNDINGBOX_FRAG_GLSL_HXX
#define BOUNDINGBOX_FRAG_GLSL_HXX

#include "shader_header.hxx"

static const std::string BOUNDINGBOX_FRAG_GLSL = FRAG_SHADER_HEADER + R"LITERAL(

uniform vec3 uColor;

in float vfDashingValue;

void main(void) {
    if (mod(floor(vfDashingValue), 2.0) != 0.0) {
        discard;
    }
    fo_FragColor = vec4(uColor, 1.0);
}
)LITERAL";

#endif

