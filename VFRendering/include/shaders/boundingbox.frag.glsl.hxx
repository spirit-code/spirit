#ifndef BOUNDINGBOX_FRAG_GLSL_HXX
#define BOUNDINGBOX_FRAG_GLSL_HXX

#include "shader_header.hxx"

static const std::string BOUNDINGBOX_FRAG_GLSL = SHADER_HEADER + R"LITERAL(

uniform vec3 uColor;

in float vfDashingValue;

out vec4 fo_FragColor;

void main(void) {
    if (int(floor(vfDashingValue)) % 2 != 0) {
        discard;
    }
    fo_FragColor = vec4(uColor, 1.0);
}
)LITERAL";

#endif

