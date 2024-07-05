#ifndef BOUNDINGBOX_VERT_GLSL_HXX
#define BOUNDINGBOX_VERT_GLSL_HXX

#include "shader_header.hxx"

static const std::string BOUNDINGBOX_VERT_GLSL = VERT_SHADER_HEADER + R"LITERAL(

uniform mat4 uProjectionMatrix;
uniform mat4 uModelviewMatrix;
in vec3 ivPosition;
in float ivDashingValue;
out float vfDashingValue;

void main(void) {
    vfDashingValue = ivDashingValue;
    gl_Position = uProjectionMatrix * uModelviewMatrix * vec4(ivPosition, 1.0);
}
)LITERAL";

#endif

