#ifndef ISOSURFACE_VERT_GLSL_HXX
#define ISOSURFACE_VERT_GLSL_HXX

#include "shader_header.hxx"

static const std::string ISOSURFACE_VERT_GLSL = VERT_SHADER_HEADER + R"LITERAL(

uniform mat4 uProjectionMatrix;
uniform mat4 uModelviewMatrix;
uniform float uFlipNormals;
in vec3 ivPosition;
in vec3 ivDirection;
in vec3 ivNormal;
out vec3 vfPosition;
out vec3 vfDirection;
out vec3 vfNormal;


void main(void) {
    vfPosition = ivPosition;
    vfDirection = normalize(ivDirection);
    vfNormal = normalize((uModelviewMatrix * vec4(ivNormal, 0.0)).xyz);
    vfNormal *= uFlipNormals;
    gl_Position = uProjectionMatrix * (uModelviewMatrix * vec4(ivPosition, 1.0));
}
)LITERAL";

#endif

