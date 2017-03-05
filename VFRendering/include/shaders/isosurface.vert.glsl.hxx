#ifndef ISOSURFACE_VERT_GLSL_HXX
#define ISOSURFACE_VERT_GLSL_HXX

static const std::string ISOSURFACE_VERT_GLSL = R"LITERAL(
#version 330

uniform mat4 uProjectionMatrix;
uniform mat4 uModelviewMatrix;
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
    gl_Position = uProjectionMatrix * (uModelviewMatrix * vec4(ivPosition, 1.0));
}
)LITERAL";

#endif

