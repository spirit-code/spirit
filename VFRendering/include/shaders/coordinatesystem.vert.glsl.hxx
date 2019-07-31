#ifndef COORDINATESYSTEM_VERT_GLSL_HXX
#define COORDINATESYSTEM_VERT_GLSL_HXX

#include "shader_header.hxx"

static const std::string COORDINATESYSTEM_VERT_GLSL = VERT_SHADER_HEADER + R"LITERAL(

uniform mat4 uProjectionMatrix;
uniform mat4 uModelviewMatrix;
uniform vec3 uOrigin;
in vec3 ivPosition;
in vec3 ivNormal;
in vec3 ivDirection;
out vec3 vfNormal;
out vec3 vfColor;

vec3 colormap(vec3 direction);

void main(void) {
    if (length(ivDirection) < 0.5) {
        vfColor = vec3(1, 1, 1);
    } else {
        vfColor = colormap(normalize(ivDirection));
    }
  vfNormal = (uModelviewMatrix * vec4(ivNormal, 0.0)).xyz;
  gl_Position = uProjectionMatrix * uModelviewMatrix * vec4(uOrigin+ivPosition, 1.0);
}
)LITERAL";

#endif

