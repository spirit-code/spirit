#ifndef COORDINATESYSTEM_VERT_GLSL_HXX
#define COORDINATESYSTEM_VERT_GLSL_HXX

static const std::string COORDINATESYSTEM_VERT_GLSL = R"LITERAL(
#version 330

uniform mat4 uProjectionMatrix;
uniform mat4 uModelviewMatrix;
uniform vec3 uOrigin;
uniform vec3 uAxisLength;
in vec3 ivPosition;
in vec3 ivDirection;
out vec3 vfColor;

vec3 colormap(vec3 direction);

void main(void) {
  vfColor = colormap(normalize(ivDirection));
  gl_Position = uProjectionMatrix * uModelviewMatrix * vec4(uOrigin+ivPosition*uAxisLength, 1.0);
}
)LITERAL";

#endif

