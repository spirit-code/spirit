#ifndef BOUNDINGBOX_VERT_GLSL_HXX
#define BOUNDINGBOX_VERT_GLSL_HXX

static const std::string BOUNDINGBOX_VERT_GLSL = R"LITERAL(
#version 330

uniform mat4 uProjectionMatrix;
uniform mat4 uModelviewMatrix;
in vec3 ivPosition;

void main(void) {
  gl_Position = uProjectionMatrix * uModelviewMatrix * vec4(ivPosition, 1.0);
}
)LITERAL";

#endif

