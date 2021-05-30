#ifndef SPHERE_BACKGROUND_VERT_GLSL_HXX
#define SPHERE_BACKGROUND_VERT_GLSL_HXX

#include "shader_header.hxx"

static const std::string SPHERE_BACKGROUND_VERT_GLSL = VERT_SHADER_HEADER + R"LITERAL(

uniform float uAspectRatio;
uniform float uInnerSphereRadius;
in vec3 ivPosition;
out vec3 vfPosition;

void main(void) {
  vfPosition = ivPosition;
  if (uAspectRatio > 1.0) {
    gl_Position = vec4(vfPosition.xy*vec2(uInnerSphereRadius/uAspectRatio, uInnerSphereRadius), 0.0, 1.0);
  } else {
    gl_Position = vec4(vfPosition.xy*vec2(uInnerSphereRadius, uInnerSphereRadius*uAspectRatio), 0.0, 1.0);
  }
}
)LITERAL";

#endif

