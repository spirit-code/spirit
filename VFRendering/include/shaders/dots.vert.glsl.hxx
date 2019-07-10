#ifndef DOTS_VERT_GLSL_HXX
#define DOTS_VERT_GLSL_HXX

#include "shader_header.hxx"

static const std::string DOT_VERT_GLSL = VERT_SHADER_HEADER + R"LITERAL(

uniform mat4 uProjectionMatrix;
uniform mat4 uModelviewMatrix;
uniform float uDotRadius;

in vec3 ivDotCoordinates;
in vec3 ivDotDirection;
out vec3 vfColor;

vec3 colormap(vec3 direction);

bool is_visible(vec3 position, vec3 direction);

void main(void) {
  float direction_length = length( ivDotDirection );
  
  if ( is_visible( ivDotCoordinates, ivDotDirection ) && direction_length > 0.0) {
    vfColor = colormap( normalize( ivDotDirection ) );
    vec3 vfPosition = ( uModelviewMatrix * vec4( ivDotCoordinates, 1.0 ) ).xyz;
    gl_Position = uProjectionMatrix * vec4( vfPosition, 1.0 );
  } else {
    gl_Position = vec4(2.0, 2.0, 2.0, 0.0);
  }
  
  gl_PointSize = uDotRadius / gl_Position.z;
  float point_size = gl_PointSize;
}
)LITERAL";

#endif
