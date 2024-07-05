#ifndef DOTS_CIRCLE_FRAG_GLSL_HXX
#define DOTS_CIRCLE_FRAG_GLSL_HXX

#include "shader_header.hxx"

static const std::string DOT_CIRCLE_FRAG_GLSL = FRAG_SHADER_HEADER + R"LITERAL(

in vec3 vfColor;

void main(void) {
  if( dot( gl_PointCoord-0.5, gl_PointCoord-0.5 ) > 0.25 ) 
    discard;
  else
   fo_FragColor = vec4( vfColor, 1.0 ); 
}
)LITERAL";

#endif

