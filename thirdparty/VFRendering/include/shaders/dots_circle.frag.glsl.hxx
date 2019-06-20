#ifndef DOTS_CIRCLE_FRAG_GLSL_HXX
#define DOTS_CIRCLE_FRAG_GLSL_HXX

static const std::string DOT_CIRCLE_FRAG_GLSL = R"LITERAL(
#version 330

in vec3 vfColor;
out vec4 fo_FragColor;

void main(void) {
  if( dot( gl_PointCoord-0.5, gl_PointCoord-0.5 ) > 0.25 ) 
    discard;
  else
   fo_FragColor = vec4( vfColor, 1.0 ); 
}
)LITERAL";

#endif

