#ifndef ARROWS_FRAG_GLSL_HXX
#define ARROWS_FRAG_GLSL_HXX

static const std::string ARROWS_FRAG_GLSL = R"LITERAL(
#version 330
uniform vec3 uLightPosition;
in vec3 vfPosition;
in vec3 vfNormal;
in vec3 vfColor;
out vec4 fo_FragColor;

void main(void) {
  vec3 cameraLocation = vec3(0, 0, 0);
  vec3 normal = normalize(vfNormal);
  vec3 lightDirection = normalize(uLightPosition-vfPosition);
  vec3 reflectionDirection = normalize(reflect(lightDirection, normal));
  float specular = 0.2*pow(max(0.0, -reflectionDirection.z), 8.0);
  float diffuse = 0.7*max(0.0, dot(normal, lightDirection));
  float ambient = 0.2;
  fo_FragColor = vec4((ambient+diffuse)*vfColor + specular*vec3(1, 1, 1), 1.0);
}

)LITERAL";

#endif

