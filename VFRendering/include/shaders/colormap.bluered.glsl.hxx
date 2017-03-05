#ifndef COLORMAP_BLUERED_GLSL_HXX
#define COLORMAP_BLUERED_GLSL_HXX

static const std::string COLORMAP_BLUERED_GLSL = R"LITERAL(
vec3 colormap(vec3 direction) {
     vec3 color_down = vec3(0.0, 0.0, 1.0);
     vec3 color_up = vec3(1.0, 0.0, 0.0);
     return mix(color_down, color_up, normalize(direction).z*0.5+0.5);
}
)LITERAL";

#endif

