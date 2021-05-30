#ifndef COLORMAP_BLUEWHITERED_GLSL_HXX
#define COLORMAP_BLUEWHITERED_GLSL_HXX

static const std::string COLORMAP_BLUEWHITERED_GLSL = R"LITERAL(
vec3 colormap(vec3 direction) {
    if (direction.z < 0.0) {
        vec3 color_down = vec3(0.0, 0.0, 1.0);
        vec3 color_up = vec3(1.0, 1.0, 1.0);
        return mix(color_down, color_up, normalize(direction).z+1.0);
    } else {
        vec3 color_down = vec3(1.0, 1.0, 1.0);
        vec3 color_up = vec3(1.0, 0.0, 0.0);
        return mix(color_down, color_up, normalize(direction).z);
    }
}

)LITERAL";

#endif

