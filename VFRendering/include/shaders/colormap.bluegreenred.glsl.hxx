#ifndef COLORMAP_BLUEGREENRED_GLSL_HXX
#define COLORMAP_BLUEGREENRED_GLSL_HXX

static const std::string COLORMAP_BLUEGREENRED_GLSL = R"LITERAL(
float atan2(float y, float x) {
    return x == 0.0 ? sign(y)*3.14159/2.0 : atan(y, x);
}
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec3 colormap(vec3 direction) {
    float hue = 1.0/3.0-normalize(direction).z/3.0;
    return hsv2rgb(vec3(hue, 1.0, 1.0));
}
)LITERAL";

#endif

