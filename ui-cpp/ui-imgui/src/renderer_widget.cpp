#include <fonts.hpp>
#include <renderer_widget.hpp>
#include <widgets.hpp>

#include <VFRendering/ArrowRenderer.hxx>
#include <VFRendering/BoundingBoxRenderer.hxx>
#include <VFRendering/CombinedRenderer.hxx>
#include <VFRendering/CoordinateSystemRenderer.hxx>
#include <VFRendering/DotRenderer.hxx>
#include <VFRendering/GlyphRenderer.hxx>
#include <VFRendering/IsosurfaceRenderer.hxx>
#include <VFRendering/ParallelepipedRenderer.hxx>
#include <VFRendering/SphereRenderer.hxx>
#include <VFRendering/SurfaceRenderer.hxx>

#include <imgui/imgui.h>

#include <glm/gtc/type_ptr.hpp>

#include <Spirit/Geometry.h>
#include <Spirit/Hamiltonian.h>

#include <fmt/format.h>

#include <string>

namespace ui
{

std::string get_lighting_implementation( bool draw_shadows )
{
    if( draw_shadows )
    {
        return "uniform vec3 uLightPosition;"
               "float lighting(vec3 position, vec3 normal)"
               "{"
               "    vec3 lightDirection = -normalize(uLightPosition-position);"
               "    float diffuse = 0.7*max(0.0, dot(normal, lightDirection));"
               "    float ambient = 0.2;"
               "    return diffuse+ambient;"
               "}";
    }

    return "float lighting(vec3 position, vec3 normal) { return 1.0; }";
}

std::string get_is_visible_implementation(
    std::shared_ptr<State> state, float filter_position_min[3], float filter_position_max[3],
    float filter_direction_min[3], float filter_direction_max[3] )
{
    const float epsilon = 1e-5;

    float b_min[3], b_max[3], b_range[3];
    Geometry_Get_Bounds( state.get(), b_min, b_max );

    float filter_pos_min[3], filter_pos_max[3];
    float filter_dir_min[3], filter_dir_max[3];
    for( int dim = 0; dim < 3; ++dim )
    {
        b_range[dim]        = b_max[dim] - b_min[dim];
        filter_pos_min[dim] = b_min[dim] + filter_position_min[dim] * b_range[dim] - epsilon;
        filter_pos_max[dim] = b_max[dim] + ( filter_position_max[dim] - 1 ) * b_range[dim] + epsilon;

        filter_dir_min[dim] = filter_direction_min[dim] - epsilon;
        filter_dir_max[dim] = filter_direction_max[dim] + epsilon;
    }
    return fmt::format(
        R"(
            bool is_visible(vec3 position, vec3 direction)
            {{
                float x_min_pos = {};
                float x_max_pos = {};
                bool is_visible_x_pos = position.x <= x_max_pos && position.x >= x_min_pos;

                float y_min_pos = {};
                float y_max_pos = {};
                bool is_visible_y_pos = position.y <= y_max_pos && position.y >= y_min_pos;

                float z_min_pos = {};
                float z_max_pos = {};
                bool is_visible_z_pos = position.z <= z_max_pos && position.z >= z_min_pos;

                float x_min_dir = {};
                float x_max_dir = {};
                bool is_visible_x_dir = direction.x <= x_max_dir && direction.x >= x_min_dir;

                float y_min_dir = {};
                float y_max_dir = {};
                bool is_visible_y_dir = direction.y <= y_max_dir && direction.y >= y_min_dir;

                float z_min_dir = {};
                float z_max_dir = {};
                bool is_visible_z_dir = direction.z <= z_max_dir && direction.z >= z_min_dir;

                return is_visible_x_pos && is_visible_y_pos && is_visible_z_pos && is_visible_x_dir && is_visible_y_dir && is_visible_z_dir;
            }}
            )",
        filter_pos_min[0], filter_pos_max[0], filter_pos_min[1], filter_pos_max[1], filter_pos_min[2],
        filter_pos_max[2], filter_direction_min[0], filter_direction_max[0], filter_direction_min[1],
        filter_direction_max[1], filter_direction_min[2], filter_direction_max[2] );
}

std::string get_colormap(
    Colormap colormap, int phi, bool invert_z, bool invert_xy, glm::vec3 cardinal_a, glm::vec3 cardinal_b,
    glm::vec3 cardinal_c, glm::vec3 monochrome_color )
{
    int sign_z  = 1 - 2 * (int)invert_z;
    int sign_xy = 1 - 2 * (int)invert_xy;

    float P = glm::radians( (float)phi ) / 3.14159;

    std::string colormap_implementation;
    switch( colormap )
    {
            // case Colormap::MONOCHROME:
            //     colormap_implementation = R"(
            // vec3 colormap(vec3 direction)
            // {{
            //     return vec3()" + std::to_string( monochrome_color.x )
            //                               + ", " + std::to_string( monochrome_color.y ) + ", "
            //                               + std::to_string( monochrome_color.z ) + R"();
            // }}
            // )";
            //     break;
            // Custom color maps not included in VFRendering:
            // case Colormap::HSV:
            //     colormap_implementation = R"(
            // float atan2(float y, float x) {
            //     return x == 0.0 ? sign(y)*3.14159/2.0 : atan(y, x);
            // }
            // vec3 hsv2rgb(vec3 c) {
            //     vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            //     vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            //     return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
            // }
            // vec3 colormap(vec3 direction) {
            //     vec3 cardinal_a = vec3()" + std::to_string( cardinal_a.x )
            //                               + ", " + std::to_string( cardinal_a.y ) + ", " + std::to_string(
            //                               cardinal_a.z )
            //                               + R"();
            //     vec3 cardinal_b = vec3()" + std::to_string( cardinal_b.x )
            //                               + ", " + std::to_string( cardinal_b.y ) + ", " + std::to_string(
            //                               cardinal_b.z )
            //                               + R"();
            //     vec3 cardinal_c = vec3()" + std::to_string( cardinal_c.x )
            //                               + ", " + std::to_string( cardinal_c.y ) + ", " + std::to_string(
            //                               cardinal_c.z )
            //                               + R"();
            //     vec3 projection = vec3( dot(direction, cardinal_a), dot(direction, cardinal_b), dot(direction,
            //     cardinal_c) ); float hue = atan2()" + std::to_string( sign_xy )
            //                               + R"(*projection.x, projection.y) / 3.14159 / 2.0 + )" + std::to_string( P
            //                               )
            //                               + R"(/2.0;
            //     float saturation = projection.z * )"
            //                               + std::to_string( sign_z ) + R"(;
            //     if (saturation > 0.0) {
            //         return hsv2rgb(vec3(hue, 1.0-saturation, 1.0));
            //     } else {
            //         return hsv2rgb(vec3(hue, 1.0, 1.0+saturation));
            //     }
            // }
            // )";
            //     break;
            // case Colormap::HSV_NO_Z:
            //     colormap_implementation = R"(
            // float atan2(float y, float x) {
            //     return x == 0.0 ? sign(y)*3.14159/2.0 : atan(y, x);
            // }
            // vec3 hsv2rgb(vec3 c) {
            //     vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            //     vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            //     return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
            // }
            // vec3 colormap(vec3 direction) {
            //     vec3 cardinal_a = vec3()" + std::to_string( cardinal_a.x )
            //                               + ", " + std::to_string( cardinal_a.y ) + ", " + std::to_string(
            //                               cardinal_a.z )
            //                               + R"();
            //     vec3 cardinal_b = vec3()" + std::to_string( cardinal_b.x )
            //                               + ", " + std::to_string( cardinal_b.y ) + ", " + std::to_string(
            //                               cardinal_b.z )
            //                               + R"();
            //     vec3 cardinal_c = vec3()" + std::to_string( cardinal_c.x )
            //                               + ", " + std::to_string( cardinal_c.y ) + ", " + std::to_string(
            //                               cardinal_c.z )
            //                               + R"();
            //     vec3 projection = vec3( dot(direction, cardinal_a), dot(direction, cardinal_b), dot(direction,
            //     cardinal_c) ); float hue = atan2()" + std::to_string( sign_xy )
            //                               + R"(*projection.x, projection.y) / 3.14159 / 2.0 + )" + std::to_string( P
            //                               )
            //                               + R"(;
            //     return hsv2rgb(vec3(hue, 1.0, 1.0));
            // }
            // )";
            //     break;
            // case Colormap::BLUE_RED:
            //     colormap_implementation = R"(
            // vec3 colormap(vec3 direction) {
            //     vec3 cardinal_a = vec3()" + std::to_string( cardinal_a.x )
            //                               + ", " + std::to_string( cardinal_a.y ) + ", " + std::to_string(
            //                               cardinal_a.z )
            //                               + R"();
            //     vec3 cardinal_b = vec3()" + std::to_string( cardinal_b.x )
            //                               + ", " + std::to_string( cardinal_b.y ) + ", " + std::to_string(
            //                               cardinal_b.z )
            //                               + R"();
            //     vec3 cardinal_c = vec3()" + std::to_string( cardinal_c.x )
            //                               + ", " + std::to_string( cardinal_c.y ) + ", " + std::to_string(
            //                               cardinal_c.z )
            //                               + R"();
            //     vec3 projection = vec3( dot(direction, cardinal_a), dot(direction, cardinal_b), dot(direction,
            //     cardinal_c) ); float z_sign = projection.z * )"
            //                               + std::to_string( sign_z ) + R"(;
            //     vec3 color_down = vec3(0.0, 0.0, 1.0);
            //     vec3 color_up = vec3(1.0, 0.0, 0.0);
            //     return mix(color_down, color_up, z_sign*0.5+0.5);
            // }
            // )";
            //     break;
            // case Colormap::BLUE_GREEN_RED:
            //     colormap_implementation = R"(
            // float atan2(float y, float x) {
            //     return x == 0.0 ? sign(y)*3.14159/2.0 : atan(y, x);
            // }
            // vec3 hsv2rgb(vec3 c) {
            //     vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            //     vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            //     return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
            // }
            // vec3 colormap(vec3 direction) {
            //     vec3 cardinal_a = vec3()" + std::to_string( cardinal_a.x )
            //                               + ", " + std::to_string( cardinal_a.y ) + ", " + std::to_string(
            //                               cardinal_a.z )
            //                               + R"();
            //     vec3 cardinal_b = vec3()" + std::to_string( cardinal_b.x )
            //                               + ", " + std::to_string( cardinal_b.y ) + ", " + std::to_string(
            //                               cardinal_b.z )
            //                               + R"();
            //     vec3 cardinal_c = vec3()" + std::to_string( cardinal_c.x )
            //                               + ", " + std::to_string( cardinal_c.y ) + ", " + std::to_string(
            //                               cardinal_c.z )
            //                               + R"();
            //     vec3 projection = vec3( dot(direction, cardinal_a), dot(direction, cardinal_b), dot(direction,
            //     cardinal_c) ); float hue = 1.0/3.0-normalize(projection).z/3.0* )"
            //                               + std::to_string( sign_z ) + R"(;
            //     return hsv2rgb(vec3(hue, 1.0, 1.0));
            // }
            // )";
            //     break;
            // case Colormap::BLUE_WHITE_RED:
            //     colormap_implementation = R"(
            // vec3 colormap(vec3 direction) {
            //     vec3 cardinal_a = vec3()" + std::to_string( cardinal_a.x )
            //                               + ", " + std::to_string( cardinal_a.y ) + ", " + std::to_string(
            //                               cardinal_a.z )
            //                               + R"();
            //     vec3 cardinal_b = vec3()" + std::to_string( cardinal_b.x )
            //                               + ", " + std::to_string( cardinal_b.y ) + ", " + std::to_string(
            //                               cardinal_b.z )
            //                               + R"();
            //     vec3 cardinal_c = vec3()" + std::to_string( cardinal_c.x )
            //                               + ", " + std::to_string( cardinal_c.y ) + ", " + std::to_string(
            //                               cardinal_c.z )
            //                               + R"();
            //     vec3 projection = vec3( dot(direction, cardinal_a), dot(direction, cardinal_b), dot(direction,
            //     cardinal_c) ); float z_sign = projection.z * )"
            //                               + std::to_string( sign_z ) + R"(;
            //     if (z_sign < 0) {
            //         vec3 color_down = vec3(0.0, 0.0, 1.0);
            //         vec3 color_up = vec3(1.0, 1.0, 1.0);
            //         return mix(color_down, color_up, z_sign+1);
            //     } else {
            //         vec3 color_down = vec3(1.0, 1.0, 1.0);
            //         vec3 color_up = vec3(1.0, 0.0, 0.0);
            //         return mix(color_down, color_up, z_sign);
            //     }
            // }
            // )";
            //     break;
            // // Default is regular HSV
            // default:
            //     colormap_implementation = R"(
            // float atan2(float y, float x) {
            //     return x == 0.0 ? sign(y)*3.14159/2.0 : atan(y, x);
            // }
            // vec3 hsv2rgb(vec3 c) {
            //     vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            //     vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            //     return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
            // }
            // vec3 colormap(vec3 direction) {
            //     vec3 cardinal_a = vec3()" + std::to_string( cardinal_a.x )
            //                               + ", " + std::to_string( cardinal_a.y ) + ", " + std::to_string(
            //                               cardinal_a.z )
            //                               + R"();
            //     vec3 cardinal_b = vec3()" + std::to_string( cardinal_b.x )
            //                               + ", " + std::to_string( cardinal_b.y ) + ", " + std::to_string(
            //                               cardinal_b.z )
            //                               + R"();
            //     vec3 cardinal_c = vec3()" + std::to_string( cardinal_c.x )
            //                               + ", " + std::to_string( cardinal_c.y ) + ", " + std::to_string(
            //                               cardinal_c.z )
            //                               + R"();
            //     vec3 projection = vec3( dot(direction, cardinal_a), dot(direction, cardinal_b), dot(direction,
            //     cardinal_c) ); float hue = atan2()" + std::to_string( sign_xy )
            //                               + R"(*projection.x, projection.y) / 3.14159 / 2.0 + )" + std::to_string( P
            //                               )
            //                               + R"(/2.0;
            //     float saturation = projection.z * )"
            //                               + std::to_string( sign_z ) + R"(;
            //     if (saturation > 0.0) {
            //         return hsv2rgb(vec3(hue, 1.0-saturation, 1.0));
            //     } else {
            //         return hsv2rgb(vec3(hue, 1.0, 1.0+saturation));
            //     }
            // }
            // )";
            //     break;

        case Colormap::MONOCHROME:
            colormap_implementation = fmt::format(
                R"(
                    vec3 colormap(vec3 direction)
                    {{
                        return vec3({}, {}, {});
                    }}
                    )",
                monochrome_color.x, monochrome_color.y, monochrome_color.z );
            break;
        case Colormap::HSV:
            colormap_implementation = fmt::format(
                R"(
                    float atan2(float y, float x)
                    {{
                        return x == 0.0 ? sign(y)*3.14159/2.0 : atan(y, x);
                    }}
                    vec3 hsv2rgb(vec3 c)
                    {{
                        vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                        vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                        return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
                    }}
                    vec3 colormap(vec3 direction)
                    {{
                        vec3 cardinal_a = vec3({}, {}, {});
                        vec3 cardinal_b = vec3({}, {}, {});
                        vec3 cardinal_c = vec3({}, {}, {});
                        vec3 projection = vec3( dot(direction, cardinal_a), dot(direction, cardinal_b),
                            dot(direction, cardinal_c) );
                        float hue = atan2({}*projection.x, projection.y) / 3.14159 / 2.0 + {}/2.0;
                        float saturation = projection.z * {}; if (saturation > 0.0)
                        {{
                            return hsv2rgb(vec3(hue, 1.0-saturation, 1.0));
                        }}
                        else
                        {{
                            return hsv2rgb(vec3(hue, 1.0, 1.0+saturation));
                        }}
                    }}
                    )",
                cardinal_a.x, cardinal_a.y, cardinal_a.z, cardinal_b.x, cardinal_b.y, cardinal_b.z, cardinal_c.x,
                cardinal_c.y, cardinal_c.z, sign_xy, P, sign_z );
            break;
        case Colormap::HSV_NO_Z:
            colormap_implementation = fmt::format(
                R"(
                    float atan2(float y, float x)
                    {{
                        return x == 0.0 ? sign(y)*3.14159/2.0 : atan(y, x);
                    }}
                    vec3 hsv2rgb(vec3 c)
                    {{
                        vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                        vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                        return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
                    }}
                    vec3 colormap(vec3 direction)
                    {{
                        vec3 cardinal_a = vec3({}, {}, {});
                        vec3 cardinal_b = vec3({}, {}, {});
                        vec3 cardinal_c = vec3({}, {}, {});
                        vec3 projection = vec3( dot(direction, cardinal_a), dot(direction, cardinal_b),
                            dot(direction, cardinal_c) );
                        float hue = atan2({}*projection.x, projection.y) / 3.14159 / 2.0 + {};
                        return hsv2rgb(vec3(hue, 1.0, 1.0));
                    }}
                    )",
                cardinal_a.x, cardinal_a.y, cardinal_a.z, cardinal_b.x, cardinal_b.y, cardinal_b.z, cardinal_c.x,
                cardinal_c.y, cardinal_c.z, sign_xy, P );
            break;
        case Colormap::BLUE_RED:
            colormap_implementation = fmt::format(
                R"(
                    vec3 colormap(vec3 direction)
                    {{
                        vec3 cardinal_a = vec3({}, {}, {});
                        vec3 cardinal_b = vec3({}, {}, {});
                        vec3 cardinal_c = vec3({}, {}, {});
                        vec3 projection = vec3( dot(direction, cardinal_a), dot(direction, cardinal_b),
                            dot(direction, cardinal_c) );
                        float z_sign = projection.z * {};
                        vec3 color_down = vec3(0.0, 0.0, 1.0);
                        vec3 color_up = vec3(1.0, 0.0, 0.0);
                        return mix(color_down, color_up, z_sign*0.5+0.5);
                    }}
                    )",
                cardinal_a.x, cardinal_a.y, cardinal_a.z, cardinal_b.x, cardinal_b.y, cardinal_b.z, cardinal_c.x,
                cardinal_c.y, cardinal_c.z, sign_z );
            break;
        case Colormap::BLUE_GREEN_RED:
            colormap_implementation = fmt::format(
                R"(
                    float atan2(float y, float x)
                    {{
                        return x == 0.0 ? sign(y)*3.14159/2.0 : atan(y, x);
                    }}
                    vec3 hsv2rgb(vec3 c)
                    {{
                        vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                        vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                        return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
                    }}
                    vec3 colormap(vec3 direction)
                    {{
                        vec3 cardinal_a = vec3({}, {}, {});
                        vec3 cardinal_b = vec3({}, {}, {});
                        vec3 cardinal_c = vec3({}, {}, {});
                        vec3 projection = vec3( dot(direction, cardinal_a), dot(direction, cardinal_b),
                            dot(direction, cardinal_c) );
                        float hue = 1.0/3.0-normalize(projection).z/3.0* {};
                        return hsv2rgb(vec3(hue, 1.0, 1.0));
                    }}
                    )",
                cardinal_a.x, cardinal_a.y, cardinal_a.z, cardinal_b.x, cardinal_b.y, cardinal_b.z, cardinal_c.x,
                cardinal_c.y, cardinal_c.z, sign_z );
            break;
        case Colormap::BLUE_WHITE_RED:
            colormap_implementation = fmt::format(
                R"(
                    vec3 colormap(vec3 direction)
                    {{
                        vec3 cardinal_a = vec3({}, {}, {});
                        vec3 cardinal_b = vec3({}, {}, {});
                        vec3 cardinal_c = vec3({}, {}, {});
                        vec3 projection = vec3( dot(direction, cardinal_a), dot(direction, cardinal_b),
                            dot(direction, cardinal_c) );
                        float z_sign = projection.z * {};
                        if (z_sign < 0)
                        {{
                            vec3 color_down = vec3(0.0, 0.0, 1.0);
                            vec3 color_up = vec3(1.0, 1.0, 1.0);
                            return mix(color_down, color_up, z_sign+1);
                        }}
                        else
                        {{
                            vec3 color_down = vec3(1.0, 1.0, 1.0);
                            vec3 color_up = vec3(1.0, 0.0, 0.0);
                            return mix(color_down, color_up, z_sign);
                        }}
                    }}
                    )",
                cardinal_a.x, cardinal_a.y, cardinal_a.z, cardinal_b.x, cardinal_b.y, cardinal_b.z, cardinal_c.x,
                cardinal_c.y, cardinal_c.z, sign_z );
            break;
        // Default is regular HSV
        default:
            colormap_implementation = fmt::format(
                R"(
                    float atan2(float y, float x)
                    {{
                        return x == 0.0 ? sign(y)*3.14159/2.0 : atan(y, x);
                    }}
                    vec3 hsv2rgb(vec3 c)
                    {{
                        vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                        vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                        return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
                    }}
                    vec3 colormap(vec3 direction)
                    {{
                        vec3 cardinal_a = vec3({}, {}, {});
                        vec3 cardinal_b = vec3({}, {}, {});
                        vec3 cardinal_c = vec3({}, {}, {});
                        vec3 projection = vec3( dot(direction, cardinal_a), dot(direction, cardinal_b),
                        dot(direction, cardinal_c) ); float hue = atan2({}*projection.x, projection.y) / 3.14159
                        / 2.0 + {}/2.0; float saturation = projection.z * {}; if (saturation > 0.0)
                        {{
                            return hsv2rgb(vec3(hue, 1.0-saturation, 1.0));
                        }}
                        else
                        {{
                            return hsv2rgb(vec3(hue, 1.0, 1.0+saturation));
                        }}
                    }}
                    )",
                cardinal_a.x, cardinal_a.y, cardinal_a.z, cardinal_b.x, cardinal_b.y, cardinal_b.z, cardinal_c.x,
                cardinal_c.y, cardinal_c.z, sign_xy, P, sign_z );
            break;
    }
    return colormap_implementation;
}

ColormapWidget::ColormapWidget()
{
    this->set_colormap( colormap );
}

void ColormapWidget::reset_colormap()
{
    this->colormap = Colormap::HSV;

    this->colormap_rotation         = 0;
    this->colormap_invert_z         = false;
    this->colormap_invert_xy        = false;
    this->colormap_cardinal_a       = glm::vec3{ 1, 0, 0 };
    this->colormap_cardinal_b       = glm::vec3{ 0, 1, 0 };
    this->colormap_cardinal_c       = glm::vec3{ 0, 0, 1 };
    this->colormap_monochrome_color = glm::vec3{ 0.5f, 0.5f, 0.5f };

    this->colormap_implementation_str = get_colormap(
        colormap, colormap_rotation, colormap_invert_z, colormap_invert_xy, colormap_cardinal_a, colormap_cardinal_b,
        colormap_cardinal_c, colormap_monochrome_color );
}

void ColormapWidget::set_colormap( Colormap colormap )
{
    this->colormap = colormap;

    this->colormap_implementation_str = get_colormap(
        colormap, colormap_rotation, colormap_invert_z, colormap_invert_xy, colormap_cardinal_a, colormap_cardinal_b,
        colormap_cardinal_c, colormap_monochrome_color );
}

bool ColormapWidget::colormap_input()
{
    std::vector<const char *> colormaps{ "HSV",      "HSV (no z)", "Blue-White-Red", "Blue-Green-Red",
                                         "Blue-Red", "Monochrome" };

    int colormap_index = int( colormap );
    ImGui::SetNextItemWidth( 120 );
    if( ImGui::Combo( "Colormap##arrows", &colormap_index, colormaps.data(), int( colormaps.size() ) ) )
    {
        set_colormap( Colormap( colormap_index ) );
        return true;
    }

    if( colormap == Colormap::MONOCHROME
        && ImGui::ColorEdit3( "Colour", &colormap_monochrome_color.x, ImGuiColorEditFlags_NoInputs ) )
    {
        set_colormap( colormap );
        return true;
    }

    return false;
}

void RendererWidget::show()
{
    ImGui::PushID( this->id );
    ImGui::Checkbox( "##checkbox_show", &show_ );
    ImGui::SameLine();
    if( ImGui::CollapsingHeader( this->name().c_str() ) )
    {
        ImGui::Indent( 25 );

        if( ImGui::Button( ICON_FA_TRASH_ALT " Remove" ) )
        {
            show_   = false;
            remove_ = true;
        }
        ImGui::SameLine();
        if( ImGui::Button( ICON_FA_REDO " Reset" ) )
            this->reset();
        ImGui::Dummy( { 0, 5 } );
        this->show_filters();
        ImGui::Dummy( { 0, 5 } );
        this->show_settings();

        ImGui::Indent( -25 );
    }
    ImGui::PopID();
}

void RendererWidget::apply_settings()
{
    renderer->setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>( get_is_visible_implementation(
        state, filter_position_min, filter_position_max, filter_direction_min, filter_direction_max ) );
}

void RendererWidget::show_filters()
{
    if( ImGui::TreeNode( "Filters" ) )
    {
        ImGui::TextUnformatted( "Orientation" );
        ImGui::Indent( 15 );
        bool update = false;
        if( widgets::RangeSliderFloat(
                "##filter_direction_x", &filter_direction_min[0], &filter_direction_max[0], -1, 1, "x: [%.3f, %.3f]" ) )
            update = true;
        if( widgets::RangeSliderFloat(
                "##filter_direction_y", &filter_direction_min[1], &filter_direction_max[1], -1, 1, "y: [%.3f, %.3f]" ) )
            update = true;
        if( widgets::RangeSliderFloat(
                "##filter_direction_z", &filter_direction_min[2], &filter_direction_max[2], -1, 1, "z: [%.3f, %.3f]" ) )
            update = true;
        ImGui::Indent( -15 );

        ImGui::TextUnformatted( "Position" );
        ImGui::Indent( 15 );
        if( widgets::RangeSliderFloat(
                "##filter_position_x", &filter_position_min[0], &filter_position_max[0], 0, 1, "x: [%.3f, %.3f]" ) )
            update = true;
        if( widgets::RangeSliderFloat(
                "##filter_position_y", &filter_position_min[1], &filter_position_max[1], 0, 1, "y: [%.3f, %.3f]" ) )
            update = true;
        if( widgets::RangeSliderFloat(
                "##filter_position_z", &filter_position_min[2], &filter_position_max[2], 0, 1, "z: [%.3f, %.3f]" ) )
            update = true;
        ImGui::Indent( -15 );

        if( update )
            RendererWidget::apply_settings();

        ImGui::TreePop();
    }
}

void RendererWidget::reset_filters()
{
    for( int dim = 0; dim < 3; ++dim )
    {
        this->filter_direction_min[dim] = -1;
        this->filter_direction_max[dim] = 1;
        this->filter_position_min[dim]  = 0;
        this->filter_position_max[dim]  = 1;
    }
    RendererWidget::apply_settings();
}

BoundingBoxRendererWidget::BoundingBoxRendererWidget(
    std::shared_ptr<State> state, const VFRendering::View & view, const VFRendering::VectorField & vectorfield )
        : RendererWidget( state )
{
    bool periodical[3];
    float b_min[3], b_max[3];
    Geometry_Get_Bounds( state.get(), b_min, b_max );
    glm::vec3 bounds_min = glm::make_vec3( b_min );
    glm::vec3 bounds_max = glm::make_vec3( b_max );
    glm::vec2 x_range{ bounds_min[0], bounds_max[0] };
    glm::vec2 y_range{ bounds_min[1], bounds_max[1] };
    glm::vec2 z_range{ bounds_min[2], bounds_max[2] };
    glm::vec3 bounding_box_center = { ( bounds_min[0] + bounds_max[0] ) / 2, ( bounds_min[1] + bounds_max[1] ) / 2,
                                      ( bounds_min[2] + bounds_max[2] ) / 2 };
    glm::vec3 bounding_box_side_lengths
        = { bounds_max[0] - bounds_min[0], bounds_max[1] - bounds_min[1], bounds_max[2] - bounds_min[2] };

    float indi_length            = glm::length( bounds_max - bounds_min ) * 0.05;
    int indi_dashes              = 5;
    float indi_dashes_per_length = (float)indi_dashes / indi_length;

    Hamiltonian_Get_Boundary_Conditions( this->state.get(), periodical );
    glm::vec3 indis{ indi_length * periodical[0], indi_length * periodical[1], indi_length * periodical[2] };

    // TODO: use proper parallelepipeds for non-cuboid geometries
    renderer = std::make_shared<VFRendering::BoundingBoxRenderer>( VFRendering::BoundingBoxRenderer::forCuboid(
        view, bounding_box_center, bounding_box_side_lengths, indis, indi_dashes_per_length ) );

    renderer->setOption<VFRendering::BoundingBoxRenderer::Option::LINE_WIDTH>( line_width );
    renderer->setOption<VFRendering::BoundingBoxRenderer::Option::LEVEL_OF_DETAIL>( level_of_detail );

    // if( draw_shadows )
    // {
    //     renderer->setOption<VFRendering::View::Option::LIGHTING_IMPLEMENTATION>(
    //         "uniform vec3 uLightPosition;"
    //         "float lighting(vec3 position, vec3 normal)"
    //         "{"
    //         "    vec3 lightDirection = -normalize(uLightPosition-position);"
    //         "    float diffuse = 0.7*max(0.0, dot(normal, lightDirection));"
    //         "    float ambient = 0.2;"
    //         "    return diffuse+ambient;"
    //         "}" );
    // }
    // else
    // {
    //     renderer->setOption<VFRendering::View::Option::LIGHTING_IMPLEMENTATION>(
    //         "float lighting(vec3 position, vec3 normal) { return 1.0; }" );
    // }
}

void BoundingBoxRendererWidget::reset()
{
    this->line_width      = 0;
    this->level_of_detail = 10;
    this->draw_shadows    = false;
}

void BoundingBoxRendererWidget::show()
{
    ImGui::PushID( this->id );
    ImGui::Checkbox( "##checkbox_show", &show_ );
    ImGui::SameLine();
    if( ImGui::CollapsingHeader( this->name().c_str() ) )
    {
        ImGui::Indent( 25 );

        ImGui::SetNextItemWidth( 100 );
        if( ImGui::SliderFloat( "thickness", &line_width, 0, 10, "%.1f" ) )
        {
            renderer->setOption<VFRendering::BoundingBoxRenderer::Option::LINE_WIDTH>( line_width );
        }
        if( line_width > 0 )
        {
            ImGui::SetNextItemWidth( 100 );
            if( ImGui::SliderInt( "level of detail", &level_of_detail, 5, 100 ) )
                renderer->setOption<VFRendering::BoundingBoxRenderer::Option::LEVEL_OF_DETAIL>( level_of_detail );

            if( ImGui::Checkbox( "draw shadows", &draw_shadows ) )
            {
                // renderer->setOption<VFRendering::BoundingBoxRenderer::Option::LIGHTING_IMPLEMENTATION>(
                //     get_lighting_implementation( draw_shadows ) );
            }
        }

        ImGui::Indent( -25 );
    }
    ImGui::PopID();
}

void BoundingBoxRendererWidget::apply_settings()
{
    RendererWidget::apply_settings();
    renderer->setOption<VFRendering::BoundingBoxRenderer::Option::LINE_WIDTH>( line_width );
}

void BoundingBoxRendererWidget::show_settings() {}

CoordinateSystemRendererWidget::CoordinateSystemRendererWidget( std::shared_ptr<State> state ) : RendererWidget( state )
{
}

void CoordinateSystemRendererWidget::show() {}

void CoordinateSystemRendererWidget::reset() {}

void CoordinateSystemRendererWidget::show_settings() {}

DotRendererWidget::DotRendererWidget(
    std::shared_ptr<State> state, const VFRendering::View & view, const VFRendering::VectorField & vectorfield )
        : RendererWidget( state ), ColormapWidget()
{
    renderer = std::make_shared<VFRendering::DotRenderer>( view, vectorfield );
    this->apply_settings();
}

void DotRendererWidget::apply_settings()
{
    RendererWidget::apply_settings();
    set_colormap( colormap );
    renderer->setOption<VFRendering::View::Option::COLORMAP_IMPLEMENTATION>( colormap_implementation_str );

    renderer->setOption<VFRendering::DotRenderer::DOT_RADIUS>( size * 1000 );
}

void DotRendererWidget::reset()
{
    this->reset_filters();
    this->reset_colormap();

    this->size = 1;

    this->apply_settings();
}

void DotRendererWidget::show_settings()
{
    ImGui::SetNextItemWidth( 100 );
    if( ImGui::SliderFloat( "size", &size, 0.01f, 100, "%.3f", 10 ) )
    {
        renderer->setOption<VFRendering::DotRenderer::DOT_RADIUS>( size * 1000 );
    }

    if( colormap_input() )
        renderer->setOption<VFRendering::View::Option::COLORMAP_IMPLEMENTATION>( colormap_implementation_str );
}

ArrowRendererWidget::ArrowRendererWidget(
    std::shared_ptr<State> state, const VFRendering::View & view, const VFRendering::VectorField & vectorfield )
        : RendererWidget( state ), ColormapWidget()
{
    renderer = std::make_shared<VFRendering::ArrowRenderer>( view, vectorfield );
    this->apply_settings();
}

void ArrowRendererWidget::apply_settings()
{
    RendererWidget::apply_settings();
    set_colormap( Colormap::MONOCHROME );
    renderer->setOption<VFRendering::View::Option::COLORMAP_IMPLEMENTATION>( colormap_implementation_str );
    this->colormap_implementation_str = get_colormap(
        Colormap::MONOCHROME, colormap_rotation, colormap_invert_z, colormap_invert_xy, colormap_cardinal_a,
        colormap_cardinal_b, colormap_cardinal_c, colormap_monochrome_color );

    renderer->setOption<VFRendering::ArrowRenderer::Option::CONE_RADIUS>( size * 0.125f );
    renderer->setOption<VFRendering::ArrowRenderer::Option::CONE_HEIGHT>( size * 0.3f );
    renderer->setOption<VFRendering::ArrowRenderer::Option::CYLINDER_RADIUS>( size * 0.0625f );
    renderer->setOption<VFRendering::ArrowRenderer::Option::CYLINDER_HEIGHT>( size * 0.35f );

    renderer->setOption<VFRendering::ArrowRenderer::Option::LEVEL_OF_DETAIL>( lod );
}

void ArrowRendererWidget::reset()
{
    this->reset_filters();
    this->reset_colormap();

    this->size = 1;
    this->lod  = 10;

    this->apply_settings();
}

void ArrowRendererWidget::show_settings()
{
    ImGui::SetNextItemWidth( 100 );
    if( ImGui::SliderFloat( "size", &size, 0.01f, 100, "%.3f", 10 ) )
    {
        renderer->setOption<VFRendering::ArrowRenderer::Option::CONE_RADIUS>( size * 0.125f );
        renderer->setOption<VFRendering::ArrowRenderer::Option::CONE_HEIGHT>( size * 0.3f );
        renderer->setOption<VFRendering::ArrowRenderer::Option::CYLINDER_RADIUS>( size * 0.0625f );
        renderer->setOption<VFRendering::ArrowRenderer::Option::CYLINDER_HEIGHT>( size * 0.35f );
    }

    ImGui::SetNextItemWidth( 100 );
    if( ImGui::SliderInt( "level of detail", &lod, 5, 100 ) )
        renderer->setOption<VFRendering::ArrowRenderer::Option::LEVEL_OF_DETAIL>( lod );

    if( colormap_input() )
        renderer->setOption<VFRendering::View::Option::COLORMAP_IMPLEMENTATION>( colormap_implementation_str );
}

ParallelepipedRendererWidget::ParallelepipedRendererWidget(
    std::shared_ptr<State> state, const VFRendering::View & view, const VFRendering::VectorField & vectorfield )
        : RendererWidget( state ), ColormapWidget()
{
    renderer = std::make_shared<VFRendering::ParallelepipedRenderer>( view, vectorfield );
    renderer->setOption<VFRendering::GlyphRenderer::Option::ROTATE_GLYPHS>( false );
    this->apply_settings();
}

void ParallelepipedRendererWidget::apply_settings()
{
    RendererWidget::apply_settings();
    set_colormap( colormap );
    renderer->setOption<VFRendering::View::Option::COLORMAP_IMPLEMENTATION>( colormap_implementation_str );

    renderer->setOption<VFRendering::ParallelepipedRenderer::Option::LENGTH_A>( size * 0.5f );
    renderer->setOption<VFRendering::ParallelepipedRenderer::Option::LENGTH_B>( size * 0.5f );
    renderer->setOption<VFRendering::ParallelepipedRenderer::Option::LENGTH_C>( size * 0.5f );
}

void ParallelepipedRendererWidget::reset()
{
    this->reset_filters();
    this->reset_colormap();

    this->size = 1;

    this->apply_settings();
}

void ParallelepipedRendererWidget::show_settings()
{
    ImGui::SetNextItemWidth( 100 );
    if( ImGui::SliderFloat( "size", &size, 0.01f, 100, "%.3f", 10 ) )
    {
        renderer->setOption<VFRendering::ParallelepipedRenderer::Option::LENGTH_A>( size * 0.5f );
        renderer->setOption<VFRendering::ParallelepipedRenderer::Option::LENGTH_B>( size * 0.5f );
        renderer->setOption<VFRendering::ParallelepipedRenderer::Option::LENGTH_C>( size * 0.5f );
    }

    if( colormap_input() )
        renderer->setOption<VFRendering::View::Option::COLORMAP_IMPLEMENTATION>( colormap_implementation_str );
}

SphereRendererWidget::SphereRendererWidget(
    std::shared_ptr<State> state, const VFRendering::View & view, const VFRendering::VectorField & vectorfield )
        : RendererWidget( state ), ColormapWidget()
{
    renderer = std::make_shared<VFRendering::SphereRenderer>( view, vectorfield );
    this->apply_settings();
}

void SphereRendererWidget::apply_settings()
{
    RendererWidget::apply_settings();
    set_colormap( colormap );
    renderer->setOption<VFRendering::View::Option::COLORMAP_IMPLEMENTATION>( colormap_implementation_str );

    renderer->setOption<VFRendering::SphereRenderer::Option::SPHERE_RADIUS>( size );
    renderer->setOption<VFRendering::SphereRenderer::Option::LEVEL_OF_DETAIL>( lod );
}

void SphereRendererWidget::reset()
{
    this->reset_filters();
    this->reset_colormap();

    this->size = 0.1f;
    this->lod  = 10;

    this->apply_settings();
}

void SphereRendererWidget::show_settings()
{
    ImGui::SetNextItemWidth( 100 );
    if( ImGui::SliderFloat( "size", &size, 0.01f, 10, "%.3f", 10 ) )
    {
        renderer->setOption<VFRendering::SphereRenderer::Option::SPHERE_RADIUS>( size );
    }

    ImGui::SetNextItemWidth( 100 );
    if( ImGui::SliderInt( "level of detail", &lod, 10, 100 ) )
        renderer->setOption<VFRendering::SphereRenderer::Option::LEVEL_OF_DETAIL>( lod );

    if( colormap_input() )
        renderer->setOption<VFRendering::View::Option::COLORMAP_IMPLEMENTATION>( colormap_implementation_str );
}

SurfaceRendererWidget::SurfaceRendererWidget(
    std::shared_ptr<State> state, const VFRendering::View & view, const VFRendering::VectorField & vectorfield )
        : RendererWidget( state ), ColormapWidget()
{
    renderer = std::make_shared<VFRendering::SurfaceRenderer>( view, vectorfield );
    this->apply_settings();
}

void SurfaceRendererWidget::apply_settings()
{
    RendererWidget::apply_settings();
    set_colormap( colormap );
    renderer->setOption<VFRendering::View::Option::COLORMAP_IMPLEMENTATION>( colormap_implementation_str );
}

void SurfaceRendererWidget::reset()
{
    this->reset_filters();
    this->reset_colormap();

    this->apply_settings();
}

void SurfaceRendererWidget::show_settings()
{
    if( colormap_input() )
        renderer->setOption<VFRendering::View::Option::COLORMAP_IMPLEMENTATION>( colormap_implementation_str );
}

IsosurfaceRendererWidget::IsosurfaceRendererWidget(
    std::shared_ptr<State> state, const VFRendering::View & view, const VFRendering::VectorField & vectorfield )
        : RendererWidget( state ), ColormapWidget()
{
    renderer = std::make_shared<VFRendering::IsosurfaceRenderer>( view, vectorfield );
    this->apply_settings();
}

void IsosurfaceRendererWidget::apply_settings()
{
    RendererWidget::apply_settings();
    set_colormap( colormap );
    renderer->setOption<VFRendering::View::Option::COLORMAP_IMPLEMENTATION>( colormap_implementation_str );

    set_lighting_implementation( draw_shadows );
    set_isocomponent( isocomponent );
    renderer->setOption<VFRendering::IsosurfaceRenderer::Option::ISOVALUE>( isovalue );
    renderer->setOption<VFRendering::IsosurfaceRenderer::Option::FLIP_NORMALS>( flip_normals );
}

void IsosurfaceRendererWidget::reset()
{
    this->reset_filters();
    this->reset_colormap();

    this->isovalue     = 0;
    this->flip_normals = false;
    this->draw_shadows = true;
    this->isocomponent = 2;

    this->apply_settings();
}

void IsosurfaceRendererWidget::show_settings()
{
    if( ImGui::SliderFloat( "isovalue", &isovalue, -1, 1 ) )
        renderer->setOption<VFRendering::IsosurfaceRenderer::Option::ISOVALUE>( isovalue );

    if( ImGui::Checkbox( "draw shadows", &draw_shadows ) )
    {
        this->set_lighting_implementation( draw_shadows );
    }

    if( ImGui::Checkbox( "flip normals", &flip_normals ) )
        renderer->setOption<VFRendering::IsosurfaceRenderer::Option::FLIP_NORMALS>( flip_normals );

    bool iso_x = isocomponent == 0;
    bool iso_y = isocomponent == 1;
    bool iso_z = isocomponent == 2;
    ImGui::TextUnformatted( "Component" );
    ImGui::SameLine();
    if( ImGui::Checkbox( "x", &iso_x ) )
        set_isocomponent( 0 );
    ImGui::SameLine();
    if( ImGui::Checkbox( "y", &iso_y ) )
        set_isocomponent( 1 );
    ImGui::SameLine();
    if( ImGui::Checkbox( "z", &iso_z ) )
        set_isocomponent( 2 );

    if( colormap_input() )
        renderer->setOption<VFRendering::View::Option::COLORMAP_IMPLEMENTATION>( colormap_implementation_str );
}

void IsosurfaceRendererWidget::set_isocomponent( int isocomponent )
{
    this->isocomponent = isocomponent;
    if( this->isocomponent == 0 )
    {
        renderer->setOption<VFRendering::IsosurfaceRenderer::Option::VALUE_FUNCTION>(
            []( const glm::vec3 & position,
                const glm::vec3 & direction ) -> VFRendering::IsosurfaceRenderer::isovalue_type {
                (void)position;
                return direction.x;
            } );
    }
    else if( this->isocomponent == 1 )
    {
        renderer->setOption<VFRendering::IsosurfaceRenderer::Option::VALUE_FUNCTION>(
            []( const glm::vec3 & position,
                const glm::vec3 & direction ) -> VFRendering::IsosurfaceRenderer::isovalue_type {
                (void)position;
                return direction.y;
            } );
    }
    else if( this->isocomponent == 2 )
    {
        renderer->setOption<VFRendering::IsosurfaceRenderer::Option::VALUE_FUNCTION>(
            []( const glm::vec3 & position,
                const glm::vec3 & direction ) -> VFRendering::IsosurfaceRenderer::isovalue_type {
                (void)position;
                return direction.z;
            } );
    }
}

void IsosurfaceRendererWidget::set_lighting_implementation( bool draw_shadows )
{
    this->draw_shadows = draw_shadows;

    renderer->setOption<VFRendering::IsosurfaceRenderer::Option::LIGHTING_IMPLEMENTATION>(
        get_lighting_implementation( draw_shadows ) );
}

} // namespace ui