#include <renderer_widget.hpp>

#include <imgui/imgui.h>

#include <glm/gtc/type_ptr.hpp>

#include <Spirit/Geometry.h>
#include <Spirit/Hamiltonian.h>

#include <fmt/format.h>

#include <string>

namespace ui
{

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
    colormap_implementation_str = get_colormap(
        colormap, colormap_rotation, colormap_invert_z, colormap_invert_xy, colormap_cardinal_a, colormap_cardinal_b,
        colormap_cardinal_c, colormap_monochrome_color );
}

void ColormapWidget::showcolormap_input()
{
    std::vector<const char *> colormaps{ "HSV",      "HSV (no z)", "Blue-White-Red", "Blue-Green-Red",
                                         "Blue-Red", "Monochrome" };

    int colormap_index = int( colormap );
    if( ImGui::Combo( "Colormap##arrows", &colormap_index, colormaps.data(), int( colormaps.size() ) ) )
    {
        colormap = Colormap( colormap_index );

        colormap_implementation_str = get_colormap(
            colormap, colormap_rotation, colormap_invert_z, colormap_invert_xy, colormap_cardinal_a,
            colormap_cardinal_b, colormap_cardinal_c, colormap_monochrome_color );

        colormap_changed = true;
    }

    if( colormap == Colormap::MONOCHROME
        && ImGui::ColorEdit3( "Colour", &colormap_monochrome_color.x, ImGuiColorEditFlags_NoInputs ) )
    {
        colormap_implementation_str = get_colormap(
            colormap, colormap_rotation, colormap_invert_z, colormap_invert_xy, colormap_cardinal_a,
            colormap_cardinal_b, colormap_cardinal_c, colormap_monochrome_color );

        colormap_changed = true;
    }
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

void BoundingBoxRendererWidget::show()
{
    ImGui::PushID( "Boundingbox" );
    ImGui::Checkbox( "Boundingbox", &show_ );
    if( !show_ )
    {
        ImGui::PopID();
        return;
    }
    ImGui::Indent( 15 );

    if( ImGui::SliderFloat( "thickness", &line_width, 0, 10, "%.1f" ) )
    {
        renderer->setOption<VFRendering::BoundingBoxRenderer::Option::LINE_WIDTH>( line_width );
    }
    if( line_width > 0 )
    {
        if( ImGui::SliderInt( "level of detail", &level_of_detail, 5, 100 ) )
            renderer->setOption<VFRendering::BoundingBoxRenderer::Option::LEVEL_OF_DETAIL>( level_of_detail );

        if( ImGui::Checkbox( "draw shadows", &draw_shadows ) )
        {
            // if( draw_shadows )
            // {
            //     renderer->setOption<VFRendering::IsosurfaceRenderer::Option::LIGHTING_IMPLEMENTATION>(
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
            //     renderer->setOption<VFRendering::IsosurfaceRenderer::Option::LIGHTING_IMPLEMENTATION>(
            //         "float lighting(vec3 position, vec3 normal) { return 1.0; }" );
            // }
        }
    }

    ImGui::Indent( -15 );
    ImGui::PopID();
}

CoordinateSystemRendererWidget::CoordinateSystemRendererWidget( std::shared_ptr<State> state ) : RendererWidget( state )
{
}

void CoordinateSystemRendererWidget::show()
{
    ImGui::PushID( "Dots" );
    ImGui::Checkbox( "Dots", &show_ );
    ImGui::SameLine();
    if( ImGui::Button( "Remove" ) )
    {
        show_   = false;
        remove_ = true;
    }

    if( !show_ )
    {
        ImGui::PopID();
        return;
    }
    ImGui::Indent( 15 );

    ImGui::Indent( -15 );
    ImGui::PopID();
}

DotRendererWidget::DotRendererWidget(
    std::shared_ptr<State> state, const VFRendering::View & view, const VFRendering::VectorField & vectorfield )
        : RendererWidget( state ), ColormapWidget()
{
    renderer = std::make_shared<VFRendering::DotRenderer>( view, vectorfield );

    renderer->setOption<VFRendering::DotRenderer::DOT_RADIUS>( dotsize * 1000 );
}

void DotRendererWidget::show()
{
    ImGui::PushID( "Dots" );
    ImGui::Checkbox( "Dots", &show_ );
    ImGui::SameLine();
    if( ImGui::Button( "Remove" ) )
    {
        show_   = false;
        remove_ = true;
    }

    if( !show_ )
    {
        ImGui::PopID();
        return;
    }
    ImGui::Indent( 15 );

    if( ImGui::SliderFloat( "dotsize", &dotsize, 0.01f, 100, "%.3f", 10 ) )
    {
        renderer->setOption<VFRendering::DotRenderer::DOT_RADIUS>( dotsize * 1000 );
    }

    showcolormap_input();
    if( colormap_changed )
        renderer->setOption<VFRendering::View::COLORMAP_IMPLEMENTATION>( colormap_implementation_str );

    ImGui::Indent( -15 );
    ImGui::PopID();
}

ArrowRendererWidget::ArrowRendererWidget(
    std::shared_ptr<State> state, const VFRendering::View & view, const VFRendering::VectorField & vectorfield )
        : RendererWidget( state ), ColormapWidget()
{
    renderer = std::make_shared<VFRendering::ArrowRenderer>( view, vectorfield );

    renderer->setOption<VFRendering::ArrowRenderer::Option::CONE_RADIUS>( arrow_size * 0.125f );
    renderer->setOption<VFRendering::ArrowRenderer::Option::CONE_HEIGHT>( arrow_size * 0.3f );
    renderer->setOption<VFRendering::ArrowRenderer::Option::CYLINDER_RADIUS>( arrow_size * 0.0625f );
    renderer->setOption<VFRendering::ArrowRenderer::Option::CYLINDER_HEIGHT>( arrow_size * 0.35f );
}

void ArrowRendererWidget::show()
{
    ImGui::PushID( "Arrows" );
    ImGui::Checkbox( "Arrows", &show_ );
    ImGui::SameLine();
    if( ImGui::Button( "Remove" ) )
    {
        show_   = false;
        remove_ = true;
    }

    if( !show_ )
    {
        ImGui::PopID();
        return;
    }
    ImGui::Indent( 15 );

    if( ImGui::SliderFloat( "arrow_size", &arrow_size, 0.01f, 100, "%.3f", 10 ) )
    {
        renderer->setOption<VFRendering::ArrowRenderer::Option::CONE_RADIUS>( arrow_size * 0.125f );
        renderer->setOption<VFRendering::ArrowRenderer::Option::CONE_HEIGHT>( arrow_size * 0.3f );
        renderer->setOption<VFRendering::ArrowRenderer::Option::CYLINDER_RADIUS>( arrow_size * 0.0625f );
        renderer->setOption<VFRendering::ArrowRenderer::Option::CYLINDER_HEIGHT>( arrow_size * 0.35f );
    }

    if( ImGui::SliderInt( "level of detail", &arrow_lod, 5, 100 ) )
        renderer->setOption<VFRendering::ArrowRenderer::Option::LEVEL_OF_DETAIL>( arrow_lod );

    showcolormap_input();
    if( colormap_changed )
        renderer->setOption<VFRendering::View::COLORMAP_IMPLEMENTATION>( colormap_implementation_str );

    ImGui::Indent( -15 );
    ImGui::PopID();
}

ParallelepipedRendererWidget::ParallelepipedRendererWidget(
    std::shared_ptr<State> state, const VFRendering::View & view, const VFRendering::VectorField & vectorfield )
        : RendererWidget( state ), ColormapWidget()
{
    renderer = std::make_shared<VFRendering::ParallelepipedRenderer>( view, vectorfield );
    renderer->setOption<VFRendering::GlyphRenderer::Option::ROTATE_GLYPHS>( false );
    renderer->setOption<VFRendering::ParallelepipedRenderer::Option::LENGTH_A>( 0.25f );
}

void ParallelepipedRendererWidget::show()
{
    ImGui::PushID( "Parallelepiped" );
    ImGui::Checkbox( "Parallelepiped", &show_ );
    ImGui::SameLine();
    if( ImGui::Button( "Remove" ) )
    {
        show_   = false;
        remove_ = true;
    }

    if( !show_ )
    {
        ImGui::PopID();
        return;
    }
    ImGui::Indent( 15 );

    showcolormap_input();
    if( colormap_changed )
        renderer->setOption<VFRendering::View::COLORMAP_IMPLEMENTATION>( colormap_implementation_str );

    ImGui::Indent( -15 );
    ImGui::PopID();
}

SphereRendererWidget::SphereRendererWidget(
    std::shared_ptr<State> state, const VFRendering::View & view, const VFRendering::VectorField & vectorfield )
        : RendererWidget( state ), ColormapWidget()
{
    renderer = std::make_shared<VFRendering::SphereRenderer>( view, vectorfield );
}

void SphereRendererWidget::show()
{
    ImGui::PushID( "Sphere" );
    ImGui::Checkbox( "Sphere", &show_ );
    ImGui::SameLine();
    if( ImGui::Button( "Remove" ) )
    {
        show_   = false;
        remove_ = true;
    }

    if( !show_ )
    {
        ImGui::PopID();
        return;
    }
    ImGui::Indent( 15 );

    showcolormap_input();
    if( colormap_changed )
        renderer->setOption<VFRendering::View::COLORMAP_IMPLEMENTATION>( colormap_implementation_str );

    ImGui::Indent( -15 );
    ImGui::PopID();
}

SurfaceRendererWidget::SurfaceRendererWidget(
    std::shared_ptr<State> state, const VFRendering::View & view, const VFRendering::VectorField & vectorfield )
        : RendererWidget( state ), ColormapWidget()
{
    renderer = std::make_shared<VFRendering::SurfaceRenderer>( view, vectorfield );
}

void SurfaceRendererWidget::show()
{
    ImGui::PushID( "Surface" );
    ImGui::Checkbox( "Surface", &show_ );
    ImGui::SameLine();
    if( ImGui::Button( "Remove" ) )
    {
        show_   = false;
        remove_ = true;
    }

    if( !show_ )
    {
        ImGui::PopID();
        return;
    }
    ImGui::Indent( 15 );

    showcolormap_input();
    if( colormap_changed )
        renderer->setOption<VFRendering::View::COLORMAP_IMPLEMENTATION>( colormap_implementation_str );

    ImGui::Indent( -15 );
    ImGui::PopID();
}

IsosurfaceRendererWidget::IsosurfaceRendererWidget(
    std::shared_ptr<State> state, const VFRendering::View & view, const VFRendering::VectorField & vectorfield )
        : RendererWidget( state ), ColormapWidget()
{
    renderer = std::make_shared<VFRendering::IsosurfaceRenderer>( view, vectorfield );

    renderer->setOption<VFRendering::IsosurfaceRenderer::Option::ISOVALUE>( isovalue );

    if( draw_shadows )
    {
        renderer->setOption<VFRendering::IsosurfaceRenderer::Option::LIGHTING_IMPLEMENTATION>(
            "uniform vec3 uLightPosition;"
            "float lighting(vec3 position, vec3 normal)"
            "{"
            "    vec3 lightDirection = -normalize(uLightPosition-position);"
            "    float diffuse = 0.7*max(0.0, dot(normal, lightDirection));"
            "    float ambient = 0.2;"
            "    return diffuse+ambient;"
            "}" );
    }
    else
    {
        renderer->setOption<VFRendering::IsosurfaceRenderer::Option::LIGHTING_IMPLEMENTATION>(
            "float lighting(vec3 position, vec3 normal) { return 1.0; }" );
    }
}

void IsosurfaceRendererWidget::show()
{
    ImGui::PushID( "Isosurface" );
    ImGui::Checkbox( "Isosurface", &show_ );
    ImGui::SameLine();
    if( ImGui::Button( "Remove" ) )
    {
        show_   = false;
        remove_ = true;
    }

    if( !show_ )
    {
        ImGui::PopID();
        return;
    }
    ImGui::Indent( 15 );

    if( ImGui::SliderFloat( "isovalue", &isovalue, -1, 1 ) )
        renderer->setOption<VFRendering::IsosurfaceRenderer::Option::ISOVALUE>( isovalue );

    if( ImGui::Checkbox( "draw shadows", &draw_shadows ) )
    {
        if( draw_shadows )
        {
            renderer->setOption<VFRendering::IsosurfaceRenderer::Option::LIGHTING_IMPLEMENTATION>(
                "uniform vec3 uLightPosition;"
                "float lighting(vec3 position, vec3 normal)"
                "{"
                "    vec3 lightDirection = -normalize(uLightPosition-position);"
                "    float diffuse = 0.7*max(0.0, dot(normal, lightDirection));"
                "    float ambient = 0.2;"
                "    return diffuse+ambient;"
                "}" );
        }
        else
        {
            renderer->setOption<VFRendering::IsosurfaceRenderer::Option::LIGHTING_IMPLEMENTATION>(
                "float lighting(vec3 position, vec3 normal) { return 1.0; }" );
        }
    }

    if( ImGui::Checkbox( "flip normals", &flip_normals ) )
        renderer->setOption<VFRendering::IsosurfaceRenderer::Option::FLIP_NORMALS>( flip_normals );

    auto set_isocomponent = [&]() {
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
    };
    bool iso_x = isocomponent == 0;
    bool iso_y = isocomponent == 1;
    bool iso_z = isocomponent == 2;
    ImGui::TextUnformatted( "Component" );
    ImGui::SameLine();
    if( ImGui::Checkbox( "x", &iso_x ) )
    {
        isocomponent = 0;
        set_isocomponent();
    }
    ImGui::SameLine();
    if( ImGui::Checkbox( "y", &iso_y ) )
    {
        isocomponent = 1;
        set_isocomponent();
    }
    ImGui::SameLine();
    if( ImGui::Checkbox( "z", &iso_z ) )
    {
        isocomponent = 2;
        set_isocomponent();
    }

    showcolormap_input();
    if( colormap_changed )
        renderer->setOption<VFRendering::View::COLORMAP_IMPLEMENTATION>( colormap_implementation_str );

    ImGui::Indent( -15 );
    ImGui::PopID();
}

} // namespace ui