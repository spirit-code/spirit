#include <iostream>
#include <GLES3/gl3.h>
#include <emscripten/html5.h>

#include "VFRendering/View.hxx"
#include "VFRendering/DotRenderer.hxx"
#include "VFRendering/ArrowRenderer.hxx"
#include "VFRendering/SphereRenderer.hxx"
#include "VFRendering/VectorSphereRenderer.hxx"
#include "VFRendering/CoordinateSystemRenderer.hxx"
#include "VFRendering/BoundingBoxRenderer.hxx"
#include "VFRendering/CombinedRenderer.hxx"
#include "VFRendering/IsosurfaceRenderer.hxx"
#include "VFRendering/ParallelepipedRenderer.hxx"
#include "VFRendering/SurfaceRenderer.hxx"

#include <Spirit/Geometry.h>
#include <Spirit/System.h>
#include <Spirit/Hamiltonian.h>

static bool needs_redraw = false;
static const int default_mode = (int)VFRendering::CameraMovementModes::ROTATE_BOUNDED;
VFRendering::View view;

// Data
std::vector<glm::vec3> positions;
std::vector<glm::vec3> directions;

std::shared_ptr<VFRendering::Geometry> geometry;
std::shared_ptr<VFRendering::VectorField> vf;

// Which to show
static bool show_bounding_box       = true;
static bool show_dots               = true;
static bool show_arrows             = true;
static bool show_boxes              = false;
static bool show_spheres            = true;
static bool show_surface            = true;
static bool show_isosurface         = true;

static bool mainview_is_system      = true;
static bool show_coordinate_system  = true;
static bool show_miniview           = true;

static int location_coordinatesystem = 2;
static int location_miniview         = 3;

// Renderers
std::shared_ptr<VFRendering::BoundingBoxRenderer>       bounding_box_renderer_ptr;
std::shared_ptr<VFRendering::CoordinateSystemRenderer>  coordinate_system_renderer_ptr;
std::shared_ptr<VFRendering::VectorSphereRenderer>      vectorsphere_renderer_ptr;
std::shared_ptr<VFRendering::DotRenderer>               dot_renderer_ptr;
std::shared_ptr<VFRendering::ArrowRenderer>             arrow_renderer_ptr;
std::shared_ptr<VFRendering::SphereRenderer>            sphere_renderer_ptr;
std::shared_ptr<VFRendering::ParallelepipedRenderer>    box_renderer_ptr;
std::shared_ptr<VFRendering::IsosurfaceRenderer>        surface_renderer_3d_ptr;
std::shared_ptr<VFRendering::SurfaceRenderer>           surface_renderer_2d_ptr;
std::shared_ptr<VFRendering::IsosurfaceRenderer>        isosurface_renderer_ptr;

void framebufferSizeCallback(EMSCRIPTEN_WEBGL_CONTEXT_HANDLE context, int width, int height)
{
    (void)context;
    view.setFramebufferSize(width, height);
    needs_redraw = true;
}

void update_renderers(State * state)
{
    int dimensionality = Geometry_Get_Dimensionality(state);

    std::vector<std::shared_ptr<VFRendering::RendererBase>> system_renderers(0);
    if( show_bounding_box )
        system_renderers.push_back(bounding_box_renderer_ptr);
    if( show_dots )
        system_renderers.push_back(dot_renderer_ptr);
    if( show_arrows )
        system_renderers.push_back(arrow_renderer_ptr);
    if( show_spheres )
        system_renderers.push_back(sphere_renderer_ptr);
    if( show_boxes )
        system_renderers.push_back(box_renderer_ptr);
    if( show_surface )
    {
        if( dimensionality == 3 )
            system_renderers.push_back(surface_renderer_3d_ptr);
        else if( dimensionality == 2 )
            system_renderers.push_back(surface_renderer_2d_ptr);
    }
    if( show_isosurface && dimensionality == 3 )
        system_renderers.push_back(isosurface_renderer_ptr);

    // ---------------

    std::array<float, 4> position_coordinatesystem;
    if( location_coordinatesystem == 0 ) // TL
        position_coordinatesystem = { 0, 0.7f, 0.25f, 0.25f };
    else if( location_coordinatesystem == 1 ) // TR
        position_coordinatesystem = { 0.75f, 0.7f, 0.25f, 0.25f };
    else if( location_coordinatesystem == 2 ) // BR
        position_coordinatesystem = { 0.75f, 0.05, 0.25f, 0.25f };
    else if( location_coordinatesystem == 3 ) // BL
        position_coordinatesystem = { 0, 0.05f, 0.25f, 0.25f };

    std::array<float, 4> position_miniview;
    if( location_miniview == 0 ) // TL
        position_miniview = { 0, 0.7f, 0.25f, 0.25f };
    else if( location_miniview == 1 ) // TR
        position_miniview = { 0.75f, 0.7f, 0.25f, 0.25f };
    else if( location_miniview == 2 ) // BR
        position_miniview = { 0.75f, 0.05f, 0.25f, 0.25f };
    else if( location_miniview == 3 ) // BL
        position_miniview = { 0, 0.05f, 0.25f, 0.25f };

    // ---------------

    std::vector<std::pair<std::shared_ptr<VFRendering::RendererBase>, std::array<float, 4>>> view_renderers(0);

    if( mainview_is_system )
    {
        view_renderers.push_back({std::make_shared<VFRendering::CombinedRenderer>(view, system_renderers), {{0, 0, 1, 1}}});
        if( show_miniview )
            view_renderers.push_back({vectorsphere_renderer_ptr, position_miniview});
    }
    else
    {
        view_renderers.push_back({vectorsphere_renderer_ptr, {{0, 0, 1, 1}}});
        if( show_miniview )
            view_renderers.push_back({std::make_shared<VFRendering::CombinedRenderer>(view, system_renderers), position_miniview});
    }

    if( show_coordinate_system )
        view_renderers.push_back({coordinate_system_renderer_ptr, position_coordinatesystem});

    // ---------------

    view.renderers(view_renderers);
}

extern "C" void mouse_move(float previous_mouse_position[2], float current_mouse_position[2], int movement_mode=default_mode)
{
    glm::vec2 previous = {previous_mouse_position[0], previous_mouse_position[1]};
    glm::vec2 current = {current_mouse_position[0], current_mouse_position[1]};
    VFRendering::CameraMovementModes mode = VFRendering::CameraMovementModes(movement_mode);
    view.mouseMove(previous, current, mode);
    // // std::cerr << "move: "<< previous.x << " " << previous.y << " " << current.x << " " << current.y <<"\n";
}

extern "C" void mouse_scroll(float delta)
{
    view.mouseScroll( delta );
}

extern "C" void set_camera(float position[3], float center[3], float up[3])
{
    VFRendering::Options options;
    options.set<VFRendering::View::Option::SYSTEM_CENTER>((geometry->min() + geometry->max()) * 0.5f);
    options.set<VFRendering::View::Option::CAMERA_POSITION>({position[0], position[1], position[2]});
    // options.set<VFRendering::View::Option::CENTER_POSITION>({center[0], center[1], center[2]});
    options.set<VFRendering::View::Option::CENTER_POSITION>((geometry->min() + geometry->max()) * 0.5f);
    options.set<VFRendering::View::Option::UP_VECTOR>({up[0], up[1], up[2]});
    view.updateOptions(options);
}

extern "C" void align_camera(float direction[3], float up[3])
{
    glm::vec3 d = view.getOption<VFRendering::View::Option::CENTER_POSITION>() - view.getOption<VFRendering::View::Option::CAMERA_POSITION>();
    float dist = glm::length(d);
    glm::vec3 dir{direction[0], direction[1], direction[2]};
    glm::normalize(dir);
    VFRendering::Options options;
    options.set<VFRendering::View::Option::CAMERA_POSITION>((geometry->min() + geometry->max()) * 0.5f - dist*dir);
    // options.set<VFRendering::View::Option::CENTER_POSITION>({center[0], center[1], center[2]});
    options.set<VFRendering::View::Option::CENTER_POSITION>((geometry->min() + geometry->max()) * 0.5f);
    options.set<VFRendering::View::Option::UP_VECTOR>({up[0], up[1], up[2]});
    view.updateOptions(options);
}

extern "C" void recenter_camera()
{
    view.setOption<VFRendering::View::Option::CENTER_POSITION>((geometry->min() + geometry->max()) * 0.5f);
}

extern "C" void set_rendermode(int mode)
{
    if( mode == 0 )
        mainview_is_system = true;
    else
        mainview_is_system = false;
}

extern "C" void set_background(float colour[3])
{
    glm::vec3 c{colour[0], colour[1], colour[2]};
    view.setOption<VFRendering::View::Option::BACKGROUND_COLOR>(c);
}

extern "C" void set_boundingbox_colour(float colour[3])
{
    glm::vec3 c{colour[0], colour[1], colour[2]};
    bounding_box_renderer_ptr->setOption<VFRendering::BoundingBoxRenderer::Option::COLOR>(c);
}

extern "C" void set_miniview(State * state, bool show, int position)
{
    show_miniview = show;
    location_miniview = position;
    update_renderers(state);
}

extern "C" void set_boundingbox(State * state, bool show, float line_width)
{
    show_bounding_box = show;

    bool periodical[3]{false, false, false};
    Hamiltonian_Get_Boundary_Conditions(state, periodical);

    float indi_length = std::max(0.1f, glm::length(geometry->max()-geometry->min())*0.1f);
    glm::vec3 indis{ indi_length*periodical[0], indi_length*periodical[1], indi_length*periodical[2] };
    int   indi_dashes = 5;
    float indi_dashes_per_length = 0.9f*(float)indi_dashes / indi_length;

    int n_cells[3];
    Geometry_Get_N_Cells(state, n_cells);
    auto cell_size = glm::vec3{0,0,0};
    for( int dim=0; dim<3; ++dim)
    {
        if( n_cells[dim] > 1 )
            cell_size[dim] = 1;
    }

    glm::vec3 c = bounding_box_renderer_ptr->getOption<VFRendering::BoundingBoxRenderer::Option::COLOR>();
    bounding_box_renderer_ptr = std::make_shared<VFRendering::BoundingBoxRenderer>(
        VFRendering::BoundingBoxRenderer::forCuboid(
            view, (geometry->min()+geometry->max())*0.5f, geometry->max()-geometry->min() + cell_size, indis, indi_dashes_per_length));
    bounding_box_renderer_ptr->setOption<VFRendering::BoundingBoxRenderer::Option::COLOR>(c);
    bounding_box_renderer_ptr->setOption<VFRendering::BoundingBoxRenderer::Option::LINE_WIDTH>(line_width);
    update_renderers(state);
}

extern "C" void set_coordinate_system(State * state, bool show, int position)
{
    show_coordinate_system = show;
    location_coordinatesystem = position;
    update_renderers(state);
}

extern "C" void set_dots(State * state, bool show)
{
    show_dots = show;
    update_renderers(state);
}

extern "C" void set_arrows(State * state, bool show)
{
    show_arrows = show;
    update_renderers(state);
}

extern "C" void set_spheres(State * state, bool show)
{
    show_spheres = show;
    update_renderers(state);
}

extern "C" void set_boxes(State * state, bool show)
{
    show_boxes = show;
    update_renderers(state);
}

extern "C" void set_isosurface(State * state, bool show)
{
    show_isosurface = show;
    update_renderers(state);
}

extern "C" void set_surface(State * state, bool show)
{
    show_surface = show;

    glm::vec3 max = geometry->max();
    glm::vec3 min = geometry->min();
    if( Geometry_Get_Dimensionality(state) == 3 )
    {
        surface_renderer_3d_ptr->setOption<VFRendering::IsosurfaceRenderer::Option::VALUE_FUNCTION>(
            [min, max] (const glm::vec3& position, const glm::vec3& direction) -> VFRendering::IsosurfaceRenderer::isovalue_type {
                (void)direction;

                glm::vec3 _min = min + glm::vec3{0.00001f, 0.00001f, 0.00001f};
                glm::vec3 _max = max - glm::vec3{0.00001f, 0.00001f, 0.00001f};

                /* Transform position in selected cuboid to position in unit cube [-1,1]^3 */
                glm::vec3 normalized_position = 2.0f * (position - _min) / (_max - _min) - 1.0f;

                /* Calculate maximum metric / Chebyshev distance */
                glm::vec3 absolute_normalized_position = glm::abs(normalized_position);
                float max_norm = glm::max(glm::max(absolute_normalized_position.x, absolute_normalized_position.y), absolute_normalized_position.z);

                /* Translate so that the selected cuboid surface has an isovalue of 0 */
                return max_norm - 1.0f;
            });
    }

    update_renderers(state);
}

extern "C" void set_colormap(int colormap)
{
    // std::string colormap_implementation(colormap);
    glm::vec3 cardinal_a{1, 0, 0};
    glm::vec3 cardinal_b{0, 1, 0};
    glm::vec3 cardinal_c{0, 0, 1};
    float sign_xy = 1;
    float P = 0;
    std::string colormap_implementation;
    if( colormap == 1 )
        colormap_implementation = R"(
            float atan2(float y, float x) {
                return x == 0.0 ? sign(y)*3.14159/2.0 : atan(y, x);
            }
            vec3 hsv2rgb(vec3 c) {
                vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
            }
            vec3 colormap(vec3 direction) {
                vec3 cardinal_a = vec3()" + std::to_string(cardinal_a.x) + ", " + std::to_string(cardinal_a.y) + ", " + std::to_string(cardinal_a.z) + R"();
                vec3 cardinal_b = vec3()" + std::to_string(cardinal_b.x) + ", " + std::to_string(cardinal_b.y) + ", " + std::to_string(cardinal_b.z) + R"();
                vec3 cardinal_c = vec3()" + std::to_string(cardinal_c.x) + ", " + std::to_string(cardinal_c.y) + ", " + std::to_string(cardinal_c.z) + R"();
                vec3 projection = vec3( dot(direction, cardinal_a), dot(direction, cardinal_b), dot(direction, cardinal_c) );
                float hue = atan2()" + std::to_string(sign_xy) + R"(*projection.x, projection.y) / 3.14159 / 2.0 + )" + std::to_string(P) + R"(;
                return hsv2rgb(vec3(hue, 1.0, 1.0));
            }
            )";
    else if( colormap == 2 )
        colormap_implementation = VFRendering::Utilities::getColormapImplementation(VFRendering::Utilities::Colormap::BLUERED);
    else if( colormap == 3 )
        colormap_implementation = VFRendering::Utilities::getColormapImplementation(VFRendering::Utilities::Colormap::BLUEGREENRED);
    else if( colormap == 4 )
        colormap_implementation = VFRendering::Utilities::getColormapImplementation(VFRendering::Utilities::Colormap::BLUEWHITERED);
    else if( colormap == 5 )
        colormap_implementation = R"(
            vec3 colormap(vec3 direction) {
                return vec3(0.5, 0, 0);
            }
            )";
    else if( colormap == 6 )
        colormap_implementation = VFRendering::Utilities::getColormapImplementation(VFRendering::Utilities::Colormap::WHITE);
    else if( colormap == 7 )
        colormap_implementation = R"(
            vec3 colormap(vec3 direction) {
                return vec3(0.5, 0.5, 0.5);
            }
            )";
    else if( colormap == 8 )
        colormap_implementation = VFRendering::Utilities::getColormapImplementation(VFRendering::Utilities::Colormap::BLACK);
    else
        colormap_implementation = VFRendering::Utilities::getColormapImplementation(VFRendering::Utilities::Colormap::HSV);

    view.setOption<VFRendering::View::COLORMAP_IMPLEMENTATION>(colormap_implementation);
}

extern "C" void set_visibility(float zmin, float zmax)
{
    if( zmin <= -1 )
        zmin = -1.1f;
    if( zmax >= 1 )
        zmax = 1.1f;
    std::string s_zmin = std::to_string(zmin);
    std::string s_zmax = std::to_string(zmax);
    std::string visibility = "bool is_visible(vec3 position, vec3 direction) { return direction.z <= "+s_zmax+" && direction.z >= "+s_zmin+"; }";
    vectorsphere_renderer_ptr->setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>(visibility);
    arrow_renderer_ptr->setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>(visibility);
    dot_renderer_ptr->setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>(visibility);
    sphere_renderer_ptr->setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>(visibility);
    box_renderer_ptr->setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>(visibility);
    surface_renderer_2d_ptr->setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>(visibility);
    surface_renderer_3d_ptr->setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>(visibility);
    isosurface_renderer_ptr->setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>(visibility);
}

extern "C" void set_vectorsphere(float pointsize_range_min, float pointsize_range_max)
{
    view.setOption<VFRendering::VectorSphereRenderer::Option::POINT_SIZE_RANGE>({pointsize_range_min, pointsize_range_max});
}

extern "C" void draw()
{
    EmscriptenWebGLContextAttributes attrs;
    emscripten_webgl_init_context_attributes(&attrs);
    attrs.majorVersion=1;
    attrs.minorVersion=0;
    EMSCRIPTEN_WEBGL_CONTEXT_HANDLE context = emscripten_webgl_create_context("#webgl-canvas", &attrs);
    int width, height;
    emscripten_webgl_get_drawing_buffer_size(context, &width, &height);
    framebufferSizeCallback(context, width, height);
    view.draw();
}


extern "C" void update_geometry(State * state)
{
    int nos = System_Get_NOS(state);
    int n_cells[3];
    Geometry_Get_N_Cells(state, n_cells);

    // float b_min[3], b_max[3];
    // Geometry_Get_Bounds(state.get(), b_min, b_max);
    // glm::vec3 bounds_min = glm::make_vec3(b_min);
    // glm::vec3 bounds_max = glm::make_vec3(b_max);

    scalar *spin_pos;
    int *atom_types;
    spin_pos = Geometry_Get_Positions(state);
    atom_types = Geometry_Get_Atom_Types(state);
    positions.resize(nos);
    int icell = 0;
    for( int cell_c=0; cell_c<n_cells[2]; cell_c++ )
    {
        for( int cell_b=0; cell_b<n_cells[1]; cell_b++ )
        {
            for( int cell_a=0; cell_a<n_cells[0]; cell_a++ )
            {
                // for( int ibasis=0; ibasis < n_cell_atoms; ++ibasis )
                // {
                    int idx = cell_a + n_cells[0]*cell_b + n_cells[0]*n_cells[1]*cell_c;
                    positions[icell] = glm::vec3(spin_pos[3*idx], spin_pos[1 + 3*idx], spin_pos[2 + 3*idx]);
                    ++icell;
                // }
            }
        }
    }

    std::vector<float> xs(n_cells[0]), ys(n_cells[1]), zs(n_cells[2]);
    for( int i = 0; i < n_cells[0]; ++i )
        xs[i] = positions[i].x;
    for( int i = 0; i < n_cells[1]; ++i )
        ys[i] = positions[i*n_cells[0]].y;
    for( int i = 0; i < n_cells[2]; ++i )
        zs[i] = positions[i*n_cells[0]*n_cells[1]].z;

    if( Geometry_Get_Dimensionality(state) == 3 )
    {
        *geometry = VFRendering::Geometry::rectilinearGeometry(xs, ys, zs);
    }
    else if( Geometry_Get_Dimensionality(state) == 2 )
    {
        *geometry = VFRendering::Geometry::rectilinearGeometry(xs, ys, zs);
    }
    vf->updateGeometry(*geometry);

    auto line_width = bounding_box_renderer_ptr->getOption<VFRendering::BoundingBoxRenderer::Option::LINE_WIDTH>();
    set_boundingbox(state, show_bounding_box, line_width);
}

extern "C" void update_directions(State * state)
{
    int nos = System_Get_NOS(state);
    // int n_cells[3];
    // Geometry_Get_N_Cells(this->state.get(), n_cells);
    // int n_cell_atoms = Geometry_Get_N_Cell_Atoms(this->state.get());

    auto dirs = System_Get_Spin_Directions(state);
    directions.resize(nos);
    // int N = directions.size();
    // std::cerr << "N = " << N << std::endl;
    for( int idx = 0; idx < nos; ++idx )
    {
        directions[idx] = glm::vec3{dirs[3*idx], dirs[3*idx + 1], dirs[3*idx + 2]};
        // if( idx < 20 )
            // std::cerr << idx << " " << directions[idx].x << " " << directions[idx].y << " " << directions[idx].z << std::endl;
    }

    // int icell = 0;
    // for (int cell_c=0; cell_c<n_cells[2]; cell_c++)
    // {
    //     for (int cell_b=0; cell_b<n_cells[1]; cell_b++)
    //     {
    //         for (int cell_a=0; cell_a<n_cells[0]; cell_a++)
    //         {
    //             int idx = cell_a + n_cells[0]*cell_b + n_cells[0]*n_cells[1]*cell_c;
    //             // std::cerr << idx << " " << icell << std::endl;
    //             directions[idx] = glm::vec3(dirs[3*idx], dirs[3*idx + 1], dirs[3*idx + 2]);
    //             // if (atom_types[idx] < 0) directions[icell] *= 0;
    //             // ++icell;
    //             std::cerr << idx << " " << directions[icell].z << std::endl;
    //         }
    //     }
    // }
    
    vf->updateVectors(directions);
}

extern "C" void initialize(State * state)
{
    EmscriptenWebGLContextAttributes attrs;
    emscripten_webgl_init_context_attributes(&attrs);
    attrs.majorVersion=1;
    attrs.minorVersion=0;

    EMSCRIPTEN_WEBGL_CONTEXT_HANDLE context = emscripten_webgl_create_context("#webgl-canvas", &attrs);
    emscripten_webgl_make_context_current(context);
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    int width, height;
    emscripten_webgl_get_drawing_buffer_size(context, &width, &height);
    framebufferSizeCallback(context, width, height);

    // Vectorfield
    positions = std::vector<glm::vec3>(1);
    directions = std::vector<glm::vec3>(1);
    auto geo = VFRendering::Geometry::cartesianGeometry({1, 1, 1}, {-1, -1, -1}, {1, 1, 1});
    geometry = std::shared_ptr<VFRendering::Geometry>(new VFRendering::Geometry(geo));
    vf = std::shared_ptr<VFRendering::VectorField>(new VFRendering::VectorField(*geometry, directions));

    // Default camera
    float pos[3]{-25.f, -25.f, 50.f};
    float center[3]{0.f, 0.f, 0.f};
    float up[3]{0.f, 0.f, 1.f};
    set_camera(pos, center, up);

    // General options
    view.setOption<VFRendering::View::Option::COLORMAP_IMPLEMENTATION>(
        VFRendering::Utilities::getColormapImplementation(VFRendering::Utilities::Colormap::HSV));
    // view.setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>(
    //     "bool is_visible(vec3 position, vec3 direction) { return position.x >= 0.0; }");

    // Background
    float colour[3]{0.5, 0.5, 0.5};
    set_background(colour);

    // --- Renderers
    // Bounding Box
    bounding_box_renderer_ptr = std::make_shared<VFRendering::BoundingBoxRenderer>(
        VFRendering::BoundingBoxRenderer::forCuboid(
            view,
            (geometry->min()+geometry->max())*0.5f,
            geometry->max()-geometry->min(),
            (geometry->max()-geometry->min())*0.2f,
            0.5f));
    // Coordinate system
    coordinate_system_renderer_ptr = std::make_shared<VFRendering::CoordinateSystemRenderer>(view);
    coordinate_system_renderer_ptr->setOption<VFRendering::CoordinateSystemRenderer::Option::AXIS_LENGTH>({1, 1, 1});
    coordinate_system_renderer_ptr->setOption<VFRendering::CoordinateSystemRenderer::Option::NORMALIZE>(true);
    auto colormap_implementation = VFRendering::Utilities::getColormapImplementation(VFRendering::Utilities::Colormap::HSV);
    coordinate_system_renderer_ptr->setOption<VFRendering::View::COLORMAP_IMPLEMENTATION>(colormap_implementation);

    // Vectorsphere
    // std::cerr << "//////////////////// INIT " << coordinate_system_renderer_ptr.get() << std::endl;
    vectorsphere_renderer_ptr = std::make_shared<VFRendering::VectorSphereRenderer>(view, *vf);
    // Dots
    dot_renderer_ptr = std::make_shared<VFRendering::DotRenderer>(view, *vf);
    dot_renderer_ptr->setOption<VFRendering::DotRenderer::DOT_RADIUS>(1000);
    // Arrows
    arrow_renderer_ptr = std::make_shared<VFRendering::ArrowRenderer>(view, *vf);
    // arrow_renderer_ptr->setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>(
    //     "bool is_visible(vec3 position, vec3 direction) { return position.x <= 0.0; }");
    // Spheres
    sphere_renderer_ptr = std::make_shared<VFRendering::SphereRenderer>(view, *vf);
    // sphere_renderer_ptr->setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>(
    //     "bool is_visible(vec3 position, vec3 direction) { return position.x <= 0.0; }");
    // Boxes
    box_renderer_ptr = std::make_shared<VFRendering::ParallelepipedRenderer>(view, *vf);
    box_renderer_ptr->setOption<VFRendering::ParallelepipedRenderer::LENGTH_A>(0.5);
    box_renderer_ptr->setOption<VFRendering::ParallelepipedRenderer::LENGTH_B>(0.5);
    box_renderer_ptr->setOption<VFRendering::ParallelepipedRenderer::LENGTH_C>(0.5);
    box_renderer_ptr->setOption<VFRendering::ParallelepipedRenderer::ROTATE_GLYPHS>(false);
    // Surface (2D)
    surface_renderer_2d_ptr = std::make_shared<VFRendering::SurfaceRenderer>(view, *vf);
    // Surface (3D)
    surface_renderer_3d_ptr = std::make_shared<VFRendering::IsosurfaceRenderer>(view, *vf);
    surface_renderer_3d_ptr->setOption<VFRendering::IsosurfaceRenderer::Option::ISOVALUE>(0.0);
    // Isosurface
    isosurface_renderer_ptr = std::make_shared<VFRendering::IsosurfaceRenderer>(view, *vf);
    // isosurface_renderer_ptr->setOption<VFRendering::IsosurfaceRenderer::Option::VALUE_FUNCTION>(
    //     [] (const glm::vec3& position, const glm::vec3& direction) -> VFRendering::IsosurfaceRenderer::isovalue_type {
    //         (void)direction;
    //         return position.z * position.z + position.y * position.y + position.x * position.x - 19*19;
    //     });
    // isosurface_renderer_ptr->setOption<VFRendering::IsosurfaceRenderer::Option::ISOVALUE>(0.0);
    // isosurface_renderer_ptr->setOption<VFRendering::IsosurfaceRenderer::Option::LIGHTING_IMPLEMENTATION>(
    //     "float lighting(vec3 position, vec3 normal) { return -normal.z; }");
    // isosurface_renderer_ptr->setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>(
    //     "bool is_visible(vec3 position, vec3 direction) { return position.x >= 0.0; }");

    isosurface_renderer_ptr->setOption<VFRendering::IsosurfaceRenderer::Option::VALUE_FUNCTION>(
        [] (const glm::vec3& position, const glm::vec3& direction) -> VFRendering::IsosurfaceRenderer::isovalue_type {
            (void)position;
            return direction.z;
        });
    isosurface_renderer_ptr->setOption<VFRendering::IsosurfaceRenderer::Option::ISOVALUE>(0.0);
    isosurface_renderer_ptr->setOption<VFRendering::IsosurfaceRenderer::Option::LIGHTING_IMPLEMENTATION>(
        "float lighting(vec3 position, vec3 normal) { return -normal.z; }");
    // isosurface_renderer_ptr->setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>(
    //     "bool is_visible(vec3 position, vec3 direction) { return position.x >= 0.0; }");

    // Combine
    update_renderers(state);

    // Get correct data
    update_directions(state);
    update_geometry(state);
}