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

#include <Spirit/Geometry.h>
#include <Spirit/System.h>

static bool needs_redraw = false;
static const int default_mode = (int)VFRendering::CameraMovementModes::ROTATE_BOUNDED;
VFRendering::View view;

// Data
std::vector<glm::vec3> positions;
std::vector<glm::vec3> directions;

std::shared_ptr<VFRendering::Geometry> geometry;
std::shared_ptr<VFRendering::VectorField> vf;

glm::ivec3 n_cells{21, 21, 21};

// Renderers
std::shared_ptr<VFRendering::BoundingBoxRenderer>       bounding_box_renderer_ptr;
std::shared_ptr<VFRendering::CoordinateSystemRenderer>  coordinate_system_renderer_ptr;
std::shared_ptr<VFRendering::DotRenderer>               dot_renderer_ptr;
std::shared_ptr<VFRendering::ArrowRenderer>             arrow_renderer_ptr;
std::shared_ptr<VFRendering::IsosurfaceRenderer>        isosurface_renderer_ptr;

void framebufferSizeCallback(EMSCRIPTEN_WEBGL_CONTEXT_HANDLE context, int width, int height)
{
    (void)context;
    view.setFramebufferSize(width, height);
    needs_redraw = true;
}

extern "C" void mouse_move(float previous_mouse_position[2], float current_mouse_position[2], int movement_mode=default_mode)
{
    glm::vec2 previous = {previous_mouse_position[0], previous_mouse_position[1]};
    glm::vec2 current = {current_mouse_position[0], current_mouse_position[1]};
    VFRendering::CameraMovementModes mode = VFRendering::CameraMovementModes(movement_mode);
    view.mouseMove(previous, current, mode);
    // // std::cerr << "move: "<< previous.x << " " << previous.y << " " << current.x << " " << current.y <<"\n";
}

extern "C" void mouse_scroll(float wheel_delta, float scale=1)
{
    view.mouseScroll( -wheel_delta * 0.1 * scale );
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
    VFRendering::Options options;
    glm::vec3 d = options.get<VFRendering::View::Option::CENTER_POSITION>() - options.get<VFRendering::View::Option::CAMERA_POSITION>();
    float dist = glm::length(d);
    glm::vec3 dir{direction[0], direction[1], direction[2]};
    glm::normalize(dir);
    options.set<VFRendering::View::Option::CAMERA_POSITION>((geometry->min() + geometry->max()) * 0.5f - dist*dir);
    // options.set<VFRendering::View::Option::CENTER_POSITION>({center[0], center[1], center[2]});
    options.set<VFRendering::View::Option::CENTER_POSITION>((geometry->min() + geometry->max()) * 0.5f);
    options.set<VFRendering::View::Option::UP_VECTOR>({up[0], up[1], up[2]});
    view.updateOptions(options);
}

extern "C" void set_boundingbox()
{
    bounding_box_renderer_ptr = std::make_shared<VFRendering::BoundingBoxRenderer>(VFRendering::BoundingBoxRenderer::forCuboid(view, (geometry->min()+geometry->max())*0.5f, geometry->max()-geometry->min(), (geometry->max()-geometry->min())*0.2f, 0.5f));
}

extern "C" void set_coordinate_system()
{
    coordinate_system_renderer_ptr = std::make_shared<VFRendering::CoordinateSystemRenderer>(view);
    coordinate_system_renderer_ptr->setOption<VFRendering::CoordinateSystemRenderer::Option::AXIS_LENGTH>({0, 20, 20});
    coordinate_system_renderer_ptr->setOption<VFRendering::CoordinateSystemRenderer::Option::NORMALIZE>(false);
}

extern "C" void set_dots()
{
    dot_renderer_ptr = std::make_shared<VFRendering::DotRenderer>(view, *vf.get());
}

extern "C" void set_arrows()
{
    arrow_renderer_ptr = std::make_shared<VFRendering::ArrowRenderer>(view, *vf.get());
    arrow_renderer_ptr->setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>(
        "bool is_visible(vec3 position, vec3 direction) { return position.x <= 0.0; }");
}

extern "C" void set_isosurface()
{
    isosurface_renderer_ptr = std::make_shared<VFRendering::IsosurfaceRenderer>(view, *vf.get());
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
}


extern "C" void initialize()
{
    EmscriptenWebGLContextAttributes attrs;
    emscripten_webgl_init_context_attributes(&attrs);
    attrs.majorVersion=2;
    attrs.minorVersion=0;

    EMSCRIPTEN_WEBGL_CONTEXT_HANDLE context = emscripten_webgl_create_context("#webgl-canvas", &attrs);
    emscripten_webgl_make_context_current(context);
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    int width, height;
    emscripten_webgl_get_drawing_buffer_size(context, &width, &height);
    framebufferSizeCallback(context, width, height);

    // Vectorfield
    positions = std::vector<glm::vec3>(0);
    directions = std::vector<glm::vec3>(0);
    for (int z = -20; z <= 20; z+=2)
    {
        for (int y = -20; y <= 20; y+=2)
        {
            for (int x = -20; x <= 20; x+=2)
            {
                positions.push_back({x, y, z});
                directions.push_back(glm::vec3{x, y, z} / glm::length(positions.back()));
            }
        }
    }
    auto geo = VFRendering::Geometry::cartesianGeometry(n_cells, {-20, -20, -20}, {20, 20, 20});
    geometry = std::shared_ptr<VFRendering::Geometry>(new VFRendering::Geometry(geo));
    vf = std::shared_ptr<VFRendering::VectorField>(new VFRendering::VectorField(*geometry.get(), directions));

    // Default camera
    float pos[3]{-50.f, -50.f, 100.f};
    float center[3]{0.f, 0.f, 0.f};
    float up[3]{0.f, 0.f, 1.f};
    set_camera(pos, center, up);

    // General options
    view.setOption<VFRendering::View::Option::COLORMAP_IMPLEMENTATION>(
        VFRendering::Utilities::getColormapImplementation(VFRendering::Utilities::Colormap::HSV));
    // view.setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>(
    //     "bool is_visible(vec3 position, vec3 direction) { return position.x >= 0.0; }");
    // glm::vec3 colour;
    // if      (background_color == Color::BLACK) colour = { 0,   0,   0   };
    // else if (background_color == Color::GRAY)  colour = { 0.5, 0.5, 0.5 };
    // else if (background_color == Color::WHITE) colour = { 1,   1,   1   };
    view.setOption<VFRendering::View::Option::BACKGROUND_COLOR>({ 0.5, 0.5, 0.5 });

    // Renderers
    set_boundingbox();
    set_coordinate_system();
    set_dots();
    set_arrows();
    set_isosurface();

    // sphere
    auto sphere_renderer_ptr = std::make_shared<VFRendering::SphereRenderer>(view, *vf.get());
    sphere_renderer_ptr->setOption<VFRendering::View::Option::IS_VISIBLE_IMPLEMENTATION>(
        "bool is_visible(vec3 position, vec3 direction) { return position.x <= 0.0; }");
    // yz-plane
    auto yzplane_renderer_ptr = std::make_shared<VFRendering::IsosurfaceRenderer>(view, *vf.get());
    yzplane_renderer_ptr->setOption<VFRendering::IsosurfaceRenderer::Option::VALUE_FUNCTION>(
        [] (const glm::vec3& position, const glm::vec3& direction) -> VFRendering::IsosurfaceRenderer::isovalue_type {
            (void)direction;
            return position.x;
        });
    yzplane_renderer_ptr->setOption<VFRendering::IsosurfaceRenderer::Option::ISOVALUE>(0.0);

    // Combine
    std::vector<std::shared_ptr<VFRendering::RendererBase>> renderers = {
            isosurface_renderer_ptr,
            yzplane_renderer_ptr,
            arrow_renderer_ptr,
            sphere_renderer_ptr,
            dot_renderer_ptr,
            bounding_box_renderer_ptr,
            coordinate_system_renderer_ptr
    };
    view.renderers({{std::make_shared<VFRendering::CombinedRenderer>(view, renderers), {{0, 0, 1, 1}}}});
}

extern "C" void draw()
{
    EmscriptenWebGLContextAttributes attrs;
    emscripten_webgl_init_context_attributes(&attrs);
    attrs.majorVersion=2;
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

}

extern "C" void update_directions(State * state)
{
    // int nos = System_Get_NOS(state);
    // int n_cells[3];
    // Geometry_Get_N_Cells(this->state.get(), n_cells);
    // int n_cell_atoms = Geometry_Get_N_Cell_Atoms(this->state.get());

    auto dirs = System_Get_Spin_Directions(state);
    int N = directions.size();
    // std::cerr << "N = " << N << std::endl;
    for( int idx = 0; idx < N; ++idx )
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