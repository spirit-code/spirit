#pragma once

#include <nanogui/opengl.h>
#include <nanogui/glutil.h>
#include <nanogui/screen.h>
#include <nanogui/window.h>
#include <nanogui/graph.h>
#include <nanogui/button.h>
#include <nanogui/layout.h>
#include <nanogui/tabwidget.h>
#include <nanogui/tabheader.h>
#include <nanogui/glcanvas.h>

#include <VFRendering/View.hxx>
#include <VFRendering/CombinedRenderer.hxx>
#include <VFRendering/BoundingBoxRenderer.hxx>
#include <VFRendering/DotRenderer.hxx>
#include <VFRendering/ArrowRenderer.hxx>
#include <VFRendering/ParallelepipedRenderer.hxx>

#include <glm/gtc/type_ptr.hpp>

#include <Spirit/Geometry.h>
#include <Spirit/System.h>
#include <Spirit/Configurations.h>
#include <Spirit/Simulation.h>
#include <Spirit/Hamiltonian.h>

class VFGLCanvas : public nanogui::GLCanvas {
public:
    VFGLCanvas(nanogui::Widget *parent, std::shared_ptr<State> state)
        : nanogui::GLCanvas(parent), state(state)
    {
        using namespace nanogui;

        // Create the VFRendering::View
        this->view = VFRendering::View();
        glm::vec3 color = { 0.5, 0.5, 0.5 };
        this->view.setOption<VFRendering::View::Option::BACKGROUND_COLOR>(color);

        int nos = System_Get_NOS(state.get());
        int n_cells[3];
        Geometry_Get_N_Cells(this->state.get(), n_cells);
        int n_cell_atoms = Geometry_Get_N_Cell_Atoms(this->state.get());

        float b_min[3], b_max[3];
        Geometry_Get_Bounds(state.get(), b_min, b_max);
        glm::vec3 bounds_min = glm::make_vec3(b_min);
        glm::vec3 bounds_max = glm::make_vec3(b_max);
        glm::vec2 x_range{bounds_min[0], bounds_max[0]};
        glm::vec2 y_range{bounds_min[1], bounds_max[1]};
        glm::vec2 z_range{bounds_min[2], bounds_max[2]};
        glm::vec3 bounding_box_center = { (bounds_min[0] + bounds_max[0]) / 2, (bounds_min[1] + bounds_max[1]) / 2, (bounds_min[2] + bounds_max[2]) / 2 };
        glm::vec3 bounding_box_side_lengths = { bounds_max[0] - bounds_min[0], bounds_max[1] - bounds_min[1], bounds_max[2] - bounds_min[2] };

        float indi_length = glm::length(bounds_max - bounds_min)*0.05;
        int   indi_dashes = 5;
        float indi_dashes_per_length = (float)indi_dashes / indi_length;

        bool periodical[3];
        Hamiltonian_Get_Boundary_Conditions(this->state.get(), periodical);
        glm::vec3 indis{ indi_length*periodical[0], indi_length*periodical[1], indi_length*periodical[2] };

        this->m_renderer_boundingbox = std::make_shared<VFRendering::BoundingBoxRenderer>(
            VFRendering::BoundingBoxRenderer::forCuboid(this->view, bounding_box_center, bounding_box_side_lengths, indis, indi_dashes_per_length));

        this->directions = std::vector<glm::vec3>(nos, glm::vec3{0,0,1});
        this->positions = std::vector<glm::vec3>(nos);
        scalar *spin_pos;
        int *atom_types;
        spin_pos = Geometry_Get_Positions(state.get());
        atom_types = Geometry_Get_Atom_Types(state.get());
        int icell = 0;
        for( int cell_c=0; cell_c<n_cells[2]; cell_c++ )
        {
            for( int cell_b=0; cell_b<n_cells[1]; cell_b++ )
            {
                for( int cell_a=0; cell_a<n_cells[0]; cell_a++ )
                {
                    for( int ibasis=0; ibasis < n_cell_atoms; ++ibasis )
                    {
                        int idx = ibasis + n_cell_atoms * (
                            + cell_a + n_cells[0]*cell_b + n_cells[0]*n_cells[1]*cell_c );
                        positions[icell] = glm::vec3(spin_pos[3*idx], spin_pos[1 + 3*idx], spin_pos[2 + 3*idx]);
                        ++icell;
                    }
                }
            }
        }
        // this->geometry = VFRendering::Geometry::cartesianGeometry({21, 21, 21}, {-20, -20, -20}, {20, 20, 20});
        this->geometry = VFRendering::Geometry(positions, {}, {}, true);
        this->vf = new VFRendering::VectorField(this->geometry, this->directions);

        // Renderers
        this->m_renderer_dots = std::shared_ptr<VFRendering::DotRenderer>(new VFRendering::DotRenderer(this->view, *this->vf));
        this->m_renderer_dots->setOption<VFRendering::DotRenderer::DOT_RADIUS>(1000);

        this->m_renderer_arrows = std::shared_ptr<VFRendering::ArrowRenderer>(new VFRendering::ArrowRenderer(this->view, *this->vf));

        this->m_renderer_parallelpipeds = std::shared_ptr<VFRendering::ParallelepipedRenderer>(new VFRendering::ParallelepipedRenderer(this->view, *this->vf));
        this->m_renderer_parallelpipeds->setOption<VFRendering::ParallelepipedRenderer::LENGTH_A>(0.5);
        this->m_renderer_parallelpipeds->setOption<VFRendering::ParallelepipedRenderer::LENGTH_B>(0.5);
        this->m_renderer_parallelpipeds->setOption<VFRendering::ParallelepipedRenderer::LENGTH_C>(0.5);
        this->m_renderer_parallelpipeds->setOption<VFRendering::ParallelepipedRenderer::ROTATE_GLYPHS>(false);

        this->updateRenderers();

        VFRendering::Options options;
        float camera_distance = 30;
        auto center = bounding_box_center;//(geometry.min() + geometry.max()) * 0.5f;
        options.set<VFRendering::View::Option::SYSTEM_CENTER>(center);

        options.set<VFRendering::View::Option::COLORMAP_IMPLEMENTATION>(VFRendering::Utilities::getColormapImplementation(VFRendering::Utilities::Colormap::HSV));
        options.set<VFRendering::View::Option::CAMERA_POSITION>(center + camera_distance * glm::vec3(0, 0, 1));
        options.set<VFRendering::View::Option::CENTER_POSITION>(center);
        options.set<VFRendering::View::Option::UP_VECTOR>({0, 1, 0});

        view.updateOptions(options);

        this->updateDirections();
    }

    ~VFGLCanvas()
    {
        delete this->vf;
    }

    virtual void drawGL() override
    {
        using namespace nanogui;
        glEnable(GL_DEPTH_TEST);
        if( Simulation_Running_On_Image(this->state.get(), System_Get_Index(this->state.get())) )
            this->updateDirections();
        this->view.draw();
        glDisable(GL_DEPTH_TEST);
    }

    void setSize(const Eigen::Vector2i & size)
    {
        nanogui::GLCanvas::setSize(size);
        nanogui::Vector2i fs = this->fixedSize();
        nanogui::Vector2i newSize = {fs[0] > 0 ? fs[0] : size[0], fs[1] > 0 ? fs[1] : size[1] };
        view.setFramebufferSize(newSize[0]*this->screen()->pixelRatio(), newSize[1]*this->screen()->pixelRatio());
    }

    void setFixedSize(const Eigen::Vector2i & size)
    {
        nanogui::GLCanvas::setFixedSize(size);
        view.setFramebufferSize(this->size()[0]*this->screen()->pixelRatio(), this->size()[1]*this->screen()->pixelRatio());
    }

    // We set a small value here to allow layout container to shrink this
    virtual nanogui::Vector2i preferredSize(NVGcontext *ctx) const override
    {
        return {100, 100};
    }

    void updateRenderers()
    {
        this->renderers = {};
        // if this->bounding_box.show:
            this->renderers.push_back(this->m_renderer_boundingbox);
        // if this->dots.show:
            this->renderers.push_back(this->m_renderer_dots);
        // if this->arrows.show:
            // this->renderers.push_back(this->m_renderer_arrows);
        // if this->cubes.show:
            // this->renderers.push_back(this->m_renderer_parallelpipeds);

        // Combine renderers
        auto renderers_system = std::shared_ptr<VFRendering::CombinedRenderer>(new VFRendering::CombinedRenderer( this->view, this->renderers ));
        this->view.renderers({{renderers_system, {0.0, 0.0, 1.0, 1.0}}}, false);
    }

    virtual bool mouseDragEvent(const Eigen::Vector2i &current_position,
        const Eigen::Vector2i &relative_position, int button, int modifiers) override
    {
        auto prev = current_position-relative_position;
        glm::vec2 previous = {prev.x(), prev.y()};
        glm::vec2 current = {current_position.x(), current_position.y()};
        if( button & (1 << GLFW_MOUSE_BUTTON_LEFT)  )
        {
            view.mouseMove(previous, current, VFRendering::CameraMovementModes::ROTATE_BOUNDED);
            return true;
        }
        else if( button & (1 << GLFW_MOUSE_BUTTON_RIGHT) )
        {
            view.mouseMove(previous, current, VFRendering::CameraMovementModes::TRANSLATE);
            return true;
        }
        return false;
    }

    virtual bool scrollEvent(const Eigen::Vector2i &p, const Eigen::Vector2f &relative_position) override
    {
        view.mouseScroll(-relative_position.y());
        return true;
    }

    void updateDirections()
    {
        int nos = System_Get_NOS(state.get());
        auto spins = System_Get_Spin_Directions(state.get());
        for(int i=0; i<nos; ++i)
        {
            this->directions[i] = glm::vec3(spins[3*i], spins[1 + 3*i], spins[2 + 3*i]);;
        }
        this->vf->updateVectors(this->directions);
    }

    void updateData()
    {
        this->updateDirections();
        this->drawGL();
    }


    VFRendering::Geometry geometry;
    std::vector<glm::vec3> directions;
    std::vector<glm::vec3> positions;

private:
    std::shared_ptr<State> state;

    VFRendering::View view;
    VFRendering::VectorField * vf;

    std::vector<std::shared_ptr<VFRendering::RendererBase>> renderers;
    std::shared_ptr<VFRendering::BoundingBoxRenderer> m_renderer_boundingbox;
    std::shared_ptr<VFRendering::ArrowRenderer> m_renderer_arrows;
    std::shared_ptr<VFRendering::ParallelepipedRenderer> m_renderer_parallelpipeds;
    std::shared_ptr<VFRendering::DotRenderer> m_renderer_dots;
};