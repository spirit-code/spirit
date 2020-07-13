#pragma once
#ifndef SPIRIT_IMGUI_MAIN_WINDOW_HPP
#define SPIRIT_IMGUI_MAIN_WINDOW_HPP

#include <enums.hpp>

#include <imgui/imgui.h>

#include <GLFW/glfw3.h>

#include <VFRendering/ArrowRenderer.hxx>
#include <VFRendering/BoundingBoxRenderer.hxx>
#include <VFRendering/CombinedRenderer.hxx>
#include <VFRendering/CoordinateSystemRenderer.hxx>
#include <VFRendering/IsosurfaceRenderer.hxx>
#include <VFRendering/SphereRenderer.hxx>
#include <VFRendering/View.hxx>

#include <Spirit/Simulation.h>

#include <memory>
#include <thread>
#include <vector>

struct State;

class MainWindow
{
public:
    MainWindow( std::shared_ptr<State> state );
    ~MainWindow();

    int run();
    void draw();
    void resize( int width, int height );

private:
    void show_menu_bar();

    void intitialize_gl();
    void draw_vfr( int display_w, int display_h );
    void draw_imgui( int display_w, int display_h );

    void update_vf_geometry();
    void update_vf_directions();

    void reset_camera();
    void handle_mouse();
    void handle_keyboard();

    void start_stop();
    void stop_all();
    void stop_current();

    GLFWwindow * glfw_window;

    ImFont * font_cousine_14 = nullptr;
    ImFont * font_karla_14   = nullptr;
    ImFont * font_karla_16   = nullptr;
    ImFont * font_karla_18   = nullptr;
    ImFont * font_mono_14    = nullptr;

    std::shared_ptr<State> state;
    std::vector<std::thread> threads_image;
    std::thread thread_chain;

    // Window settings
    int selected_solver_min = Solver_VP_OSO;
    int selected_solver_llg = Solver_Depondt;
    GUI_Mode selected_mode  = GUI_Mode::Minimizer;

    bool show_overlays               = true;
    bool show_parameters_settings    = true;
    bool show_visualisation_settings = true;
    bool show_hamiltonian_settings   = true;
    bool show_geometry_settings      = true;
    bool show_plots                  = true;

    bool show_demo_window = false;

    bool show_keybindings = false;
    bool show_about       = false;

    bool dark_mode  = true;
    bool maximized  = false;
    bool fullscreen = false;

    // Visualisation settings
    bool vfr_needs_redraw       = true;
    bool vfr_needs_data         = true;
    glm::vec4 background_colour = glm::vec4{ 0.4f, 0.4f, 0.4f, 0.f };

    VFRendering::View vfr_view;
    VFRendering::Geometry vfr_geometry;
    VFRendering::VectorField vfr_vectorfield        = VFRendering::VectorField( {}, {} );
    VFRendering::VectorField vfr_vectorfield_surf2D = VFRendering::VectorField( {}, {} );

    std::shared_ptr<VFRendering::ArrowRenderer> vfr_arrow_renderer_ptr;
    std::vector<std::shared_ptr<VFRendering::RendererBase>> vfr_system_renderers{};

    int n_cell_step = 1;

    // Interaction
    bool m_dragging = false;
};

#endif