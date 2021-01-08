#pragma once
#ifndef SPIRIT_IMGUI_MAIN_WINDOW_HPP
#define SPIRIT_IMGUI_MAIN_WINDOW_HPP

#include <configurations_widget.hpp>
#include <geometry_widget.hpp>
#include <glfw_window.hpp>
#include <parameters_widget.hpp>
#include <plots_widget.hpp>
#include <rendering_layer.hpp>
#include <ui_config_file.hpp>
#include <ui_shared_state.hpp>
#include <visualisation_widget.hpp>

#include <imgui/imgui.h>

#include <memory>
#include <thread>
#include <vector>

struct State;

namespace ui
{

class MainWindow : GlfwWindow
{
public:
    MainWindow( std::shared_ptr<State> state );
    ~MainWindow();

    int run();
    void draw();
    void resize( int width, int height );

private:
    void show_menu_bar();
    void show_notifications();

    void draw_imgui( int display_w, int display_h );

    void handle_mouse();
    void handle_keyboard();

    void start_stop();
    void stop_all();
    void stop_current();

    void cut_image();
    void paste_image();
    void insert_image_left();
    void insert_image_right();
    void delete_image();

    UiSharedState ui_shared_state;
    RenderingLayer rendering_layer;
    UiConfigFile ui_config_file;

    ConfigurationsWidget configurations_widget;
    ParametersWidget parameters_widget;
    GeometryWidget geometry_widget;
    PlotsWidget plots_widget;
    VisualisationWidget visualisation_widget;

    ImFont * font_cousine_14 = nullptr;
    ImFont * font_karla_14   = nullptr;
    ImFont * font_karla_16   = nullptr;
    ImFont * font_karla_18   = nullptr;
    ImFont * font_mono_14    = nullptr;

    std::shared_ptr<State> state;
    std::vector<std::thread> threads_image;
    std::thread thread_chain;

    bool show_imgui_demo_window  = false;
    bool show_implot_demo_window = false;

    bool show_keybindings    = false;
    bool show_about          = false;
    bool calculation_running = false;
};

} // namespace ui

#endif