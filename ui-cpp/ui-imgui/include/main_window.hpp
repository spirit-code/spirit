#pragma once
#ifndef SPIRIT_IMGUI_MAIN_WINDOW_HPP
#define SPIRIT_IMGUI_MAIN_WINDOW_HPP

#include <rendering_layer.hpp>
#include <ui_config_file.hpp>
#include <ui_shared_state.hpp>

#include <imgui/imgui.h>

#include <GLFW/glfw3.h>

#include <memory>
#include <thread>
#include <vector>

struct State;

namespace ui
{

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
    void show_notifications();

    void draw_imgui( int display_w, int display_h );

    void reset_camera();
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

    GLFWwindow * glfw_window;
    UiSharedState ui_shared_state;

    UiConfigFile ui_config_file;
    RenderingLayer rendering_layer;

    ImFont * font_cousine_14 = nullptr;
    ImFont * font_karla_14   = nullptr;
    ImFont * font_karla_16   = nullptr;
    ImFont * font_karla_18   = nullptr;
    ImFont * font_mono_14    = nullptr;

    std::shared_ptr<State> state;
    std::vector<std::thread> threads_image;
    std::thread thread_chain;

    bool show_demo_window = false;

    bool show_keybindings = false;
    bool show_about       = false;
};

} // namespace ui

#endif