#pragma once
#ifndef SPIRIT_IMGUI_MAIN_WINDOW_HPP
#define SPIRIT_IMGUI_MAIN_WINDOW_HPP

#include <imgui/imgui.h>

#include <memory>
#include <thread>
#include <vector>

struct State;

class main_window
{
public:
    main_window( std::shared_ptr<State> state );
    // ~main_window();
    int run();
    void loop();

    std::shared_ptr<State> state;
    std::vector<std::thread> threads_image;
    std::thread thread_chain;

private:
    void quit();

    void intitialize_gl();
    void draw_vfr( int display_w, int display_h );
    void draw_imgui( int display_w, int display_h );

    void update_vf_geometry();
    void update_vf_directions();

    void reset_camera();
    void handle_mouse();
    void handle_keyboard();

    void start_stop();
    void stop_current();

    int selected_method;
    int selected_solver;
};

#endif