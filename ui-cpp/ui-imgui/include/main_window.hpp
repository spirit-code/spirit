#pragma once
#ifndef SPIRIT_IMGUI_MAIN_WINDOW_HPP
#define SPIRIT_IMGUI_MAIN_WINDOW_HPP

#include <imgui/imgui.h>

#include <memory>

struct State;

class main_window
{
public:
    main_window( std::shared_ptr<State> state );
    // ~main_window();
    int run();

private:
    void intitialize_gl();
    void draw_gl( int display_w, int display_h );

    std::shared_ptr<State> state;
};

#endif