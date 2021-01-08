#pragma once
#ifndef SPIRIT_IMGUI_GLFW_WINDOW_HPP
#define SPIRIT_IMGUI_GLFW_WINDOW_HPP

#include <GLFW/glfw3.h>

namespace ui
{

class GlfwWindow
{
public:
    GlfwWindow();

protected:
    ~GlfwWindow();

    GLFWwindow * glfw_window;
};

} // namespace ui

#endif