#pragma once
#ifndef METHODWIDGET_HPP
#define METHODWIDGET_HPP

#include "Spirit/Simulation.h"
#include <nanogui/window.h>
#include <thread>

class MethodWidget : public nanogui::Window
{
    // Enum and string vectors must be in same order!
    enum Method
    {
        LLG  = 0,
        MC   = 1,
        GNEB = 2,
        MMF  = 3,
        EMA  = 4
    };
    const std::vector<std::string> method_strings = {"LLG", "MC", "GNEB", "MMF", "EMA"};

    enum Solver
    {
        VP          = Solver_VP,
        SIB         = Solver_SIB,
        Depondt     = Solver_Depondt,
        Heun        = Solver_Heun,
        RungeKutta4 = Solver_RungeKutta4
    };
    const std::vector<std::string> solver_strings = {"VP", "SIB", "Depondt", "Heun", "RK4"};

public:
    MethodWidget(nanogui::Widget * parent, std::shared_ptr<State> state);
    ~MethodWidget();
    void updateThreads();
    void update();
    void start_stop();
    virtual void draw(NVGcontext *ctx) override;
    // This window should not be draggable
    bool mouseDragEvent(const Eigen::Vector2i& /* p */, const Eigen::Vector2i& /* rel */, int /* button */, int /* modifiers*/) override
    { return true; }

protected:
    nanogui::Button * start_stop_button;
    nanogui::ComboBox * method_select;
    nanogui::ComboBox * solver_select;
    nanogui::Label * image_label;
    int idx_img   = -1;
    int idx_chain = -1;
    Method selected_method = Method::LLG;
    Solver selected_solver = Solver::VP;
    std::shared_ptr<State> state;

    std::vector<std::thread> threads_image;
    std::thread thread_chain;
};

#endif