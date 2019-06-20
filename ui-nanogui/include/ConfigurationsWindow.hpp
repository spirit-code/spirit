#pragma once

#include <nanogui/window.h>
#include <nanogui/button.h>

#include <Spirit/Configurations.h>

class ConfigurationsWindow : public nanogui::Window
{
public:
    ConfigurationsWindow(nanogui::Screen *parent, std::shared_ptr<State> state)
        : nanogui::Window(parent, "Configurations"), state(state)
    {
        using namespace nanogui;

        Button *b_plus_z = new Button(this, "+z");
        b_plus_z->setCallback([&] {
            Configuration_PlusZ(this->state.get());
            this->mUpdateCallback();
        });
        // b_plus_z->setTooltip("set a homogeneous configuration in +z direction");

        Button *b_skyrmion = new Button(this, "Skyrmion");
        b_skyrmion->setCallback([&] {
            Configuration_Skyrmion(this->state.get(), 5, 1, -90, false, false, false);
            this->mUpdateCallback();
        });
        // b_skyrmion->setTooltip("set a skyrmion configuration");
    }

    void setUpdateCallback(const std::function<void()> &callback)
    {
        mUpdateCallback = callback;
    }

private:
    std::shared_ptr<State> state;
    std::function<void()> mUpdateCallback;
};