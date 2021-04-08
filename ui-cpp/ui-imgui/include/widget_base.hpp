#pragma once
#ifndef SPIRIT_IMGUI_WIDGET_BASE_HPP
#define SPIRIT_IMGUI_WIDGET_BASE_HPP
#include <string>
#include <array>
#include <imgui/imgui.h>
#include <iostream>
#include <functional>

namespace ui
{

class WidgetBase
{
    public:
    WidgetBase(bool & show_) : show_(show_) 
    {
        title = "Base Widget";
        size_min = { 50, 50 };
        size_max = { 800, 999999 };
    };

    bool & show_;
    bool dragging = false; // Is the window being dragged?
    bool docked = false; // Is the window docked to another window?
    bool wants_to_dock = false; // Should the window dock to the sidebar in the next frame? (Triggered via context menu 'attach')
    std::string title;
    ImVec2 size_min;
    ImVec2 size_max;

    virtual void show()
    {
        hook_pre_show();
        if(!show_)
        {
            return;
        }

        ImGui::SetNextWindowSizeConstraints( size_min, size_max );
        ImGui::Begin( title.c_str(), &show_ );
        docked = ImGui::IsWindowDocked();
        if(ImGui::IsItemHovered() && ImGui::IsMouseDragging(0))
        {
            dragging = true;
        } else {
            dragging = false;
        }
        if( ImGui::BeginPopupContextWindow() )
        {
            if( ImGui::MenuItem( "Attach", NULL, false ) )
                wants_to_dock=true;
            ImGui::EndPopup();
        }
        show_content();
        ImGui::End();
        hook_post_show();
    }

    virtual void show_content() = 0;
    virtual void hook_pre_show() {};
    virtual void hook_post_show() {};

};

}

#endif