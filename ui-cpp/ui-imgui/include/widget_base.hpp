#pragma once
#ifndef SPIRIT_IMGUI_WIDGET_BASE_HPP
#define SPIRIT_IMGUI_WIDGET_BASE_HPP
#include <string>
#include <array>
#include <imgui/imgui.h>

namespace ui
{

class WidgetBase
{
    public:
    enum class LayoutMode
    {
        FREE,
        STACKED
    };

    WidgetBase(bool & show_) : show_(show_) 
    {
        title = "Visualisation settings";
        size_min = { 300, 300 };
        size_max = { 800, 999999 };
        m_layout = LayoutMode::FREE;
    };

    bool & show_;

    LayoutMode m_layout;
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
        if(m_layout == LayoutMode::FREE)
        {
            ImGui::SetNextWindowSizeConstraints( size_min, size_max );
            ImGui::Begin( title.c_str(), &show_ );
            show_content();
            ImGui::End();
        } else if (m_layout == LayoutMode::STACKED)
        {
            if (ImGui::CollapsingHeader(title.c_str()))
            {
                show_content();
            }
        }
        hook_post_show();
    }
    void show(LayoutMode layout)
    {
        m_layout = layout;
        show();
    }
    virtual void show_content() = 0;
    virtual void hook_pre_show() {};
    virtual void hook_post_show() {};

};

}

#endif