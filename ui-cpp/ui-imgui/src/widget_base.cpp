#include <widget_base.hpp>

#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>

namespace ui
{

ImGuiID WidgetBase::get_root_dock_node_id()
{
    if( ImGui::IsWindowDocked() )
    {
        ImGuiDockNode * node   = ImGui::GetCurrentWindow()->DockNode;
        ImGuiDockNode * parent = node->ParentNode;

        while( parent )
        {
            node   = parent;
            parent = node->ParentNode;
        }

        return node->ID;
    }

    return 0;
}

} // namespace ui