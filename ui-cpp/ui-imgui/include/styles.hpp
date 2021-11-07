#pragma once
#ifndef SPIRIT_IMGUI_STYLES_HPP
#define SPIRIT_IMGUI_STYLES_HPP

#include <imgui/imgui.h>

namespace styles
{

void apply_light( ImGuiStyle * dst = nullptr );
void apply_charcoal( ImGuiStyle * dst = nullptr );

} // namespace styles

#endif