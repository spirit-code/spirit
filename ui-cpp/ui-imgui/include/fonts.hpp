#pragma once
#ifndef SPIRIT_IMGUI_FONTS_HPP
#define SPIRIT_IMGUI_FONTS_HPP

#include "imgui_impl/fontawesome5_icons.h"

#include <imgui/imgui.h>

namespace fonts
{

ImFont * imgui_default( float size_px );
ImFont * cousine_regular( float size_px );
ImFont * fontawesome_icons( float size_px );
ImFont * font_combined( float size_px );

} // namespace fonts

#endif