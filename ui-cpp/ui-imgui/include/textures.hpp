#pragma once
#ifndef SPIRIT_IMGUI_TEXTURES_HPP
#define SPIRIT_IMGUI_TEXTURES_HPP

namespace textures
{

// Load an image into an OpenGL texture with common settings, using stb_image
bool load_from_file( const char * filename, unsigned int * out_texture, int * out_width, int * out_height );

} // namespace textures

#endif