#include <glad/glad.h>

#include <images.hpp>

#include <cmrc/cmrc.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image_resize2.h>

#include <fmt/format.h>

#include <array>
#include <memory>
#include <vector>

CMRC_DECLARE( resources );

namespace images
{

Image::Image( std::string file_name ) : file_name( file_name )
{
    auto fs   = cmrc::resources::get_filesystem();
    auto file = fs.open( file_name );

    file_data  = std::string( file.cbegin(), file.cend() );
    image_data = stbi_load_from_memory(
        reinterpret_cast<const unsigned char *>( file_data.c_str() ),
        file_data.size() * sizeof( char ) / sizeof( unsigned char ), &width, &height, NULL, 4 );

    if( !image_data )
    {
        fmt::print( "Image: could not load image data from \"{}\"\n", file_name );
    }
}

Image::~Image()
{
    stbi_image_free( image_data );
}

bool Image::get_gl_texture( unsigned int & out_texture )
{
    if( !image_data )
    {
        fmt::print( "Cannot get gl texture without image data\n" );
        return false;
    }

    GLuint image_texture;
    glGenTextures( 1, &image_texture );
    glBindTexture( GL_TEXTURE_2D, image_texture );

    // Setup filtering parameters for display
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

    // Upload pixels into texture
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data );
    out_texture = image_texture;

    return true;
}

bool glfw_set_app_icon( GLFWwindow * glfw_window )
{
    images::Image icon_338( "res/Logo_Ghost_Final_Notext.png" );

    static constexpr int num_channels = 4;
    static constexpr std::array<int, 15> resolutions{ 16, 20, 24, 28, 30, 31, 32, 40, 42, 47, 48, 56, 60, 63, 84 };
    std::vector<std::vector<stbir_uint8>> data( resolutions.size() );
    std::array<GLFWimage, resolutions.size()> glfw_images;

    if( icon_338.image_data )
    {
        for( int i = 0; i < resolutions.size(); ++i )
        {
            // fmt::print( "resizing to {}\n", resolutions[i] );
            // data[i] = std::vector<stbir_uint8>( 40 * resolutions[i] );
            glfw_images[i].pixels = new unsigned char
                [num_channels * resolutions[i] * resolutions[i]]; // data[i].data(); // icon_338.image_data;
            glfw_images[i].width  = resolutions[i];
            glfw_images[i].height = resolutions[i];

            stbir_resize_uint8_linear(
                icon_338.image_data, icon_338.width, icon_338.height, 0, glfw_images[i].pixels, resolutions[i],
                resolutions[i], 0, static_cast<stbir_pixel_layout>( num_channels ) );

            // fmt::print( "resized     {}\n", resolutions[i] );
        }

        glfwSetWindowIcon( glfw_window, glfw_images.size(), glfw_images.data() );
        return true;
    }

    return false;
}

} // namespace images
