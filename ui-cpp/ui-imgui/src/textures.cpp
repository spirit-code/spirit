
#include <textures.hpp>

#include <cmrc/cmrc.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#include <glad/glad.h>

CMRC_DECLARE( resources );

namespace textures
{

bool load_from_file( const char * filename, unsigned int * out_texture, int * out_width, int * out_height )
{
    // Load from file
    int image_width  = 0;
    int image_height = 0;

    auto fs          = cmrc::resources::get_filesystem();
    auto file        = fs.open( filename );
    std::string data = std::string( file.cbegin(), file.cend() );

    unsigned char * image_data = stbi_load_from_memory(
        reinterpret_cast<const unsigned char *>( data.c_str() ), data.size() * sizeof( char ) / sizeof( unsigned char ),
        &image_width, &image_height, NULL, 4 );

    if( image_data == NULL )
        return false;

    // Create a OpenGL texture identifier
    GLuint image_texture;
    glGenTextures( 1, &image_texture );
    glBindTexture( GL_TEXTURE_2D, image_texture );

    // Setup filtering parameters for display
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

    // Upload pixels into texture
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data );
    stbi_image_free( image_data );

    *out_texture = image_texture;
    *out_width   = image_width;
    *out_height  = image_height;

    return true;
}

} // namespace textures