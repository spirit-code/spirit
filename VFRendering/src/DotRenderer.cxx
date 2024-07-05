#include "VFRendering/DotRenderer.hxx"

#ifndef __EMSCRIPTEN__
#include <glad/glad.h>
#else
#include <GLES3/gl3.h>
#endif

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "VFRendering/Utilities.hxx"

#include "shaders/dots.vert.glsl.hxx"
#include "shaders/dots_circle.frag.glsl.hxx"
#include "shaders/dots_square.frag.glsl.hxx"

namespace VFRendering {

DotRenderer::DotRenderer( const View& view, const VectorField& vf ) :
    VectorFieldRenderer( view, vf ) { 
    }

void DotRenderer::initialize() 
{
    if ( m_is_initialized ) return;
   
    m_is_initialized = true;

    // VAO
    glGenVertexArrays(1, &m_vao);
    glBindVertexArray(m_vao);
   
    // Instance positions VBO
    glGenBuffers(1, &m_instance_position_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_instance_position_vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, nullptr);
    glEnableVertexAttribArray(0);
  
    // Instance direction VBO
    glGenBuffers(1, &m_instance_direction_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_instance_direction_vbo);
    glVertexAttribPointer(1, 3, GL_FLOAT, false, 0, nullptr);
    glEnableVertexAttribArray(1);
    
    m_num_instances = 0;
    updateShaderProgram(); 

    update( false );
}

DotRenderer::~DotRenderer() 
{
    if ( !m_is_initialized ) return;

    glDeleteVertexArrays(1, &m_vao);
    glDeleteBuffers(1, &m_instance_position_vbo);
    glDeleteBuffers(1, &m_instance_direction_vbo);
    glDeleteProgram(m_program);
}

void DotRenderer::optionsHaveChanged( const std::vector<int>& changed_options )
{
    if ( !m_is_initialized ) return;
    
    bool update_shader = false;
    for (auto option_index : changed_options) {
        switch (option_index) {
        case View::Option::COLORMAP_IMPLEMENTATION:
        case View::Option::IS_VISIBLE_IMPLEMENTATION:
        case DotRenderer::Option::DOT_STYLE: 
            update_shader = true;
            break;
        }
    }
    if (update_shader) {
        updateShaderProgram();
    }
}

void DotRenderer::update( bool keep_geometry ) 
{
    if (!m_is_initialized) {
        return;
    }
    glBindVertexArray( m_vao );
   
    // If geometry is changed copy the new data into position's VBO
    if ( !keep_geometry ) {
        glBindBuffer( GL_ARRAY_BUFFER, m_instance_position_vbo );
        glBufferData( GL_ARRAY_BUFFER, sizeof(glm::vec3) * positions().size(), 
            positions().data(), GL_STREAM_DRAW );
    }
   
    // New data into direction's VBO
    glBindBuffer( GL_ARRAY_BUFFER, m_instance_direction_vbo );
    glBufferData( GL_ARRAY_BUFFER, sizeof(glm::vec3) * directions().size(), 
        directions().data(), GL_STREAM_DRAW );

    m_num_instances = std::min( positions().size(), directions().size() );
}

void DotRenderer::draw( float aspect_ratio )
{
    initialize();
    if ( m_num_instances <= 0 ) return;
    
    glBindVertexArray( m_vao );
    glUseProgram( m_program );

    auto matrices = Utilities::getMatrices( options(), aspect_ratio );
    auto model_view_matrix = matrices.first;
    auto projection_matrix = matrices.second;
    
    float dot_radius = options().get<DotRenderer::Option::DOT_RADIUS>();
    glm::vec2 frame_size = m_view.getFramebufferSize();
    dot_radius *= std::min( frame_size[0], frame_size[1] ) / 1000;

    // Set shader's uniforms
    glUniformMatrix4fv( glGetUniformLocation( m_program, "uProjectionMatrix" ), 
        1, false, glm::value_ptr(projection_matrix));
    glUniformMatrix4fv( glGetUniformLocation( m_program, "uModelviewMatrix" ), 
        1, false, glm::value_ptr(model_view_matrix));
    glUniform1f( glGetUniformLocation( m_program, "uDotRadius" ), dot_radius);

    glDisable( GL_CULL_FACE );
#ifndef __EMSCRIPTEN__
    glEnable( GL_PROGRAM_POINT_SIZE );
#endif
    glDrawArrays( GL_POINTS, 0, m_num_instances ); 
    glEnable( GL_CULL_FACE );
}

void DotRenderer::updateShaderProgram()
{
    if ( !m_is_initialized ) return;

    if ( m_program ) glDeleteProgram( m_program );

    // Vertex shader options
    std::string vertex_shader_source = DOT_VERT_GLSL; 
    vertex_shader_source += 
        options().get<View::Option::COLORMAP_IMPLEMENTATION>();
    vertex_shader_source +=
        options().get<View::Option::IS_VISIBLE_IMPLEMENTATION>();

    // Fragment shader options 
    std::string fragment_shader_source = getDotStyle(options().get<DotRenderer::Option::DOT_STYLE>());
    
    // Compile & link shader Program. Pass uniforms. 
    m_program = Utilities::createProgram( vertex_shader_source, 
        fragment_shader_source, { "ivDotCoordinates", "ivDotDirection" } );
}

std::string DotRenderer::getDotStyle(const DotStyle& dotstyle)
{
    switch(dotstyle) {
    case DotStyle::CIRCLE:
        return DOT_CIRCLE_FRAG_GLSL;
    case DotStyle::SQUARE:
        return DOT_SQUARE_FRAG_GLSL;
    default:
        return DOT_CIRCLE_FRAG_GLSL;
    }
}

} // namespace VFRendering
