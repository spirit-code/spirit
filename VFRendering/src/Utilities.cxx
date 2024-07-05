#include "VFRendering/Utilities.hxx"

#include <iostream>
#ifndef __EMSCRIPTEN__
#include <glad/glad.h>
#else
#include <GLES3/gl3.h>
#endif

#include <glm/gtc/matrix_transform.hpp>

#include "VFRendering/View.hxx"

#include "shaders/colormap.hsv.glsl.hxx"
#include "shaders/colormap.bluered.glsl.hxx"
#include "shaders/colormap.bluegreenred.glsl.hxx"
#include "shaders/colormap.bluewhitered.glsl.hxx"
#include "shaders/colormap.black.glsl.hxx"
#include "shaders/colormap.white.glsl.hxx"

namespace VFRendering {
namespace Utilities {
static GLuint createShader(GLenum shader_type, const std::string& shader_source) {
    const char* shader_source_c_str = shader_source.c_str();
    GLuint shader = glCreateShader(shader_type);
    glShaderSource(shader, 1, &shader_source_c_str, nullptr);
    glCompileShader(shader);

    GLint status = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (!status) {
        std::string message = "shader failed to compile!";
        message += "\nshader source:\n";
        message += shader_source;

        GLsizei length = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        if (length > 0) {
            char* info_log = new char[length];
            glGetShaderInfoLog(shader, length, nullptr, info_log);
            message += "\nshader info log:\n";
            message += info_log;
            delete[] info_log;
        }
#ifdef __EMSCRIPTEN__
        std::cerr << message << std::endl;
#endif
        throw OpenGLException(message);
    }
    return shader;
}

OpenGLException::OpenGLException(const std::string& message) : std::runtime_error(message) {}

unsigned int createProgram(const std::string& vertex_shader_source, const std::string& fragment_shader_source, const std::vector<std::string>& attributes) {
    GLuint vertex_shader = createShader(GL_VERTEX_SHADER, vertex_shader_source);
    GLuint fragment_shader = createShader(GL_FRAGMENT_SHADER, fragment_shader_source);

    GLuint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    for (std::vector<std::string>::size_type i = 0; i < attributes.size(); i++) {
        glBindAttribLocation(program, i, attributes[i].c_str());
    }
#ifndef EMSCRIPTEN
    glBindFragDataLocation(program, 0, "fo_FragColor");
#endif
    glLinkProgram(program);
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    GLint status = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (!status) {
        std::string message = "program failed to link!";
        message += "\nvertex nshader source:\n";
        message += vertex_shader_source;
        message += "\nfragment nshader source:\n";
        message += fragment_shader_source;

        GLsizei length = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
        if (length > 0) {
            char* info_log = new char[length];
            glGetProgramInfoLog(program, length, nullptr, info_log);
            message += "\nprogram info log:\n";
            message += info_log;
            delete[] info_log;
        }
#ifdef __EMSCRIPTEN__
        std::cerr << message << std::endl;
#endif
        throw OpenGLException(message);
    }
    return program;
}

std::string getColormapImplementation(const Colormap& colormap) {
    switch (colormap) {
    case Colormap::BLUERED:
        return COLORMAP_BLUERED_GLSL;
    case Colormap::BLUEGREENRED:
        return COLORMAP_BLUEGREENRED_GLSL;
    case Colormap::BLUEWHITERED:
        return COLORMAP_BLUEWHITERED_GLSL;
    case Colormap::HSV:
        return COLORMAP_HSV_GLSL;
    case Colormap::BLACK:
        return COLORMAP_BLACK_GLSL;
    case Colormap::WHITE:
        return COLORMAP_WHITE_GLSL;
    case Colormap::DEFAULT:
    default:
        return "vec3 colormap(vec3 direction) {return vec3(1.0, 1.0, 1.0);}";
    }
}

std::pair<glm::mat4, glm::mat4> getMatrices(const VFRendering::Options& options, float aspect_ratio) {
    auto vertical_field_of_view = options.get<View::Option::VERTICAL_FIELD_OF_VIEW>();
    auto camera_position = options.get<View::Option::CAMERA_POSITION>();
    auto center_position = options.get<View::Option::CENTER_POSITION>();
    auto up_vector = options.get<View::Option::UP_VECTOR>();

    glm::mat4 projection_matrix;
    if (vertical_field_of_view > 0) {
        projection_matrix = glm::perspective(glm::radians(vertical_field_of_view), aspect_ratio, 0.1f, 10000.0f);
        if (aspect_ratio < 1) {
            projection_matrix[0][0] *= aspect_ratio;
            projection_matrix[1][1] *= aspect_ratio;
        }
    } else {
        float camera_distance = glm::length(camera_position - center_position);
        float left_right = camera_distance * aspect_ratio;
        float bottom_top = camera_distance;
        projection_matrix = glm::ortho(-left_right, left_right, -bottom_top, bottom_top, -10000.0f, 10000.0f);
    }
    glm::mat4 model_view_matrix = glm::lookAt(camera_position, center_position, up_vector);
    return {
               model_view_matrix, projection_matrix
    };
}
}
}
