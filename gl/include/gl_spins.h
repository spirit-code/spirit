#ifndef GL_SPINS_H
#define GL_SPINS_H

#ifndef __gl_h_
#include <glad/glad.h>
#endif

#include "glm/glm.hpp"
#include "Camera.h"
#include "data/Spin_System.h"
#include "data/Geometry.h"


class GLSpins
{
public:
    GLSpins(std::shared_ptr<Data::Spin_System> s, int width, int height);
    ~GLSpins();

    void rotate_model(float angle);
    void camera_updated();
    void update_projection_matrix(int width, int height);
    void draw();
	void update_spin_system(std::shared_ptr<Data::Spin_System> s);

    Camera camera;

private:
	std::shared_ptr<Data::Spin_System> s;
	GLuint nos;
	glm::vec3 center;
	glm::vec3 bounds_min;
	glm::vec3 bounds_max;

    GLuint vertex_array_id;
    GLuint program_id;
    GLuint vertex_position_modelspace_id;
    GLuint vertex_normal_id;
	GLuint instance_offset_id;
	GLuint instance_direction_id;
    GLuint light_color_id;
    GLuint light_direction_cameraspace_id;
    GLuint mv_matrix_id;
    GLuint mvp_matrix_id;
    GLuint normal_mv_matrix_id;
    glm::mat4 projection_matrix;
    glm::mat4 model_matrix;
    glm::mat4 mv_matrix;
    glm::mat4 mvp_matrix;
    glm::mat3 normal_mv_matrix;
    GLuint vertex_buffer;
    GLuint normal_buffer;
	GLuint instance_offset_vbo;
	GLuint instance_direction_vbo;

    // All the triangles
    std::vector<GLfloat> vertex_data;
    std::vector<GLfloat> normal_data;

    // Light color
    GLfloat light_color[3] = {1.0f, 1.0f, 1.0f};
    // Light direction given in camera coordinates (coordinate system after applying model and view matrix)
    GLfloat light_direction_cameraspace[3] = {-0.57735027f, 0.57735027f, 0.57735027f};

};

#endif
