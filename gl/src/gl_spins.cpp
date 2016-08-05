#include <iostream>
#include <cmath>
#include <memory>

// Include GLM
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/matrix_inverse.hpp"
#include "glm/gtx/string_cast.hpp"

#include "Camera.h"
#include "Geometry.h"
#include "load_shader.h"
#include "gl_spins.h"

#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif // !M_PI


GLSpins::GLSpins(std::shared_ptr<Data::Spin_System> s, int width, int height)
{
	// Copy Positions from geometry
	this->s = s;
	std::shared_ptr<Data::Geometry> g = s->geometry;
	this->nos = g->nos;

	// Spin positions
	std::vector<glm::vec3> translations(nos);
	for (unsigned int i = 0; i < nos; ++i)
	{
		translations[i] = glm::vec3(g->spin_pos[0][i], g->spin_pos[1][i], g->spin_pos[2][i]);
	}

	// Spin orientations
	std::vector<glm::vec3> directions(nos);
    auto spins = *s->spins;
	for (unsigned int i = 0; i < nos; ++i)
	{
		directions[i] = glm::vec3(spins[i], spins[g->nos + i], spins[2*g->nos + i]);
	}

	// Copy Center and bounds
	center = glm::vec3(g->center[0], g->center[1], g->center[2]);
	bounds_min = glm::vec3(g->bounds_min[0], g->bounds_min[1], g->bounds_min[2]);
	bounds_max = glm::vec3(g->bounds_max[0], g->bounds_max[1], g->bounds_max[2]);

	this->camera.lookAt(glm::vec3(center.x, center.y, 30.0f),
						center,
						glm::vec3(0.0f, 1.0f, 0.0f));

	/*this->camera.lookAt(glm::vec3(00.0f, 00.0f, 20.0f),
						glm::vec3(0.0f, 0.0f, 0.0f),
						glm::vec3(0.0f, 1.0f, 0.0f));*/
	
	// Initialize buffer data
    //init_data();
    //Geometry::Generate_Cube(vertex_data, normal_data);
	//Geometry::Generate_Cylinder(vertex_data, normal_data);
    //Geometry::Generate_Cone(vertex_data, normal_data);
	Geometry::Generate_Arrow(vertex_data, normal_data);

	



    // Initialize glad
    if (!gladLoadGL()) {
        std::cerr << "Failed to initialize glad" << std::endl;
        // return 1;
    }

    // Dark blue background
    //glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
	// Dark gray background
	glClearColor(0.1f, 0.1f, 0.1f, 0.0f);

    // Enable depth test
    glEnable(GL_DEPTH_TEST);
    // Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS);

    // Create and compile our GLSL program from the shaders
	const char* vertex_source_ptr =
	#include "vertex.txt"
	;
	const char* fragment_source_ptr =
	#include "fragment.txt"
	;
    program_id = load_program(
        vertex_source_ptr,
		fragment_source_ptr
    );
    if(!program_id) {
        std::cerr << "GLSL program compilation/linking failed" << std::endl;
		std::cerr << "This may just mean the files were not found" << std::endl;
        // return 2;
    }

    // Use our shader
    glUseProgram(program_id);

    // Get handles for the vertex shader input variables
	vertex_position_modelspace_id = glGetAttribLocation(program_id, "vertex_position_modelspace");
	vertex_normal_id = glGetAttribLocation(program_id, "vertex_normal");
	instance_offset_id = glGetAttribLocation(program_id, "instance_offset");
	instance_direction_id = glGetAttribLocation(program_id, "instance_direction");

    // Handle for light color
    light_color_id = glGetUniformLocation(program_id, "light_color");
    glUniform3fv(light_color_id, 1, light_color);

    // Handle for light direction
    light_direction_cameraspace_id = glGetUniformLocation(program_id, "light_direction_cameraspace");
    glUniform3fv(light_direction_cameraspace_id, 1, light_direction_cameraspace);

    // Get a handle for our "mv", "mvp" and "normal_mv" uniform (modelview, modelviewprojection and normal_modelview matrices)
    mv_matrix_id = glGetUniformLocation(program_id, "mv_matrix");
    mvp_matrix_id = glGetUniformLocation(program_id, "mvp_matrix");
    normal_mv_matrix_id = glGetUniformLocation(program_id, "normal_mv_matrix");     // normal vectors need to be transformed with an extra matrix

    // Model matrix : an identity matrix (model will be at the origin)
    model_matrix = glm::mat4(1.0f);
    // Product of camera and model matrix
    mv_matrix = camera * model_matrix;
    glUniformMatrix4fv(mv_matrix_id, 1, GL_FALSE, &mv_matrix[0][0]);
    // Normal matrix
    normal_mv_matrix = glm::inverseTranspose(glm::mat3(mv_matrix));
    glUniformMatrix3fv(normal_mv_matrix_id, 1, GL_FALSE, &normal_mv_matrix[0][0]);
    // Projection matrix (function updates mvp matrix as well)
    update_projection_matrix(width, height);

    // Transfer vertex data to the GPU memory
    glGenBuffers(1, &vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, vertex_data.size()*sizeof(GLfloat), vertex_data.data(), GL_STATIC_DRAW);

    // Transfer normal data to the GPU memory
    glGenBuffers(1, &normal_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, normal_buffer);
    glBufferData(GL_ARRAY_BUFFER, normal_data.size()*sizeof(GLfloat), normal_data.data(), GL_STATIC_DRAW);


	glGenBuffers(1, &instance_offset_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, instance_offset_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * nos, &translations[0], GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &instance_direction_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, instance_direction_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * nos, &directions[0], GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Modern OpenGL requires a vertex array; it can be used for fast state restore operations
    glGenVertexArrays(1, &vertex_array_id);
    glBindVertexArray(vertex_array_id);     // The following calls are recorded for a fast restore in draw calls

    // Enable input variables of the vertex shader stage
    // 1rst attribute buffer : vertices
    glEnableVertexAttribArray(vertex_position_modelspace_id);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glVertexAttribPointer(
        vertex_position_modelspace_id,  // attribute
        3,                              // size
        GL_FLOAT,                       // type
        GL_FALSE,                       // normalized?
        0,                              // stride
        (void*) 0                       // array buffer offset
    );

    // 2nd attribute buffer : colors
    glEnableVertexAttribArray(vertex_normal_id);
    glBindBuffer(GL_ARRAY_BUFFER, normal_buffer);
    glVertexAttribPointer(
        vertex_normal_id,    // attribute
        3,                  // size
        GL_FLOAT,           // type
        GL_FALSE,           // normalized?
        0,                  // stride
        (void*) 0           // array buffer offset
    );

	glEnableVertexAttribArray(instance_offset_id);
	glBindBuffer(GL_ARRAY_BUFFER, instance_offset_vbo);
	glVertexAttribPointer(
		instance_offset_id,
		3,
		GL_FLOAT,
		GL_FALSE,
		3 * sizeof(GLfloat),
		(GLvoid*)0
	);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glVertexAttribDivisor(instance_offset_id, 1);

	glEnableVertexAttribArray(instance_direction_id);
	glBindBuffer(GL_ARRAY_BUFFER, instance_direction_vbo);
	glVertexAttribPointer(
		instance_direction_id,
		3,
		GL_FLOAT,
		GL_FALSE,
		3 * sizeof(GLfloat),
		(GLvoid*)0
	);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glVertexAttribDivisor(instance_direction_id, 1);

    glBindVertexArray(0);   // Save point; everything between this and the previous glBindVertexArray call is saved as a state
}



void GLSpins::rotate_model(float angle) {
    // Rotate model around the y axis
    model_matrix = glm::rotate(glm::mat4(1.0f), angle * ((float) M_PI / 180.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    mv_matrix = camera * model_matrix;
    normal_mv_matrix = glm::inverseTranspose(glm::mat3(mv_matrix));
    mvp_matrix = projection_matrix * mv_matrix; // Remember, matrix multiplication is the other way around

    // Use our shader
    glUseProgram(program_id);

    glUniformMatrix4fv(mv_matrix_id, 1, GL_FALSE, &mv_matrix[0][0]);
    glUniformMatrix3fv(normal_mv_matrix_id, 1, GL_FALSE, &normal_mv_matrix[0][0]);
    glUniformMatrix4fv(mvp_matrix_id, 1, GL_FALSE, &mvp_matrix[0][0]);
}

void GLSpins::camera_updated() {
    mv_matrix = camera * model_matrix;
    normal_mv_matrix = glm::inverseTranspose(glm::mat3(mv_matrix));
    mvp_matrix = projection_matrix * mv_matrix; // Remember, matrix multiplication is the other way around

    // Use our shader
    glUseProgram(program_id);

    glUniformMatrix4fv(mv_matrix_id, 1, GL_FALSE, &mv_matrix[0][0]);
    glUniformMatrix3fv(normal_mv_matrix_id, 1, GL_FALSE, &normal_mv_matrix[0][0]);
    glUniformMatrix4fv(mvp_matrix_id, 1, GL_FALSE, &mvp_matrix[0][0]);
}

void GLSpins::update_projection_matrix(int width, int height) {
    // Projection matrix : 45° Field of View, width:height ratio, display range : 0.1 unit <-> 100 units
    projection_matrix = glm::perspective(45.0f, ((float) width) / height, 0.1f, 10000.0f);
    // Our ModelViewProjection : multiplication of our 3 matrices
    mvp_matrix = projection_matrix * mv_matrix; // Remember, matrix multiplication is the other way around

    // Use our shader
    glUseProgram(program_id);

    // Send our transformation to the currently bound shader,
    // in the "mvp_matrix" uniform
    glUniformMatrix4fv(mvp_matrix_id, 1, GL_FALSE, &mvp_matrix[0][0]);
}

void GLSpins::draw() {
    // Clear the screen and the depth buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Use our shader
    glUseProgram(program_id);

    // Restore the saved state (see init)
    glBindVertexArray(vertex_array_id);

	// Update directions
	std::vector<glm::vec3> directions(nos);
    auto spins = *s->spins;
	for (unsigned int i = 0; i < nos; ++i)
	{
		directions[i] = glm::vec3(spins[i], spins[nos + i], spins[2 * nos + i]);
	}
	glBindBuffer(GL_ARRAY_BUFFER, instance_direction_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * nos, &directions[0], GL_STATIC_DRAW);

    // Draw the triangles
    //glDrawArrays(GL_TRIANGLES, 0, this->vertex_data.size() / 3); // 12*3 indices starting at 0 -> 12 triangles
	glDrawArraysInstanced(GL_TRIANGLES, 0, this->vertex_data.size() / 3, nos);

    glBindVertexArray(0);
}

void GLSpins::update_spin_system(std::shared_ptr<Data::Spin_System> s)
{
	this->s = s;
}

GLSpins::~GLSpins() {
    // Cleanup VBO and shader
    glDeleteBuffers(1, &vertex_buffer);
    glDeleteBuffers(1, &normal_buffer);
	glDeleteBuffers(1, &instance_offset_vbo);
	glDeleteBuffers(1, &instance_direction_vbo);
    glDeleteProgram(program_id);
    glDeleteVertexArrays(1, &vertex_array_id);
}
