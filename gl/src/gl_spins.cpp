#include <iostream>
#include <cmath>
#include <cassert>
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
#include "ISpinRenderer.h"
#include "ArrowSpinRenderer.h"

#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif // !M_PI


GLSpins::GLSpins(std::shared_ptr<Data::Spin_System> s, int width, int height)
{
	// Copy Positions from geometry
	this->s = s;
	std::shared_ptr<Data::Geometry> g = s->geometry;
	this->nos = g->nos;

	// Copy Center and bounds
	center = glm::vec3(g->center[0], g->center[1], g->center[2]);
	bounds_min = glm::vec3(g->bounds_min[0], g->bounds_min[1], g->bounds_min[2]);
	bounds_max = glm::vec3(g->bounds_max[0], g->bounds_max[1], g->bounds_max[2]);

	this->camera.lookAt(glm::vec3(center.x, center.y, 30.0f),
						center,
						glm::vec3(0.0f, 1.0f, 0.0f));

	



    // Initialize glad
    if (!gladLoadGL()) {
        std::cerr << "Failed to initialize glad" << std::endl;
        // return 1;
    }
  
  
  // Spin positions
  std::vector<glm::vec3> translations(nos);
  for (unsigned int i = 0; i < nos; ++i)
  {
    translations[i] = glm::vec3(g->spin_pos[0][i], g->spin_pos[1][i], g->spin_pos[2][i]);
  }
  
  // Spin orientations
  std::vector<glm::vec3> directions(nos);
  for (unsigned int i = 0; i < nos; ++i)
  {
    directions[i] = glm::vec3(s->spins[i], s->spins[g->nos + i], s->spins[2*g->nos + i]);
  }
  
  renderer = std::make_shared<ArrowSpinRenderer>();
  renderer->initGL();
  renderer->updateSpins(translations, directions);

    // Dark blue background
    //glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
	// Dark gray background
	glClearColor(0.1f, 0.1f, 0.1f, 0.0f);

    // Enable depth test
    glEnable(GL_DEPTH_TEST);
    // Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS);
}



void GLSpins::rotate_model(float angle) {
    // Rotate model around the y axis
    model_matrix = glm::rotate(glm::mat4(1.0f), angle * ((float) M_PI / 180.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    mv_matrix = camera * model_matrix;
    normal_mv_matrix = glm::inverseTranspose(glm::mat3(mv_matrix));
    mvp_matrix = projection_matrix * mv_matrix; // Remember, matrix multiplication is the other way around

}

void GLSpins::camera_updated() {
    mv_matrix = camera * model_matrix;
    normal_mv_matrix = glm::inverseTranspose(glm::mat3(mv_matrix));
    mvp_matrix = projection_matrix * mv_matrix; // Remember, matrix multiplication is the other way around

}

void GLSpins::update_projection_matrix(int width, int height) {
    // Projection matrix : 45° Field of View, width:height ratio, display range : 0.1 unit <-> 100 units
    projection_matrix = glm::perspective(45.0f, ((float) width) / height, 0.1f, 10000.0f);
    // Our ModelViewProjection : multiplication of our 3 matrices
    mvp_matrix = projection_matrix * mv_matrix; // Remember, matrix multiplication is the other way around

}

void GLSpins::draw() {
  // Clear the screen and the depth buffer
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  std::shared_ptr<Data::Geometry> g = s->geometry;
  // Spin positions
  std::vector<glm::vec3> translations(nos);
  for (unsigned int i = 0; i < nos; ++i)
  {
    translations[i] = glm::vec3(g->spin_pos[0][i], g->spin_pos[1][i], g->spin_pos[2][i]);
  }
  
  // Spin orientations
  std::vector<glm::vec3> directions(nos);
  for (unsigned int i = 0; i < nos; ++i)
  {
    directions[i] = glm::vec3(s->spins[i], s->spins[g->nos + i], s->spins[2*g->nos + i]);
  }
  // TODO: aspect ratio
  renderer->updateSpins(translations, directions);
  renderer->draw(1.0);
  assert(!glGetError());
}

void GLSpins::update_spin_system(std::shared_ptr<Data::Spin_System> s)
{
	this->s = s;
}

GLSpins::~GLSpins() {
}
