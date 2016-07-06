#ifndef VIS_GEOMETRY_H
#define VIS_GEOMETRY_H

#include <vector>
#include <glad/glad.h>

namespace Geometry
{

	void Generate_Single_Triangle(std::vector<float> &vertices, std::vector<float> &normals);
	void Generate_Cube(std::vector<GLfloat> & vertex_data, std::vector<GLfloat> & normal_data);
	void Generate_Cone(std::vector<GLfloat> & vertex_data, std::vector<GLfloat> & normal_data);
	void Generate_Cylinder(std::vector<GLfloat> & vertex_data, std::vector<GLfloat> & normal_data);
	void Generate_Arrow(std::vector<GLfloat> & vertex_data, std::vector<GLfloat> & normal_data);

}
#endif
