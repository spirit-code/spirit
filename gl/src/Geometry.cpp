#include "Geometry.h"
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

namespace Geometry
{

void Generate_Single_Triangle(std::vector<float> &vertices, std::vector<float> &normals) {
    float v[] = {-1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    float n[] = {0, 0, 1, 0, 0, 1, 0, 0, 1};

    for(int i = 0; i < 9; i++) {
        vertices.push_back(v[i]);
        normals.push_back(n[i]);
    }
}

void Generate_Cylinder(std::vector<GLfloat> & vertex_data, std::vector<GLfloat> & normal_data)
{

    // Parameters
    unsigned int resolution = 20;
    float length = 0.7;
    float radius = 0.125;
    float zTop = 0.05; // Cylinder length/2 - Cone length/2
    float zBase = zTop-length;

    auto vertex_offset = vertex_data.size();
    auto normal_offset = normal_data.size();

    if (radius <= 0) return;

    vertex_data.resize(vertex_offset+27*resolution);
    normal_data.resize(normal_offset+27*resolution);

    // Base
    for(unsigned int i = 0; i < resolution; i++)
    {
        vertex_data[vertex_offset+9*i + 3*0 + 0] = 0;
        vertex_data[vertex_offset+9*i + 3*0 + 1] = 0;
        vertex_data[vertex_offset+9*i + 3*0 + 2] = zBase;
        vertex_data[vertex_offset+9*i + 3*1 + 0] = radius*cos(3.14159*2*i/resolution);
        vertex_data[vertex_offset+9*i + 3*1 + 1] = radius*sin(3.14159*2*i/resolution);
        vertex_data[vertex_offset+9*i + 3*1 + 2] = zBase;
        vertex_data[vertex_offset+9*i + 3*2 + 0] = radius*cos(3.14159*2*((i+1)%resolution)/resolution);
        vertex_data[vertex_offset+9*i + 3*2 + 1] = radius*sin(3.14159*2*((i+1)%resolution)/resolution);
        vertex_data[vertex_offset+9*i + 3*2 + 2] = zBase;
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                normal_data[normal_offset+9*i + 3*j + k] = (k==2 ? -1 : 0);
            }
        }
    }
    vertex_offset += 9*resolution;
    normal_offset += 9*resolution;
    // Side
    for(unsigned int i = 0; i < resolution; i++)
    {
        vertex_data[vertex_offset+18*i + 3*0 + 0] = radius*cos(3.14159*2*i/resolution);
        vertex_data[vertex_offset+18*i + 3*0 + 1] = radius*sin(3.14159*2*i/resolution);
        vertex_data[vertex_offset+18*i + 3*0 + 2] = zBase;
        vertex_data[vertex_offset+18*i + 3*1 + 0] = radius*cos(3.14159*2*((i+1)%resolution)/resolution);
        vertex_data[vertex_offset+18*i + 3*1 + 1] = radius*sin(3.14159*2*((i+1)%resolution)/resolution);
        vertex_data[vertex_offset+18*i + 3*1 + 2] = zBase;
        vertex_data[vertex_offset+18*i + 3*2 + 0] = radius*cos(3.14159*2*i/resolution);
        vertex_data[vertex_offset+18*i + 3*2 + 1] = radius*sin(3.14159*2*i/resolution);
        vertex_data[vertex_offset+18*i + 3*2 + 2] = zTop;
        vertex_data[vertex_offset+18*i + 3*3 + 0] = radius*cos(3.14159*2*i/resolution);
        vertex_data[vertex_offset+18*i + 3*3 + 1] = radius*sin(3.14159*2*i/resolution);
        vertex_data[vertex_offset+18*i + 3*3 + 2] = zTop;
        vertex_data[vertex_offset+18*i + 3*4 + 0] = radius*cos(3.14159*2*((i+1)%resolution)/resolution);
        vertex_data[vertex_offset+18*i + 3*4 + 1] = radius*sin(3.14159*2*((i+1)%resolution)/resolution);
        vertex_data[vertex_offset+18*i + 3*4 + 2] = zBase;
        vertex_data[vertex_offset+18*i + 3*5 + 0] = radius*cos(3.14159*2*((i+1)%resolution)/resolution);
        vertex_data[vertex_offset+18*i + 3*5 + 1] = radius*sin(3.14159*2*((i+1)%resolution)/resolution);
        vertex_data[vertex_offset+18*i + 3*5 + 2] = zTop;
        for (int j = 0; j < 6; j++) {
            for (int k = 0; k < 3; k++) {
                normal_data[normal_offset+18*i + 3*j + k] = (k==2 ? 0 : vertex_data[vertex_offset+18*i + 3*j + k]/radius);
            }
        }
    }
}

void Generate_Cone(std::vector<GLfloat> & vertex_data, std::vector<GLfloat> & normal_data)
{
    // Parameters
    unsigned int resolution = 20;
    float length = 0.6;
    float radius = 0.25;
    float zBase = 0.05; // Cylinder length/2 - Cone length/2
    float zTop = zBase+length;
    auto vertices = std::vector<glm::vec3>(0);
    auto indices = std::vector<glm::vec3>(0);
    auto normals = std::vector<glm::vec3>(0);;

    auto vertex_offset = vertex_data.size();
    auto normal_offset = normal_data.size();
    vertex_data.resize(vertex_offset+18*resolution);
    normal_data.resize(normal_offset+18*resolution);

    float l = sqrt(radius*radius+length*length);
    float f1 = radius/l;
    float f2 = length/l;

    // Vertices and Normals
    //      Rim
    for (unsigned int i=0; i<resolution; ++i)
    {
        vertices.push_back(glm::vec3
                           (
                            radius*glm::cos(2*glm::pi<float>()/resolution*i),
                            radius*glm::sin(2*glm::pi<float>()/resolution*i),
                            zBase
                            ));
        normals.push_back(glm::vec3
                          (
                           f2*glm::cos(2*glm::pi<float>()/resolution*i),
                           f2*glm::sin(2*glm::pi<float>()/resolution*i),
                           f1
                           ));
    }
    //      Centers
    vertices.push_back(glm::vec3(0.0, 0.0, zBase));
    vertices.push_back(glm::vec3(0.0, 0.0, zTop));
    normals.push_back(glm::vec3(0.0, 0.0, -1.0));
    normals.push_back(glm::vec3(0.0, 0.0, 0.0));
    // Hat
    for(unsigned int i = 0; i < resolution; i++)
    {
        for(int j = 0; j < 2; j++) {
            for (int k=0; k<3; ++k) {
                vertex_data[vertex_offset+9*i + 3*j + k] = vertices[(i+j) % resolution][k];
                normal_data[normal_offset+9*i + 3*j + k] = normals[(i+j) % resolution][k];
            }
        }
        // auto apex_normal = glm::normalize(normals[i] + normals[(i+1) % resolution]);
        auto apex_normal = glm::vec3(0.0f, 0.0f, 0.0f);
        for (int k=0; k<3; ++k) {
            vertex_data[vertex_offset+9*i + 3*2 + k] = vertices[resolution +1][k];
            normal_data[normal_offset+9*i + 3*2 + k] = apex_normal[k];
        }
    }
    // Base
    for(unsigned int i = resolution; i < 2* resolution; i++)
    {
        for(int j = 0; j < 2; j++) {
            for (int k=0; k<3; ++k) {
                vertex_data[vertex_offset+9*i + 3*j + k] = vertices[(i+j) % resolution][k];
                normal_data[normal_offset+9*i + 3*j + k] = normals[resolution][k];
            }
        }
        for (int k=0; k<3; ++k) {
            vertex_data[vertex_offset+9*i + 3*2 + k] = vertices[resolution][k];
            normal_data[normal_offset+9*i + 3*2 + k] = normals[resolution][k];
        }
    }
}

void Generate_Arrow(std::vector<GLfloat> & vertex_data, std::vector<GLfloat> & normal_data)
{
    vertex_data = std::vector<GLfloat>(0);
    normal_data = std::vector<GLfloat>(0);
    Generate_Cone(vertex_data, normal_data);
    Generate_Cylinder(vertex_data, normal_data);
}


void Generate_Cube(std::vector<GLfloat> & vertex_data, std::vector<GLfloat> & normal_data)
{
    std::vector<unsigned int> vertex_indices = std::vector<unsigned int>(36, 0);
    std::vector<GLfloat>      vertices       = std::vector<GLfloat>(24, 0.0);

    // Each vertex index generates 3 float values (x, y, z) -> 3*36 = 108 float values
    vertex_data = std::vector<GLfloat>(108, 0.0);
    // One normal for each vertex.
    normal_data = std::vector<GLfloat>(108, 0.0);

    // Vertex indices by hand
    vertex_indices = std::vector<unsigned int> {    // Each line is one triangle
        0, 6, 4,
        0, 6, 2,
        1, 7, 5,
        1, 3, 7,
        0, 5, 4,
        0, 1, 5,
        2, 7, 3,
        2, 7, 6,
        0, 2, 1,
        1, 2, 3,
        4, 7, 6,
        4, 5, 7
    };

    GLfloat x, y, z;
    // Generate cube vertices
    for(int i = 0; i < 8; i++)
    {
        x = (i & 1) ? 1.0f : -1.0f;
        y = (i & 2) ? 1.0f : -1.0f;
        z = (i & 4) ? 1.0f : -1.0f;
        vertices[3*i + 0] = x;
        vertices[3*i + 1] = y;
        vertices[3*i + 2] = z;
    }

    // Generate cube vertex data
    for(int i = 0; i < 36; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            vertex_data[3*i + j] = vertices[3 * vertex_indices[i] + j];
        }
    }

    // One cube face consists of two triangles -> 4 vertices share the same normal vector
    for(int i = 0; i < 6; i++)
    {
        x = ((i / 2 == 0) ? 1.0f : 0.0f) * (((i % 2) == 0) ? -1.0f : 1.0f);
        y = ((i / 2 == 1) ? 1.0f : 0.0f) * (((i % 2) == 0) ? -1.0f : 1.0f);
        z = ((i / 2 == 2) ? 1.0f : 0.0f) * (((i % 2) == 0) ? -1.0f : 1.0f);
        for(int j = 0; j < 6; j++)
        {
            normal_data[18*i + 3*j + 0] = x;
            normal_data[18*i + 3*j + 1] = y;
            normal_data[18*i + 3*j + 2] = z;
        }
    }
}

}
