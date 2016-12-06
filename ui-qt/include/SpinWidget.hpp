#pragma once
#ifndef SPIN_WIDGET_H
#define SPIN_WIDGET_H


#include <memory>
#include <QOpenGLWidget>
#include "glm/glm.hpp"

#include <VFRendering/View.hxx>
#include <VFRendering/ArrowRenderer.hxx>
#include <VFRendering/BoundingBoxRenderer.hxx>
#include <VFRendering/CombinedRenderer.hxx>
#include <VFRendering/CoordinateSystemRenderer.hxx>
#include <VFRendering/IsosurfaceRenderer.hxx>
#include <VFRendering/RendererBase.hxx>
#include <VFRendering/SurfaceRenderer.hxx>
#include <VFRendering/VectorSphereRenderer.hxx>


// TODO: remove this
class GLSpins
{
public:
  enum WidgetLocation {
    BOTTOM_LEFT,
    BOTTOM_RIGHT,
    TOP_LEFT,
    TOP_RIGHT
  };
  enum VisualizationMode {
    SYSTEM,
    SPHERE
  };
};

struct State;

class SpinWidget : public QOpenGLWidget
{
  Q_OBJECT
  
public:

    enum Colormap {
        HSV,
        HSV_NO_Z,
        BLUE_RED,
        BLUE_GREEN_RED,
        BLUE_WHITE_RED,
        OTHER
    };


  SpinWidget(std::shared_ptr<State> state, QWidget *parent = 0);
  void updateData();
  void initializeGL();
  void resizeGL(int width, int height);
  void paintGL();
  float getFramesPerSecond() const;
  
  // Camera
  void setCameraToDefault();
  void setCameraToX();
  void setCameraToY();
  void setCameraToZ();
  void setCameraPositon(const glm::vec3& camera_position);
  void setCameraFocus(const glm::vec3& center_position);
  void setCameraUpVector(const glm::vec3& up_vector);
  glm::vec3 getCameraPositon();
  glm::vec3 getCameraFocus();
  glm::vec3 getCameraUpVector();
  float verticalFieldOfView() const;
  void setVerticalFieldOfView(float vertical_field_of_view);
  // Mode
  void setVisualizationMode(GLSpins::VisualizationMode visualization_mode);
  bool show_miniview, show_coordinatesystem;
  // MiniView
  void setVisualizationMiniview(bool show, std::array<float, 4> position);
  // System
  void setVisualizationSystem(bool arrows, bool boundingbox, bool surface, bool isosurface);
  bool show_arrows, show_boundingbox, show_surface, show_isosurface;
  //    Bounding Box
  bool isBoundingBoxEnabled() const;
  void enableBoundingBox(bool enabled);
  //    Arrows
  glm::vec2 zRange() const;
  void setZRange(glm::vec2 z_range);
  //    Isosurface
  float isovalue() const;
  void setIsovalue(float isovalue);
  // Sphere
  glm::vec2 spherePointSizeRange() const;
  void setSpherePointSizeRange(glm::vec2 sphere_point_size_range);
  // Coordinate System
  void setVisualizationCoordinatesystem(bool show, std::array<float, 4> position);
  bool isCoordinateSystemEnabled() const;
  void enableCoordinateSystem(bool enabled);
  GLSpins::WidgetLocation coordinateSystemPosition() const;
  void setCoordinateSystemPosition(GLSpins::WidgetLocation coordinatesystem_position);
  // Colors
  glm::vec3 backgroundColor() const;
  void setBackgroundColor(glm::vec3 background_color);
  Colormap colormap() const;
  void setColormap(Colormap colormap);
  glm::vec3 boundingBoxColor() const;
  void setBoundingBoxColor(glm::vec3 bounding_box_color);
  
protected:
  virtual void mouseMoveEvent(QMouseEvent *event);
  virtual void mousePressEvent(QMouseEvent *event);
  virtual void wheelEvent(QWheelEvent *event);
  
  protected slots:
  void teardownGL();
  
private:
  std::shared_ptr<State> state;
  QPoint m_previous_mouse_position;
  bool _reset_camera;
  
  // Renderers
  std::shared_ptr<VFRendering::RendererBase> m_mainview;
  std::shared_ptr<VFRendering::RendererBase> m_miniview;
  std::array<float, 4> m_miniview_position;
  std::shared_ptr<VFRendering::CoordinateSystemRenderer> m_coordinatecross;
  std::array<float, 4> m_coordinatecross_position;
  std::shared_ptr<VFRendering::VectorSphereRenderer> m_sphere;

  std::shared_ptr<VFRendering::CombinedRenderer> m_system;
  std::shared_ptr<VFRendering::ArrowRenderer> m_renderer_arrows;
  std::shared_ptr<VFRendering::BoundingBoxRenderer> m_renderer_boundingbox;
  std::shared_ptr<VFRendering::IsosurfaceRenderer> m_renderer_surface;
  std::shared_ptr<VFRendering::IsosurfaceRenderer> m_renderer_isosurface;

  void setupRenderers();

  const VFRendering::Options& options() const;
  
  Colormap m_colormap;
    glm::vec2 m_z_range;
  
  // Visualisation
  VFRendering::View m_view;
};

#endif
