#pragma once
#ifndef SPIN_WIDGET_H
#define SPIN_WIDGET_H


#include <memory>
#include <QOpenGLWidget>
#include "glm/glm.hpp"

#include <VFRendering/View.hxx>


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
    SURFACE,
    ISOSURFACE,
    ARROWS,
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
  glm::vec3 backgroundColor() const;
  void setBackgroundColor(glm::vec3 background_color);
  glm::vec3 boundingBoxColor() const;
  void setBoundingBoxColor(glm::vec3 bounding_box_color);
  bool isMiniviewEnabled() const;
  void enableMiniview(bool enabled);
  bool isCoordinateSystemEnabled() const;
  void enableCoordinateSystem(bool enabled);
  bool isBoundingBoxEnabled() const;
  void enableBoundingBox(bool enabled);
  GLSpins::WidgetLocation miniviewPosition() const;
  void setMiniviewPosition(GLSpins::WidgetLocation miniview_position);
  GLSpins::WidgetLocation coordinateSystemPosition() const;
  void setCoordinateSystemPosition(GLSpins::WidgetLocation coordinatesystem_position);
  GLSpins::VisualizationMode visualizationMode() const;
  void setVisualizationMode(GLSpins::VisualizationMode visualization_mode);
  glm::vec2 zRange() const;
  void setZRange(glm::vec2 z_range);
  float isovalue() const;
  void setIsovalue(float isovalue);
  Colormap colormap() const;
  void setColormap(Colormap colormap);
  glm::vec2 spherePointSizeRange() const;
  void setSpherePointSizeRange(glm::vec2 sphere_point_size_range);
  
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
  bool initialized;
  
  const VFRendering::Options& options() const;
  VFRendering::Options default_options;
  
  Colormap m_colormap;
    glm::vec2 m_z_range;
  
  // Visualisation
  std::shared_ptr<VFRendering::View> m_view;
};

#endif
