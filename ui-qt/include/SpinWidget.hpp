#pragma once
#ifndef SPIN_WIDGET_H
#define SPIN_WIDGET_H


#include <memory>
#include <QOpenGLWidget>
#include "glm/glm.hpp"

#include "GLSpins.hpp"

struct State;

class SpinWidget : public QOpenGLWidget
{
  Q_OBJECT
  
public:
  SpinWidget(std::shared_ptr<State> state, QWidget *parent = 0);
  void initializeGL();
  void resizeGL(int width, int height);
  void paintGL();
  double getFramesPerSecond() const;
  
  void setCameraToDefault();
  void setCameraToX();
  void setCameraToY();
  void setCameraToZ();
  double verticalFieldOfView() const;
  void setVerticalFieldOfView(double vertical_field_of_view);
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
  GLSpins::Colormap colormap() const;
  void setColormap(GLSpins::Colormap colormap);
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
  QPoint _previous_pos;
  bool _reset_camera;
  
  const Options<GLSpins>& options() const;
  const Options<GLSpins> default_options;
  
  // Visualisation
  std::shared_ptr<GLSpins> gl_spins;
};

#endif
