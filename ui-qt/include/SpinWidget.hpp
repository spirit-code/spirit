#pragma once
#ifndef SPIN_WIDGET_H
#define SPIN_WIDGET_H

#include <memory>
#include <set>

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


struct State;

class SpinWidget : public QOpenGLWidget
{
  Q_OBJECT
  
public:

    enum class Colormap {
        HSV,
        HSV_NO_Z,
        BLUE_WHITE_RED,
        BLUE_GREEN_RED,
        BLUE_RED,
        WHITE,
        GRAY,
        BLACK,
        OTHER
    };

	enum class Color {
		BLACK,
		GRAY,
		WHITE,
		OTHER
	};

	enum class WidgetLocation {
		BOTTOM_LEFT,
		BOTTOM_RIGHT,
		TOP_LEFT,
		TOP_RIGHT
	};

	enum class VisualizationMode {
		SYSTEM,
		SPHERE
	};

	enum class VisualizationSource {
		SPINS,
		EFF_FIELD
	};


  SpinWidget(std::shared_ptr<State> state, QWidget *parent = 0);
  void updateData();
  void initializeGL();
  void resizeGL(int width, int height);
  void paintGL();
  void screenShot(std::string filename);
  float getFramesPerSecond() const;
  
  void setVisualisationSource(int source);
  int m_source;

  const VFRendering::View * view();

  void addIsosurface(std::shared_ptr<VFRendering::IsosurfaceRenderer> renderer);
  void removeIsosurface(std::shared_ptr<VFRendering::IsosurfaceRenderer>);

  // --- Mode
  void setVisualizationMode(SpinWidget::VisualizationMode visualization_mode);
  SpinWidget::VisualizationMode visualizationMode();
  SpinWidget::VisualizationMode visMode;
  bool show_miniview, show_coordinatesystem;
  // --- MiniView
  void setVisualizationMiniview(bool show, SpinWidget::WidgetLocation location);
  bool isMiniviewEnabled() const;
  void enableMiniview(bool enabled);
  WidgetLocation miniviewPosition() const;
  void setMiniviewPosition(WidgetLocation position);
  // --- Coordinate System
  void setVisualizationCoordinatesystem(bool show, SpinWidget::WidgetLocation location);
  bool isCoordinateSystemEnabled() const;
  void enableCoordinateSystem(bool enabled);
  WidgetLocation coordinateSystemPosition() const;
  void setCoordinateSystemPosition(WidgetLocation position);

  // --- System
  void enableSystem(bool arrows, bool boundingbox, bool surface, bool isosurface);
  void cycleSystem(bool forward=true);
  void moveSlab(int amount);
  bool show_arrows, show_boundingbox, show_surface, show_isosurface;
  //    Arrows
  void setArrows(float size=1, int lod=20);
  float arrowSize() const;
  int arrowLOD() const;
  glm::vec2 xRangeDirection() const;
  glm::vec2 yRangeDirection() const;
  glm::vec2 zRangeDirection() const;
  void setOverallDirectionRange(glm::vec2 x_range, glm::vec2 y_range, glm::vec2 z_range);
  glm::vec2 xRangePosition() const;
  glm::vec2 yRangePosition() const;
  glm::vec2 zRangePosition() const;
  void setOverallPositionRange(glm::vec2 x_range, glm::vec2 y_range, glm::vec2 z_range);
  //    Bounding Box
  bool isBoundingBoxEnabled() const;
  void enableBoundingBox(bool enabled);
  //    Surface
  void setSurface(glm::vec2 x_range, glm::vec2 y_range, glm::vec2 z_range);
  glm::vec2 surfaceXRange() const;
  glm::vec2 surfaceYRange() const;
  glm::vec2 surfaceZRange() const;

  // --- Sphere
  glm::vec2 spherePointSizeRange() const;
  void setSpherePointSizeRange(glm::vec2 sphere_point_size_range);

  // --- Colors
  Colormap colormap() const;
  void setColormap(Colormap colormap);
  float colormap_rotation();
  std::array<bool, 2> colormap_inverted();
  void setColormapRotationInverted(int phi=0, bool invert_z=false, bool invert_xy=false);
  Color backgroundColor() const;
  void setBackgroundColor(Color background_color);
  Color boundingBoxColor() const;
  void setBoundingBoxColor(Color bounding_box_color);
  void updateBoundingBoxIndicators();

  // --- Camera
  void cycleCamera();
  void setCameraToDefault();
  void setCameraToX(bool inverted=false);
  void setCameraToY(bool inverted=false);
  void setCameraToZ(bool inverted=false);
  void setCameraPositon(const glm::vec3& camera_position);
  void setCameraFocus(const glm::vec3& center_position);
  void setCameraUpVector(const glm::vec3& up_vector);
  glm::vec3 getCameraPositon();
  glm::vec3 getCameraFocus();
  glm::vec3 getCameraUpVector();
  float verticalFieldOfView() const;
  void setVerticalFieldOfView(float vertical_field_of_view);
  // --- Move Camera
  void moveCamera(float backforth, float rightleft, float updown);
  void rotateCamera(float theta, float phi);
  bool getCameraRotationType();
  void setCameraRotationType(bool free);

  // --- Light
  void setLightPosition(float theta, float phi);
  std::array<float,2> getLightPosition();
  
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
  bool m_camera_rotate_free;
  float m_light_theta, m_light_phi;
  
  // temporaries for system cycle
  void setSystemCycle(int idx);
  void setSlabRanges();
  int idx_cycle;
  bool user_show_arrows, user_show_boundingbox, user_show_surface, user_show_isosurface;
  float user_fov;
  glm::vec3 slab_displacements;

  // Renderers
  std::shared_ptr<VFRendering::RendererBase> m_mainview;
  std::shared_ptr<VFRendering::RendererBase> m_miniview;
  WidgetLocation m_location_miniview;
  std::shared_ptr<VFRendering::CoordinateSystemRenderer> m_coordinatesystem;
  WidgetLocation m_location_coordinatesystem;
  std::shared_ptr<VFRendering::VectorSphereRenderer> m_sphere;

  std::shared_ptr<VFRendering::CombinedRenderer> m_system;
  std::shared_ptr<VFRendering::ArrowRenderer> m_renderer_arrows;
  std::shared_ptr<VFRendering::BoundingBoxRenderer> m_renderer_boundingbox;
  std::shared_ptr<VFRendering::RendererBase> m_renderer_surface;
  std::shared_ptr<VFRendering::IsosurfaceRenderer> m_renderer_surface_3D;
  std::shared_ptr<VFRendering::SurfaceRenderer> m_renderer_surface_2D;
  std::set<std::shared_ptr<VFRendering::IsosurfaceRenderer>> m_renderers_isosurface;

  void setupRenderers();
  bool m_gl_initialized;

  const VFRendering::Options& options() const;
  
  // Parameters
  Colormap m_colormap;
  int m_colormap_rotation;
  bool m_colormap_invert_z;
  bool m_colormap_invert_xy;
  glm::vec2 m_x_range_direction;
  glm::vec2 m_y_range_direction;
  glm::vec2 m_z_range_direction;
  glm::vec2 m_x_range_position;
  glm::vec2 m_y_range_position;
  glm::vec2 m_z_range_position;
  glm::vec2 m_surface_x_range;
  glm::vec2 m_surface_y_range;
  glm::vec2 m_surface_z_range;
  
  // Visualisation
  VFRendering::View m_view;
  
	// Persistent Settings
	void writeSettings();
	void readSettings();

protected:
	void closeEvent(QCloseEvent *event);
};

#endif
