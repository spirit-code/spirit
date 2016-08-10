#include "SpinWidget.h"

#include <QTimer>
#include <QMouseEvent>
#include "Interface_Geometry.h"
#include "Interface_State.h"
#include "ISpinRenderer.h"
#include "BoundingBoxRenderer.h"
#include "SphereSpinRenderer.h"
#include "utilities.h"

SpinWidget::SpinWidget(std::shared_ptr<State> state, QWidget *parent) : QOpenGLWidget(parent)
{
	this->state = state;

	setFocusPolicy(Qt::StrongFocus);
}

void SpinWidget::initializeGL() {
	makeCurrent();
	this->gl_spins = std::make_shared<GLSpins>();
  _reset_camera = true;
}

void SpinWidget::teardownGL() {
	// GLSpins::terminate();
}

void SpinWidget::resizeGL(int width, int height) {
  gl_spins->setFramebufferSize(width*devicePixelRatio(), height*devicePixelRatio());
  update();
}

void SpinWidget::paintGL() {
  // Update the pointer to our Data
  auto s = state->active_image;
  auto& spins = *s->spins;
  
  std::vector<glm::vec3> positions(s->geometry->nos);
  for (int i = 0; i < s->geometry->nos; ++i)
  {
    positions[i] = glm::vec3(s->geometry->spin_pos[0][i], s->geometry->spin_pos[1][i], s->geometry->spin_pos[2][i]);
  }
  std::vector<glm::vec3> directions(s->geometry->nos);
  for (int i = 0; i < s->geometry->nos; ++i)
  {
    directions[i] = glm::vec3(spins[i], spins[s->geometry->nos + i], spins[2*s->geometry->nos + i]);
  }

  gl_spins->updateSpins(positions, directions);

  glm::vec3 bounds_min;
  glm::vec3 bounds_max;
  Geometry_Get_Bounds(state.get(), &bounds_min.x, &bounds_min.y, &bounds_min.z, &bounds_max.x, &bounds_max.y, &bounds_max.z);
  glm::vec3 center = (bounds_min+bounds_max) * 0.5f;

  gl_spins->updateSystemGeometry(bounds_min, center, bounds_max);
  if (_reset_camera) {
    gl_spins->setCameraToDefault();
    _reset_camera = false;
  }
  gl_spins->draw();
  QTimer::singleShot(1, this, SLOT(update()));
}

void SpinWidget::mousePressEvent(QMouseEvent *event) {
  _previous_pos = event->pos();
}

void SpinWidget::mouseMoveEvent(QMouseEvent *event) {
  auto current_pos = event->pos();
  auto position_before = glm::vec2(_previous_pos.x(), _previous_pos.y()) * (float)devicePixelRatio();
  auto position_after = glm::vec2(current_pos.x(), current_pos.y()) * (float)devicePixelRatio();
  GLSpins::CameraMovementModes mode = GLSpins::CameraMovementModes::ROTATE;
  if ((event->modifiers() & Qt::AltModifier) == Qt::AltModifier) {
    mode = GLSpins::CameraMovementModes::TRANSLATE;
  }
  gl_spins->mouseMove(position_before, position_after, mode);
  _previous_pos = current_pos;
}

void SpinWidget::setCameraToDefault() {
  gl_spins->setCameraToDefault();
}

void SpinWidget::setCameraToX() {
  gl_spins->setCameraToX();
}

void SpinWidget::setCameraToY() {
  gl_spins->setCameraToY();
}

void SpinWidget::setCameraToZ() {
  gl_spins->setCameraToZ();
}

double SpinWidget::getFramesPerSecond() const {
  return gl_spins->getFramerate();
}

void SpinWidget::wheelEvent(QWheelEvent *event) {
  double wheel_delta = event->delta();
  gl_spins->mouseScroll(wheel_delta*0.1);
}

const Options<GLSpins>& SpinWidget::options() const {
  if (gl_spins) {
    return gl_spins->options();
  }
  return default_options;
  
}

double SpinWidget::verticalFieldOfView() const {
  return options().get<ISpinRenderer::Option::VERTICAL_FIELD_OF_VIEW>();
}

void SpinWidget::setVerticalFieldOfView(double vertical_field_of_view) {
	makeCurrent();
  auto option = Options<GLSpins>::withOption<ISpinRenderer::Option::VERTICAL_FIELD_OF_VIEW>(vertical_field_of_view);
  gl_spins->updateOptions(option);
}

glm::vec3 SpinWidget::backgroundColor() const {
  return options().get<ISpinRenderer::Option::BACKGROUND_COLOR>();
}

void SpinWidget::setBackgroundColor(glm::vec3 background_color) {
	makeCurrent();
  auto option = Options<GLSpins>::withOption<ISpinRenderer::Option::BACKGROUND_COLOR>(background_color);
  gl_spins->updateOptions(option);
}

glm::vec3 SpinWidget::boundingBoxColor() const {
  return options().get<BoundingBoxRenderer::Option::COLOR>();
}

void SpinWidget::setBoundingBoxColor(glm::vec3 bounding_box_color) {
	makeCurrent();
  auto option = Options<GLSpins>::withOption<BoundingBoxRenderer::Option::COLOR>(bounding_box_color);
  gl_spins->updateOptions(option);
}

bool SpinWidget::isMiniviewEnabled() const {
  return options().get<GLSpins::Option::SHOW_MINIVIEW>();
}

void SpinWidget::enableMiniview(bool enabled) {
	makeCurrent();
  auto option = Options<GLSpins>::withOption<GLSpins::Option::SHOW_MINIVIEW>(enabled);
  gl_spins->updateOptions(option);
}

bool SpinWidget::isCoordinateSystemEnabled() const {
  return options().get<GLSpins::Option::SHOW_COORDINATE_SYSTEM>();
}

void SpinWidget::enableCoordinateSystem(bool enabled) {
	makeCurrent();
  auto option = Options<GLSpins>::withOption<GLSpins::Option::SHOW_COORDINATE_SYSTEM>(enabled);
  gl_spins->updateOptions(option);
}

bool SpinWidget::isBoundingBoxEnabled() const {
  return options().get<GLSpins::Option::SHOW_BOUNDING_BOX>();
}

void SpinWidget::enableBoundingBox(bool enabled) {
	makeCurrent();
  auto option = Options<GLSpins>::withOption<GLSpins::Option::SHOW_BOUNDING_BOX>(enabled);
  gl_spins->updateOptions(option);
}

GLSpins::WidgetLocation SpinWidget::miniviewPosition() const {
  return options().get<GLSpins::Option::MINIVIEW_LOCATION>();
}

void SpinWidget::setMiniviewPosition(GLSpins::WidgetLocation miniview_position) {
	makeCurrent();
  auto option = Options<GLSpins>::withOption<GLSpins::Option::MINIVIEW_LOCATION>(miniview_position);
  gl_spins->updateOptions(option);
}

GLSpins::WidgetLocation SpinWidget::coordinateSystemPosition() const {
  return options().get<GLSpins::Option::COORDINATE_SYSTEM_LOCATION>();
}

void SpinWidget::setCoordinateSystemPosition(GLSpins::WidgetLocation coordinatesystem_position) {
	makeCurrent();
  auto option = Options<GLSpins>::withOption<GLSpins::Option::COORDINATE_SYSTEM_LOCATION>(coordinatesystem_position);
  gl_spins->updateOptions(option);
}

GLSpins::VisualizationMode SpinWidget::visualizationMode() const {
  return options().get<GLSpins::Option::VISUALIZATION_MODE>();
}

void SpinWidget::setVisualizationMode(GLSpins::VisualizationMode visualization_mode) {
	makeCurrent();
  auto option = Options<GLSpins>::withOption<GLSpins::Option::VISUALIZATION_MODE>(visualization_mode);
  gl_spins->updateOptions(option);
}

glm::vec2 SpinWidget::zRange() const {
  return options().get<ISpinRenderer::Option::Z_RANGE>();
}

void SpinWidget::setZRange(glm::vec2 z_range) {
	makeCurrent();
  auto option = Options<GLSpins>::withOption<ISpinRenderer::Option::Z_RANGE>(z_range);
  gl_spins->updateOptions(option);
}


GLSpins::Colormap SpinWidget::colormap() const {
  auto colormap_implementation = options().get<ISpinRenderer::Option::COLORMAP_IMPLEMENTATION>();
  if (colormap_implementation == getColormapImplementation("hsv")) {
    return GLSpins::Colormap::HSV;
  } else if (colormap_implementation == getColormapImplementation("redblue")) {
    return GLSpins::Colormap::RED_BLUE;
  }
  return GLSpins::Colormap::OTHER;
}

void SpinWidget::setColormap(GLSpins::Colormap colormap) {
	makeCurrent();
  std::string colormap_implementation = getColormapImplementation("hsv");
  switch (colormap) {
    case GLSpins::Colormap::HSV:
      break;
    case GLSpins::Colormap::RED_BLUE:
      colormap_implementation = getColormapImplementation("redblue");
      break;
    case GLSpins::Colormap::OTHER:
      break;
    default:
      break;
  }
  auto option = Options<GLSpins>::withOption<ISpinRenderer::Option::COLORMAP_IMPLEMENTATION>(colormap_implementation);
  gl_spins->updateOptions(option);
}

glm::vec2 SpinWidget::spherePointSizeRange() const {
  return options().get<SphereSpinRenderer::Option::POINT_SIZE_RANGE>();
}

void SpinWidget::setSpherePointSizeRange(glm::vec2 sphere_point_size_range) {
	makeCurrent();
  auto option = Options<GLSpins>::withOption<SphereSpinRenderer::Option::POINT_SIZE_RANGE>(sphere_point_size_range);
  gl_spins->updateOptions(option);
}
