#include "SpinWidget.h"

#include <QTimer>
#include <QMouseEvent>
#include "Interface_Geometry.h"
#include "Interface_State.h"

SpinWidget::SpinWidget(std::shared_ptr<State> state, QWidget *parent) : QOpenGLWidget(parent)
{
	this->state = state;

	setFocusPolicy(Qt::StrongFocus);
}

void SpinWidget::initializeGL() {
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
  
  std::vector<glm::vec3> positions(s->geometry->nos);
  for (unsigned int i = 0; i < s->geometry->nos; ++i)
  {
    positions[i] = glm::vec3(s->geometry->spin_pos[0][i], s->geometry->spin_pos[1][i], s->geometry->spin_pos[2][i]);
  }
  std::vector<glm::vec3> directions(s->geometry->nos);
  for (unsigned int i = 0; i < s->geometry->nos; ++i)
  {
    directions[i] = glm::vec3(s->spins[i], s->spins[s->geometry->nos + i], s->spins[2*s->geometry->nos + i]);
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
  auto position_before = glm::vec2(_previous_pos.x(), _previous_pos.y());
  auto position_after = glm::vec2(current_pos.x(), current_pos.y());
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
