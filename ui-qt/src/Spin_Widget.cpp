#include "Spin_Widget.h"

#include <QOpenGLWidget>
#include <QMouseEvent>

#include "gl_spins.h"

using Data::Spin_System;

Spin_Widget::Spin_Widget(std::shared_ptr<Data::Spin_System_Chain> c, QWidget *parent) : QOpenGLWidget(parent) {
	this->c = c;
	this->s = c->images[c->active_image];

	setFocusPolicy(Qt::StrongFocus);
}

void Spin_Widget::initializeGL() {
	this->gl_spins = std::shared_ptr<GLSpins>(new GLSpins(s, width(), height()));
}

void Spin_Widget::teardownGL() {
	// GLSpins::terminate();
}

void Spin_Widget::resizeGL(int width, int height) {
  gl_spins->setFramebufferSize(width*devicePixelRatio(), height*devicePixelRatio());
  update();
}

void Spin_Widget::paintGL() {
  // Update the pointer to our Data
  this->s = c->images[c->active_image];
  this->gl_spins->update_spin_system(this->s);
  gl_spins->draw();
  QTimer::singleShot(1, this, SLOT(update()));
}

void Spin_Widget::mousePressEvent(QMouseEvent *event) {
  _previous_pos = event->pos();
}

void Spin_Widget::mouseMoveEvent(QMouseEvent *event) {
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

void Spin_Widget::SetCameraToDefault() {
}

void Spin_Widget::SetCameraToX() {
}

void Spin_Widget::SetCameraToY() {
}

void Spin_Widget::SetCameraToZ() {
}

double Spin_Widget::getFramesPerSecond() const {
  return gl_spins->getFramerate();
}

void Spin_Widget::wheelEvent(QWheelEvent *event) {
  double wheel_delta = event->delta();
  gl_spins->mouseScroll(wheel_delta*0.1);
}
