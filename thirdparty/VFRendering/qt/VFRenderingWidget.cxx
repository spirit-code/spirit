#include "VFRenderingWidget.hxx"

#include <VFRendering/ArrowRenderer.hxx>

#include <vector>
#include <QWheelEvent>

VFRenderingWidget::VFRenderingWidget(QWidget *parent) : QOpenGLWidget(parent), m_vf({}, {}) {}

VFRenderingWidget::~VFRenderingWidget() {}

void VFRenderingWidget::initializeGL() {
  setMouseTracking(true);
  m_view.renderers({{ std::make_shared<VFRendering::ArrowRenderer>(m_view, m_vf), {{0, 0, 1, 1}} }});
}

void VFRenderingWidget::resizeGL(int width, int height) {
  m_view.setFramebufferSize(width, height);
}

void VFRenderingWidget::update(const VFRendering::Geometry& geometry, const std::vector<glm::vec3>& vectors) {
  m_vf.update(geometry, vectors);
}


void VFRenderingWidget::updateVectors(const std::vector<glm::vec3>& vectors) {
  m_vf.updateVectors(vectors);
}

void VFRenderingWidget::updateOptions(const VFRendering::Options& options) {
  m_view.updateOptions(options);
}

void VFRenderingWidget::paintGL() {
  m_view.draw();
}

float VFRenderingWidget::getFramerate() const {
  return m_view.getFramerate();
}

void VFRenderingWidget::wheelEvent(QWheelEvent *event) {
  float delta = event->angleDelta().y()*0.1;
  m_view.mouseScroll(delta);
  ((QWidget *)this)->update();
}

void VFRenderingWidget::mousePressEvent(QMouseEvent *event) {
  m_previous_mouse_position = event->pos();
}

void VFRenderingWidget::mouseMoveEvent(QMouseEvent *event) {
  glm::vec2 current_mouse_position(event->pos().x(), event->pos().y());
  glm::vec2 previous_mouse_position(m_previous_mouse_position.x(), m_previous_mouse_position.y());
  m_previous_mouse_position = event->pos();
  
  if (event->buttons() & Qt::LeftButton) {
    auto movement_mode = VFRendering::CameraMovementModes::ROTATE;
    if (event->modifiers() & Qt::AltModifier) {
      movement_mode = VFRendering::CameraMovementModes::TRANSLATE;
    }
    m_view.mouseMove(previous_mouse_position, current_mouse_position, movement_mode);
    ((QWidget *)this)->update();
  }
}
