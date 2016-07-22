#pragma once
#ifndef SPIN_WIDGET_H
#define SPIN_WIDGET_H


#include <memory>

#include <QTimer>
#include <QOpenGLWidget>

#include "gl_spins.h"

#include "Spin_System_Chain.h"

class Spin_Widget : public QOpenGLWidget
{
  Q_OBJECT
  
public:
  Spin_Widget(std::shared_ptr<Data::Spin_System_Chain> c, QWidget *parent = 0);
  void initializeGL();
  void resizeGL(int width, int height);
  void paintGL();
  double getFramesPerSecond() const;
  
  void SetCameraToDefault();
  void SetCameraToX();
  void SetCameraToY();
  void SetCameraToZ();
  
protected:
  virtual void mouseMoveEvent(QMouseEvent *event);
  virtual void mousePressEvent(QMouseEvent *event);
  virtual void wheelEvent(QWheelEvent *event);
  
  protected slots:
  void teardownGL();
  
private:
  std::shared_ptr<Data::Spin_System_Chain> c;
  std::shared_ptr<Data::Spin_System> s;
  QPoint _previous_pos;
  
  // Visualisation
  std::shared_ptr<GLSpins> gl_spins;
};

#endif
