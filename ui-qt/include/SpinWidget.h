#pragma once
#ifndef SPIN_WIDGET_H
#define SPIN_WIDGET_H


#include <memory>
#include <QOpenGLWidget>

#include "GLSpins.h"

class State;

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
  
  // Visualisation
  std::shared_ptr<GLSpins> gl_spins;
};

#endif
