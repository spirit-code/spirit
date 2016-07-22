#pragma once
#ifndef SPIN_WIDGET_H
#define SPIN_WIDGET_H

#include <QTimer>

#include <QOpenGLWidget>
#include <QOpenGLFunctions>

#include <memory>
#include <deque>

#include "gl_spins.h"

#include "Spin_System_Chain.h"
#include "InputHandler.h"
#include "Interface_State.h"
//#include "Transform.h"

class Spin_Widget : public QOpenGLWidget, protected QOpenGLFunctions
{
	Q_OBJECT

public:
	Spin_Widget(std::shared_ptr<State> state, QWidget *parent = 0);
	void initializeGL();
	void resizeGL(int w, int h);
	void paintGL();
    void mouseMoveEvent(QMouseEvent *event);
	double getFramesPerSecond();

	void SetCameraToDefault();
	void SetCameraToX();
	void SetCameraToY();
	void SetCameraToZ();

protected:
	void keyPressEvent(QKeyEvent *event);
	void keyReleaseEvent(QKeyEvent *event);
	void mousePressEvent(QMouseEvent *event);
	void mouseReleaseEvent(QMouseEvent *event);
    void wheelEvent(QWheelEvent *event);

	protected slots:
	void teardownGL();
	void update();

	private slots:
	void on_timer();

private:
	std::shared_ptr<State> state;
	std::shared_ptr<Data::Spin_System_Chain> c;
	std::shared_ptr<Data::Spin_System> s;
	QTimer *timer_;

	// Visualisation
    std::shared_ptr<GLSpins> gl_spins;

	// Private Helpers
	void UpdateInput();
	void printVersionInformation();
    void Update_Projection_Matrix(int width, int height);
	int m_frame;
	double fps;
	std::deque<std::chrono::time_point<std::chrono::system_clock>> t_frames;

};

#endif
