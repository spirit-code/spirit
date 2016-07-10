#include "Spin_Widget.h"

#include <QOpenGLWidget>
#include <QtGui/QOpenGLFunctions>
#include <QtGui/QOpenGLShaderProgram>

#include <QTimer>
#include <QKeyEvent>

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>

#include "gl_spins.h"

#define M_PI_DIV180 0.017453292519943295769236907684886127134428718885417254560971


using Data::Spin_System;

Spin_Widget::Spin_Widget(std::shared_ptr<Data::Spin_System_Chain> c, QWidget *parent) : QOpenGLWidget(parent)
{
	this->c = c;
	this->s = c->images[c->active_image];

	setFocusPolicy(Qt::StrongFocus);

	this->timer_ = new QTimer(this);
	this->connect(this->timer_, SIGNAL(timeout()), this, SLOT(on_timer()));
	this->timer_->start(10);

	this->t_frames.push_back(system_clock::now());
	this->t_frames.push_back(system_clock::now());
	this->t_frames.push_back(system_clock::now());
	this->t_frames.push_back(system_clock::now());
	this->t_frames.push_back(system_clock::now());
	this->t_frames.push_back(system_clock::now());
	this->fps = 0;
}


void Spin_Widget::initializeGL()
{
	this->gl_spins = std::shared_ptr<GLSpins>(new GLSpins(s, width(), height()));
}

void Spin_Widget::teardownGL()
{
	// GLSpins::terminate();
}

void Spin_Widget::resizeGL(int width, int height)
{
	gl_spins->update_projection_matrix(width, height);
	gl_spins->camera.setWindowSize(width, height);
}

void Spin_Widget::paintGL()
{
	gl_spins->draw();

	// Update frame times
	this->t_frames.pop_front();
	this->t_frames.push_back(system_clock::now());
}

void Spin_Widget::on_timer()
{
	static float rotation_angle = 0.0f;
	// rotation_angle += 1.0f;
	rotation_angle -= (int)(rotation_angle / 360.0f) * 360.0f;
	// std::cerr << rotation_angle << std::endl;
	gl_spins->rotate_model(rotation_angle);
	update();
}


void Spin_Widget::mouseMoveEvent(QMouseEvent *event)
{
	update();
}


void Spin_Widget::update()
{
	// Update the pointer to our Data
	this->s = c->images[c->active_image];
	this->gl_spins->update_spin_system(this->s);

	// Update Keyboard and Mouse Input
	UpdateInput();

	// Schedule a redraw
	QOpenGLWidget::update();
}

void Spin_Widget::UpdateInput()
{
	InputHandler::update();
	QPoint mousePosition = mapFromGlobal(InputHandler::mousePosition());
	QPoint mouseDelta = InputHandler::mouseDelta();
	//qDebug() << "update";

	// Camera Transformation
	if (InputHandler::buttonTriggered(Qt::LeftButton))
	{
		gl_spins->camera.startTrackball(mousePosition.x(), mousePosition.y());
	}
	else if (InputHandler::buttonPressed(Qt::LeftButton))
	{
		gl_spins->camera.updateTrackball(mousePosition.x(), mousePosition.y());
		gl_spins->camera_updated();
	}
	else if (InputHandler::buttonPressed(Qt::RightButton))
	{
		gl_spins->camera.translate(mouseDelta.x(), mouseDelta.y());
		gl_spins->camera_updated();
	}
	else if (InputHandler::mouseWheelTurned())
	{
		gl_spins->camera.translate(InputHandler::mouseWheelDelta());
	}

}

void Spin_Widget::SetCameraToDefault()
{
	//// ToDo: make the default camera view cool & angled?
	//auto camera = this->m_renderer->GetActiveCamera();

	//// ToDo: BUILD A SWITCH FOR WHICH CAMERA WE WANT TO USE -- WITH OR WITHOUT PROJECTION
	////camera->SetViewAngle(0.1);

	//// How to determine the camera Distance from s->geometry->bounds??
	////double cameraDistance = 6000000.0;
	////double cameraDistance = 60.0;
	//camera->SetFocalPoint(s->geometry->center[0], s->geometry->center[1], s->geometry->center[2]);
	//camera->SetPosition(s->geometry->center[0], s->geometry->center[1], CAMERA_DISTANCE);
	//camera->SetViewUp(0, 1, 0);
	//if (PARALLEL_PROJECTION) camera->SetParallelScale(PARALLEL_SCALE);
}

void Spin_Widget::SetCameraToX()
{
	//auto camera = this->m_renderer->GetActiveCamera();

	//// How to determine the camera Distance from s->geometry->bounds??
	//double cameraDistance = 60.0;
	//camera->SetFocalPoint(s->geometry->center[0], s->geometry->center[1], s->geometry->center[2]);
	//camera->SetPosition(cameraDistance, s->geometry->center[1], s->geometry->center[2]);
	//camera->SetViewUp(0, 0, 1);
	//if (PARALLEL_PROJECTION) camera->SetParallelScale(PARALLEL_SCALE);
}

void Spin_Widget::SetCameraToY()
{
	//auto camera = this->m_renderer->GetActiveCamera();

	//// How to determine the camera Distance from s->geometry->bounds??
	////double cameraDistance = 60.0;
	//camera->SetFocalPoint(s->geometry->center[0], s->geometry->center[1], s->geometry->center[2]);
	//camera->SetPosition(s->geometry->center[0], CAMERA_DISTANCE, s->geometry->center[2]);
	//camera->SetViewUp(0, 0, 1);
	//if (PARALLEL_PROJECTION) camera->SetParallelScale(PARALLEL_SCALE);
}

void Spin_Widget::SetCameraToZ()
{
	//auto camera = this->m_renderer->GetActiveCamera();

	//// How to determine the camera Distance from s->geometry->bounds??
	////double cameraDistance = 60.0;
	//camera->SetFocalPoint(s->geometry->center[0], s->geometry->center[1], s->geometry->center[2]);
	//camera->SetPosition(s->geometry->center[0], s->geometry->center[1], CAMERA_DISTANCE);
	//camera->SetViewUp(0, 1, 0);
	//if (PARALLEL_PROJECTION) camera->SetParallelScale(PARALLEL_SCALE);
}

double Spin_Widget::getFramesPerSecond()
{
	double l_fps = 0.0;
	for (int i = 0; i < t_frames.size()-1; ++i)
	{
		l_fps += Utility::Timing::SecondsPassed(t_frames[i], t_frames[i+1]);
	}
	this->fps = 1.0 / (l_fps / (t_frames.size() - 1));
	return this->fps;
}

/*******************************************************************************
* Key Press Handlers
******************************************************************************/
void Spin_Widget::keyPressEvent(QKeyEvent *event)
{
	if (
		event->isAutoRepeat() ||
		// TODO: somehow let the InputHandler handle the rest?
		event->matches(QKeySequence::Copy) ||
		event->matches(QKeySequence::Cut) ||
		event->matches(QKeySequence::Paste) ||
		(event->modifiers() & Qt::ControlModifier && (event->key() == Qt::Key_Left || event->key() == Qt::Key_Right)) ||
		event->key() == Qt::Key_Escape ||
		event->key() == Qt::Key_Up ||
		event->key() == Qt::Key_Left ||
		event->key() == Qt::Key_Right ||
		event->key() == Qt::Key_Down ||
		event->key() == Qt::Key_0 ||
		event->key() == Qt::Key_Space ||
		event->key() == Qt::Key_F1 ||
		event->key() == Qt::Key_F2 ||
		event->key() == Qt::Key_F3 ||
		event->key() == Qt::Key_F4 ||
		event->key() == Qt::Key_1 ||
		event->key() == Qt::Key_2 ||
		event->key() == Qt::Key_3 ||
		event->key() == Qt::Key_4 ||
		event->key() == Qt::Key_5 ||
		event->key() == Qt::Key_Delete
		)
	{
		// Propagate to parentWidget
		event->ignore();
	}
	else
	{
		InputHandler::registerKeyPress(event->key());
	}
}

void Spin_Widget::keyReleaseEvent(QKeyEvent *event)
{
	if (event->isAutoRepeat())
	{
		event->ignore();
	}
	else
	{
		InputHandler::registerKeyRelease(event->key());
	}
}

void Spin_Widget::mousePressEvent(QMouseEvent *event)
{
	InputHandler::registerMousePress(event->button());
}

void Spin_Widget::mouseReleaseEvent(QMouseEvent *event)
{
	InputHandler::registerMouseRelease(event->button());
}

void Spin_Widget::wheelEvent(QWheelEvent *event)
{
	InputHandler::registerMouseWheel(event->delta());
}


/*******************************************************************************
* Private Helpers
******************************************************************************/

void Spin_Widget::printVersionInformation()
{
	QString glType;
	QString glVersion;
	QString glProfile;

	// Get Version Information
	glType = (context()->isOpenGLES()) ? "OpenGL ES" : "OpenGL";
	glVersion = reinterpret_cast<const char*>(glGetString(GL_VERSION));

	// Get Profile Information
#define CASE(c) case QSurfaceFormat::c: glProfile = #c; break
	switch (format().profile())
	{
		CASE(NoProfile);
		CASE(CoreProfile);
		CASE(CompatibilityProfile);
	}
#undef CASE

	// qPrintable() will print our QString w/o quotes around it.
	qDebug() << qPrintable(glType) << qPrintable(glVersion) << "(" << qPrintable(glProfile) << ")";


	/*// DEBUG INFO: OpenGL and GLSL Versions
	const GLubyte* renderer = glGetString (GL_RENDERER); // get renderer string
	const GLubyte* version = glGetString (GL_VERSION); // version as a string
	printf("Renderer: %s\n", renderer);
	printf("OpenGL version supported %s\n", version);
	//qDebug() << this->format().majorVersion() << "." << this->format().minorVersion();*/
}
