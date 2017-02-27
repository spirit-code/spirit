#pragma once
#ifndef MOUSE_DECORATOR_WIDGET_H
#define MOUSE_DECORATOR_WIDGET_H

#include <QWidget>
#include <QTimer>

class MouseDecoratorWidget : public QWidget
{
public:
	MouseDecoratorWidget();
	void MouseDecoratorWidget::paintEvent(QPaintEvent *);
private slots:
	void incrementRotation();
private:
	QTimer * m_rotation_timer;
	int m_rotation_angle;
	int m_rpm;
	int m_rotation_updates_per_second;
};

#endif