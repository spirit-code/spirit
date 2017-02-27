#include "MouseDecoratorWidget.hpp"

#include <QPainter>

MouseDecoratorWidget::MouseDecoratorWidget()
{
	this->m_rotation_angle = 0;
	this->m_rotation_updates_per_second = 40;
	this->m_rpm = 30;

	this->m_rotation_timer = new QTimer(this);
	connect(this->m_rotation_timer, &QTimer::timeout, this, &MouseDecoratorWidget::incrementRotation);
	this->m_rotation_timer->start(std::max(1,1000/this->m_rotation_updates_per_second));
}

void MouseDecoratorWidget::paintEvent(QPaintEvent *)
{
	float radius = 80;
	int shift = (int)std::min(5.0f, radius-5);


    QPainter painter(this);
	
	/*painter.setFont(QFont("Arial", 30));
	painter.drawText(rect(), Qt::AlignCenter, "Qt");*/

	QPen pen;

	// Draw a solid background circle (gray)
	pen = QPen(Qt::gray, 3, Qt::PenStyle::SolidLine);
    painter.setPen(pen);
	painter.drawArc(shift, shift, 2 * (radius - shift), 2 * (radius - shift), 0, 16 * 360);

	// Draw a dashed front circle (blue)
	pen = QPen(Qt::blue, 3, Qt::PenStyle::CustomDashLine);
	pen.setDashPattern({ 25, 25 });
	painter.setPen(pen);
	painter.drawArc(shift, shift, 2 * (radius - shift), 2 * (radius - shift), 16 * this->m_rotation_angle, 16 * 360);
}

void MouseDecoratorWidget::incrementRotation()
{
	int add_angle =  6 * m_rpm / m_rotation_updates_per_second;
	this->m_rotation_angle = (m_rotation_angle + add_angle) % 360;
	this->update();
}