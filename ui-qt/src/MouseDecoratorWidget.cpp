#include "MouseDecoratorWidget.hpp"

#include <QPainter>

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
	painter.drawArc(shift, shift, 2 * (radius - shift), 2 * (radius - shift), 0, 16 * 360);
}