#pragma once
#ifndef SPIRIT_MOUSEDECORATORWIDGET_HPP
#define SPIRIT_MOUSEDECORATORWIDGET_HPP

#include <QtCore/QTimer>
#include <QtWidgets/QWidget>

class MouseDecoratorWidget : public QWidget
{
public:
    MouseDecoratorWidget( float radius );
    void paintEvent( QPaintEvent * );
    void setRadius( float radius );
    void setColors( Qt::GlobalColor one, Qt::GlobalColor two );

private slots:
    void incrementRotation();

private:
    float m_radius;
    float m_dash_length;
    QTimer * m_rotation_timer;
    int m_rotation_angle;
    int m_rpm;
    int m_rotation_updates_per_second;
    Qt::GlobalColor color_one;
    Qt::GlobalColor color_two;
};

#endif