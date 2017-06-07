#pragma once
#ifndef ISOSURFACEWIDGET_H
#define ISOSURFACEWIDGET_H

#include <QWidget>

#include <memory>

#include <VFRendering/IsosurfaceRenderer.hxx>

#include "ui_IsosurfaceWidget.h"

struct State;
class SpinWidget;

class IsosurfaceWidget : public QWidget, private Ui::IsosurfaceWidget
{
    Q_OBJECT

public:
	IsosurfaceWidget(std::shared_ptr<State> state, SpinWidget *spinWidget);

	std::shared_ptr<State> state;
	SpinWidget * spinWidget;
    
	bool showIsosurface();
	void setShowIsosurface(bool show);

	float isovalue();
	void setIsovalue(float value);

	int isocomponent();
	void setIsocomponent(int component);

	bool drawShadows();
	void setDrawShadows(bool show);

signals:
	void closedSignal();

private slots:
	void slot_setIsovalue_slider();
	void slot_setIsovalue_lineedit();
	void slot_setIsocomponent();

private:
	void setupSlots();
	void setupInputValidators();

	bool m_show_isosurface;
	float m_isovalue;
	int m_isocomponent;
	bool m_draw_shadows;
	std::shared_ptr<VFRendering::IsosurfaceRenderer> m_renderer;
	QRegularExpressionValidator * number_validator;

protected:
	void closeEvent(QCloseEvent *event);
};

#endif