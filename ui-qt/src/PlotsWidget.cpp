#include <QtWidgets>

#include "PlotsWidget.h"
#include "PlotWidget.h"

PlotsWidget::PlotsWidget(std::shared_ptr<State> state)
{
	this->state = state;
	this->c = this->state->c;
    
	// Setup User Interface
    this->setupUi(this);

    this->energyPlot = new PlotWidget(this->c);
	this->gridLayout_Energy_Plots->addWidget(energyPlot, 0, 0, 1, 1);

	connect(this->pushButton_Refresh, SIGNAL(clicked()), this, SLOT(RefreshClicked()));
}


void PlotsWidget::RefreshClicked()
{
	this->c->Update_Data();
	this->energyPlot->update();
}

void PlotsWidget::ChangeInterpolationClicked()
{

}