#include <QtWidgets>

#include "PlotsWidget.h"
#include "PlotWidget.h"
#include "Interface_Chain.h"

PlotsWidget::PlotsWidget(std::shared_ptr<State> state)
{
	this->state = state;
    
	// Setup User Interface
    this->setupUi(this);

    this->energyPlot = new PlotWidget(this->state);
	this->gridLayout_Energy_Plots->addWidget(energyPlot, 0, 0, 1, 1);

	connect(this->pushButton_Refresh, SIGNAL(clicked()), this, SLOT(RefreshClicked()));
}


void PlotsWidget::RefreshClicked()
{
	Chain_Update_Data(this->state.get());
	this->energyPlot->update();
}

void PlotsWidget::ChangeInterpolationClicked()
{

}