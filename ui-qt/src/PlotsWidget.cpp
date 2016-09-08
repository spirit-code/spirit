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
	connect(this->checkBox_InterpolateEnergies, SIGNAL(stateChanged(int)), this, SLOT(ChangeInterpolationClicked()));

	// Update Timer
	auto timer = new QTimer(this);
	connect(timer, &QTimer::timeout, this, &PlotsWidget::updatePlots);
	timer->start(500);
}

void PlotsWidget::updatePlots()
{
	// TODO: check which plot is active -> which we should update
	this->energyPlot->update();
}

void PlotsWidget::RefreshClicked()
{
	Chain_Update_Data(this->state.get());
}

void PlotsWidget::ChangeInterpolationClicked()
{
	this->energyPlot->set_interpolating(this->checkBox_InterpolateEnergies->isChecked());
}