
#include "PlotsWidget.hpp"
#include "PlotWidget.hpp"

#include <QtWidgets>

#include <Spirit/Chain.h>

PlotsWidget::PlotsWidget( std::shared_ptr<State> state )
{
    this->state = state;

    // Setup User Interface
    this->setupUi( this );

    this->energyPlot = new PlotWidget( this->state );
    this->gridLayout_Energy_Plots->addWidget( energyPlot, 0, 0, 1, 1 );

    this->checkBox_ImageEnergies->setChecked( this->energyPlot->plot_image_energies );
    this->checkBox_InterpolateEnergies->setChecked( this->energyPlot->plot_interpolated );
    this->spinBox_InterpolateEnergies_N->setValue( this->energyPlot->plot_interpolated_n );

    connect( this->pushButton_Refresh, SIGNAL( clicked() ), this, SLOT( refreshClicked() ) );
    connect( this->checkBox_ImageEnergies, SIGNAL( stateChanged( int ) ), this, SLOT( updatePlotSettings() ) );
    connect( this->checkBox_InterpolateEnergies, SIGNAL( stateChanged( int ) ), this, SLOT( updatePlotSettings() ) );
    connect( this->spinBox_InterpolateEnergies_N, SIGNAL( editingFinished() ), this, SLOT( updatePlotSettings() ) );

    // Update Timer
    auto timer = new QTimer( this );
    connect( timer, &QTimer::timeout, this, &PlotsWidget::updatePlotData );
    timer->start( 200 );
}

void PlotsWidget::updatePlotData()
{
    // TODO: check which plot is active -> which we should update
    this->energyPlot->updateData();
}

void PlotsWidget::refreshClicked()
{
    Chain_Update_Data( this->state.get() );
}

void PlotsWidget::updatePlotSettings()
{
    this->energyPlot->plot_image_energies = this->checkBox_ImageEnergies->isChecked();
    this->energyPlot->plot_interpolated   = this->checkBox_InterpolateEnergies->isChecked();
    this->energyPlot->plot_interpolated_n = this->spinBox_InterpolateEnergies_N->value();
}