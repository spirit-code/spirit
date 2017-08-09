#include <QtWidgets>

#include "GeometryWidget.hpp"

#include <Spirit/Geometry.h>

GeometryWidget::GeometryWidget(std::shared_ptr<State> state, SpinWidget * spinWidget)
{
	this->state = state;
	this->spinWidget = spinWidget;

	// Setup User Interface
    this->setupUi(this);
	// We use a regular expression (regex) to filter the input into the lineEdits
	QRegularExpression re("[+|-]?[\\d]*[\\.]?[\\d]*");
	this->number_validator = new QRegularExpressionValidator(re);
	QRegularExpression re2("[\\d]*[\\.]?[\\d]*");
	this->number_validator_unsigned = new QRegularExpressionValidator(re2);
	QRegularExpression re3("[+|-]?[\\d]*");
	this->number_validator_int = new QRegularExpressionValidator(re3);
	QRegularExpression re4("[\\d]*");
	this->number_validator_int_unsigned = new QRegularExpressionValidator(re4);
	// Setup the validators for the various input fields
	 this->Setup_Input_Validators();

	// Load variables from SpinWidget and State
	this->updateData();

	// Connect signals and slots
	 this->Setup_Slots();
}

void GeometryWidget::updateData()
{
}

void GeometryWidget::setNCells()
{
	int n_cells[3]{ 10, 30, 1 };
	Geometry_Set_N_Cells(this->state.get(), n_cells);
	this->spinWidget->initializeGL();
	this->spinWidget->updateData();
}

void GeometryWidget::Setup_Input_Validators()
{
	this->lineEdit_n_cells_a->setValidator(this->number_validator_unsigned);
	this->lineEdit_n_cells_b->setValidator(this->number_validator_unsigned);
	this->lineEdit_n_cells_c->setValidator(this->number_validator_unsigned);

}


void GeometryWidget::Setup_Slots()
{
	connect(this->lineEdit_n_cells_a, SIGNAL(returnPressed()), this, SLOT(setNCells()));
	connect(this->lineEdit_n_cells_b, SIGNAL(returnPressed()), this, SLOT(setNCells()));
	connect(this->lineEdit_n_cells_c, SIGNAL(returnPressed()), this, SLOT(setNCells()));
}
