#include "GeometryWidget.hpp"

#include <QtWidgets>

#include <Spirit/Chain.h>
#include <Spirit/Geometry.h>
#include <Spirit/System.h>

GeometryWidget::GeometryWidget( std::shared_ptr<State> state, SpinWidget * spinWidget )
{
    this->state      = state;
    this->spinWidget = spinWidget;

    // Setup User Interface
    this->setupUi( this );

    // We use a regular expression (regex) to filter the input into the lineEdits
    QRegularExpression re( "[+|-]?[\\d]*[\\.]?[\\d]*" );
    this->number_validator = new QRegularExpressionValidator( re );
    QRegularExpression re2( "[\\d]*[\\.]?[\\d]*" );
    this->number_validator_unsigned = new QRegularExpressionValidator( re2 );
    QRegularExpression re3( "[+|-]?[\\d]*" );
    this->number_validator_int = new QRegularExpressionValidator( re3 );
    QRegularExpression re4( "[\\d]*" );
    this->number_validator_int_unsigned = new QRegularExpressionValidator( re4 );
    // Setup the validators for the various input fields
    this->Setup_Input_Validators();

    // Load variables from SpinWidget and State
    this->updateData();

    // Connect signals and slots
    this->Setup_Slots();
}

void GeometryWidget::updateData()
{
    // Boundary conditions
    bool boundary_conditions[3];
    Geometry_Get_Boundary_Conditions( state.get(), boundary_conditions );
    this->checkBox_periodical_a->setChecked( boundary_conditions[0] );
    this->checkBox_periodical_b->setChecked( boundary_conditions[1] );
    this->checkBox_periodical_c->setChecked( boundary_conditions[2] );

    int n_cells[3];
    Geometry_Get_N_Cells( this->state.get(), n_cells );
    this->lineEdit_n_cells_a->setText( QString::number( n_cells[0] ) );
    this->lineEdit_n_cells_b->setText( QString::number( n_cells[1] ) );
    this->lineEdit_n_cells_c->setText( QString::number( n_cells[2] ) );
}

void GeometryWidget::setNCells()
{
    // Get some reference values for the position filter of SpinWidget
    auto x_range = this->spinWidget->xRangePosition();
    auto y_range = this->spinWidget->yRangePosition();
    auto z_range = this->spinWidget->zRangePosition();
    scalar pc_x  = x_range.y - x_range.x;
    scalar pc_y  = y_range.y - y_range.x;
    scalar pc_z  = z_range.y - z_range.x;
    scalar b_min[3], b_max[3], b_range[3];
    Geometry_Get_Bounds( state.get(), b_min, b_max );
    if( std::abs( b_max[0] - b_min[0] ) > 0 )
        pc_x /= std::abs( b_max[0] - b_min[0] );
    if( std::abs( b_max[1] - b_min[1] ) > 0 )
        pc_y /= std::abs( b_max[1] - b_min[1] );
    if( std::abs( b_max[2] - b_min[2] ) > 0 )
        pc_z /= std::abs( b_max[2] - b_min[2] );

    // Update the geometry in the core
    int n_cells[3]{ this->lineEdit_n_cells_a->text().toInt(), this->lineEdit_n_cells_b->text().toInt(),
                    this->lineEdit_n_cells_c->text().toInt() };
    Geometry_Set_N_Cells( this->state.get(), n_cells );

    this->spinWidget->setCellFilter( 0, n_cells[0] - 1, 0, n_cells[1] - 1, 0, n_cells[2] - 1 );

    // Update geometry and arrays in SpinWidget
    this->spinWidget->initializeGL();
    this->spinWidget->updateData();

    // Update the position filter of SpinWidget
    scalar b_min_new[3], b_max_new[3], b_range_new[3];
    Geometry_Get_Bounds( state.get(), b_min_new, b_max_new );
    x_range.x = pc_x * b_min_new[0];
    x_range.y = pc_x * b_max_new[0];
    y_range.x = pc_y * b_min_new[1];
    y_range.y = pc_y * b_max_new[1];
    z_range.x = pc_z * b_min_new[2];
    z_range.y = pc_z * b_max_new[2];
    this->spinWidget->setOverallPositionRange( x_range, y_range, z_range );

    // Update all widgets
    emit updateNeeded();
}

void GeometryWidget::set_boundary_conditions()
{
    // Boundary conditions
    const std::array<bool, 3> boundary_conditions{ this->checkBox_periodical_a->isChecked(),
                                                   this->checkBox_periodical_b->isChecked(),
                                                   this->checkBox_periodical_c->isChecked() };

    // The Hamiltonians, for each image in the chain, have a shared pointer to the geometry.
    // So technically we only have to update the geometry object for one image.
    // *But* updating the boundary conditions involves some bookkeeping in the
    // hamiltonian interactions and therefore we call `Geometry_Set_Boundary_Conditions` for each image in the chain.
    // This way we can be sure that the necessary bookkeeping is performed for all hamiltonians
    for( int idx_image = 0; idx_image < Chain_Get_NOI( state.get() ); idx_image++ )
    {
        Geometry_Set_Boundary_Conditions( state.get(), boundary_conditions.data(), idx_image );
    }
    this->spinWidget->updateBoundingBoxIndicators();
}

void GeometryWidget::Setup_Input_Validators()
{
    this->lineEdit_n_cells_a->setValidator( this->number_validator_unsigned );
    this->lineEdit_n_cells_b->setValidator( this->number_validator_unsigned );
    this->lineEdit_n_cells_c->setValidator( this->number_validator_unsigned );
}

void GeometryWidget::Setup_Slots()
{
    // Boundary Conditions
    connect( this->checkBox_periodical_a, SIGNAL( stateChanged( int ) ), this, SLOT( set_boundary_conditions() ) );
    connect( this->checkBox_periodical_b, SIGNAL( stateChanged( int ) ), this, SLOT( set_boundary_conditions() ) );
    connect( this->checkBox_periodical_c, SIGNAL( stateChanged( int ) ), this, SLOT( set_boundary_conditions() ) );

    connect( this->lineEdit_n_cells_a, SIGNAL( returnPressed() ), this, SLOT( setNCells() ) );
    connect( this->lineEdit_n_cells_b, SIGNAL( returnPressed() ), this, SLOT( setNCells() ) );
    connect( this->lineEdit_n_cells_c, SIGNAL( returnPressed() ), this, SLOT( setNCells() ) );
}
