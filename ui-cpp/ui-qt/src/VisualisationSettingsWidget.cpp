#include "VisualisationSettingsWidget.hpp"

#include <QtWidgets>

#include <Spirit/Geometry.h>
#include <Spirit/Log.h>
#include <Spirit/System.h>

// Small function for normalization of vectors
#define Exception_Division_by_zero 6666
template<typename T>
void normalize( T v[3] )
{
    T len = 0.0;
    for( int i = 0; i < 3; ++i )
        len += v[i] * v[i];
    if( len == 0.0 )
        throw Exception_Division_by_zero;
    for( int i = 0; i < 3; ++i )
        v[i] /= std::sqrt( len );
}

VisualisationSettingsWidget::VisualisationSettingsWidget( std::shared_ptr<State> state, SpinWidget * spinWidget )
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

    // Defaults
    m_isosurfaceshadows = false;
    add_isosurface();

    this->camera_position_last = spinWidget->getCameraPositon();
    this->camera_focus_last    = spinWidget->getCameraFocus();
    this->camera_upvector_last = spinWidget->getCameraUpVector();

    // Load variables from SpinWidget and State
    this->updateData();

    // Connect signals and slots
    this->Setup_Visualization_Slots();

    auto camera_timer = new QTimer( this );
    connect( camera_timer, &QTimer::timeout, this, &VisualisationSettingsWidget::read_camera );
    camera_timer->start( 200 );
}

void VisualisationSettingsWidget::updateData()
{
    this->Load_Visualization_Contents();
}

void VisualisationSettingsWidget::Setup_Input_Validators()
{
    // Visualisation
    //      Arrows
    this->lineEdit_arrows_lod->setValidator( this->number_validator_int_unsigned );
    //      Colormap
    this->lineEdit_cm_axis_x->setValidator( this->number_validator );
    this->lineEdit_cm_axis_y->setValidator( this->number_validator );
    this->lineEdit_cm_axis_z->setValidator( this->number_validator );
    this->lineEdit_colormap_rotate_phi->setValidator( this->number_validator_int_unsigned );
    //      Camera
    this->lineEdit_camera_pos_x->setValidator( this->number_validator );
    this->lineEdit_camera_pos_y->setValidator( this->number_validator );
    this->lineEdit_camera_pos_z->setValidator( this->number_validator );
    this->lineEdit_camera_focus_x->setValidator( this->number_validator );
    this->lineEdit_camera_focus_y->setValidator( this->number_validator );
    this->lineEdit_camera_focus_z->setValidator( this->number_validator );

    this->lineEdit_cell_a_min->setValidator( this->number_validator_int_unsigned );
    this->lineEdit_cell_a_max->setValidator( this->number_validator_int_unsigned );
    this->lineEdit_cell_b_min->setValidator( this->number_validator_int_unsigned );
    this->lineEdit_cell_b_max->setValidator( this->number_validator_int_unsigned );
    this->lineEdit_cell_c_min->setValidator( this->number_validator_int_unsigned );
    this->lineEdit_cell_c_max->setValidator( this->number_validator_int_unsigned );

    this->lineEdit_overall_pos_xmin->setValidator( this->number_validator_unsigned );
    this->lineEdit_overall_pos_xmax->setValidator( this->number_validator_unsigned );
    this->lineEdit_overall_pos_ymin->setValidator( this->number_validator_unsigned );
    this->lineEdit_overall_pos_ymax->setValidator( this->number_validator_unsigned );
    this->lineEdit_overall_pos_zmin->setValidator( this->number_validator_unsigned );
    this->lineEdit_overall_pos_zmax->setValidator( this->number_validator_unsigned );

    this->lineEdit_overall_dir_xmin->setValidator( this->number_validator );
    this->lineEdit_overall_dir_xmax->setValidator( this->number_validator );
    this->lineEdit_overall_dir_ymin->setValidator( this->number_validator );
    this->lineEdit_overall_dir_ymax->setValidator( this->number_validator );
    this->lineEdit_overall_dir_zmin->setValidator( this->number_validator );
    this->lineEdit_overall_dir_zmax->setValidator( this->number_validator );
}

void VisualisationSettingsWidget::Load_Visualization_Contents()
{
    // Mode
    if( this->spinWidget->visualizationMode() == SpinWidget::VisualizationMode::SYSTEM )
        this->radioButton_vismode_system->setChecked( true );
    else
        this->radioButton_vismode_sphere->setChecked( true );

    // N_cell_steps (draw every N'th unit cell)
    this->spinBox_n_cell_steps->setValue( this->spinWidget->visualisationNCellSteps() );

    // System
    bool show_arrows      = spinWidget->show_arrows;
    bool show_boundingbox = spinWidget->show_boundingbox;
    bool show_surface     = spinWidget->show_surface;
    bool show_isosurface  = spinWidget->show_isosurface;
    this->checkBox_show_arrows->setChecked( show_arrows );
    this->checkBox_showBoundingBox->setChecked( show_boundingbox );
    this->checkBox_show_surface->setChecked( show_surface );
    this->checkBox_show_isosurface->setChecked( show_isosurface );
    this->checkBox_isosurfaceshadows->setChecked( this->m_isosurfaceshadows );

    // Miniview
    this->checkBox_showMiniView->setChecked( spinWidget->isMiniviewEnabled() );
    this->comboBox_miniViewPosition->setCurrentIndex( (int)spinWidget->miniviewPosition() );

    // Coordinate System
    this->checkBox_showCoordinateSystem->setChecked( spinWidget->isCoordinateSystemEnabled() );
    this->comboBox_coordinateSystemPosition->setCurrentIndex( (int)spinWidget->coordinateSystemPosition() );

    // Range Arrows
    auto x_range = spinWidget->xRangeDirection();
    auto y_range = spinWidget->yRangeDirection();
    auto z_range = spinWidget->zRangeDirection();
    x_range.x    = std::max( -1.0f, std::min( 1.0f, x_range.x ) );
    x_range.y    = std::max( -1.0f, std::min( 1.0f, x_range.y ) );
    y_range.x    = std::max( -1.0f, std::min( 1.0f, y_range.x ) );
    y_range.y    = std::max( -1.0f, std::min( 1.0f, y_range.y ) );
    z_range.x    = std::max( -1.0f, std::min( 1.0f, z_range.x ) );
    z_range.y    = std::max( -1.0f, std::min( 1.0f, z_range.y ) );

    // Overall direction filter X
    horizontalSlider_overall_dir_xmin->setInvertedAppearance( true );
    horizontalSlider_overall_dir_xmin->setRange( -100, 100 );
    horizontalSlider_overall_dir_xmin->setValue( (int)( -x_range.x * 100 ) );
    horizontalSlider_overall_dir_xmax->setRange( -100, 100 );
    horizontalSlider_overall_dir_xmax->setValue( (int)( x_range.y * 100 ) );
    horizontalSlider_overall_dir_xmin->setTracking( true );
    horizontalSlider_overall_dir_xmax->setTracking( true );
    // Overall direction filter Y
    horizontalSlider_overall_dir_ymin->setInvertedAppearance( true );
    horizontalSlider_overall_dir_ymin->setRange( -100, 100 );
    horizontalSlider_overall_dir_ymin->setValue( (int)( -y_range.x * 100 ) );
    horizontalSlider_overall_dir_ymax->setRange( -100, 100 );
    horizontalSlider_overall_dir_ymax->setValue( (int)( y_range.y * 100 ) );
    horizontalSlider_overall_dir_ymin->setTracking( true );
    horizontalSlider_overall_dir_ymax->setTracking( true );
    // Overall direction filter Z
    horizontalSlider_overall_dir_zmin->setInvertedAppearance( true );
    horizontalSlider_overall_dir_zmin->setRange( -100, 100 );
    horizontalSlider_overall_dir_zmin->setValue( (int)( -z_range.x * 100 ) );
    horizontalSlider_overall_dir_zmax->setRange( -100, 100 );
    horizontalSlider_overall_dir_zmax->setValue( (int)( z_range.y * 100 ) );
    horizontalSlider_overall_dir_zmin->setTracking( true );
    horizontalSlider_overall_dir_zmax->setTracking( true );

    x_range = spinWidget->xRangePosition();
    y_range = spinWidget->yRangePosition();
    z_range = spinWidget->zRangePosition();

    scalar b_min[3], b_max[3], b_range[3];
    Geometry_Get_Bounds( state.get(), b_min, b_max );
    for( int dim = 0; dim < 3; ++dim )
        b_range[dim] = b_max[dim] - b_min[dim];

    scalar range_min = horizontalSlider_overall_pos_xmin->value() / 10000.0;
    scalar range_max = horizontalSlider_overall_pos_xmax->value() / 10000.0;

    // Overall position filter X
    // horizontalSlider_overall_pos_xmin->setInvertedAppearance(true);
    range_min = ( x_range.x - b_min[0] ) / b_range[0];
    range_max = ( x_range.y - b_min[0] ) / b_range[0];
    horizontalSlider_overall_pos_xmin->setRange( 0, 10000 );
    horizontalSlider_overall_pos_xmin->setValue( (int)( range_min * 10000 ) );
    horizontalSlider_overall_pos_xmax->setRange( 0, 10000 );
    horizontalSlider_overall_pos_xmax->setValue( (int)( range_max * 10000 ) );
    horizontalSlider_overall_pos_xmin->setTracking( true );
    horizontalSlider_overall_pos_xmax->setTracking( true );
    // Overall position filter Y
    // horizontalSlider_overall_pos_ymin->setInvertedAppearance(true);
    range_min = ( y_range.x - b_min[1] ) / b_range[1];
    range_max = ( y_range.y - b_min[1] ) / b_range[1];
    horizontalSlider_overall_pos_ymin->setRange( 0, 10000 );
    horizontalSlider_overall_pos_ymin->setValue( (int)( range_min * 10000 ) );
    horizontalSlider_overall_pos_ymax->setRange( 0, 10000 );
    horizontalSlider_overall_pos_ymax->setValue( (int)( range_max * 10000 ) );
    horizontalSlider_overall_pos_ymin->setTracking( true );
    horizontalSlider_overall_pos_ymax->setTracking( true );
    // Overall position filter Z
    // horizontalSlider_overall_pos_zmin->setInvertedAppearance(true);
    range_min = ( z_range.x - b_min[2] ) / b_range[2];
    range_max = ( z_range.y - b_min[2] ) / b_range[2];
    horizontalSlider_overall_pos_zmin->setRange( 0, 10000 );
    horizontalSlider_overall_pos_zmin->setValue( (int)( range_min * 10000 ) );
    horizontalSlider_overall_pos_zmax->setRange( 0, 10000 );
    horizontalSlider_overall_pos_zmax->setValue( (int)( range_max * 10000 ) );
    horizontalSlider_overall_pos_zmin->setTracking( true );
    horizontalSlider_overall_pos_zmax->setTracking( true );

    // Cells filter
    auto cell_ranges = spinWidget->getCellFilter();
    int n_cells[3];
    Geometry_Get_N_Cells( state.get(), n_cells );
    horizontalSlider_cell_a_min->setRange( 1, n_cells[0] );
    horizontalSlider_cell_a_max->setRange( 1, n_cells[0] );
    horizontalSlider_cell_b_min->setRange( 1, n_cells[1] );
    horizontalSlider_cell_b_max->setRange( 1, n_cells[1] );
    horizontalSlider_cell_c_min->setRange( 1, n_cells[2] );
    horizontalSlider_cell_c_max->setRange( 1, n_cells[2] );

    horizontalSlider_cell_a_min->setValue( cell_ranges[0] + 1 );
    horizontalSlider_cell_a_max->setValue( cell_ranges[1] + 1 );
    horizontalSlider_cell_b_min->setValue( cell_ranges[2] + 1 );
    horizontalSlider_cell_b_max->setValue( cell_ranges[3] + 1 );
    horizontalSlider_cell_c_min->setValue( cell_ranges[4] + 1 );
    horizontalSlider_cell_c_max->setValue( cell_ranges[5] + 1 );

    lineEdit_cell_a_min->setText( QString::number( cell_ranges[0] + 1 ) );
    lineEdit_cell_a_max->setText( QString::number( cell_ranges[1] + 1 ) );
    lineEdit_cell_b_min->setText( QString::number( cell_ranges[2] + 1 ) );
    lineEdit_cell_b_max->setText( QString::number( cell_ranges[3] + 1 ) );
    lineEdit_cell_c_min->setText( QString::number( cell_ranges[4] + 1 ) );
    lineEdit_cell_c_max->setText( QString::number( cell_ranges[5] + 1 ) );
    checkBox_cell_a->setChecked( true );
    checkBox_cell_b->setChecked( true );
    checkBox_cell_c->setChecked( true );

    scalar bounds_min[3], bounds_max[3];
    Geometry_Get_Bounds( state.get(), bounds_min, bounds_max );
    glm::vec3 sys_size{ bounds_max[0] - bounds_min[0], bounds_max[1] - bounds_min[1], bounds_max[2] - bounds_min[2] };
    horizontalSlider_surface_xmin->blockSignals( true );
    horizontalSlider_surface_xmax->blockSignals( true );
    horizontalSlider_surface_ymin->blockSignals( true );
    horizontalSlider_surface_ymax->blockSignals( true );
    horizontalSlider_surface_zmin->blockSignals( true );
    horizontalSlider_surface_zmax->blockSignals( true );
    // X Range Surface
    auto surface_x_range = spinWidget->surfaceXRange();
    horizontalSlider_surface_xmin->setRange( 1, 99999 );
    horizontalSlider_surface_xmin->setValue( (int)( surface_x_range[0] / sys_size[0] * 100000 ) );
    horizontalSlider_surface_xmax->setRange( 1, 99999 );
    horizontalSlider_surface_xmax->setValue( (int)( surface_x_range[1] / sys_size[0] * 100000 ) );
    horizontalSlider_surface_xmin->setTracking( true );
    horizontalSlider_surface_xmax->setTracking( true );
    // Y Range Surface
    auto surface_y_range = spinWidget->surfaceYRange();
    horizontalSlider_surface_ymin->setRange( 1, 99999 );
    horizontalSlider_surface_ymin->setValue( (int)( surface_y_range[0] / sys_size[1] * 100000 ) );
    horizontalSlider_surface_ymax->setRange( 1, 99999 );
    horizontalSlider_surface_ymax->setValue( (int)( surface_y_range[1] / sys_size[1] * 100000 ) );
    horizontalSlider_surface_ymin->setTracking( true );
    horizontalSlider_surface_ymax->setTracking( true );
    // Z Range Surface
    auto surface_z_range = spinWidget->surfaceZRange();
    horizontalSlider_surface_zmin->setRange( 1, 99999 );
    horizontalSlider_surface_zmin->setValue( (int)( surface_z_range[0] / sys_size[2] * 100000 ) );
    horizontalSlider_surface_zmax->setRange( 1, 99999 );
    horizontalSlider_surface_zmax->setValue( (int)( surface_z_range[1] / sys_size[2] * 100000 ) );
    horizontalSlider_surface_zmin->setTracking( true );
    horizontalSlider_surface_zmax->setTracking( true );
    horizontalSlider_surface_xmin->blockSignals( false );
    horizontalSlider_surface_xmax->blockSignals( false );
    horizontalSlider_surface_ymin->blockSignals( false );
    horizontalSlider_surface_ymax->blockSignals( false );
    horizontalSlider_surface_zmin->blockSignals( false );

    // Colormaps
    int idx_cmg = (int)spinWidget->colormap_general();
    comboBox_colormap_general->setCurrentIndex( idx_cmg );
    int idx_cma = (int)spinWidget->colormap_arrows();
    if( idx_cma == idx_cmg )
        idx_cma = -1;
    comboBox_colormap_arrows->setCurrentIndex( idx_cma + 1 );
    float cm_rotation       = spinWidget->colormap_rotation();
    auto cm_inverted        = spinWidget->colormap_inverted();
    glm::vec3 cm_cardinal_c = spinWidget->colormap_cardinal_c();
    lineEdit_cm_axis_x->setText( QString::number( cm_cardinal_c.x ) );
    lineEdit_cm_axis_y->setText( QString::number( cm_cardinal_c.y ) );
    lineEdit_cm_axis_z->setText( QString::number( cm_cardinal_c.z ) );
    horizontalSlider_colormap_rotate_phi->setRange( 0, 360 );
    horizontalSlider_colormap_rotate_phi->setValue( cm_rotation );
    lineEdit_colormap_rotate_phi->setText( QString::number( cm_rotation ) );
    checkBox_colormap_invert_z->setChecked( cm_inverted[0] );
    checkBox_colormap_invert_xy->setChecked( cm_inverted[1] );

    // Perspective / FOV
    if( spinWidget->cameraProjection() )
        radioButton_perspectiveProjection->setChecked( true );
    else
        radioButton_orthographicProjection->setChecked( true );
    this->horizontalSlider_camera_fov->setRange( 0, 160 );
    this->lineEdit_camera_fov->setText( QString::number( spinWidget->verticalFieldOfView() ) );
    this->horizontalSlider_camera_fov->setValue( (int)( spinWidget->verticalFieldOfView() ) );

    // Arrows: size and lod
    this->horizontalSlider_arrowsize->setRange( 0, 20 );
    float logs = std::log10( spinWidget->arrowSize() );
    this->horizontalSlider_arrowsize->setValue( (int)( ( logs + 1 ) * 10 ) );
    this->lineEdit_arrows_lod->setText( QString::number( spinWidget->arrowLOD() ) );

    // Sphere
    this->horizontalSlider_spherePointSize->setRange( 1, 10 );
    this->horizontalSlider_spherePointSize->setValue( (int)spinWidget->spherePointSizeRange().y );

    // Light
    this->horizontalSlider_light_theta->setRange( 0, 180 );
    this->horizontalSlider_light_phi->setRange( 0, 360 );

    // Bounding Box
    // checkBox_showBoundingBox->setChecked(spinWidget->isBoundingBoxEnabled());

    // Background
    int idx_bg = (int)spinWidget->backgroundColor();
    comboBox_backgroundColor->setCurrentIndex( idx_bg );

    // Camera
    this->read_camera();

    // Light
    auto angles = this->spinWidget->getLightPosition();
    this->horizontalSlider_light_theta->setValue( (int)angles[0] );
    this->horizontalSlider_light_phi->setValue( (int)angles[1] );
}

// -----------------------------------------------------------------------------------
// --------------------- Visualization -----------------------------------------------
// -----------------------------------------------------------------------------------
void VisualisationSettingsWidget::set_visualisation_source()
{
    this->spinWidget->setVisualisationSource( this->comboBox_VisualisationSource->currentIndex() );
}

void VisualisationSettingsWidget::set_visualisation_n_cell_steps()
{
    // N_cell_steps (draw every N'th unit cell)
    this->spinWidget->setVisualisationNCellSteps( this->spinBox_n_cell_steps->value() );
}

void VisualisationSettingsWidget::set_visualization_mode()
{
    SpinWidget::VisualizationMode mode;

    if( this->radioButton_vismode_sphere->isChecked() )
        mode = SpinWidget::VisualizationMode::SPHERE;
    else
        mode = SpinWidget::VisualizationMode::SYSTEM;

    this->spinWidget->setVisualizationMode( mode );
}

void VisualisationSettingsWidget::set_visualization_perspective()
{
    // Perspective / FOV
    if( radioButton_orthographicProjection->isChecked() )
    {
        spinWidget->setCameraProjection( false );
    }
    else
    {
        spinWidget->setCameraProjection( true );
    }
}

void VisualisationSettingsWidget::set_visualization_miniview()
{
    bool miniview;
    SpinWidget::WidgetLocation pos;

    miniview = this->checkBox_showMiniView->isChecked();
    if( this->comboBox_miniViewPosition->currentText() == "Bottom Left" )
    {
        pos = SpinWidget::WidgetLocation::BOTTOM_LEFT;
    }
    else if( this->comboBox_miniViewPosition->currentText() == "Bottom Right" )
    {
        pos = SpinWidget::WidgetLocation::BOTTOM_RIGHT;
    }
    else if( this->comboBox_miniViewPosition->currentText() == "Top Left" )
    {
        pos = SpinWidget::WidgetLocation::TOP_LEFT;
    }
    else if( this->comboBox_miniViewPosition->currentText() == "Top Right" )
    {
        pos = SpinWidget::WidgetLocation::TOP_RIGHT;
    }

    this->spinWidget->setVisualizationMiniview( miniview, pos );
}

void VisualisationSettingsWidget::set_visualization_coordinatesystem()
{
    bool coordinatesystem;
    SpinWidget::WidgetLocation pos;

    coordinatesystem = this->checkBox_showCoordinateSystem->isChecked();
    if( this->comboBox_coordinateSystemPosition->currentText() == "Bottom Left" )
    {
        pos = SpinWidget::WidgetLocation::BOTTOM_LEFT;
    }
    else if( this->comboBox_coordinateSystemPosition->currentText() == "Bottom Right" )
    {
        pos = SpinWidget::WidgetLocation::BOTTOM_RIGHT;
    }
    else if( this->comboBox_coordinateSystemPosition->currentText() == "Top Left" )
    {
        pos = SpinWidget::WidgetLocation::TOP_LEFT;
    }
    else if( this->comboBox_coordinateSystemPosition->currentText() == "Top Right" )
    {
        pos = SpinWidget::WidgetLocation::TOP_RIGHT;
    }

    this->spinWidget->setVisualizationCoordinatesystem( coordinatesystem, pos );
}

void VisualisationSettingsWidget::set_visualization_system()
{
    bool arrows, boundingbox, surface, isosurface;

    arrows      = this->checkBox_show_arrows->isChecked();
    boundingbox = this->checkBox_showBoundingBox->isChecked();
    surface     = this->checkBox_show_surface->isChecked();
    isosurface  = this->checkBox_show_isosurface->isChecked();

    this->spinWidget->enableSystem( arrows, boundingbox, surface, isosurface );
}

void VisualisationSettingsWidget::set_visualization_system_arrows()
{
    float exponent  = horizontalSlider_arrowsize->value() / 10.0f - 1.0f;
    float arrowsize = std::pow( 10.0f, exponent );
    int arrowlod    = lineEdit_arrows_lod->text().toInt();
    this->spinWidget->setArrows( arrowsize, arrowlod );
}
void VisualisationSettingsWidget::set_visualization_system_boundingbox() {}
void VisualisationSettingsWidget::set_visualization_system_surface()
{
    scalar bounds_min[3], bounds_max[3];
    Geometry_Get_Bounds( state.get(), bounds_min, bounds_max );
    float s_min, s_max;

    // X
    s_min = horizontalSlider_surface_xmin->value();
    s_max = horizontalSlider_surface_xmax->value();
    if( s_min > s_max )
    {
        float t = s_min;
        s_min   = s_max;
        s_max   = t;
    }
    horizontalSlider_surface_xmin->blockSignals( true );
    horizontalSlider_surface_xmax->blockSignals( true );
    horizontalSlider_surface_xmin->setValue( (int)( s_min ) );
    horizontalSlider_surface_xmax->setValue( (int)( s_max ) );
    horizontalSlider_surface_xmin->blockSignals( false );
    horizontalSlider_surface_xmax->blockSignals( false );
    float x_min = bounds_min[0] + ( s_min / 100000.0 ) * ( bounds_max[0] - bounds_min[0] );
    float x_max = bounds_min[0] + ( s_max / 100000.0 ) * ( bounds_max[0] - bounds_min[0] );
    // Y
    s_min = horizontalSlider_surface_ymin->value();
    s_max = horizontalSlider_surface_ymax->value();
    if( s_min > s_max )
    {
        float t = s_min;
        s_min   = s_max;
        s_max   = t;
    }
    horizontalSlider_surface_ymin->blockSignals( true );
    horizontalSlider_surface_ymax->blockSignals( true );
    horizontalSlider_surface_ymin->setValue( (int)( s_min ) );
    horizontalSlider_surface_ymax->setValue( (int)( s_max ) );
    horizontalSlider_surface_ymin->blockSignals( false );
    horizontalSlider_surface_ymax->blockSignals( false );
    float y_min = bounds_min[1] + ( s_min / 100000.0 ) * ( bounds_max[1] - bounds_min[1] );
    float y_max = bounds_min[1] + ( s_max / 100000.0 ) * ( bounds_max[1] - bounds_min[1] );
    // Z
    s_min = horizontalSlider_surface_zmin->value();
    s_max = horizontalSlider_surface_zmax->value();
    if( s_min > s_max )
    {
        float t = s_min;
        s_min   = s_max;
        s_max   = t;
    }
    horizontalSlider_surface_zmin->blockSignals( true );
    horizontalSlider_surface_zmax->blockSignals( true );
    horizontalSlider_surface_zmin->setValue( (int)( s_min ) );
    horizontalSlider_surface_zmax->setValue( (int)( s_max ) );
    horizontalSlider_surface_zmin->blockSignals( false );
    horizontalSlider_surface_zmax->blockSignals( false );
    float z_min = bounds_min[2] + ( s_min / 100000.0 ) * ( bounds_max[2] - bounds_min[2] );
    float z_max = bounds_min[2] + ( s_max / 100000.0 ) * ( bounds_max[2] - bounds_min[2] );

    // Set
    glm::vec2 x_range( x_min, x_max );
    glm::vec2 y_range( y_min, y_max );
    glm::vec2 z_range( z_min, z_max );
    spinWidget->setSurface( x_range, y_range, z_range );
}

void VisualisationSettingsWidget::set_visualization_system_overall_direction_line_edits()
{
    float range_xmin = lineEdit_overall_dir_xmin->text().toFloat();
    float range_xmax = lineEdit_overall_dir_xmax->text().toFloat();
    float range_ymin = lineEdit_overall_dir_ymin->text().toFloat();
    float range_ymax = lineEdit_overall_dir_ymax->text().toFloat();
    float range_zmin = lineEdit_overall_dir_zmin->text().toFloat();
    float range_zmax = lineEdit_overall_dir_zmax->text().toFloat();
    horizontalSlider_overall_dir_xmin->setValue( -100 * range_xmax );
    horizontalSlider_overall_dir_xmax->setValue( 100 * range_xmin );
    horizontalSlider_overall_dir_ymin->setValue( -100 * range_ymin );
    horizontalSlider_overall_dir_ymax->setValue( 100 * range_ymax );
    horizontalSlider_overall_dir_zmin->setValue( -100 * range_zmin );
    horizontalSlider_overall_dir_zmax->setValue( 100 * range_zmax );
    set_visualization_system_overall_direction(
        range_xmin, range_xmax, range_ymin, range_ymax, range_zmin, range_zmax );
}

void VisualisationSettingsWidget::set_visualization_system_overall_direction_sliders()
{
    float range_xmin = -horizontalSlider_overall_dir_xmin->value() / 100.0;
    float range_xmax = horizontalSlider_overall_dir_xmax->value() / 100.0;
    float range_ymin = -horizontalSlider_overall_dir_ymin->value() / 100.0;
    float range_ymax = horizontalSlider_overall_dir_ymax->value() / 100.0;
    float range_zmin = -horizontalSlider_overall_dir_zmin->value() / 100.0;
    float range_zmax = horizontalSlider_overall_dir_zmax->value() / 100.0;
    lineEdit_overall_dir_xmin->setText( QString::number( range_xmin ) );
    lineEdit_overall_dir_xmax->setText( QString::number( range_xmax ) );
    lineEdit_overall_dir_ymin->setText( QString::number( range_ymin ) );
    lineEdit_overall_dir_ymax->setText( QString::number( range_ymax ) );
    lineEdit_overall_dir_zmin->setText( QString::number( range_zmin ) );
    lineEdit_overall_dir_zmax->setText( QString::number( range_zmax ) );
    set_visualization_system_overall_direction(
        range_xmin, range_xmax, range_ymin, range_ymax, range_zmin, range_zmax );
}

void VisualisationSettingsWidget::set_visualization_system_overall_direction(
    float range_xmin, float range_xmax, float range_ymin, float range_ymax, float range_zmin, float range_zmax )
{
    // X
    float range_min;
    float range_max;

    range_min = range_xmin;
    range_max = range_xmax;

    if( range_min > range_max )
    {
        float t   = range_min;
        range_min = range_max;
        range_max = t;
    }
    horizontalSlider_overall_dir_xmin->blockSignals( true );
    horizontalSlider_overall_dir_xmax->blockSignals( true );
    horizontalSlider_overall_dir_xmin->setValue( (int)( -range_min * 100 ) );
    horizontalSlider_overall_dir_xmax->setValue( (int)( range_max * 100 ) );
    horizontalSlider_overall_dir_xmin->blockSignals( false );
    horizontalSlider_overall_dir_xmax->blockSignals( false );
    glm::vec2 x_range( range_min, range_max );

    // Y
    range_min = range_ymin;
    range_max = range_ymax;
    if( range_min > range_max )
    {
        float t   = range_min;
        range_min = range_max;
        range_max = t;
    }
    horizontalSlider_overall_dir_ymin->blockSignals( true );
    horizontalSlider_overall_dir_ymax->blockSignals( true );
    horizontalSlider_overall_dir_ymin->setValue( (int)( -range_min * 100 ) );
    horizontalSlider_overall_dir_ymax->setValue( (int)( range_max * 100 ) );
    horizontalSlider_overall_dir_ymin->blockSignals( false );
    horizontalSlider_overall_dir_ymax->blockSignals( false );
    glm::vec2 y_range( range_min, range_max );

    // Z
    range_min = range_zmin;
    range_max = range_zmax;

    if( range_min > range_max )
    {
        float t   = range_min;
        range_min = range_max;
        range_max = t;
    }
    horizontalSlider_overall_dir_zmin->blockSignals( true );
    horizontalSlider_overall_dir_zmax->blockSignals( true );
    horizontalSlider_overall_dir_zmin->setValue( (int)( -range_min * 100 ) );
    horizontalSlider_overall_dir_zmax->setValue( (int)( range_max * 100 ) );
    horizontalSlider_overall_dir_zmin->blockSignals( false );
    horizontalSlider_overall_dir_zmax->blockSignals( false );
    glm::vec2 z_range( range_min, range_max );

    spinWidget->setOverallDirectionRange( x_range, y_range, z_range );
}

void VisualisationSettingsWidget::set_visualization_system_overall_position_line_edits()
{
    float range_xmax = lineEdit_overall_pos_xmax->text().toFloat();
    float range_xmin = lineEdit_overall_pos_xmin->text().toFloat();
    float range_ymin = lineEdit_overall_pos_ymin->text().toFloat();
    float range_ymax = lineEdit_overall_pos_ymax->text().toFloat();
    float range_zmin = lineEdit_overall_pos_zmin->text().toFloat();
    float range_zmax = lineEdit_overall_pos_zmax->text().toFloat();
    horizontalSlider_overall_pos_xmin->setValue( 10000 * range_xmin );
    horizontalSlider_overall_pos_xmax->setValue( 10000 * range_xmax );
    horizontalSlider_overall_pos_ymin->setValue( 10000 * range_ymin );
    horizontalSlider_overall_pos_ymax->setValue( 10000 * range_ymax );
    horizontalSlider_overall_pos_zmin->setValue( 10000 * range_zmin );
    horizontalSlider_overall_pos_zmax->setValue( 10000 * range_zmax );
    set_visualization_system_overall_position( range_xmin, range_xmax, range_ymin, range_ymax, range_zmin, range_zmax );
}

void VisualisationSettingsWidget::set_visualization_system_overall_position_sliders()
{
    float range_xmin = horizontalSlider_overall_pos_xmin->value() / 10000.0;
    float range_xmax = horizontalSlider_overall_pos_xmax->value() / 10000.0;
    float range_ymin = horizontalSlider_overall_pos_ymin->value() / 10000.0;
    float range_ymax = horizontalSlider_overall_pos_ymax->value() / 10000.0;
    float range_zmin = horizontalSlider_overall_pos_zmin->value() / 10000.0;
    float range_zmax = horizontalSlider_overall_pos_zmax->value() / 10000.0;
    lineEdit_overall_pos_xmin->setText( QString::number( range_xmin ) );
    lineEdit_overall_pos_xmax->setText( QString::number( range_xmax ) );
    lineEdit_overall_pos_ymin->setText( QString::number( range_ymin ) );
    lineEdit_overall_pos_ymax->setText( QString::number( range_ymax ) );
    lineEdit_overall_pos_zmin->setText( QString::number( range_zmin ) );
    lineEdit_overall_pos_zmax->setText( QString::number( range_zmax ) );
    set_visualization_system_overall_position( range_xmin, range_xmax, range_ymin, range_ymax, range_zmin, range_zmax );
}

void VisualisationSettingsWidget::set_visualization_system_overall_position(
    float range_xmin, float range_xmax, float range_ymin, float range_ymax, float range_zmin, float range_zmax )
{
    scalar b_min[3], b_max[3], b_range[3];
    Geometry_Get_Bounds( state.get(), b_min, b_max );
    for( int dim = 0; dim < 3; ++dim )
        b_range[dim] = b_max[dim] - b_min[dim];
    float range_min;
    float range_max;
    // X
    range_min = range_xmin;
    range_max = range_xmax;
    if( range_min > range_max )
    {
        float t   = range_min;
        range_min = range_max;
        range_max = t;
    }
    horizontalSlider_overall_pos_xmin->blockSignals( true );
    horizontalSlider_overall_pos_xmax->blockSignals( true );
    horizontalSlider_overall_pos_xmin->setValue( (int)( range_min * 10000 ) );
    horizontalSlider_overall_pos_xmax->setValue( (int)( range_max * 10000 ) );
    horizontalSlider_overall_pos_xmin->blockSignals( false );
    horizontalSlider_overall_pos_xmax->blockSignals( false );
    glm::vec2 x_range( b_min[0] + range_min * b_range[0], b_min[0] + range_max * b_range[0] );

    // Y
    range_min = range_ymin;
    range_max = range_ymax;
    if( range_min > range_max )
    {
        float t   = range_min;
        range_min = range_max;
        range_max = t;
    }
    horizontalSlider_overall_pos_ymin->blockSignals( true );
    horizontalSlider_overall_pos_ymax->blockSignals( true );
    horizontalSlider_overall_pos_ymin->setValue( (int)( range_min * 10000 ) );
    horizontalSlider_overall_pos_ymax->setValue( (int)( range_max * 10000 ) );
    horizontalSlider_overall_pos_ymin->blockSignals( false );
    horizontalSlider_overall_pos_ymax->blockSignals( false );
    glm::vec2 y_range( b_min[1] + range_min * b_range[1], b_min[1] + range_max * b_range[1] );

    // Z
    range_min = range_zmin;
    range_max = range_zmax;
    if( range_min > range_max )
    {
        float t   = range_min;
        range_min = range_max;
        range_max = t;
    }
    horizontalSlider_overall_pos_zmin->blockSignals( true );
    horizontalSlider_overall_pos_zmax->blockSignals( true );
    horizontalSlider_overall_pos_zmin->setValue( (int)( range_min * 10000 ) );
    horizontalSlider_overall_pos_zmax->setValue( (int)( range_max * 10000 ) );
    horizontalSlider_overall_pos_zmin->blockSignals( false );
    horizontalSlider_overall_pos_zmax->blockSignals( false );
    glm::vec2 z_range( b_min[2] + range_min * b_range[2], b_min[2] + range_max * b_range[2] );

    spinWidget->setOverallPositionRange( x_range, y_range, z_range );
}

void VisualisationSettingsWidget::set_visualization_system_cells_line_edits()
{
    int n_cells[3];
    Geometry_Get_N_Cells( state.get(), n_cells );
    std::array<QLineEdit *, 6> lineEdits = { lineEdit_cell_a_min, lineEdit_cell_a_max, lineEdit_cell_b_min,
                                             lineEdit_cell_b_max, lineEdit_cell_c_min, lineEdit_cell_c_max };
    std::array<QSlider *, 6> horizontalSliders
        = { horizontalSlider_cell_a_min, horizontalSlider_cell_a_max, horizontalSlider_cell_b_min,
            horizontalSlider_cell_b_max, horizontalSlider_cell_c_min, horizontalSlider_cell_c_max };
    std::array<int, 6> cell;
    for( int i = 0; i < 6; i++ )
    {
        cell[i] = lineEdits[i]->text().toFloat();
        if( cell[i] > n_cells[i / 2] )
            lineEdits[i]->setText( QString::number( n_cells[i / 2] ) );
        else if( cell[i] < 1 )
            lineEdits[i]->setText( QString::number( 1 ) );
        horizontalSliders[i]->setValue( cell[i] ); // Note: This invokes set_visualization_system_cells_sliders()
    }
}

void VisualisationSettingsWidget::set_visualization_system_cells_sliders()
{
    int n_cells[3];
    Geometry_Get_N_Cells( state.get(), n_cells );
    std::array<QLineEdit *, 6> lineEdits = { lineEdit_cell_a_min, lineEdit_cell_a_max, lineEdit_cell_b_min,
                                             lineEdit_cell_b_max, lineEdit_cell_c_min, lineEdit_cell_c_max };
    std::array<QSlider *, 6> horizontalSliders
        = { horizontalSlider_cell_a_min, horizontalSlider_cell_a_max, horizontalSlider_cell_b_min,
            horizontalSlider_cell_b_max, horizontalSlider_cell_c_min, horizontalSlider_cell_c_max };
    std::array<QCheckBox *, 3> checkBoxes = { checkBox_cell_a, checkBox_cell_b, checkBox_cell_c };
    std::array<int, 6> cell               = {};

    for( int i = 0; i < 3; i++ )
    {
        cell[2 * i] = horizontalSliders[2 * i]->value();
        if( !checkBoxes[i]->isChecked() )
        {
            cell[2 * i + 1] = cell[2 * i];
            horizontalSliders[2 * i + 1]->blockSignals( true );
            horizontalSliders[2 * i + 1]->setValue( cell[2 * i] );
            horizontalSliders[2 * i + 1]->blockSignals( false );
        }
        else
        {
            cell[2 * i + 1] = horizontalSliders[2 * i + 1]->value();
        }
        if( cell[2 * i] > cell[2 * i + 1] )
        {
            auto temp       = cell[2 * i + 1];
            cell[2 * i + 1] = cell[2 * i];
            cell[2 * i]     = temp;
            horizontalSliders[2 * i]->blockSignals( true );
            horizontalSliders[2 * i + 1]->blockSignals( true );

            horizontalSliders[2 * i]->setValue( cell[2 * i] );
            horizontalSliders[2 * i + 1]->setValue( cell[2 * i + 1] );

            horizontalSliders[2 * i]->blockSignals( false );
            horizontalSliders[2 * i + 1]->blockSignals( false );
        }
        lineEdits[2 * i]->setText( QString::number( cell[2 * i] ) );
        lineEdits[2 * i + 1]->setText( QString::number( cell[2 * i + 1] ) );
    }
    spinWidget->setCellFilter( cell[0] - 1, cell[1] - 1, cell[2] - 1, cell[3] - 1, cell[4] - 1, cell[5] - 1 );
    spinWidget->updateData();
}

void VisualisationSettingsWidget::reset_visualization_system_overall_direction()
{
    std::array<QSlider *, 6> horizontalSliders
        = { horizontalSlider_overall_dir_xmin, horizontalSlider_overall_dir_xmax, horizontalSlider_overall_dir_ymin,
            horizontalSlider_overall_dir_ymax, horizontalSlider_overall_dir_zmin, horizontalSlider_overall_dir_zmax };
    for( int i = 0; i < 3; i++ )
    {
        horizontalSliders[2 * i]->blockSignals( true );
        horizontalSliders[2 * i + 1]->blockSignals( true );
        horizontalSliders[2 * i]->setValue( 100 );
        horizontalSliders[2 * i + 1]->setValue( 100 );
        horizontalSliders[2 * i]->blockSignals( false );
        horizontalSliders[2 * i + 1]->blockSignals( false );
    }
    set_visualization_system_overall_direction_sliders();
}

void VisualisationSettingsWidget::reset_visualization_system_overall_position()
{
    std::array<QSlider *, 6> horizontalSliders
        = { horizontalSlider_overall_pos_xmin, horizontalSlider_overall_pos_xmax, horizontalSlider_overall_pos_ymin,
            horizontalSlider_overall_pos_ymax, horizontalSlider_overall_pos_zmin, horizontalSlider_overall_pos_zmax };
    for( int i = 0; i < 3; i++ )
    {
        horizontalSliders[2 * i]->blockSignals( true );
        horizontalSliders[2 * i + 1]->blockSignals( true );
        horizontalSliders[2 * i]->setValue( 0 );
        horizontalSliders[2 * i + 1]->setValue( 10000 );
        horizontalSliders[2 * i]->blockSignals( false );
        horizontalSliders[2 * i + 1]->blockSignals( false );
    }
    set_visualization_system_overall_position_sliders();
}

void VisualisationSettingsWidget::reset_visualization_system_cells()
{
    std::array<QLineEdit *, 6> lineEdits = { lineEdit_cell_a_min, lineEdit_cell_a_max, lineEdit_cell_b_min,
                                             lineEdit_cell_b_max, lineEdit_cell_c_min, lineEdit_cell_c_max };
    std::array<QSlider *, 6> horizontalSliders
        = { horizontalSlider_cell_a_min, horizontalSlider_cell_a_max, horizontalSlider_cell_b_min,
            horizontalSlider_cell_b_max, horizontalSlider_cell_c_min, horizontalSlider_cell_c_max };
    std::array<QCheckBox *, 3> checkBoxes = { checkBox_cell_a, checkBox_cell_b, checkBox_cell_c };

    std::array<int, 3> n_cells;
    Geometry_Get_N_Cells( state.get(), n_cells.data() );

    for( int i = 0; i < 3; i++ )
    {
        checkBoxes[i]->blockSignals( true );
        checkBoxes[i]->setChecked( true );
        checkBoxes[i]->blockSignals( false );
        horizontalSliders[2 * i + 1]->setEnabled( true );
        lineEdits[2 * i + 1]->setEnabled( true );

        horizontalSliders[2 * i]->blockSignals( true );
        horizontalSliders[2 * i + 1]->blockSignals( true );
        horizontalSliders[2 * i]->setValue( 1 );
        horizontalSliders[2 * i + 1]->setValue( n_cells[i] );
        horizontalSliders[2 * i]->blockSignals( false );
        horizontalSliders[2 * i + 1]->blockSignals( false );
    }
    set_visualization_system_cells_sliders();
}

void VisualisationSettingsWidget::cell_on_checkbox()
{
    if( checkBox_cell_a->isChecked() )
    {
        horizontalSlider_cell_a_max->setEnabled( true );
        lineEdit_cell_a_max->setEnabled( true );
    }
    else
    {
        horizontalSlider_cell_a_max->setEnabled( false );
        lineEdit_cell_a_max->setEnabled( false );
    }
    if( checkBox_cell_b->isChecked() )
    {
        horizontalSlider_cell_b_max->setEnabled( true );
        lineEdit_cell_b_max->setEnabled( true );
    }
    else
    {
        horizontalSlider_cell_b_max->setEnabled( false );
        lineEdit_cell_b_max->setEnabled( false );
    }
    if( checkBox_cell_c->isChecked() )
    {
        horizontalSlider_cell_c_max->setEnabled( true );
        lineEdit_cell_c_max->setEnabled( true );
    }
    else
    {
        horizontalSlider_cell_c_max->setEnabled( false );
        lineEdit_cell_c_max->setEnabled( false );
    }
    set_visualization_system_cells_sliders();
}

void VisualisationSettingsWidget::set_visualization_system_isosurface()
{
    this->m_isosurfaceshadows = this->checkBox_isosurfaceshadows->isChecked();
    for( auto & isoWidget : this->isosurfaceWidgets )
        isoWidget->setDrawShadows( this->m_isosurfaceshadows );
}

void VisualisationSettingsWidget::add_isosurface()
{
    IsosurfaceWidget * iso = new IsosurfaceWidget( state, spinWidget );
    connect( iso, SIGNAL( closedSignal() ), this, SLOT( update_isosurfaces() ) );
    iso->setDrawShadows( this->m_isosurfaceshadows );
    this->isosurfaceWidgets.push_back( iso );
    this->verticalLayout_isosurface->addWidget( isosurfaceWidgets.back() );
    // this->set_visualization_system();
}

void VisualisationSettingsWidget::update_isosurfaces()
{
    // std::cerr << "........................" << std::endl;
    QObject * obj = sender();

    for( unsigned int i = 0; i < this->isosurfaceWidgets.size(); ++i )
    {
        if( this->isosurfaceWidgets[i] == obj )
            this->isosurfaceWidgets.erase( this->isosurfaceWidgets.begin() + i );
        else
            int x = 0;
    }
}

void VisualisationSettingsWidget::set_visualization_sphere()
{
    // This function does not make any sense, does it?
    // Only possibility: draw/dont draw the sphere, only draw the points
}
void VisualisationSettingsWidget::set_visualization_sphere_pointsize()
{
    this->spinWidget->setSpherePointSizeRange( { 0.2, this->horizontalSlider_spherePointSize->value() } );
}

void VisualisationSettingsWidget::set_visualization_colormap()
{
    SpinWidget::Colormap colormap_general = SpinWidget::Colormap::HSV;
    if( comboBox_colormap_general->currentText() == "HSV, no z-component" )
        colormap_general = SpinWidget::Colormap::HSV_NO_Z;
    if( comboBox_colormap_general->currentText() == "Z-Component: Blue-Red" )
        colormap_general = SpinWidget::Colormap::BLUE_RED;
    if( comboBox_colormap_general->currentText() == "Z-Component: Blue-Green-Red" )
        colormap_general = SpinWidget::Colormap::BLUE_GREEN_RED;
    if( comboBox_colormap_general->currentText() == "Z-Component: Blue-White-Red" )
        colormap_general = SpinWidget::Colormap::BLUE_WHITE_RED;
    if( comboBox_colormap_general->currentText() == "White" )
        colormap_general = SpinWidget::Colormap::WHITE;
    if( comboBox_colormap_general->currentText() == "Gray" )
        colormap_general = SpinWidget::Colormap::GRAY;
    if( comboBox_colormap_general->currentText() == "Black" )
        colormap_general = SpinWidget::Colormap::BLACK;

    spinWidget->setColormapGeneral( colormap_general );

    SpinWidget::Colormap colormap_arrows = SpinWidget::Colormap::HSV;

    if( comboBox_colormap_arrows->currentText() == "Same as General" )
        colormap_arrows = colormap_general;
    if( comboBox_colormap_arrows->currentText() == "HSV, no z-component" )
        colormap_arrows = SpinWidget::Colormap::HSV_NO_Z;
    if( comboBox_colormap_arrows->currentText() == "Z-Component: Blue-Red" )
        colormap_arrows = SpinWidget::Colormap::BLUE_RED;
    if( comboBox_colormap_arrows->currentText() == "Z-Component: Blue-Green-Red" )
        colormap_arrows = SpinWidget::Colormap::BLUE_GREEN_RED;
    if( comboBox_colormap_arrows->currentText() == "Z-Component: Blue-White-Red" )
        colormap_arrows = SpinWidget::Colormap::BLUE_WHITE_RED;
    if( comboBox_colormap_arrows->currentText() == "White" )
        colormap_arrows = SpinWidget::Colormap::WHITE;
    if( comboBox_colormap_arrows->currentText() == "Gray" )
        colormap_arrows = SpinWidget::Colormap::GRAY;
    if( comboBox_colormap_arrows->currentText() == "Black" )
        colormap_arrows = SpinWidget::Colormap::BLACK;

    spinWidget->setColormapArrows( colormap_arrows );
}

void VisualisationSettingsWidget::set_visualization_colormap_rotation_slider()
{
    int phi        = this->horizontalSlider_colormap_rotate_phi->value();
    bool invert_z  = this->checkBox_colormap_invert_z->isChecked();
    bool invert_xy = this->checkBox_colormap_invert_xy->isChecked();

    this->lineEdit_colormap_rotate_phi->setText( QString::number( phi ) );

    this->set_visualization_colormap_axis();
}

void VisualisationSettingsWidget::set_visualization_colormap_rotation_lineEdit()
{
    int phi = this->lineEdit_colormap_rotate_phi->text().toInt();
    this->horizontalSlider_colormap_rotate_phi->setValue( phi );

    this->set_visualization_colormap_axis();
}

void VisualisationSettingsWidget::set_visualization_colormap_axis()
{
    const float epsilon = 1e-5;

    const glm::vec3 ex{ 1, 0, 0 };
    const glm::vec3 ey{ 0, 1, 0 };
    const glm::vec3 ez{ 0, 0, 1 };

    glm::vec3 temp1{ 1, 0, 0 };
    glm::vec3 temp2{ 0, 1, 0 };
    glm::vec3 cardinal_c{ 0, 0, 1 };
    cardinal_c.x = this->lineEdit_cm_axis_x->text().toFloat();
    cardinal_c.y = this->lineEdit_cm_axis_y->text().toFloat();
    cardinal_c.z = this->lineEdit_cm_axis_z->text().toFloat();
    if( glm::length( cardinal_c ) > 0 )
        cardinal_c = glm::normalize( cardinal_c );

    if( cardinal_c[2] == 0 )
    {
        temp1 = ez;
        temp2 = glm::cross( cardinal_c, ez );
    }
    // Else it's either above or below the xy-plane.
    //      if it's above the xy-plane, it points in z-direction
    //      the vectors should be: cardinal_c, ex, -ey
    else if( cardinal_c[2] > 0 )
    {
        temp1 = ex;
        temp2 = -ey;
    }
    //      if it's below the xy-plane, it points in -z-direction
    //      the vectors should be: cardinal_c, ex, ey
    else
    {
        temp1 = ex;
        temp2 = ey;
    }

    // First vector: orthogonalize temp1 w.r.t. cardinal_c
    glm::vec3 cardinal_a = temp1 - glm::dot( temp1, cardinal_c ) * cardinal_c;
    cardinal_a           = glm::normalize( cardinal_a );

    // Second vector: orthogonalize temp2 w.r.t. cardinal_c and cardinal_a
    glm::vec3 cardinal_b
        = temp2 - glm::dot( temp2, cardinal_c ) * cardinal_c - glm::dot( temp2, cardinal_a ) * cardinal_a;
    cardinal_b = glm::normalize( cardinal_b );

    // Set
    int phi        = this->lineEdit_colormap_rotate_phi->text().toInt();
    bool invert_z  = this->checkBox_colormap_invert_z->isChecked();
    bool invert_xy = this->checkBox_colormap_invert_xy->isChecked();
    this->spinWidget->setColormapRotationInverted( phi, invert_z, invert_xy, cardinal_a, cardinal_b, cardinal_c );
}

void VisualisationSettingsWidget::set_visualization_background()
{
    SpinWidget::Color color;
    SpinWidget::Color invcolor;
    if( comboBox_backgroundColor->currentText() == "Black" )
    {
        color    = SpinWidget::Color::BLACK;
        invcolor = SpinWidget::Color::WHITE;
    }
    else if( comboBox_backgroundColor->currentText() == "Gray" )
    {
        color    = SpinWidget::Color::GRAY;
        invcolor = SpinWidget::Color::WHITE;
    }
    else
    {
        color    = SpinWidget::Color::WHITE;
        invcolor = SpinWidget::Color::BLACK;
    }
    spinWidget->setBackgroundColor( color );
    spinWidget->setBoundingBoxColor( invcolor );
}

// -----------------------------------------------------------------------------------
// --------------------- Camera ------------------------------------------------------
// -----------------------------------------------------------------------------------

void VisualisationSettingsWidget::read_camera()
{
    auto camera_position = spinWidget->getCameraPositon();
    auto camera_focus    = spinWidget->getCameraFocus();
    auto camera_upvector = spinWidget->getCameraUpVector();

    if( camera_position != camera_position_last )
    {
        this->lineEdit_camera_pos_x->setText( QString::number( camera_position.x, 'f', 2 ) );
        this->lineEdit_camera_pos_y->setText( QString::number( camera_position.y, 'f', 2 ) );
        this->lineEdit_camera_pos_z->setText( QString::number( camera_position.z, 'f', 2 ) );
        camera_position_last = camera_position;
    }

    if( camera_focus != camera_focus_last )
    {
        this->lineEdit_camera_focus_x->setText( QString::number( camera_focus.x, 'f', 2 ) );
        this->lineEdit_camera_focus_y->setText( QString::number( camera_focus.y, 'f', 2 ) );
        this->lineEdit_camera_focus_z->setText( QString::number( camera_focus.z, 'f', 2 ) );
        camera_focus_last = camera_focus;
    }

    if( camera_upvector != camera_upvector_last )
    {
        this->lineEdit_camera_upvector_x->setText( QString::number( camera_upvector.x, 'f', 2 ) );
        this->lineEdit_camera_upvector_y->setText( QString::number( camera_upvector.y, 'f', 2 ) );
        this->lineEdit_camera_upvector_z->setText( QString::number( camera_upvector.z, 'f', 2 ) );
        camera_upvector_last = camera_upvector;
    }
}

void VisualisationSettingsWidget::save_camera()
{
    QSettings settings( "Spirit Code", "Spirit" );

    auto camera_position  = spinWidget->getCameraPositon();
    auto center_position  = spinWidget->getCameraFocus();
    auto camera_up_vector = spinWidget->getCameraUpVector();

    settings.beginGroup( "Visualisation Settings Camera" );
    // Pos
    settings.setValue( "camera position x", camera_position.x );
    settings.setValue( "camera position y", camera_position.y );
    settings.setValue( "camera position z", camera_position.z );
    // Focus
    settings.setValue( "camera focus x", center_position.x );
    settings.setValue( "camera focus y", center_position.y );
    settings.setValue( "camera focus z", center_position.z );
    // Up
    settings.setValue( "camera up vector x", camera_up_vector.x );
    settings.setValue( "camera up vector y", camera_up_vector.y );
    settings.setValue( "camera up vector z", camera_up_vector.z );
    settings.endGroup();
}

void VisualisationSettingsWidget::load_camera()
{
    glm::vec3 camera_position;
    glm::vec3 center_position;
    glm::vec3 camera_up_vector;

    QSettings settings( "Spirit Code", "Spirit" );

    settings.beginGroup( "Visualisation Settings Camera" );
    // Pos
    camera_position.x = settings.value( "camera position x", 0 ).toFloat();
    camera_position.y = settings.value( "camera position y", 0 ).toFloat();
    camera_position.z = settings.value( "camera position z", 30 ).toFloat();
    // Focus
    center_position.x = settings.value( "camera focus x", 0 ).toFloat();
    center_position.y = settings.value( "camera focus y", 0 ).toFloat();
    center_position.z = settings.value( "camera focus z", 0 ).toFloat();
    // Up
    camera_up_vector.x = settings.value( "camera up vector x", 1 ).toFloat();
    camera_up_vector.y = settings.value( "camera up vector y", 0 ).toFloat();
    camera_up_vector.z = settings.value( "camera up vector z", 0 ).toFloat();
    settings.endGroup();

    // Set the line-edits
    this->lineEdit_camera_pos_x->setText( QString::number( camera_position.x, 'f', 2 ) );
    this->lineEdit_camera_pos_y->setText( QString::number( camera_position.y, 'f', 2 ) );
    this->lineEdit_camera_pos_z->setText( QString::number( camera_position.z, 'f', 2 ) );

    this->lineEdit_camera_focus_x->setText( QString::number( center_position.x, 'f', 2 ) );
    this->lineEdit_camera_focus_y->setText( QString::number( center_position.y, 'f', 2 ) );
    this->lineEdit_camera_focus_z->setText( QString::number( center_position.z, 'f', 2 ) );

    this->lineEdit_camera_upvector_x->setText( QString::number( camera_up_vector.x, 'f', 2 ) );
    this->lineEdit_camera_upvector_y->setText( QString::number( camera_up_vector.y, 'f', 2 ) );
    this->lineEdit_camera_upvector_z->setText( QString::number( camera_up_vector.z, 'f', 2 ) );

    // Update the view
    set_camera_position();
    set_camera_focus();
    set_camera_upvector();
}

void VisualisationSettingsWidget::set_camera_position()
{
    float x = this->lineEdit_camera_pos_x->text().toFloat();
    float y = this->lineEdit_camera_pos_y->text().toFloat();
    float z = this->lineEdit_camera_pos_z->text().toFloat();
    this->spinWidget->setCameraPosition( { x, y, z } );
}

void VisualisationSettingsWidget::set_camera_focus()
{
    float x = this->lineEdit_camera_focus_x->text().toFloat();
    float y = this->lineEdit_camera_focus_y->text().toFloat();
    float z = this->lineEdit_camera_focus_z->text().toFloat();
    this->spinWidget->setCameraFocus( { x, y, z } );
}

void VisualisationSettingsWidget::set_camera_upvector()
{
    float x = this->lineEdit_camera_upvector_x->text().toFloat();
    float y = this->lineEdit_camera_upvector_y->text().toFloat();
    float z = this->lineEdit_camera_upvector_z->text().toFloat();
    this->spinWidget->setCameraUpVector( { x, y, z } );
}

void VisualisationSettingsWidget::set_camera_fov_slider()
{
    float fov = this->horizontalSlider_camera_fov->value();
    this->lineEdit_camera_fov->setText( QString::number( fov ) );
    spinWidget->setVerticalFieldOfView( fov );
}

void VisualisationSettingsWidget::set_camera_fov_lineedit()
{
    float fov = this->lineEdit_camera_fov->text().toFloat();
    horizontalSlider_camera_fov->setValue( (int)( fov ) );
    spinWidget->setVerticalFieldOfView( fov );
}

// -----------------------------------------------------------------------------------
// --------------------- Light -------------------------------------------------------
// -----------------------------------------------------------------------------------

void VisualisationSettingsWidget::set_light_position()
{
    float theta = this->horizontalSlider_light_theta->value();
    float phi   = this->horizontalSlider_light_phi->value();
    this->spinWidget->setLightPosition( theta, phi );
}

void VisualisationSettingsWidget::Setup_Visualization_Slots()
{
    connect(
        comboBox_VisualisationSource, SIGNAL( currentIndexChanged( int ) ), this, SLOT( set_visualisation_source() ) );
    connect( spinBox_n_cell_steps, SIGNAL( valueChanged( int ) ), this, SLOT( set_visualisation_n_cell_steps() ) );
    // Mode
    // connect(radioButton_vismode_sphere, SIGNAL(toggled(bool)), this,
    // SLOT(set_visualization_mode()));
    connect( radioButton_vismode_system, SIGNAL( toggled( bool ) ), this, SLOT( set_visualization_mode() ) );
    connect(
        radioButton_orthographicProjection, SIGNAL( toggled( bool ) ), this, SLOT( set_visualization_perspective() ) );
    // Miniview
    connect( checkBox_showMiniView, SIGNAL( stateChanged( int ) ), this, SLOT( set_visualization_miniview() ) );
    connect(
        comboBox_miniViewPosition, SIGNAL( currentIndexChanged( int ) ), this, SLOT( set_visualization_miniview() ) );
    // Coordinate System
    connect(
        checkBox_showCoordinateSystem, SIGNAL( stateChanged( int ) ), this,
        SLOT( set_visualization_coordinatesystem() ) );
    connect(
        comboBox_coordinateSystemPosition, SIGNAL( currentIndexChanged( int ) ), this,
        SLOT( set_visualization_coordinatesystem() ) );
    // System
    connect( checkBox_show_arrows, SIGNAL( stateChanged( int ) ), this, SLOT( set_visualization_system() ) );
    connect( checkBox_showBoundingBox, SIGNAL( stateChanged( int ) ), this, SLOT( set_visualization_system() ) );
    connect( checkBox_show_surface, SIGNAL( stateChanged( int ) ), this, SLOT( set_visualization_system() ) );
    connect( checkBox_show_isosurface, SIGNAL( stateChanged( int ) ), this, SLOT( set_visualization_system() ) );
    //      arrows
    connect(
        horizontalSlider_arrowsize, SIGNAL( valueChanged( int ) ), this, SLOT( set_visualization_system_arrows() ) );
    connect( lineEdit_arrows_lod, SIGNAL( returnPressed() ), this, SLOT( set_visualization_system_arrows() ) );
    //      bounding box
    //      surface
    connect(
        horizontalSlider_surface_xmin, SIGNAL( valueChanged( int ) ), this,
        SLOT( set_visualization_system_surface() ) );
    connect(
        horizontalSlider_surface_xmax, SIGNAL( valueChanged( int ) ), this,
        SLOT( set_visualization_system_surface() ) );
    connect(
        horizontalSlider_surface_ymin, SIGNAL( valueChanged( int ) ), this,
        SLOT( set_visualization_system_surface() ) );
    connect(
        horizontalSlider_surface_ymax, SIGNAL( valueChanged( int ) ), this,
        SLOT( set_visualization_system_surface() ) );
    connect(
        horizontalSlider_surface_zmin, SIGNAL( valueChanged( int ) ), this,
        SLOT( set_visualization_system_surface() ) );
    connect(
        horizontalSlider_surface_zmax, SIGNAL( valueChanged( int ) ), this,
        SLOT( set_visualization_system_surface() ) );
    //      overall direction
    connect(
        horizontalSlider_overall_dir_xmin, SIGNAL( valueChanged( int ) ), this,
        SLOT( set_visualization_system_overall_direction_sliders() ) );
    connect(
        horizontalSlider_overall_dir_xmax, SIGNAL( valueChanged( int ) ), this,
        SLOT( set_visualization_system_overall_direction_sliders() ) );
    connect(
        horizontalSlider_overall_dir_ymin, SIGNAL( valueChanged( int ) ), this,
        SLOT( set_visualization_system_overall_direction_sliders() ) );
    connect(
        horizontalSlider_overall_dir_ymax, SIGNAL( valueChanged( int ) ), this,
        SLOT( set_visualization_system_overall_direction_sliders() ) );
    connect(
        horizontalSlider_overall_dir_zmin, SIGNAL( valueChanged( int ) ), this,
        SLOT( set_visualization_system_overall_direction_sliders() ) );
    connect(
        horizontalSlider_overall_dir_zmax, SIGNAL( valueChanged( int ) ), this,
        SLOT( set_visualization_system_overall_direction_sliders() ) );
    connect(
        this->lineEdit_overall_dir_xmin, SIGNAL( returnPressed() ), this,
        SLOT( set_visualization_system_overall_direction_line_edits() ) );
    connect(
        this->lineEdit_overall_dir_xmax, SIGNAL( returnPressed() ), this,
        SLOT( set_visualization_system_overall_direction_line_edits() ) );
    connect(
        this->lineEdit_overall_dir_ymin, SIGNAL( returnPressed() ), this,
        SLOT( set_visualization_system_overall_direction_line_edits() ) );
    connect(
        this->lineEdit_overall_dir_ymax, SIGNAL( returnPressed() ), this,
        SLOT( set_visualization_system_overall_direction_line_edits() ) );
    connect(
        this->lineEdit_overall_dir_zmin, SIGNAL( returnPressed() ), this,
        SLOT( set_visualization_system_overall_direction_line_edits() ) );
    connect(
        this->lineEdit_overall_dir_zmax, SIGNAL( returnPressed() ), this,
        SLOT( set_visualization_system_overall_direction_line_edits() ) );
    connect(
        pushButton_reset_direction_filter, SIGNAL( clicked() ), this,
        SLOT( reset_visualization_system_overall_direction() ) );

    //      overall position
    connect(
        horizontalSlider_overall_pos_xmin, SIGNAL( valueChanged( int ) ), this,
        SLOT( set_visualization_system_overall_position_sliders() ) );
    connect(
        horizontalSlider_overall_pos_xmax, SIGNAL( valueChanged( int ) ), this,
        SLOT( set_visualization_system_overall_position_sliders() ) );
    connect(
        horizontalSlider_overall_pos_ymin, SIGNAL( valueChanged( int ) ), this,
        SLOT( set_visualization_system_overall_position_sliders() ) );
    connect(
        horizontalSlider_overall_pos_ymax, SIGNAL( valueChanged( int ) ), this,
        SLOT( set_visualization_system_overall_position_sliders() ) );
    connect(
        horizontalSlider_overall_pos_zmin, SIGNAL( valueChanged( int ) ), this,
        SLOT( set_visualization_system_overall_position_sliders() ) );
    connect(
        horizontalSlider_overall_pos_zmax, SIGNAL( valueChanged( int ) ), this,
        SLOT( set_visualization_system_overall_position_sliders() ) );
    connect(
        this->lineEdit_overall_pos_xmin, SIGNAL( returnPressed() ), this,
        SLOT( set_visualization_system_overall_position_line_edits() ) );
    connect(
        this->lineEdit_overall_pos_xmax, SIGNAL( returnPressed() ), this,
        SLOT( set_visualization_system_overall_position_line_edits() ) );
    connect(
        this->lineEdit_overall_pos_ymin, SIGNAL( returnPressed() ), this,
        SLOT( set_visualization_system_overall_position_line_edits() ) );
    connect(
        this->lineEdit_overall_pos_ymax, SIGNAL( returnPressed() ), this,
        SLOT( set_visualization_system_overall_position_line_edits() ) );
    connect(
        this->lineEdit_overall_pos_zmin, SIGNAL( returnPressed() ), this,
        SLOT( set_visualization_system_overall_position_line_edits() ) );
    connect(
        this->lineEdit_overall_pos_zmax, SIGNAL( returnPressed() ), this,
        SLOT( set_visualization_system_overall_position_line_edits() ) );
    connect(
        pushButton_reset_position_filter, SIGNAL( clicked() ), this,
        SLOT( reset_visualization_system_overall_position() ) );

    // cell filter
    connect(
        this->lineEdit_cell_a_min, SIGNAL( returnPressed() ), this,
        SLOT( set_visualization_system_cells_line_edits() ) );
    connect(
        this->lineEdit_cell_a_max, SIGNAL( returnPressed() ), this,
        SLOT( set_visualization_system_cells_line_edits() ) );
    connect(
        this->lineEdit_cell_b_min, SIGNAL( returnPressed() ), this,
        SLOT( set_visualization_system_cells_line_edits() ) );
    connect(
        this->lineEdit_cell_b_max, SIGNAL( returnPressed() ), this,
        SLOT( set_visualization_system_cells_line_edits() ) );
    connect(
        this->lineEdit_cell_c_min, SIGNAL( returnPressed() ), this,
        SLOT( set_visualization_system_cells_line_edits() ) );
    connect(
        this->lineEdit_cell_c_max, SIGNAL( returnPressed() ), this,
        SLOT( set_visualization_system_cells_line_edits() ) );

    connect(
        this->horizontalSlider_cell_a_min, SIGNAL( valueChanged( int ) ), this,
        SLOT( set_visualization_system_cells_sliders() ) );
    connect(
        this->horizontalSlider_cell_a_max, SIGNAL( valueChanged( int ) ), this,
        SLOT( set_visualization_system_cells_sliders() ) );
    connect(
        this->horizontalSlider_cell_b_min, SIGNAL( valueChanged( int ) ), this,
        SLOT( set_visualization_system_cells_sliders() ) );
    connect(
        this->horizontalSlider_cell_b_max, SIGNAL( valueChanged( int ) ), this,
        SLOT( set_visualization_system_cells_sliders() ) );
    connect(
        this->horizontalSlider_cell_c_min, SIGNAL( valueChanged( int ) ), this,
        SLOT( set_visualization_system_cells_sliders() ) );
    connect(
        this->horizontalSlider_cell_c_max, SIGNAL( valueChanged( int ) ), this,
        SLOT( set_visualization_system_cells_sliders() ) );

    connect( this->checkBox_cell_a, SIGNAL( stateChanged( int ) ), this, SLOT( cell_on_checkbox() ) );
    connect( this->checkBox_cell_b, SIGNAL( stateChanged( int ) ), this, SLOT( cell_on_checkbox() ) );
    connect( this->checkBox_cell_c, SIGNAL( stateChanged( int ) ), this, SLOT( cell_on_checkbox() ) );
    connect( pushButton_reset_cell_filter, SIGNAL( clicked() ), this, SLOT( reset_visualization_system_cells() ) );

    //      isosurface
    connect(
        checkBox_isosurfaceshadows, SIGNAL( stateChanged( int ) ), this,
        SLOT( set_visualization_system_isosurface() ) );
    connect( pushButton_addIsosurface, SIGNAL( clicked() ), this, SLOT( add_isosurface() ) );
    // Sphere
    connect(
        horizontalSlider_spherePointSize, SIGNAL( valueChanged( int ) ), this,
        SLOT( set_visualization_sphere_pointsize() ) );
    // Colors
    connect(
        comboBox_backgroundColor, SIGNAL( currentIndexChanged( int ) ), this, SLOT( set_visualization_background() ) );
    connect(
        comboBox_colormap_general, SIGNAL( currentIndexChanged( int ) ), this, SLOT( set_visualization_colormap() ) );
    connect(
        comboBox_colormap_arrows, SIGNAL( currentIndexChanged( int ) ), this, SLOT( set_visualization_colormap() ) );
    connect(
        horizontalSlider_colormap_rotate_phi, SIGNAL( valueChanged( int ) ), this,
        SLOT( set_visualization_colormap_rotation_slider() ) );
    connect( this->lineEdit_cm_axis_x, SIGNAL( returnPressed() ), this, SLOT( set_visualization_colormap_axis() ) );
    connect( this->lineEdit_cm_axis_y, SIGNAL( returnPressed() ), this, SLOT( set_visualization_colormap_axis() ) );
    connect( this->lineEdit_cm_axis_z, SIGNAL( returnPressed() ), this, SLOT( set_visualization_colormap_axis() ) );
    connect(
        this->lineEdit_colormap_rotate_phi, SIGNAL( returnPressed() ), this,
        SLOT( set_visualization_colormap_rotation_lineEdit() ) );
    connect(
        this->checkBox_colormap_invert_z, SIGNAL( stateChanged( int ) ), this,
        SLOT( set_visualization_colormap_rotation_lineEdit() ) );
    connect(
        this->checkBox_colormap_invert_xy, SIGNAL( stateChanged( int ) ), this,
        SLOT( set_visualization_colormap_rotation_slider() ) );
    // Camera
    connect( this->lineEdit_camera_pos_x, SIGNAL( returnPressed() ), this, SLOT( set_camera_position() ) );
    connect( this->lineEdit_camera_pos_y, SIGNAL( returnPressed() ), this, SLOT( set_camera_position() ) );
    connect( this->lineEdit_camera_pos_z, SIGNAL( returnPressed() ), this, SLOT( set_camera_position() ) );
    connect( this->lineEdit_camera_focus_x, SIGNAL( returnPressed() ), this, SLOT( set_camera_focus() ) );
    connect( this->lineEdit_camera_focus_y, SIGNAL( returnPressed() ), this, SLOT( set_camera_focus() ) );
    connect( this->lineEdit_camera_focus_z, SIGNAL( returnPressed() ), this, SLOT( set_camera_focus() ) );
    connect( this->lineEdit_camera_upvector_x, SIGNAL( returnPressed() ), this, SLOT( set_camera_upvector() ) );
    connect( this->lineEdit_camera_upvector_y, SIGNAL( returnPressed() ), this, SLOT( set_camera_upvector() ) );
    connect( this->lineEdit_camera_upvector_z, SIGNAL( returnPressed() ), this, SLOT( set_camera_upvector() ) );
    connect( this->pushButton_save_camera, SIGNAL( clicked() ), this, SLOT( save_camera() ) );
    connect( this->pushButton_load_camera, SIGNAL( clicked() ), this, SLOT( load_camera() ) );
    connect( this->lineEdit_camera_fov, SIGNAL( returnPressed() ), this, SLOT( set_camera_fov_lineedit() ) );

    connect( horizontalSlider_camera_fov, SIGNAL( valueChanged( int ) ), this, SLOT( set_camera_fov_slider() ) );
    // Light
    connect( horizontalSlider_light_theta, SIGNAL( valueChanged( int ) ), this, SLOT( set_light_position() ) );
    connect( horizontalSlider_light_phi, SIGNAL( valueChanged( int ) ), this, SLOT( set_light_position() ) );
}

void VisualisationSettingsWidget::incrementNCellStep( int increment )
{
    this->spinBox_n_cell_steps->setValue( this->spinBox_n_cell_steps->value() + increment );
}
