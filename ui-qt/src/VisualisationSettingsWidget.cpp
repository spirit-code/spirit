#include <QtWidgets>

#include "VisualisationSettingsWidget.hpp"

#include <Spirit/Parameters.h>
#include <Spirit/System.h>
#include <Spirit/Geometry.h>
#include <Spirit/Chain.h>
#include <Spirit/Collection.h>
#include <Spirit/Log.h>
#include <Spirit/Exception.h>
#include <Spirit/Hamiltonian.h> // remove when transition of stt and temperature to Parameters is complete

// Small function for normalization of vectors
template <typename T>
void normalize(T v[3])
{
	T len = 0.0;
	for (int i = 0; i < 3; ++i) len += std::pow(v[i], 2);
	if (len == 0.0) throw Exception_Division_by_zero;
	for (int i = 0; i < 3; ++i) v[i] /= std::sqrt(len);
}

VisualisationSettingsWidget::VisualisationSettingsWidget(std::shared_ptr<State> state, SpinWidget * spinWidget)
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

	// Defaults
	m_isosurfaceshadows = false;
	add_isosurface();

	// Load variables from SpinWidget and State
	this->updateData();

	// Connect signals and slots
	this->Setup_Visualization_Slots();
}

void VisualisationSettingsWidget::updateData()
{
	this->Load_Visualization_Contents();
}


void VisualisationSettingsWidget::Setup_Input_Validators()
{
	// Visualisation
	//		Arrows
	this->lineEdit_arrows_lod->setValidator(this->number_validator_int_unsigned);
	//		Colormap
	this->lineEdit_colormap_rotate_phi->setValidator(this->number_validator_int_unsigned);
	//		Camera
	this->lineEdit_camera_pos_x->setValidator(this->number_validator);
	this->lineEdit_camera_pos_y->setValidator(this->number_validator);
	this->lineEdit_camera_pos_z->setValidator(this->number_validator);
	this->lineEdit_camera_focus_x->setValidator(this->number_validator);
	this->lineEdit_camera_focus_y->setValidator(this->number_validator);
	this->lineEdit_camera_focus_z->setValidator(this->number_validator);
}



void VisualisationSettingsWidget::Load_Visualization_Contents()
{
	// Mode
	if (this->spinWidget->visualizationMode() == SpinWidget::VisualizationMode::SYSTEM)
		this->radioButton_vismode_system->setChecked(true);
	else
		this->radioButton_vismode_sphere->setChecked(true);

	// N_cell_steps (draw every N'th unit cell)
	this->spinBox_n_cell_steps->setValue(this->spinWidget->visualisationNCellSteps());

	// System
	bool show_arrows = spinWidget->show_arrows;
	bool show_boundingbox = spinWidget->show_boundingbox;
	bool show_surface = spinWidget->show_surface;
	bool show_isosurface = spinWidget->show_isosurface;
	this->checkBox_show_arrows->setChecked(show_arrows);
	this->checkBox_showBoundingBox->setChecked(show_boundingbox);
	this->checkBox_show_surface->setChecked(show_surface);
	this->checkBox_show_isosurface->setChecked(show_isosurface);
	this->checkBox_isosurfaceshadows->setChecked(this->m_isosurfaceshadows);

	// Miniview
	this->checkBox_showMiniView->setChecked(spinWidget->isMiniviewEnabled());
	this->comboBox_miniViewPosition->setCurrentIndex((int)spinWidget->miniviewPosition());

	// Coordinate System
	this->checkBox_showCoordinateSystem->setChecked(spinWidget->isCoordinateSystemEnabled());
	this->comboBox_coordinateSystemPosition->setCurrentIndex((int)spinWidget->coordinateSystemPosition());

	// Z Range Arrows
	auto x_range = spinWidget->xRangeDirection();
	auto y_range = spinWidget->yRangeDirection();
	auto z_range = spinWidget->zRangeDirection();
	x_range.x = std::max(-1.0f, std::min(1.0f, x_range.x));
	x_range.y = std::max(-1.0f, std::min(1.0f, x_range.y));
	y_range.x = std::max(-1.0f, std::min(1.0f, y_range.x));
	y_range.y = std::max(-1.0f, std::min(1.0f, y_range.y));
	z_range.x = std::max(-1.0f, std::min(1.0f, z_range.x));
	z_range.y = std::max(-1.0f, std::min(1.0f, z_range.y));

	// Overall direction filter X
	horizontalSlider_overall_dir_xmin->setInvertedAppearance(true);
	horizontalSlider_overall_dir_xmin->setRange(-100, 100);
	horizontalSlider_overall_dir_xmin->setValue((int)(-x_range.x * 100));
	horizontalSlider_overall_dir_xmax->setRange(-100, 100);
	horizontalSlider_overall_dir_xmax->setValue((int)(x_range.y * 100));
	horizontalSlider_overall_dir_xmin->setTracking(true);
	horizontalSlider_overall_dir_xmax->setTracking(true);
	// Overall direction filter Y
	horizontalSlider_overall_dir_ymin->setInvertedAppearance(true);
	horizontalSlider_overall_dir_ymin->setRange(-100, 100);
	horizontalSlider_overall_dir_ymin->setValue((int)(-y_range.x * 100));
	horizontalSlider_overall_dir_ymax->setRange(-100, 100);
	horizontalSlider_overall_dir_ymax->setValue((int)(y_range.y * 100));
	horizontalSlider_overall_dir_ymin->setTracking(true);
	horizontalSlider_overall_dir_ymax->setTracking(true);
	// Overall direction filter Z
	horizontalSlider_overall_dir_zmin->setInvertedAppearance(true);
	horizontalSlider_overall_dir_zmin->setRange(-100, 100);
	horizontalSlider_overall_dir_zmin->setValue((int)(-z_range.x * 100));
	horizontalSlider_overall_dir_zmax->setRange(-100, 100);
	horizontalSlider_overall_dir_zmax->setValue((int)(z_range.y * 100));
	horizontalSlider_overall_dir_zmin->setTracking(true);
	horizontalSlider_overall_dir_zmax->setTracking(true);

	x_range = spinWidget->xRangePosition();
	y_range = spinWidget->yRangePosition();
	z_range = spinWidget->zRangePosition();

	float b_min[3], b_max[3], b_range[3];
	Geometry_Get_Bounds(state.get(), b_min, b_max);
	for (int dim = 0; dim < 3; ++dim) b_range[dim] = b_max[dim] - b_min[dim];

	float range_min = horizontalSlider_overall_pos_xmin->value() / 10000.0;
	float range_max = horizontalSlider_overall_pos_xmax->value() / 10000.0;

	// Overall position filter X
	//horizontalSlider_overall_pos_xmin->setInvertedAppearance(true);
	range_min = x_range.x / b_range[0] - b_min[0];
	range_max = x_range.y / b_range[0] - b_min[0];
	horizontalSlider_overall_pos_xmin->setRange(0, 10000);
	horizontalSlider_overall_pos_xmin->setValue((int)(range_min * 10000));
	horizontalSlider_overall_pos_xmax->setRange(0, 10000);
	horizontalSlider_overall_pos_xmax->setValue((int)(range_max * 10000));
	horizontalSlider_overall_pos_xmin->setTracking(true);
	horizontalSlider_overall_pos_xmax->setTracking(true);
	// Overall position filter Y
	//horizontalSlider_overall_pos_ymin->setInvertedAppearance(true);
	range_min = y_range.x / b_range[1] - b_min[1];
	range_max = y_range.y / b_range[1] - b_min[1];
	horizontalSlider_overall_pos_ymin->setRange(0, 10000);
	horizontalSlider_overall_pos_ymin->setValue((int)(range_min * 10000));
	horizontalSlider_overall_pos_ymax->setRange(0, 10000);
	horizontalSlider_overall_pos_ymax->setValue((int)(range_max * 10000));
	horizontalSlider_overall_pos_ymin->setTracking(true);
	horizontalSlider_overall_pos_ymax->setTracking(true);
	// Overall position filter Z
	//horizontalSlider_overall_pos_zmin->setInvertedAppearance(true);
	range_min = z_range.x / b_range[2] - b_min[2];
	range_max = z_range.y / b_range[2] - b_min[2];
	horizontalSlider_overall_pos_zmin->setRange(0, 10000);
	horizontalSlider_overall_pos_zmin->setValue((int)(range_min * 10000));
	horizontalSlider_overall_pos_zmax->setRange(0, 10000);
	horizontalSlider_overall_pos_zmax->setValue((int)(range_max * 10000));
	horizontalSlider_overall_pos_zmin->setTracking(true);
	horizontalSlider_overall_pos_zmax->setTracking(true);

	float bounds_min[3], bounds_max[3];
	Geometry_Get_Bounds(state.get(), bounds_min, bounds_max);
	glm::vec3 sys_size{ bounds_max[0] - bounds_min[0], bounds_max[1] - bounds_min[1], bounds_max[2] - bounds_min[2] };
	horizontalSlider_surface_xmin->blockSignals(true);
	horizontalSlider_surface_xmax->blockSignals(true);
	horizontalSlider_surface_ymin->blockSignals(true);
	horizontalSlider_surface_ymax->blockSignals(true);
	horizontalSlider_surface_zmin->blockSignals(true);
	horizontalSlider_surface_zmax->blockSignals(true);
	// X Range Surface
	auto surface_x_range = spinWidget->surfaceXRange();
	horizontalSlider_surface_xmin->setRange(1, 99999);
	horizontalSlider_surface_xmin->setValue((int)(surface_x_range[0] / sys_size[0] * 100000));
	horizontalSlider_surface_xmax->setRange(1, 99999);
	horizontalSlider_surface_xmax->setValue((int)(surface_x_range[1] / sys_size[0] * 100000));
	horizontalSlider_surface_xmin->setTracking(true);
	horizontalSlider_surface_xmax->setTracking(true);
	// Y Range Surface
	auto surface_y_range = spinWidget->surfaceYRange();
	horizontalSlider_surface_ymin->setRange(1, 99999);
	horizontalSlider_surface_ymin->setValue((int)(surface_y_range[0] / sys_size[1] * 100000));
	horizontalSlider_surface_ymax->setRange(1, 99999);
	horizontalSlider_surface_ymax->setValue((int)(surface_y_range[1] / sys_size[1] * 100000));
	horizontalSlider_surface_ymin->setTracking(true);
	horizontalSlider_surface_ymax->setTracking(true);
	// Z Range Surface
	auto surface_z_range = spinWidget->surfaceZRange();
	horizontalSlider_surface_zmin->setRange(1, 99999);
	horizontalSlider_surface_zmin->setValue((int)(surface_z_range[0] / sys_size[2] * 100000));
	horizontalSlider_surface_zmax->setRange(1, 99999);
	horizontalSlider_surface_zmax->setValue((int)(surface_z_range[1] / sys_size[2] * 100000));
	horizontalSlider_surface_zmin->setTracking(true);
	horizontalSlider_surface_zmax->setTracking(true);
	horizontalSlider_surface_xmin->blockSignals(false);
	horizontalSlider_surface_xmax->blockSignals(false);
	horizontalSlider_surface_ymin->blockSignals(false);
	horizontalSlider_surface_ymax->blockSignals(false);
	horizontalSlider_surface_zmin->blockSignals(false);

	// Colormap
	int idx_cm = (int)spinWidget->colormap();
	comboBox_colormap->setCurrentIndex(idx_cm);
	float cm_rotation = spinWidget->colormap_rotation();
	auto cm_inverted = spinWidget->colormap_inverted();
	horizontalSlider_colormap_rotate_phi->setRange(0, 360);
	horizontalSlider_colormap_rotate_phi->setValue(cm_rotation);
	lineEdit_colormap_rotate_phi->setText(QString::number(cm_rotation));
	checkBox_colormap_invert_z->setChecked(cm_inverted[0]);
	checkBox_colormap_invert_xy->setChecked(cm_inverted[1]);

	// Perspective / FOV
	if (spinWidget->cameraProjection())
	{
		radioButton_perspectiveProjection->setChecked(true);
	}
	else
	{
		radioButton_orthographicProjection->setChecked(true);
	}
	this->horizontalSlider_camera_fov->setRange(0, 160);
	this->lineEdit_camera_fov->setText(QString::number(spinWidget->verticalFieldOfView()));
	this->horizontalSlider_camera_fov->setValue((int)(spinWidget->verticalFieldOfView()));


	// Arrows: size and lod
	horizontalSlider_arrowsize->setRange(0, 20);
	float logs = std::log10(spinWidget->arrowSize());
	horizontalSlider_arrowsize->setValue((int)((logs + 1) * 10));
	lineEdit_arrows_lod->setText(QString::number(spinWidget->arrowLOD()));

	// Sphere
	horizontalSlider_spherePointSize->setRange(1, 10);
	horizontalSlider_spherePointSize->setValue((int)spinWidget->spherePointSizeRange().y);

	// Light
	horizontalSlider_light_theta->setRange(0, 180);
	horizontalSlider_light_phi->setRange(0, 360);

	// Bounding Box
	//checkBox_showBoundingBox->setChecked(spinWidget->isBoundingBoxEnabled());

	// Background
	int idx_bg = (int)spinWidget->backgroundColor();
	comboBox_backgroundColor->setCurrentIndex(idx_bg);

	// Camera
	this->read_camera();
	if (this->spinWidget->getCameraRotationType())
		this->radioButton_camera_rotate_free->setChecked(true);
	else
		this->radioButton_camera_rotate_bounded->setChecked(true);

	// Light
	auto angles = this->spinWidget->getLightPosition();
	this->horizontalSlider_light_theta->setValue((int)angles[0]);
	this->horizontalSlider_light_phi->setValue((int)angles[1]);
}


// -----------------------------------------------------------------------------------
// --------------------- Visualization -----------------------------------------------
// -----------------------------------------------------------------------------------
void VisualisationSettingsWidget::set_visualisation_source()
{
	this->spinWidget->setVisualisationSource(this->comboBox_VisualisationSource->currentIndex());
}

void VisualisationSettingsWidget::set_visualisation_n_cell_steps()
{
	// N_cell_steps (draw every N'th unit cell)
	this->spinWidget->setVisualisationNCellSteps(this->spinBox_n_cell_steps->value());
}

void VisualisationSettingsWidget::set_visualization_mode()
{
	SpinWidget::VisualizationMode mode;

	if (this->radioButton_vismode_sphere->isChecked())
		mode = SpinWidget::VisualizationMode::SPHERE;
	else
		mode = SpinWidget::VisualizationMode::SYSTEM;

	this->spinWidget->setVisualizationMode(mode);
}

void VisualisationSettingsWidget::set_visualization_perspective()
{
	// Perspective / FOV
	if (radioButton_orthographicProjection->isChecked())
	{
		spinWidget->setCameraProjection(false);
	}
	else
	{
		spinWidget->setCameraProjection(true);
	}
}

void VisualisationSettingsWidget::set_visualization_miniview()
{
	bool miniview;
	SpinWidget::WidgetLocation pos;

	miniview = this->checkBox_showMiniView->isChecked();
	if (this->comboBox_miniViewPosition->currentText() == "Bottom Left")
	{
		pos = SpinWidget::WidgetLocation::BOTTOM_LEFT;
	}
	else if (this->comboBox_miniViewPosition->currentText() == "Bottom Right")
	{
		pos = SpinWidget::WidgetLocation::BOTTOM_RIGHT;
	}
	else if (this->comboBox_miniViewPosition->currentText() == "Top Left")
	{
		pos = SpinWidget::WidgetLocation::TOP_LEFT;
	}
	else if (this->comboBox_miniViewPosition->currentText() == "Top Right")
	{
		pos = SpinWidget::WidgetLocation::TOP_RIGHT;
	}

	this->spinWidget->setVisualizationMiniview(miniview, pos);
}

void VisualisationSettingsWidget::set_visualization_coordinatesystem()
{
	bool coordinatesystem;
	SpinWidget::WidgetLocation pos;

	coordinatesystem = this->checkBox_showCoordinateSystem->isChecked();
	if (this->comboBox_coordinateSystemPosition->currentText() == "Bottom Left")
	{
		pos = SpinWidget::WidgetLocation::BOTTOM_LEFT;
	}
	else if (this->comboBox_coordinateSystemPosition->currentText() == "Bottom Right")
	{
		pos = SpinWidget::WidgetLocation::BOTTOM_RIGHT;
	}
	else if (this->comboBox_coordinateSystemPosition->currentText() == "Top Left")
	{
		pos = SpinWidget::WidgetLocation::TOP_LEFT;
	}
	else if (this->comboBox_coordinateSystemPosition->currentText() == "Top Right")
	{
		pos = SpinWidget::WidgetLocation::TOP_RIGHT;
	}

	this->spinWidget->setVisualizationCoordinatesystem(coordinatesystem, pos);
}

void VisualisationSettingsWidget::set_visualization_system()
{
	bool arrows, boundingbox, surface, isosurface;

	arrows = this->checkBox_show_arrows->isChecked();
	boundingbox = this->checkBox_showBoundingBox->isChecked();
	surface = this->checkBox_show_surface->isChecked();
	isosurface = this->checkBox_show_isosurface->isChecked();

	this->spinWidget->enableSystem(arrows, boundingbox, surface, isosurface);
}

void VisualisationSettingsWidget::set_visualization_system_arrows()
{
	float exponent = horizontalSlider_arrowsize->value() / 10.0f - 1.0f;
	float arrowsize = std::pow(10.0f, exponent);
	int arrowlod = lineEdit_arrows_lod->text().toInt();
	this->spinWidget->setArrows(arrowsize, arrowlod);
}
void VisualisationSettingsWidget::set_visualization_system_boundingbox()
{

}
void VisualisationSettingsWidget::set_visualization_system_surface()
{
	float bounds_min[3], bounds_max[3];
	Geometry_Get_Bounds(state.get(), bounds_min, bounds_max);
	float s_min, s_max;

	// X
	s_min = horizontalSlider_surface_xmin->value();
	s_max = horizontalSlider_surface_xmax->value();
	if (s_min > s_max)
	{
		float t = s_min;
		s_min = s_max;
		s_max = t;
	}
	horizontalSlider_surface_xmin->blockSignals(true);
	horizontalSlider_surface_xmax->blockSignals(true);
	horizontalSlider_surface_xmin->setValue((int)(s_min));
	horizontalSlider_surface_xmax->setValue((int)(s_max));
	horizontalSlider_surface_xmin->blockSignals(false);
	horizontalSlider_surface_xmax->blockSignals(false);
	float x_min = bounds_min[0] + (s_min / 100000.0) * (bounds_max[0] - bounds_min[0]);
	float x_max = bounds_min[0] + (s_max / 100000.0) * (bounds_max[0] - bounds_min[0]);
	// Y
	s_min = horizontalSlider_surface_ymin->value();
	s_max = horizontalSlider_surface_ymax->value();
	if (s_min > s_max)
	{
		float t = s_min;
		s_min = s_max;
		s_max = t;
	}
	horizontalSlider_surface_ymin->blockSignals(true);
	horizontalSlider_surface_ymax->blockSignals(true);
	horizontalSlider_surface_ymin->setValue((int)(s_min));
	horizontalSlider_surface_ymax->setValue((int)(s_max));
	horizontalSlider_surface_ymin->blockSignals(false);
	horizontalSlider_surface_ymax->blockSignals(false);
	float y_min = bounds_min[1] + (s_min / 100000.0) * (bounds_max[1] - bounds_min[1]);
	float y_max = bounds_min[1] + (s_max / 100000.0) * (bounds_max[1] - bounds_min[1]);
	// Z
	s_min = horizontalSlider_surface_zmin->value();
	s_max = horizontalSlider_surface_zmax->value();
	if (s_min > s_max)
	{
		float t = s_min;
		s_min = s_max;
		s_max = t;
	}
	horizontalSlider_surface_zmin->blockSignals(true);
	horizontalSlider_surface_zmax->blockSignals(true);
	horizontalSlider_surface_zmin->setValue((int)(s_min));
	horizontalSlider_surface_zmax->setValue((int)(s_max));
	horizontalSlider_surface_zmin->blockSignals(false);
	horizontalSlider_surface_zmax->blockSignals(false);
	float z_min = bounds_min[2] + (s_min / 100000.0) * (bounds_max[2] - bounds_min[2]);
	float z_max = bounds_min[2] + (s_max / 100000.0) * (bounds_max[2] - bounds_min[2]);

	// Set
	glm::vec2 x_range(x_min, x_max);
	glm::vec2 y_range(y_min, y_max);
	glm::vec2 z_range(z_min, z_max);
	spinWidget->setSurface(x_range, y_range, z_range);
}

void VisualisationSettingsWidget::set_visualization_system_overall_direction()
{
	// X
	float range_min = -horizontalSlider_overall_dir_xmin->value() / 100.0;
	float range_max = horizontalSlider_overall_dir_xmax->value() / 100.0;
	if (range_min > range_max)
	{
		float t = range_min;
		range_min = range_max;
		range_max = t;
	}
	horizontalSlider_overall_dir_xmin->blockSignals(true);
	horizontalSlider_overall_dir_xmax->blockSignals(true);
	horizontalSlider_overall_dir_xmin->setValue((int)(-range_min * 100));
	horizontalSlider_overall_dir_xmax->setValue((int)(range_max * 100));
	horizontalSlider_overall_dir_xmin->blockSignals(false);
	horizontalSlider_overall_dir_xmax->blockSignals(false);
	glm::vec2 x_range(range_min, range_max);

	// Y
	range_min = -horizontalSlider_overall_dir_ymin->value() / 100.0;
	range_max = horizontalSlider_overall_dir_ymax->value() / 100.0;
	if (range_min > range_max)
	{
		float t = range_min;
		range_min = range_max;
		range_max = t;
	}
	horizontalSlider_overall_dir_ymin->blockSignals(true);
	horizontalSlider_overall_dir_ymax->blockSignals(true);
	horizontalSlider_overall_dir_ymin->setValue((int)(-range_min * 100));
	horizontalSlider_overall_dir_ymax->setValue((int)(range_max * 100));
	horizontalSlider_overall_dir_ymin->blockSignals(false);
	horizontalSlider_overall_dir_ymax->blockSignals(false);
	glm::vec2 y_range(range_min, range_max);

	// Z
	range_min = -horizontalSlider_overall_dir_zmin->value() / 100.0;
	range_max = horizontalSlider_overall_dir_zmax->value() / 100.0;
	if (range_min > range_max)
	{
		float t = range_min;
		range_min = range_max;
		range_max = t;
	}
	horizontalSlider_overall_dir_zmin->blockSignals(true);
	horizontalSlider_overall_dir_zmax->blockSignals(true);
	horizontalSlider_overall_dir_zmin->setValue((int)(-range_min * 100));
	horizontalSlider_overall_dir_zmax->setValue((int)(range_max * 100));
	horizontalSlider_overall_dir_zmin->blockSignals(false);
	horizontalSlider_overall_dir_zmax->blockSignals(false);
	glm::vec2 z_range(range_min, range_max);

	spinWidget->setOverallDirectionRange(x_range, y_range, z_range);
}

void VisualisationSettingsWidget::set_visualization_system_overall_position()
{
	float b_min[3], b_max[3], b_range[3];
	Geometry_Get_Bounds(state.get(), b_min, b_max);
	for (int dim = 0; dim < 3; ++dim) b_range[dim] = b_max[dim] - b_min[dim];

	// X
	float range_min = horizontalSlider_overall_pos_xmin->value() / 10000.0;
	float range_max = horizontalSlider_overall_pos_xmax->value() / 10000.0;
	if (range_min > range_max)
	{
		float t = range_min;
		range_min = range_max;
		range_max = t;
	}
	horizontalSlider_overall_pos_xmin->blockSignals(true);
	horizontalSlider_overall_pos_xmax->blockSignals(true);
	horizontalSlider_overall_pos_xmin->setValue((int)(range_min * 10000));
	horizontalSlider_overall_pos_xmax->setValue((int)(range_max * 10000));
	horizontalSlider_overall_pos_xmin->blockSignals(false);
	horizontalSlider_overall_pos_xmax->blockSignals(false);
	glm::vec2 x_range(b_min[0] + range_min*b_range[0], b_min[0] + range_max*b_range[0]);

	// Y
	range_min = horizontalSlider_overall_pos_ymin->value() / 10000.0;
	range_max = horizontalSlider_overall_pos_ymax->value() / 10000.0;
	if (range_min > range_max)
	{
		float t = range_min;
		range_min = range_max;
		range_max = t;
	}
	horizontalSlider_overall_pos_ymin->blockSignals(true);
	horizontalSlider_overall_pos_ymax->blockSignals(true);
	horizontalSlider_overall_pos_ymin->setValue((int)(range_min * 10000));
	horizontalSlider_overall_pos_ymax->setValue((int)(range_max * 10000));
	horizontalSlider_overall_pos_ymin->blockSignals(false);
	horizontalSlider_overall_pos_ymax->blockSignals(false);
	glm::vec2 y_range(b_min[1] + range_min*b_range[1], b_min[1] + range_max*b_range[1]);

	// Z
	range_min = horizontalSlider_overall_pos_zmin->value() / 10000.0;
	range_max = horizontalSlider_overall_pos_zmax->value() / 10000.0;
	if (range_min > range_max)
	{
		float t = range_min;
		range_min = range_max;
		range_max = t;
	}
	horizontalSlider_overall_pos_zmin->blockSignals(true);
	horizontalSlider_overall_pos_zmax->blockSignals(true);
	horizontalSlider_overall_pos_zmin->setValue((int)(range_min * 10000));
	horizontalSlider_overall_pos_zmax->setValue((int)(range_max * 10000));
	horizontalSlider_overall_pos_zmin->blockSignals(false);
	horizontalSlider_overall_pos_zmax->blockSignals(false);
	glm::vec2 z_range(b_min[2] + range_min*b_range[2], b_min[2] + range_max*b_range[2]);

	spinWidget->setOverallPositionRange(x_range, y_range, z_range);
}


void VisualisationSettingsWidget::set_visualization_system_isosurface()
{
	this->m_isosurfaceshadows = this->checkBox_isosurfaceshadows->isChecked();
	for (auto& isoWidget : this->isosurfaceWidgets) isoWidget->setDrawShadows(this->m_isosurfaceshadows);
}

void VisualisationSettingsWidget::add_isosurface()
{
	IsosurfaceWidget * iso = new IsosurfaceWidget(state, spinWidget);
	connect(iso, SIGNAL(closedSignal()), this, SLOT(update_isosurfaces()));
	iso->setDrawShadows(this->m_isosurfaceshadows);
	this->isosurfaceWidgets.push_back(iso);
	this->verticalLayout_isosurface->addWidget(isosurfaceWidgets.back());
	//this->set_visualization_system();
}

void VisualisationSettingsWidget::update_isosurfaces()
{
	// std::cerr << "........................" << std::endl;
	QObject* obj = sender();


	for (unsigned int i = 0; i < this->isosurfaceWidgets.size(); ++i)
	{
		if (this->isosurfaceWidgets[i] == obj)
			this->isosurfaceWidgets.erase(this->isosurfaceWidgets.begin() + i);
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
	this->spinWidget->setSpherePointSizeRange({ 0.2, this->horizontalSlider_spherePointSize->value() });
}

void VisualisationSettingsWidget::set_visualization_colormap()
{
	SpinWidget::Colormap colormap = SpinWidget::Colormap::HSV;
	if (comboBox_colormap->currentText() == "HSV, no z-component")
	{
		colormap = SpinWidget::Colormap::HSV_NO_Z;
	}
	if (comboBox_colormap->currentText() == "Z-Component: Blue-Red")
	{
		colormap = SpinWidget::Colormap::BLUE_RED;
	}
	if (comboBox_colormap->currentText() == "Z-Component: Blue-Green-Red")
	{
		colormap = SpinWidget::Colormap::BLUE_GREEN_RED;
	}
	if (comboBox_colormap->currentText() == "Z-Component: Blue-White-Red")
	{
		colormap = SpinWidget::Colormap::BLUE_WHITE_RED;
	}
	if (comboBox_colormap->currentText() == "White")
	{
		colormap = SpinWidget::Colormap::WHITE;
	}
	if (comboBox_colormap->currentText() == "Gray")
	{
		colormap = SpinWidget::Colormap::GRAY;
	}
	if (comboBox_colormap->currentText() == "Black")
	{
		colormap = SpinWidget::Colormap::BLACK;
	}
	spinWidget->setColormap(colormap);
}


void VisualisationSettingsWidget::set_visualization_colormap_rotation_slider()
{
	int phi = this->horizontalSlider_colormap_rotate_phi->value();
	bool invert_z = this->checkBox_colormap_invert_z->isChecked();
	bool invert_xy = this->checkBox_colormap_invert_xy->isChecked();

	this->lineEdit_colormap_rotate_phi->setText(QString::number(phi));

	this->spinWidget->setColormapRotationInverted(phi, invert_z, invert_xy);
}

void VisualisationSettingsWidget::set_visualization_colormap_rotation_lineEdit()
{
	int phi = this->lineEdit_colormap_rotate_phi->text().toInt();
	bool invert_z = this->checkBox_colormap_invert_z->isChecked();
	bool invert_xy = this->checkBox_colormap_invert_xy->isChecked();

	this->horizontalSlider_colormap_rotate_phi->setValue(phi);

	this->spinWidget->setColormapRotationInverted(phi, invert_z, invert_xy);
}

void VisualisationSettingsWidget::set_visualization_background()
{
	SpinWidget::Color color;
	SpinWidget::Color invcolor;
	if (comboBox_backgroundColor->currentText() == "Black")
	{
		color = SpinWidget::Color::BLACK;
		invcolor = SpinWidget::Color::WHITE;
	}
	else if (comboBox_backgroundColor->currentText() == "Gray")
	{
		color = SpinWidget::Color::GRAY;
		invcolor = SpinWidget::Color::WHITE;
	}
	else
	{
		color = SpinWidget::Color::WHITE;
		invcolor = SpinWidget::Color::BLACK;
	}
	spinWidget->setBackgroundColor(color);
	spinWidget->setBoundingBoxColor(invcolor);
}

// -----------------------------------------------------------------------------------
// --------------------- Camera ------------------------------------------------------
// -----------------------------------------------------------------------------------

void VisualisationSettingsWidget::set_camera()
{
	set_camera_position();
	set_camera_focus();
	set_camera_upvector();
}

void VisualisationSettingsWidget::read_camera()
{
	auto camera_position = spinWidget->getCameraPositon();
	auto center_position = spinWidget->getCameraFocus();
	auto up_vector = spinWidget->getCameraUpVector();

	this->lineEdit_camera_pos_x->setText(QString::number(camera_position.x, 'f', 2));
	this->lineEdit_camera_pos_y->setText(QString::number(camera_position.y, 'f', 2));
	this->lineEdit_camera_pos_z->setText(QString::number(camera_position.z, 'f', 2));
	this->lineEdit_camera_focus_x->setText(QString::number(center_position.x, 'f', 2));
	this->lineEdit_camera_focus_y->setText(QString::number(center_position.y, 'f', 2));
	this->lineEdit_camera_focus_z->setText(QString::number(center_position.z, 'f', 2));
	this->lineEdit_camera_upvector_x->setText(QString::number(up_vector.x, 'f', 2));
	this->lineEdit_camera_upvector_y->setText(QString::number(up_vector.y, 'f', 2));
	this->lineEdit_camera_upvector_z->setText(QString::number(up_vector.z, 'f', 2));
}

void VisualisationSettingsWidget::set_camera_position()
{
	float x = this->lineEdit_camera_pos_x->text().toFloat();
	float y = this->lineEdit_camera_pos_y->text().toFloat();
	float z = this->lineEdit_camera_pos_z->text().toFloat();
	this->spinWidget->setCameraPosition({ x, y, z });
}

void VisualisationSettingsWidget::set_camera_focus()
{
	float x = this->lineEdit_camera_focus_x->text().toFloat();
	float y = this->lineEdit_camera_focus_y->text().toFloat();
	float z = this->lineEdit_camera_focus_z->text().toFloat();
	this->spinWidget->setCameraFocus({ x, y, z });
}

void VisualisationSettingsWidget::set_camera_upvector()
{
	float x = this->lineEdit_camera_upvector_x->text().toFloat();
	float y = this->lineEdit_camera_upvector_y->text().toFloat();
	float z = this->lineEdit_camera_upvector_z->text().toFloat();
	this->spinWidget->setCameraUpVector({ x, y, z });
}

void VisualisationSettingsWidget::set_camera_fov_slider()
{
	float fov = this->horizontalSlider_camera_fov->value();
	this->lineEdit_camera_fov->setText(QString::number(fov));
	spinWidget->setVerticalFieldOfView(fov);
}

void VisualisationSettingsWidget::set_camera_fov_lineedit()
{
	float fov = this->lineEdit_camera_fov->text().toFloat();
	horizontalSlider_camera_fov->setValue((int)(fov));
	spinWidget->setVerticalFieldOfView(fov);
}

void VisualisationSettingsWidget::set_camera_rotation()
{
	if (this->radioButton_camera_rotate_free->isChecked())
		this->spinWidget->setCameraRotationType(true);
	else
		this->spinWidget->setCameraRotationType(false);
}


// -----------------------------------------------------------------------------------
// --------------------- Light -------------------------------------------------------
// -----------------------------------------------------------------------------------

void VisualisationSettingsWidget::set_light_position()
{
	float theta = this->horizontalSlider_light_theta->value();
	float phi = this->horizontalSlider_light_phi->value();
	this->spinWidget->setLightPosition(theta, phi);
}


void VisualisationSettingsWidget::Setup_Visualization_Slots()
{
	connect(comboBox_VisualisationSource, SIGNAL(currentIndexChanged(int)), this, SLOT(set_visualisation_source()));
	connect(spinBox_n_cell_steps, SIGNAL(valueChanged(int)), this, SLOT(set_visualisation_n_cell_steps()));
	// Mode
	//connect(radioButton_vismode_sphere, SIGNAL(toggled(bool)), this, SLOT(set_visualization_mode()));
	connect(radioButton_vismode_system, SIGNAL(toggled(bool)), this, SLOT(set_visualization_mode()));
	connect(radioButton_orthographicProjection, SIGNAL(toggled(bool)), this, SLOT(set_visualization_perspective()));
	// Miniview
	connect(checkBox_showMiniView, SIGNAL(stateChanged(int)), this, SLOT(set_visualization_miniview()));
	connect(comboBox_miniViewPosition, SIGNAL(currentIndexChanged(int)), this, SLOT(set_visualization_miniview()));
	// Coordinate System
	connect(checkBox_showCoordinateSystem, SIGNAL(stateChanged(int)), this, SLOT(set_visualization_coordinatesystem()));
	connect(comboBox_coordinateSystemPosition, SIGNAL(currentIndexChanged(int)), this, SLOT(set_visualization_coordinatesystem()));
	// System
	connect(checkBox_show_arrows, SIGNAL(stateChanged(int)), this, SLOT(set_visualization_system()));
	connect(checkBox_showBoundingBox, SIGNAL(stateChanged(int)), this, SLOT(set_visualization_system()));
	connect(checkBox_show_surface, SIGNAL(stateChanged(int)), this, SLOT(set_visualization_system()));
	connect(checkBox_show_isosurface, SIGNAL(stateChanged(int)), this, SLOT(set_visualization_system()));
	//		arrows
	connect(horizontalSlider_arrowsize, SIGNAL(valueChanged(int)), this, SLOT(set_visualization_system_arrows()));
	connect(lineEdit_arrows_lod, SIGNAL(returnPressed()), this, SLOT(set_visualization_system_arrows()));
	//		bounding box
	//		surface
	connect(horizontalSlider_surface_xmin, SIGNAL(valueChanged(int)), this, SLOT(set_visualization_system_surface()));
	connect(horizontalSlider_surface_xmax, SIGNAL(valueChanged(int)), this, SLOT(set_visualization_system_surface()));
	connect(horizontalSlider_surface_ymin, SIGNAL(valueChanged(int)), this, SLOT(set_visualization_system_surface()));
	connect(horizontalSlider_surface_ymax, SIGNAL(valueChanged(int)), this, SLOT(set_visualization_system_surface()));
	connect(horizontalSlider_surface_zmin, SIGNAL(valueChanged(int)), this, SLOT(set_visualization_system_surface()));
	connect(horizontalSlider_surface_zmax, SIGNAL(valueChanged(int)), this, SLOT(set_visualization_system_surface()));
	//		overall direction
	connect(horizontalSlider_overall_dir_xmin, SIGNAL(valueChanged(int)), this, SLOT(set_visualization_system_overall_direction()));
	connect(horizontalSlider_overall_dir_xmax, SIGNAL(valueChanged(int)), this, SLOT(set_visualization_system_overall_direction()));
	connect(horizontalSlider_overall_dir_ymin, SIGNAL(valueChanged(int)), this, SLOT(set_visualization_system_overall_direction()));
	connect(horizontalSlider_overall_dir_ymax, SIGNAL(valueChanged(int)), this, SLOT(set_visualization_system_overall_direction()));
	connect(horizontalSlider_overall_dir_zmin, SIGNAL(valueChanged(int)), this, SLOT(set_visualization_system_overall_direction()));
	connect(horizontalSlider_overall_dir_zmax, SIGNAL(valueChanged(int)), this, SLOT(set_visualization_system_overall_direction()));
	//		overall position
	connect(horizontalSlider_overall_pos_xmin, SIGNAL(valueChanged(int)), this, SLOT(set_visualization_system_overall_position()));
	connect(horizontalSlider_overall_pos_xmax, SIGNAL(valueChanged(int)), this, SLOT(set_visualization_system_overall_position()));
	connect(horizontalSlider_overall_pos_ymin, SIGNAL(valueChanged(int)), this, SLOT(set_visualization_system_overall_position()));
	connect(horizontalSlider_overall_pos_ymax, SIGNAL(valueChanged(int)), this, SLOT(set_visualization_system_overall_position()));
	connect(horizontalSlider_overall_pos_zmin, SIGNAL(valueChanged(int)), this, SLOT(set_visualization_system_overall_position()));
	connect(horizontalSlider_overall_pos_zmax, SIGNAL(valueChanged(int)), this, SLOT(set_visualization_system_overall_position()));
	//		isosurface
	connect(checkBox_isosurfaceshadows, SIGNAL(stateChanged(int)), this, SLOT(set_visualization_system_isosurface()));
	connect(pushButton_addIsosurface, SIGNAL(clicked()), this, SLOT(add_isosurface()));
	// Sphere
	connect(horizontalSlider_spherePointSize, SIGNAL(valueChanged(int)), this, SLOT(set_visualization_sphere_pointsize()));
	// Colors
	connect(comboBox_backgroundColor, SIGNAL(currentIndexChanged(int)), this, SLOT(set_visualization_background()));
	connect(comboBox_colormap, SIGNAL(currentIndexChanged(int)), this, SLOT(set_visualization_colormap()));
	connect(horizontalSlider_colormap_rotate_phi, SIGNAL(valueChanged(int)), this, SLOT(set_visualization_colormap_rotation_slider()));
	connect(this->lineEdit_colormap_rotate_phi, SIGNAL(returnPressed()), this, SLOT(set_visualization_colormap_rotation_lineEdit()));
	connect(this->checkBox_colormap_invert_z, SIGNAL(stateChanged(int)), this, SLOT(set_visualization_colormap_rotation_lineEdit()));
	connect(this->checkBox_colormap_invert_xy, SIGNAL(stateChanged(int)), this, SLOT(set_visualization_colormap_rotation_slider()));
	// Camera
	connect(this->lineEdit_camera_pos_x, SIGNAL(returnPressed()), this, SLOT(set_camera_position()));
	connect(this->lineEdit_camera_pos_y, SIGNAL(returnPressed()), this, SLOT(set_camera_position()));
	connect(this->lineEdit_camera_pos_z, SIGNAL(returnPressed()), this, SLOT(set_camera_position()));
	connect(this->lineEdit_camera_focus_x, SIGNAL(returnPressed()), this, SLOT(set_camera_focus()));
	connect(this->lineEdit_camera_focus_y, SIGNAL(returnPressed()), this, SLOT(set_camera_focus()));
	connect(this->lineEdit_camera_focus_z, SIGNAL(returnPressed()), this, SLOT(set_camera_focus()));
	connect(this->lineEdit_camera_upvector_x, SIGNAL(returnPressed()), this, SLOT(set_camera_upvector()));
	connect(this->lineEdit_camera_upvector_y, SIGNAL(returnPressed()), this, SLOT(set_camera_upvector()));
	connect(this->lineEdit_camera_upvector_z, SIGNAL(returnPressed()), this, SLOT(set_camera_upvector()));
	connect(this->pushButton_set_camera, SIGNAL(clicked()), this, SLOT(set_camera()));
	connect(this->pushButton_read_camera, SIGNAL(clicked()), this, SLOT(read_camera()));
	connect(this->lineEdit_camera_fov, SIGNAL(returnPressed()), this, SLOT(set_camera_fov_lineedit()));
	connect(horizontalSlider_camera_fov, SIGNAL(valueChanged(int)), this, SLOT(set_camera_fov_slider()));
	connect(radioButton_camera_rotate_free, SIGNAL(toggled(bool)), this, SLOT(set_camera_rotation()));
	connect(radioButton_camera_rotate_bounded, SIGNAL(toggled(bool)), this, SLOT(set_camera_rotation()));
	// Light
	connect(horizontalSlider_light_theta, SIGNAL(valueChanged(int)), this, SLOT(set_light_position()));
	connect(horizontalSlider_light_phi, SIGNAL(valueChanged(int)), this, SLOT(set_light_position()));
}

void VisualisationSettingsWidget::incrementNCellStep(int increment)
{
	this->spinBox_n_cell_steps->setValue(this->spinBox_n_cell_steps->value() + increment);
}