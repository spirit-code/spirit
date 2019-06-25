#pragma once
#ifndef VisualisationSettingsWidget_H
#define VisualisationSettingsWidget_H

#include <QtWidgets/QWidget>
#include <QRegularExpressionValidator>

#include <memory>
#include <thread>

#include "SpinWidget.hpp"
#include "IsosurfaceWidget.hpp"
//#include "SettingsWidget.hpp"

#include "ui_VisualisationSettingsWidget.h"

struct State;

/*
	Converts a QString to an std::string.
	This function is needed sometimes due to weird behaviour of QString::toStdString().
*/
std::string string_q2std(QString qs);

class VisualisationSettingsWidget : public QWidget, private Ui::VisualisationSettingsWidget
{
    Q_OBJECT

public:
	VisualisationSettingsWidget(std::shared_ptr<State> state, SpinWidget * spinWidget);
	void updateData();
	void incrementNCellStep(int increment);

private slots:
	// Visualization
	void set_visualisation_source();
	void set_visualisation_n_cell_steps();
	void set_visualization_mode();
	void set_visualization_perspective();
	void set_visualization_miniview();
	void set_visualization_coordinatesystem();
	void set_visualization_system();
	void set_visualization_system_arrows();
	void set_visualization_system_boundingbox();
	void set_visualization_system_surface();
	void set_visualization_system_overall_direction();
	void set_visualization_system_overall_position();
	void set_visualization_system_isosurface();
	void add_isosurface();
	void update_isosurfaces();
	void set_visualization_sphere();
	void set_visualization_sphere_pointsize();
	void set_visualization_colormap();
	void set_visualization_colormap_rotation_slider();
	void set_visualization_colormap_rotation_lineEdit();
	void set_visualization_colormap_axis();
	void set_visualization_background();

	// Visualisation - Camera
	void read_camera();
	void save_camera();
	void load_camera();
	void set_camera_position();
	void set_camera_focus();
	void set_camera_upvector();
	void set_camera_fov_lineedit();
	void set_camera_fov_slider();

	// Light
	void set_light_position();

private:
	void Setup_Input_Validators();
	void Setup_Visualization_Slots();
	void Load_Visualization_Contents();

	std::shared_ptr<State> state;
	SpinWidget * spinWidget;
	//SettingsWidget * settingsWidget;

	bool m_isosurfaceshadows;
	std::vector<IsosurfaceWidget *> isosurfaceWidgets;

	// Validator for Input into lineEdits
	QRegularExpressionValidator * number_validator;
	QRegularExpressionValidator * number_validator_unsigned;
	QRegularExpressionValidator * number_validator_int;
	QRegularExpressionValidator * number_validator_int_unsigned;

	// The camera values of the last update
	glm::vec3 camera_position_last;
	glm::vec3 camera_focus_last;
	glm::vec3 camera_upvector_last;
};

#endif