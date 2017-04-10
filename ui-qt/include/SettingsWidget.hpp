#pragma once
#ifndef SETTINGSWIDGET_H
#define SETTINGSWIDGET_H

#include <QWidget>

#include <memory>

#include "IsosurfaceWidget.hpp"

#include "ui_SettingsWidget.h"

class SpinWidget;
struct State;

class SettingsWidget : public QWidget, private Ui::SettingsWidget
{
    Q_OBJECT

public:
	SettingsWidget(std::shared_ptr<State> state, SpinWidget *spinWidget);
	void updateData();
	void SelectTab(int index);

	std::shared_ptr<State> state;

public slots:
	// Configurations
	void configurationAddNoise();
	void randomPressed();

private:
	// Setup Input Validators
	void Setup_Input_Validators();
	// Setup Slots
	void Setup_Configurations_Slots();
	void Setup_Transitions_Slots();
	void Setup_Hamiltonian_Isotropic_Slots();
	void Setup_Hamiltonian_Anisotropic_Slots();
	void Setup_Parameters_Slots();
	void Setup_Visualization_Slots();
	// Load a set of parameters from the spin systems
	void Load_Hamiltonian_Isotropic_Contents();
	void Load_Hamiltonian_Anisotropic_Contents();
	void Load_Parameters_Contents();
	void Load_Visualization_Contents();
	// Validator for Input into lineEdits
	QRegularExpressionValidator * number_validator;
	QRegularExpressionValidator * number_validator_unsigned;
	QRegularExpressionValidator * number_validator_int;
	QRegularExpressionValidator * number_validator_int_unsigned;
	SpinWidget *_spinWidget;
	bool m_isosurfaceshadows;
	std::vector<IsosurfaceWidget *> isosurfaceWidgets;
	// Helpers
	std::array<float,3> get_position();
	std::array<float,3> get_border_rectangular();
	float get_border_cylindrical();
	float get_border_spherical();
	float get_inverted();

private slots:
	// Parameters
	void set_parameters();
	// Configurations
	void set_hamiltonian_iso();
	// void set_hamiltonian_aniso();
	void set_hamiltonian_aniso_bc();
	void set_hamiltonian_aniso_mu_s();
	void set_hamiltonian_aniso_field();
	void set_hamiltonian_aniso_ani();
	void set_hamiltonian_aniso_stt();
	void set_hamiltonian_aniso_temp();
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
	void set_visualization_sphere();
	void set_visualization_sphere_pointsize();
	void set_visualization_colormap();
	void set_visualization_colormap_rotation_slider();
	void set_visualization_colormap_rotation_lineEdit();
	void set_visualization_background();

	// Visualisation - Camera
	void set_camera();
	void read_camera();
	void set_camera_position();
	void set_camera_focus();
	void set_camera_upvector();
	void set_camera_fov_lineedit();
	void set_camera_fov_slider();
	void set_camera_rotation();

	// Light
	void set_light_position();

	// Configurations
	void addNoisePressed();
	void domainPressed();
	void plusZ();
	void minusZ();
	void create_Hopfion();
	void create_Skyrmion();
	void create_SpinSpiral();
	// Transitions
	void homogeneousTransitionPressed();
	// Debug?
	void print_Energies_to_console();
};

#endif