#pragma once
#ifndef SPIRIT_CONFIGURATIONSWIDGET_HPP
#define SPIRIT_CONFIGURATIONSWIDGET_HPP

#include "ui_ConfigurationsWidget.h"

#include "IsosurfaceWidget.hpp"
#include "SpinWidget.hpp"

#include <QRegularExpressionValidator>
#include <QtWidgets/QWidget>

#include <Spirit/Spirit_Defines.h>

#include <memory>
#include <thread>

struct State;

/*
    Converts a QString to an std::string.
    This function is needed sometimes due to weird behaviour of QString::toStdString().
*/
std::string string_q2std( QString qs );

class ConfigurationsWidget : public QWidget, private Ui::ConfigurationsWidget
{
    Q_OBJECT

public:
    ConfigurationsWidget( std::shared_ptr<State> state, SpinWidget * spinWidget );
    void updateData();

public slots:
    // Configurations
    void configurationAddNoise();
    void randomPressed();
    void lastConfiguration();

private slots:
    // Configurations
    void addNoisePressed();
    void domainPressed();
    void plusZ();
    void minusZ();
    void create_Hopfion();
    void create_Skyrmion();
    void create_SpinSpiral();
    // Pinning and atom types
    void set_atom_type_pressed();
    void set_pinned_pressed();
    // Transitions
    void homogeneousTransitionPressed();
    void homogeneousTransitionFirstLastPressed();
    void homogeneousTransitionInterpolatePressed();
    void DW_Width_CheckBox_State_Changed();

private:
    void Setup_Input_Validators();
    void Setup_Configurations_Slots();
    void Setup_Transitions_Slots();

    // Helpers
    std::array<scalar, 3> get_position();
    std::array<scalar, 3> get_border_rectangular();
    scalar get_border_cylindrical();
    scalar get_border_spherical();
    scalar get_inverted();

    // Debug?
    void print_Energies_to_console();

    std::shared_ptr<State> state;
    SpinWidget * spinWidget;
    // SettingsWidget * settingsWidget;

    // Last used configuration
    std::string last_configuration;

    // Validator for Input into lineEdits
    QRegularExpressionValidator * number_validator;
    QRegularExpressionValidator * number_validator_unsigned;
    QRegularExpressionValidator * number_validator_int;
    QRegularExpressionValidator * number_validator_int_unsigned;
};

#endif