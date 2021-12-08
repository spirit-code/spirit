#pragma once
#ifndef SPIRIT_IMGUI_PARAMETERS_WIDGET_HPP
#define SPIRIT_IMGUI_PARAMETERS_WIDGET_HPP

#include <enums.hpp>
#include <rendering_layer.hpp>

#include <Spirit/IO.h>
#include <memory>
#include <widget_base.hpp>

struct State;

namespace ui
{

struct Parameters
{
    struct Common
    {
        // --------------- Iterations ------------
        // Number of iterations carried out when pressing "play" or calling "iterate"
        int n_iterations = 1e6;
        // Number of iterations after which the Method should save data
        int n_iterations_log = 1e3;

        // Maximum walltime for Iterate in seconds
        int max_walltime_sec = 0;

        // Force convergence criterium
        float force_convergence = 1e-10;

        // ----------------- Output --------------
        // Data output folder
        std::string output_folder = "output";
        // Put a tag in front of output files (if "<time>" is used then the tag is the timestamp)
        std::string output_file_tag = "<time>";
        // Save any output when logging
        bool output_any = false;
        // Save output at initial state
        bool output_initial = false;
        // Save output at final state
        bool output_final = false;
        // Vectorfield output file format
        int output_vf_filetype = IO_Fileformat_OVF_text;
    };

    struct MC : Common
    {
        // Temperature [K]
        float temperature = 0;
        // Seed for RNG
        int rng_seed = 2006;

        // Whether to sample spins randomly or in sequence in Metropolis algorithm
        bool metropolis_random_sample = true;
        // Whether to use the adaptive cone radius (otherwise just uses full sphere sampling)
        bool metropolis_step_cone = true;
        // Whether to adapt the metropolis cone angle throughout a MC run to try to hit a target acceptance ratio
        bool metropolis_cone_adaptive = true;
        // The metropolis cone angle
        float metropolis_cone_angle = 30;

        // Target acceptance ratio of mc steps for adaptive cone angle
        float acceptance_ratio_target = 0.5;

        // ----------------- Output --------------
        // Energy output settings
        bool output_energy_step                  = false;
        bool output_energy_archive               = false;
        bool output_energy_spin_resolved         = false;
        bool output_energy_divide_by_nspins      = true;
        bool output_energy_add_readability_lines = false;
        // Spin configurations output settings
        bool output_configuration_step    = false;
        bool output_configuration_archive = false;
    };

    struct LLG : Common
    {
        // Time step per iteration [ps]
        float dt = 1e-3;

        // Damping
        float damping = 0.3;
        float beta    = 0;

        // Seed for RNG
        int rng_seed = 2006;

        // Temperature [K]
        float temperature = 0;
        // Temperature gradient [K]
        float temperature_gradient_direction[3]{ 1, 0, 0 };
        float temperature_gradient_inclination = 0;

        // - true:  use gradient approximation for STT
        // - false: use pinned monolayer approximation with current in z-direction
        bool stt_use_gradient = true;
        // Spin transfer torque parameter (prop to injected current density)
        float stt_magnitude = 0;
        // Spin current polarisation normal vector
        float stt_polarisation_normal[3]{ 1, 0, 0 };

        // Do direct minimization instead of dynamics
        bool direct_minimization = false;

        // ----------------- Output --------------
        // Energy output settings
        bool output_energy_step                  = false;
        bool output_energy_archive               = false;
        bool output_energy_spin_resolved         = false;
        bool output_energy_divide_by_nspins      = true;
        bool output_energy_add_readability_lines = false;
        // Spin configurations output settings
        bool output_configuration_step    = false;
        bool output_configuration_archive = false;
    };

    struct GNEB : Common
    {
        // Time step per iteration [ps]
        float dt = 1e-3;

        // Strength of springs between images
        float spring_constant = 1;

        // The ratio of energy to reaction coordinate in the spring force
        //      0 is Rx only, 1 is E only
        float spring_force_ratio = 0;

        // With which minimum norm per spin the path shortening force should be applied
        float path_shortening_constant = 0;

        // Number of Energy interpolations between Images
        int n_enery_interpolations = 10;

        // Temperature [K]
        float temperature = 0;
        // Seed for RNG
        int rng_seed = 2006;

        // ----------------- Output --------------
        bool output_energies_step                  = false;
        bool output_energies_divide_by_nspins      = true;
        bool output_energies_add_readability_lines = false;
        bool output_energies_interpolated          = false;
        bool output_chain_step                     = false;
    };

    struct MMF : Common
    {
        // Time step per iteration [ps]
        float dt = 1e-3;

        // Which mode to follow (based on some conditions)
        int n_mode_follow = 0;
        // Number of lowest modes to calculate
        int n_modes = 10;

        // ----------------- Output --------------
        // Energy output settings
        bool output_energy_step                  = false;
        bool output_energy_archive               = false;
        bool output_energy_spin_resolved         = false;
        bool output_energy_divide_by_nspins      = true;
        bool output_energy_add_readability_lines = false;
        // Spin configurations output settings
        bool output_configuration_step    = false;
        bool output_configuration_archive = false;
    };

    struct EMA : Common
    {
        int n_modes       = 10;
        int n_mode_follow = 0;
        float frequency   = 0.02;
        float amplitude   = 1;
        bool snapshot     = false;

        // ----------------- Output --------------
        // Energy output settings
        bool output_energy_step             = false;
        bool output_energy_archive          = false;
        bool output_energy_spin_resolved    = false;
        bool output_energy_divide_by_nspins = true;

        // Spin configurations output settings
        bool output_configuration_step    = false;
        bool output_configuration_archive = false;
    };
};

struct ParametersWidget : public WidgetBase
{
    ParametersWidget( bool & show, std::shared_ptr<State> state, UiSharedState & ui_shared_state );
    void show_content() override;
    void update_data();

    std::shared_ptr<State> state;
    UiSharedState & ui_shared_state;

    int gneb_image_type = 0;

    Parameters::MC parameters_mc;
    Parameters::LLG parameters_llg;
    Parameters::GNEB parameters_gneb;
    Parameters::MMF parameters_mmf;
    Parameters::EMA parameters_ema;
};

} // namespace ui

#endif