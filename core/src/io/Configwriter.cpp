#include <engine/Neighbours.hpp>
#include <engine/Vectormath.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/IO.hpp>
#include <utility/Constants.hpp>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace IO
{

void Folders_to_Config(
    const std::string & config_file, const std::shared_ptr<Data::Parameters_Method_LLG> parameters_llg,
    const std::shared_ptr<Data::Parameters_Method_MC> parameters_mc,
    const std::shared_ptr<Data::Parameters_Method_GNEB> parameters_gneb,
    const std::shared_ptr<Data::Parameters_Method_MMF> parameters_mmf )
{
    std::string config = "";
    config += "################# Output Folders #################\n";
    config += "output_file_tag    " + Log.file_tag + "\n";
    config += "log_output_folder  " + Log.output_folder + "\n";
    config += "llg_output_folder  " + parameters_llg->output_folder + "\n";
    config += "mc_output_folder   " + parameters_mc->output_folder + "\n";
    config += "gneb_output_folder " + parameters_gneb->output_folder + "\n";
    config += "mmf_output_folder  " + parameters_mmf->output_folder + "\n";
    config += "############### End Output Folders ###############";
    Append_String_to_File( config, config_file );
}

void Log_Levels_to_Config( const std::string & config_file )
{
    std::string config = "";
    config += "############### Logging Parameters ###############\n";
    config += fmt::format( "{:<22} {}\n", "log_to_file", (int)Log.messages_to_file );
    config += fmt::format( "{:<22} {}\n", "log_file_level", (int)Log.level_file );
    config += fmt::format( "{:<22} {}\n", "log_to_console", (int)Log.messages_to_console );
    config += fmt::format( "{:<22} {}\n", "log_console_level", (int)Log.level_console );
    config += fmt::format( "{:<22} {}\n", "log_input_save_initial", (int)Log.save_input_initial );
    config += fmt::format( "{:<22} {}\n", "log_input_save_final", (int)Log.save_input_final );
    config += "############# End Logging Parameters #############";
    Append_String_to_File( config, config_file );
}

void Geometry_to_Config( const std::string & config_file, const std::shared_ptr<Data::Geometry> geometry )
{
    // TODO: this needs to be updated!
    std::string config = "";
    config += "#################### Geometry ####################\n";

    // Bravais lattice/vectors
    if( geometry->classifier == Data::BravaisLatticeType::SC )
        config += "bravais_lattice sc\n";
    else if( geometry->classifier == Data::BravaisLatticeType::FCC )
        config += "bravais_lattice fcc\n";
    else if( geometry->classifier == Data::BravaisLatticeType::BCC )
        config += "bravais_lattice bcc\n";
    else if( geometry->classifier == Data::BravaisLatticeType::Hex2D )
        config += "bravais_lattice hex2d120\n";
    else
    {
        config += "bravais_vectors\n";
        config += fmt::format(
            "{0}\n{1}\n{2}\n", geometry->bravais_vectors[0].transpose(), geometry->bravais_vectors[1].transpose(),
            geometry->bravais_vectors[2].transpose() );
    }

    // Number of cells
    config
        += fmt::format( "n_basis_cells {} {} {}\n", geometry->n_cells[0], geometry->n_cells[1], geometry->n_cells[2] );

    // Optionally basis
    if( geometry->n_cell_atoms > 1 )
    {
        config += "basis\n";
        config += fmt::format( "{}\n", geometry->n_cell_atoms );
        for( int i = 0; i < geometry->n_cell_atoms; ++i )
        {
            config += fmt::format( "{}\n", geometry->cell_atoms[i].transpose() );
        }
    }

    // Magnetic moment
    if( !geometry->cell_composition.disordered )
    {
        config += "mu_s                     ";
        for( int i = 0; i < geometry->n_cell_atoms; ++i )
            config += fmt::format( " {}", geometry->mu_s[i] );
        config += "\n";
    }
    else
    {
        auto & iatom         = geometry->cell_composition.iatom;
        auto & atom_type     = geometry->cell_composition.atom_type;
        auto & mu_s          = geometry->cell_composition.mu_s;
        auto & concentration = geometry->cell_composition.concentration;
        config += fmt::format( "atom_types    {}\n", iatom.size() );
        for( std::size_t i = 0; i < iatom.size(); ++i )
            config += fmt::format( "{}   {}   {}   {}\n", iatom[i], atom_type[i], mu_s[i], concentration[i] );
    }

    // Optionally lattice constant
    if( std::abs( geometry->lattice_constant - 1 ) > 1e-6 )
        config += fmt::format( "lattice_constant {}\n", geometry->lattice_constant );

    config += "################## End Geometry ##################";
    Append_String_to_File( config, config_file );
}

void Parameters_Method_LLG_to_Config(
    const std::string & config_file, const std::shared_ptr<Data::Parameters_Method_LLG> parameters )
{
    std::string config = "";
    config += "################# LLG Parameters #################\n";
    config += fmt::format( "{:<35} {:d}\n", "llg_output_any", parameters->output_any );
    config += fmt::format( "{:<35} {:d}\n", "llg_output_initial", parameters->output_initial );
    config += fmt::format( "{:<35} {:d}\n", "llg_output_final", parameters->output_final );
    config += fmt::format( "{:<35} {:d}\n", "llg_output_energy_step", parameters->output_energy_step );
    config += fmt::format( "{:<35} {:d}\n", "llg_output_energy_archive", parameters->output_energy_archive );
    config
        += fmt::format( "{:<35} {:d}\n", "llg_output_energy_spin_resolved", parameters->output_energy_spin_resolved );
    config += fmt::format(
        "{:<35} {:d}\n", "llg_output_energy_divide_by_nspins", parameters->output_energy_divide_by_nspins );
    config += fmt::format( "{:<35} {:d}\n", "llg_output_configuration_step", parameters->output_configuration_step );
    config
        += fmt::format( "{:<35} {:d}\n", "llg_output_configuration_archive", parameters->output_configuration_archive );
    config += fmt::format( "{:<35} {:e}\n", "llg_force_convergence", parameters->force_convergence );
    config += fmt::format( "{:<35} {}\n", "llg_n_iterations", parameters->n_iterations );
    config += fmt::format( "{:<35} {}\n", "llg_n_iterations_log", parameters->n_iterations_log );
    config += fmt::format( "{:<35} {}\n", "llg_seed", parameters->rng_seed );
    config += fmt::format( "{:<35} {}\n", "llg_temperature", parameters->temperature );
    config += fmt::format( "{:<35} {}\n", "llg_damping", parameters->damping );
    config += fmt::format(
        "{:<35} {}\n", "llg_dt", parameters->dt * Utility::Constants::mu_B / Utility::Constants::gamma );
    config += fmt::format( "{:<35} {}\n", "llg_stt_magnitude", parameters->stt_magnitude );
    config
        += fmt::format( "{:<35} {}\n", "llg_stt_polarisation_normal", parameters->stt_polarisation_normal.transpose() );
    config += "############### End LLG Parameters ###############";
    Append_String_to_File( config, config_file );
}

void Parameters_Method_MC_to_Config(
    const std::string & config_file, const std::shared_ptr<Data::Parameters_Method_MC> parameters )
{
    std::string config = "";
    config += "################# MC Parameters ##################\n";
    config += fmt::format( "{:<35} {:d}\n", "mc_output_any", parameters->output_any );
    config += fmt::format( "{:<35} {:d}\n", "mc_output_initial", parameters->output_initial );
    config += fmt::format( "{:<35} {:d}\n", "mc_output_final", parameters->output_final );
    config += fmt::format( "{:<35} {:d}\n", "mc_output_energy_step", parameters->output_energy_step );
    config += fmt::format( "{:<35} {:d}\n", "mc_output_energy_archive", parameters->output_energy_archive );
    config += fmt::format( "{:<35} {:d}\n", "mc_output_energy_spin_resolved", parameters->output_energy_spin_resolved );
    config += fmt::format(
        "{:<35} {:d}\n", "mc_output_energy_divide_by_nspins", parameters->output_energy_divide_by_nspins );
    config += fmt::format( "{:<35} {:d}\n", "mc_output_configuration_step", parameters->output_configuration_step );
    config
        += fmt::format( "{:<35} {:d}\n", "mc_output_configuration_archive", parameters->output_configuration_archive );
    config += fmt::format( "{:<35} {}\n", "mc_n_iterations", parameters->n_iterations );
    config += fmt::format( "{:<35} {}\n", "mc_n_iterations_log", parameters->n_iterations_log );
    config += fmt::format( "{:<35} {}\n", "mc_seed", parameters->rng_seed );
    config += fmt::format( "{:<35} {}\n", "mc_temperature", parameters->temperature );
    config += fmt::format( "{:<35} {}\n", "mc_acceptance_ratio", parameters->acceptance_ratio_target );
    config += "############### End MC Parameters ################";
    Append_String_to_File( config, config_file );
}

void Parameters_Method_GNEB_to_Config(
    const std::string & config_file, const std::shared_ptr<Data::Parameters_Method_GNEB> parameters )
{
    std::string config = "";
    config += "################# GNEB Parameters ################\n";
    config += fmt::format( "{:<38} {:d}\n", "gneb_output_any", parameters->output_any );
    config += fmt::format( "{:<38} {:d}\n", "gneb_output_initial", parameters->output_initial );
    config += fmt::format( "{:<38} {:d}\n", "gneb_output_final", parameters->output_final );
    config += fmt::format( "{:<38} {:d}\n", "gneb_output_energies_step", parameters->output_energies_step );
    config += fmt::format(
        "{:<38} {:d}\n", "gneb_output_energies_interpolated", parameters->output_energies_interpolated );
    config += fmt::format(
        "{:<38} {:d}\n", "gneb_output_energies_divide_by_nspins", parameters->output_energies_divide_by_nspins );
    config += fmt::format( "{:<38} {:d}\n", "gneb_output_chain_step", parameters->output_chain_step );
    config += fmt::format( "{:<38} {:e}\n", "gneb_force_convergence", parameters->force_convergence );
    config += fmt::format( "{:<38} {}\n", "gneb_n_iterations", parameters->n_iterations );
    config += fmt::format( "{:<38} {}\n", "gneb_n_iterations_log", parameters->n_iterations_log );
    config += fmt::format( "{:<38} {}\n", "gneb_spring_constant", parameters->spring_constant );
    config += fmt::format( "{:<38} {}\n", "gneb_n_energy_interpolations", parameters->n_E_interpolations );
    config += "############### End GNEB Parameters ##############";
    Append_String_to_File( config, config_file );
}

void Parameters_Method_MMF_to_Config(
    const std::string & config_file, const std::shared_ptr<Data::Parameters_Method_MMF> parameters )
{
    std::string config = "";
    config += "################# MMF Parameters #################\n";
    config += fmt::format( "{:<38} {:d}\n", "mmf_output_any", parameters->output_any );
    config += fmt::format( "{:<38} {:d}\n", "mmf_output_initial", parameters->output_initial );
    config += fmt::format( "{:<38} {:d}\n", "mmf_output_final", parameters->output_final );
    config += fmt::format( "{:<38} {:d}\n", "mmf_output_energy_step", parameters->output_energy_step );
    config += fmt::format( "{:<38} {:d}\n", "mmf_output_energy_archive", parameters->output_energy_archive );
    config += fmt::format(
        "{:<38} {:d}\n", "mmf_output_energy_divide_by_nspins", parameters->output_energy_divide_by_nspins );
    config += fmt::format( "{:<38} {:d}\n", "mmf_output_configuration_step", parameters->output_configuration_step );
    config
        += fmt::format( "{:<38} {:d}\n", "mmf_output_configuration_archive", parameters->output_configuration_archive );
    config += fmt::format( "{:<38} {:e}\n", "mmf_force_convergence", parameters->force_convergence );
    config += fmt::format( "{:<38} {}\n", "mmf_n_iterations", parameters->n_iterations );
    config += fmt::format( "{:<38} {}\n", "mmf_n_iterations_log", parameters->n_iterations_log );
    config += "############### End MMF Parameters ###############";
    Append_String_to_File( config, config_file );
}

void Hamiltonian_to_Config(
    const std::string & config_file, const std::shared_ptr<Engine::Hamiltonian> hamiltonian,
    const std::shared_ptr<Data::Geometry> geometry )
{
    std::string config = "";
    config += "################### Hamiltonian ##################\n";
    std::string name;
    if( hamiltonian->Name() == "Heisenberg" )
        name = "heisenberg_pairs";
    else if( hamiltonian->Name() == "Gaussian" )
        name = "gaussian";
    config += fmt::format( "{:<25} {}\n", "hamiltonian", name );
    config += fmt::format(
        "{:<25} {} {} {}\n", "boundary_conditions", hamiltonian->boundary_conditions[0],
        hamiltonian->boundary_conditions[1], hamiltonian->boundary_conditions[2] );
    Append_String_to_File( config, config_file );

    if( hamiltonian->Name() == "Heisenberg" )
        Hamiltonian_Heisenberg_to_Config( config_file, hamiltonian, geometry );
    else if( hamiltonian->Name() == "Gaussian" )
        Hamiltonian_Gaussian_to_Config( config_file, hamiltonian );

    config = "################# End Hamiltonian ################";
    Append_String_to_File( config, config_file );
}

void Hamiltonian_Heisenberg_to_Config(
    const std::string & config_file, const std::shared_ptr<Engine::Hamiltonian> hamiltonian,
    const std::shared_ptr<Data::Geometry> geometry )
{
    int n_cells_tot    = geometry->n_cells[0] * geometry->n_cells[1] * geometry->n_cells[2];
    std::string config = "";

    auto * ham = dynamic_cast<Engine::Hamiltonian_Heisenberg *>( hamiltonian.get() );

    // External Field
    config += "###    External Field:\n";
    config += fmt::format(
        "{:<25} {}\n", "external_field_magnitude", ham->external_field_magnitude / Utility::Constants::mu_B );
    config += fmt::format( "{:<25} {}\n", "external_field_normal", ham->external_field_normal.transpose() );

    // Anisotropy
    config += "###    Anisotropy:\n";
    scalar K = 0;
    Vector3 K_normal{ 0, 0, 0 };
    if( !ham->anisotropy_indices.empty() )
    {
        K        = ham->anisotropy_magnitudes[0];
        K_normal = ham->anisotropy_normals[0];
    }
    config += fmt::format( "{:<25} {}\n", "anisotropy_magnitude", K );
    config += fmt::format( "{:<25} {}\n", "anisotropy_normal", K_normal.transpose() );

    config += "###    Interaction pairs:\n";
    config += fmt::format( "n_interaction_pairs {}\n", ham->exchange_pairs.size() + ham->dmi_pairs.size() );
    if( ham->exchange_pairs.size() + ham->dmi_pairs.size() > 0 )
    {
        config += fmt::format(
            "{:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15}    {:^15} {:^15} {:^15} {:^15}\n", "i", "j", "da", "db", "dc",
            "Jij", "Dij", "Dijx", "Dijy", "Dijz" );
        // Exchange
        for( unsigned int i = 0; i < ham->exchange_pairs.size(); ++i )
        {
            config += fmt::format(
                "{:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15.8f}    {:^15.8f} {:^15.8f} {:^15.8f} {:^15.8f}\n",
                ham->exchange_pairs[i].i, ham->exchange_pairs[i].j, ham->exchange_pairs[i].translations[0],
                ham->exchange_pairs[i].translations[1], ham->exchange_pairs[i].translations[2],
                ham->exchange_magnitudes[i], 0.0, 0.0, 0.0, 0.0 );
        }
        // DMI
        for( unsigned int i = 0; i < ham->dmi_pairs.size(); ++i )
        {
            config += fmt::format(
                "{:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15.8f}    {:^15.8f} {:^15.8f} {:^15.8f} {:^15.8f}\n",
                ham->dmi_pairs[i].i, ham->dmi_pairs[i].j, ham->dmi_pairs[i].translations[0],
                ham->dmi_pairs[i].translations[1], ham->dmi_pairs[i].translations[2], 0.0, ham->dmi_magnitudes[i],
                ham->dmi_normals[i][0], ham->dmi_normals[i][1], ham->dmi_normals[i][2] );
        }
    }

    // Dipole-dipole
    std::string ddi_method;
    if( ham->ddi_method == Engine::DDI_Method::None )
        ddi_method = "none";
    else if( ham->ddi_method == Engine::DDI_Method::FFT )
        ddi_method = "fft";
    else if( ham->ddi_method == Engine::DDI_Method::FMM )
        ddi_method = "fmm";
    else if( ham->ddi_method == Engine::DDI_Method::Cutoff )
        ddi_method = "cutoff";
    config += "### Dipole-dipole interaction caclulation method\n### (fft, fmm, cutoff, none)";
    config += fmt::format( "ddi_method                 {}\n", ddi_method );
    config += "### DDI number of periodic images in (a b c)";
    config += fmt::format(
        "ddi_n_periodic_images      {} {} {}\n", ham->ddi_n_periodic_images[0], ham->ddi_n_periodic_images[1],
        ham->ddi_n_periodic_images[2] );
    config += "### DDI cutoff radius (if cutoff is used)";
    config += fmt::format( "ddi_radius                 {}\n", ham->ddi_cutoff_radius );

    // Quadruplets
    config += "###    Quadruplets:\n";
    config += fmt::format( "n_interaction_quadruplets {}\n", ham->quadruplets.size() );
    if( !ham->quadruplets.empty() )
    {
        config += fmt::format(
            "{:^3} {:^3} {:^3} {:^3}    {:^3} {:^3} {:^3}    {:^3} {:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15}\n", "i",
            "j", "k", "l", "da_j", "db_j", "dc_j", "da_k", "db_k", "dc_k", "da_l", "db_l", "dc_l", "Q" );
        for( unsigned int i = 0; i < ham->quadruplets.size(); ++i )
        {
            config += fmt::format(
                "{:^3} {:^3} {:^3} {:^3}    {:^3} {:^3} {:^3}    {:^3} {:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15.8f}\n",
                ham->quadruplets[i].i, ham->quadruplets[i].j, ham->quadruplets[i].k, ham->quadruplets[i].l,
                ham->quadruplets[i].d_j[0], ham->quadruplets[i].d_j[1], ham->quadruplets[i].d_j[2],
                ham->quadruplets[i].d_k[0], ham->quadruplets[i].d_k[1], ham->quadruplets[i].d_k[2],
                ham->quadruplets[i].d_l[0], ham->quadruplets[i].d_l[1], ham->quadruplets[i].d_l[2],
                ham->quadruplet_magnitudes[i] );
        }
    }

    Append_String_to_File( config, config_file );
}

void Hamiltonian_Gaussian_to_Config(
    const std::string & config_file, const std::shared_ptr<Engine::Hamiltonian> hamiltonian )
{
    std::string config = "";

    auto * ham_gaussian = dynamic_cast<Engine::Hamiltonian_Gaussian *>( hamiltonian.get() );
    config += fmt::format( "n_gaussians {}\n", ham_gaussian->n_gaussians );
    if( ham_gaussian->n_gaussians > 0 )
    {
        config += "gaussians\n";
        for( int i = 0; i < ham_gaussian->n_gaussians; ++i )
        {
            config += fmt::format(
                "{} {} {}\n", ham_gaussian->amplitude[i], ham_gaussian->width[i], ham_gaussian->center[i].transpose() );
        }
    }
    Append_String_to_File( config, config_file );
}

} // namespace IO