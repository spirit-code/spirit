#include <engine/Neighbours.hpp>
#include <engine/Vectormath.hpp>
#include <engine/spin/Hamiltonian.hpp>
#include <io/Configwriter.hpp>
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
    append_to_file( config, config_file );
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
    append_to_file( config, config_file );
}

void Geometry_to_Config( const std::string & config_file, const Data::Geometry & geometry )
{
    // TODO: this needs to be updated!
    std::string config = "";
    config += "#################### Geometry ####################\n";

    // Bravais lattice/vectors
    if( geometry.classifier == Data::BravaisLatticeType::SC )
        config += "bravais_lattice sc\n";
    else if( geometry.classifier == Data::BravaisLatticeType::FCC )
        config += "bravais_lattice fcc\n";
    else if( geometry.classifier == Data::BravaisLatticeType::BCC )
        config += "bravais_lattice bcc\n";
    else if( geometry.classifier == Data::BravaisLatticeType::Hex2D )
        config += "bravais_lattice hex2d120\n";
    else
    {
        config += "bravais_vectors\n";
        config += fmt::format(
            "{0}\n{1}\n{2}\n", geometry.bravais_vectors[0].transpose(), geometry.bravais_vectors[1].transpose(),
            geometry.bravais_vectors[2].transpose() );
    }

    // Number of cells
    config
        += fmt::format( "n_basis_cells {} {} {}\n", geometry.n_cells[0], geometry.n_cells[1], geometry.n_cells[2] );

    // Optionally basis
    if( geometry.n_cell_atoms > 1 )
    {
        config += "basis\n";
        config += fmt::format( "{}\n", geometry.n_cell_atoms );
        for( int i = 0; i < geometry.n_cell_atoms; ++i )
        {
            config += fmt::format( "{}\n", geometry.cell_atoms[i].transpose() );
        }
    }

    // Magnetic moment
    if( !geometry.cell_composition.disordered )
    {
        config += "mu_s                     ";
        for( int i = 0; i < geometry.n_cell_atoms; ++i )
            config += fmt::format( " {}", geometry.mu_s[i] );
        config += "\n";
    }
    else
    {
        const auto & iatom         = geometry.cell_composition.iatom;
        const auto & atom_type     = geometry.cell_composition.atom_type;
        const auto & mu_s          = geometry.cell_composition.mu_s;
        const auto & concentration = geometry.cell_composition.concentration;
        config += fmt::format( "atom_types    {}\n", iatom.size() );
        for( std::size_t i = 0; i < iatom.size(); ++i )
            config += fmt::format( "{}   {}   {}   {}\n", iatom[i], atom_type[i], mu_s[i], concentration[i] );
    }

    // Optionally lattice constant
    if( std::abs( geometry.lattice_constant - 1 ) > 1e-6 )
        config += fmt::format( "lattice_constant {}\n", geometry.lattice_constant );

    config += "################## End Geometry ##################";
    append_to_file( config, config_file );
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
    append_to_file( config, config_file );
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
    append_to_file( config, config_file );
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
    append_to_file( config, config_file );
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
    append_to_file( config, config_file );
}

void Hamiltonian_to_Config(
    const std::string & config_file, const std::shared_ptr<Engine::Spin::HamiltonianVariant> hamiltonian )
{
    std::string config = "";
    config += "################### Hamiltonian ##################\n";
    std::string name;
    if( hamiltonian->Name() == "Heisenberg" )
        name = "heisenberg_pairs";
    else if( hamiltonian->Name() == "Gaussian" )
        name = "gaussian";
    config += fmt::format( "{:<25} {}\n", "hamiltonian", name );
    config += []( const auto & bc ) {
        return fmt::format( "{:<25} {} {} {}\n", "boundary_conditions", bc[0], bc[1], bc[2] );
    }( hamiltonian->get_boundary_conditions() );
    append_to_file( config, config_file );

    if( hamiltonian->Name() == "Heisenberg" )
        Hamiltonian_Heisenberg_to_Config( config_file, hamiltonian );
    else if( hamiltonian->Name() == "Gaussian" )
        Hamiltonian_Gaussian_to_Config( config_file, hamiltonian );

    config = "################# End Hamiltonian ################";
    append_to_file( config, config_file );
}

void Hamiltonian_Heisenberg_to_Config(
    const std::string & config_file, const std::shared_ptr<Engine::Spin::HamiltonianVariant> hamiltonian )
{
    std::string config = "";

    // External Field
    if( const auto * data = hamiltonian->data<Engine::Spin::Interaction::Zeeman>(); data != nullptr )
    {
        const scalar & external_field_magnitude = data->external_field_magnitude;
        const Vector3 & external_field_normal   = data->external_field_normal;

        config += "###    External Field:\n";
        config += fmt::format(
            "{:<25} {}\n", "external_field_magnitude", external_field_magnitude / Utility::Constants::mu_B );
        config += fmt::format( "{:<25} {}\n", "external_field_normal", external_field_normal.transpose() );
    }
    // Anisotropy

    if( const auto * data = hamiltonian->data<Engine::Spin::Interaction::Anisotropy>(); data != nullptr )
    {
        config += "###    Anisotropy:\n";
        const intfield & anisotropy_indices       = data->indices;
        const scalarfield & anisotropy_magnitudes = data->magnitudes;
        const vectorfield & anisotropy_normals    = data->normals;

        scalar K = 0;
        Vector3 K_normal{ 0, 0, 0 };
        if( !anisotropy_indices.empty() )
        {
            K        = anisotropy_magnitudes[0];
            K_normal = anisotropy_normals[0];
        }
        config += fmt::format( "{:<25} {}\n", "anisotropy_magnitude", K );
        config += fmt::format( "{:<25} {}\n", "anisotropy_normal", K_normal.transpose() );
    }
    // Biaxial Anisotropy
    if( const auto * data = hamiltonian->data<Engine::Spin::Interaction::Biaxial_Anisotropy>(); data != nullptr )
    {
        const intfield & indices                        = data->indices;
        const field<PolynomialBasis> & polynomial_bases = data->bases;
        const field<unsigned int> & polynomial_site_p   = data->site_p;
        const field<PolynomialTerm> & polynomial_terms  = data->terms;

        const auto n_anisotropy_axes  = indices.size();
        const auto n_anisotropy_terms = polynomial_terms.size();

        assert( n_anisotropy_axes != 0 || n_anisotropy_terms == 0 );

        config += "###   Biaxial Anisotropy Axes\n";
        config += fmt::format( "n_anisotropy_axes {}\n", n_anisotropy_axes );

        if( n_anisotropy_axes > 0 )
        {
            config += fmt::format(
                "{:^3}   {:^15} {:^15} {:^15}  {:^15} {:^15} {:^15}\n", "i", "K1x", "K1y", "K1z", "K2x", "K2y", "K2z" );

            for( std::size_t i = 0; i < n_anisotropy_axes; ++i )
            {
                config += fmt::format(
                    "{:^3}   {:^15.8f} {:^15.8f} {:^15.8f}  {:^15.8f} {:^15.8f} {:^15.8f}\n", indices[i],
                    polynomial_bases[i].k1[0], polynomial_bases[i].k1[1], polynomial_bases[i].k1[2],
                    polynomial_bases[i].k2[0], polynomial_bases[i].k2[1], polynomial_bases[i].k2[2] );
            }
        }
        config += "###   Biaxial Anisotropy Terms\n";
        config += fmt::format( "n_anisotropy_terms {}\n", n_anisotropy_terms );

        if( n_anisotropy_axes > 0 )
        {
            config += fmt::format( "{:^3}  {:^3} {:^3} {:^3}  {:^15}\n", "i", "n1", "n2", "n3", "k" );
            for( std::size_t i = 0; i < n_anisotropy_axes; ++i )
            {
                for( std::size_t j = polynomial_site_p[i]; j < polynomial_site_p[i + 1]; ++j )
                {
                    const auto & p = polynomial_terms[j];
                    config
                        += fmt::format( "{:^3}  {:^3} {:^3} {:^3}  {:^15.8f}\n", i, p.n1, p.n2, p.n3, p.coefficient );
                }
            }
        }
    }

    // Pair interactions (Exchange & DMI)
    {
        const auto * exchange_cache = hamiltonian->cache<Engine::Spin::Interaction::Exchange>();
        const auto * dmi_cache      = hamiltonian->cache<Engine::Spin::Interaction::DMI>();

        if( exchange_cache != nullptr && dmi_cache != nullptr )
        {
            const auto & exchange_pairs      = exchange_cache->pairs;
            const auto & exchange_magnitudes = exchange_cache->magnitudes;
            const auto & dmi_pairs           = dmi_cache->pairs;
            const auto & dmi_magnitudes      = dmi_cache->magnitudes;
            const auto & dmi_normals         = dmi_cache->normals;

            config += "###    Interaction pairs:\n";
            config += fmt::format( "n_interaction_pairs {}\n", exchange_pairs.size() + dmi_pairs.size() );
            if( exchange_pairs.size() + dmi_pairs.size() > 0 )
            {
                config += fmt::format(
                    "{:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15}    {:^15} {:^15} {:^15} {:^15}\n", "i", "j", "da", "db",
                    "dc", "Jij", "Dij", "Dijx", "Dijy", "Dijz" );
                // Exchange
                for( unsigned int i = 0; i < exchange_pairs.size(); ++i )
                {
                    config += fmt::format(
                        "{:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15.8f}    {:^15.8f} {:^15.8f} {:^15.8f} {:^15.8f}\n",
                        exchange_pairs[i].i, exchange_pairs[i].j, exchange_pairs[i].translations[0],
                        exchange_pairs[i].translations[1], exchange_pairs[i].translations[2], exchange_magnitudes[i],
                        0.0, 0.0, 0.0, 0.0 );
                }
                // DMI
                for( unsigned int i = 0; i < dmi_pairs.size(); ++i )
                {
                    config += fmt::format(
                        "{:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15.8f}    {:^15.8f} {:^15.8f} {:^15.8f} {:^15.8f}\n",
                        dmi_pairs[i].i, dmi_pairs[i].j, dmi_pairs[i].translations[0], dmi_pairs[i].translations[1],
                        dmi_pairs[i].translations[2], 0.0, dmi_magnitudes[i], dmi_normals[i][0], dmi_normals[i][1],
                        dmi_normals[i][2] );
                }
            }
        }
    }

    // Dipole-dipole
    if( const auto * data = hamiltonian->data<Engine::Spin::Interaction::DDI>(); data != nullptr )
    {
        std::string ddi_method;

        const auto & ddi_method_id         = data->method;
        const auto & ddi_n_periodic_images = data->n_periodic_images;
        const auto & ddi_pb_zero_padding   = data->pb_zero_padding;
        const auto & ddi_cutoff_radius     = data->cutoff_radius;

        if( ddi_method_id == Engine::Spin::DDI_Method::None )
            ddi_method = "none";
        else if( ddi_method_id == Engine::Spin::DDI_Method::FFT )
            ddi_method = "fft";
        else if( ddi_method_id == Engine::Spin::DDI_Method::FMM )
            ddi_method = "fmm";
        else if( ddi_method_id == Engine::Spin::DDI_Method::Cutoff )
            ddi_method = "cutoff";
        config += "### Dipole-dipole interaction caclulation method\n### (fft, fmm, cutoff, none)";
        config += fmt::format( "ddi_method                 {}\n", ddi_method );
        config += "### DDI number of periodic images in (a b c)";
        config += fmt::format(
            "ddi_n_periodic_images      {} {} {}\n", ddi_n_periodic_images[0], ddi_n_periodic_images[1],
            ddi_n_periodic_images[2] );
        config += "### DDI cutoff radius (if cutoff is used)";
        config += fmt::format( "ddi_radius                 {}\n", ddi_cutoff_radius );
    }
    // Quadruplets
    if( const auto * data = hamiltonian->data<Engine::Spin::Interaction::Quadruplet>(); data != nullptr )
    {
        const auto & quadruplets           = data->quadruplets;
        const auto & quadruplet_magnitudes = data->magnitudes;

        config += "###    Quadruplets:\n";
        config += fmt::format( "n_interaction_quadruplets {}\n", quadruplets.size() );
        if( !quadruplets.empty() )
        {
            config += fmt::format(
                "{:^3} {:^3} {:^3} {:^3}    {:^3} {:^3} {:^3}    {:^3} {:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15}\n",
                "i", "j", "k", "l", "da_j", "db_j", "dc_j", "da_k", "db_k", "dc_k", "da_l", "db_l", "dc_l", "Q" );
            for( unsigned int i = 0; i < quadruplets.size(); ++i )
            {
                // clang-format off
                config += fmt::format(
                    "{:^3} {:^3} {:^3} {:^3}    {:^3} {:^3} {:^3}    {:^3} {:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15.8f}\n",
                    quadruplets[i].i, quadruplets[i].j, quadruplets[i].k, quadruplets[i].l,
                    quadruplets[i].d_j[0], quadruplets[i].d_j[1], quadruplets[i].d_j[2],
                    quadruplets[i].d_k[0], quadruplets[i].d_k[1], quadruplets[i].d_k[2],
                    quadruplets[i].d_l[0], quadruplets[i].d_l[1], quadruplets[i].d_l[2],
                    quadruplet_magnitudes[i] );
                // clang-format on
            }
        }
    }
    append_to_file( config, config_file );
}

void Hamiltonian_Gaussian_to_Config(
    const std::string & config_file, const std::shared_ptr<Engine::Spin::HamiltonianVariant> hamiltonian )
{
    std::string config = "";

    if( const auto * data = hamiltonian->data<Engine::Spin::Interaction::Gaussian>(); data != nullptr )
    {
        const scalarfield & amplitude = data->amplitude;
        const scalarfield & width     = data->width;
        const vectorfield & center    = data->center;

        const auto n_gaussians = amplitude.size();

        config += fmt::format( "n_gaussians {}\n", n_gaussians );
        if( n_gaussians > 0 )
        {
            config += "gaussians\n";
            for( std::size_t i = 0; i < n_gaussians; ++i )
            {
                config += fmt::format( "{} {} {}\n", amplitude[i], width[i], center[i].transpose() );
            }
        }
    }
    append_to_file( config, config_file );
}

} // namespace IO
