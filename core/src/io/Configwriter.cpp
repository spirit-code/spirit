#include <io/IO.hpp>
#include <io/Filter_File_Handle.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Neighbours.hpp>
#include <utility/Constants.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include <fmt/format.h>
#include <fmt/ostream.h>

using namespace Utility;

namespace IO
{
    void Folders_to_Config(const std::string configFile,
            const std::shared_ptr<Data::Parameters_Method_LLG> parameters_llg,
            const std::shared_ptr<Data::Parameters_Method_MC> parameters_mc,
            const std::shared_ptr<Data::Parameters_Method_GNEB> parameters_gneb,
            const std::shared_ptr<Data::Parameters_Method_MMF> parameters_mmf)
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
        Append_String_to_File(config, configFile);
    }// End Folders_to_Config


    void Log_Levels_to_Config(const std::string configFile)
    {
        std::string config = "";
        config += "############### Logging Parameters ###############\n";
        config += fmt::format("{:<22} {}\n", "log_to_file",            (int)Log.messages_to_file);
        config += fmt::format("{:<22} {}\n", "log_file_level",         (int)Log.level_file);
        config += fmt::format("{:<22} {}\n", "log_to_console",         (int)Log.messages_to_console);
        config += fmt::format("{:<22} {}\n", "log_console_level",      (int)Log.level_console);
        config += fmt::format("{:<22} {}\n", "log_input_save_initial", (int)Log.save_input_initial);
        config += fmt::format("{:<22} {}\n", "log_input_save_final",   (int)Log.save_input_final);
        config += "############# End Logging Parameters #############";
        Append_String_to_File(config, configFile);
    }// End Log_Levels_to_Config


    void Geometry_to_Config(const std::string configFile, const std::shared_ptr<Data::Geometry> geometry)
    {
        std::string config = "";
        config += "#################### Geometry ####################\n";
        config += "basis\n";
        config += fmt::format("{0}\n{1}\n{2}\n", geometry->basis[0].transpose(), geometry->basis[1].transpose(), geometry->basis[2].transpose());
        config += fmt::format("{}\n", geometry->n_spins_basic_domain);
        for (int i=0; i<geometry->n_spins_basic_domain; ++i)
        {
            config += fmt::format("{}\n", geometry->basis_atoms[i].transpose());
        }
        config += "translation_vectors\n";
        for (int i=0; i<3; ++i)
            config += fmt::format("{} {}\n", geometry->translation_vectors[i].transpose(), geometry->n_cells[i]);
        config += "################## End Geometry ##################";
        Append_String_to_File(config, configFile);
    }// end Geometry_to_Config


    void Parameters_Method_LLG_to_Config(const std::string configFile, const std::shared_ptr<Data::Parameters_Method_LLG> parameters)
    {
        std::string config = "";
        config += "################# LLG Parameters #################\n";
        config += fmt::format("{:<35} {:d}\n", "llg_output_any",                      parameters->output_any);
        config += fmt::format("{:<35} {:d}\n", "llg_output_initial",                  parameters->output_initial);
        config += fmt::format("{:<35} {:d}\n", "llg_output_final",                    parameters->output_final);
        config += fmt::format("{:<35} {:d}\n", "llg_output_energy_step",              parameters->output_energy_step);
        config += fmt::format("{:<35} {:d}\n", "llg_output_energy_archive",           parameters->output_energy_archive);
        config += fmt::format("{:<35} {:d}\n", "llg_output_energy_spin_resolved",     parameters->output_energy_spin_resolved);
        config += fmt::format("{:<35} {:d}\n", "llg_output_energy_divide_by_nspins",  parameters->output_energy_divide_by_nspins);
        config += fmt::format("{:<35} {:d}\n", "llg_output_configuration_step",       parameters->output_configuration_step);
        config += fmt::format("{:<35} {:d}\n", "llg_output_configuration_archive",    parameters->output_configuration_archive);
        config += fmt::format("{:<35} {:e}\n", "llg_force_convergence",               parameters->force_convergence);
        config += fmt::format("{:<35} {}\n",   "llg_n_iterations",                    parameters->n_iterations);
        config += fmt::format("{:<35} {}\n",   "llg_n_iterations_log",                parameters->n_iterations_log);
        config += fmt::format("{:<35} {}\n",   "llg_seed",                            parameters->rng_seed);
        config += fmt::format("{:<35} {}\n",   "llg_temperature",                     parameters->temperature);
        config += fmt::format("{:<35} {}\n",   "llg_damping",                         parameters->damping);
        config += fmt::format("{:<35} {}\n",   "llg_dt",                              parameters->dt/std::pow(10, -12) * Constants::mu_B/1.760859644/std::pow(10, 11));
        config += fmt::format("{:<35} {}\n",   "llg_stt_magnitude",                   parameters->stt_magnitude);
        config += fmt::format("{:<35} {}\n",   "llg_stt_polarisation_normal",         parameters->stt_polarisation_normal.transpose());
        config += "############### End LLG Parameters ###############";
        Append_String_to_File(config, configFile);
    }// end Parameters_Method_LLG_to_Config

    void Parameters_Method_MC_to_Config(const std::string configFile, const std::shared_ptr<Data::Parameters_Method_MC> parameters)
    {
        std::string config = "";
        config += "################# MC Parameters ##################\n";
        config += fmt::format("{:<35} {:d}\n", "mc_output_any",                      parameters->output_any);
        config += fmt::format("{:<35} {:d}\n", "mc_output_initial",                  parameters->output_initial);
        config += fmt::format("{:<35} {:d}\n", "mc_output_final",                    parameters->output_final);
        config += fmt::format("{:<35} {:d}\n", "mc_output_energy_step",              parameters->output_energy_step);
        config += fmt::format("{:<35} {:d}\n", "mc_output_energy_archive",           parameters->output_energy_archive);
        config += fmt::format("{:<35} {:d}\n", "mc_output_energy_spin_resolved",     parameters->output_energy_spin_resolved);
        config += fmt::format("{:<35} {:d}\n", "mc_output_energy_divide_by_nspins",  parameters->output_energy_divide_by_nspins);
        config += fmt::format("{:<35} {:d}\n", "mc_output_configuration_step",       parameters->output_configuration_step);
        config += fmt::format("{:<35} {:d}\n", "mc_output_configuration_archive",    parameters->output_configuration_archive);
        config += fmt::format("{:<35} {}\n",   "mc_n_iterations",                    parameters->n_iterations);
        config += fmt::format("{:<35} {}\n",   "mc_n_iterations_log",                parameters->n_iterations_log);
        config += fmt::format("{:<35} {}\n",   "mc_seed",                            parameters->rng_seed);
        config += fmt::format("{:<35} {}\n",   "mc_temperature",                     parameters->temperature);
        config += fmt::format("{:<35} {}\n",   "mc_acceptance_ratio",                parameters->acceptance_ratio_target);
        config += "############### End MC Parameters ################";
        Append_String_to_File(config, configFile);
    }// end Parameters_Method_MC_to_Config

    void Parameters_Method_GNEB_to_Config(const std::string configFile, const std::shared_ptr<Data::Parameters_Method_GNEB> parameters)
    {
        std::string config = "";
        config += "################# GNEB Parameters ################\n";
        config += fmt::format("{:<38} {:d}\n", "gneb_output_any",                       parameters->output_any);
        config += fmt::format("{:<38} {:d}\n", "gneb_output_initial",                   parameters->output_initial);
        config += fmt::format("{:<38} {:d}\n", "gneb_output_final",                     parameters->output_final);
        config += fmt::format("{:<38} {:d}\n", "gneb_output_energies_step",             parameters->output_energies_step);
        config += fmt::format("{:<38} {:d}\n", "gneb_output_energies_interpolated",     parameters->output_energies_interpolated);
        config += fmt::format("{:<38} {:d}\n", "gneb_output_energies_divide_by_nspins", parameters->output_energies_divide_by_nspins);
        config += fmt::format("{:<38} {:d}\n", "gneb_output_chain_step",                parameters->output_chain_step);
        config += fmt::format("{:<38} {:e}\n", "gneb_force_convergence",                parameters->force_convergence);
        config += fmt::format("{:<38} {}\n",   "gneb_n_iterations",                     parameters->n_iterations);
        config += fmt::format("{:<38} {}\n",   "gneb_n_iterations_log",                 parameters->n_iterations_log);
        config += fmt::format("{:<38} {}\n",   "gneb_spring_constant",                  parameters->spring_constant);
        config += fmt::format("{:<38} {}\n",   "gneb_n_energy_interpolations",          parameters->n_E_interpolations);
        config += "############### End GNEB Parameters ##############";
        Append_String_to_File(config, configFile);
    }// end Parameters_Method_GNEB_to_Config

    void Parameters_Method_MMF_to_Config(const std::string configFile, const std::shared_ptr<Data::Parameters_Method_MMF> parameters)
    {
        std::string config = "";
        config += "################# MMF Parameters #################\n";
        config += fmt::format("{:<38} {:d}\n", "mmf_output_any",                     parameters->output_any);
        config += fmt::format("{:<38} {:d}\n", "mmf_output_initial",                 parameters->output_initial);
        config += fmt::format("{:<38} {:d}\n", "mmf_output_final",                   parameters->output_final);
        config += fmt::format("{:<38} {:d}\n", "mmf_output_energy_step",             parameters->output_energy_step);
        config += fmt::format("{:<38} {:d}\n", "mmf_output_energy_archive",          parameters->output_energy_archive);
        config += fmt::format("{:<38} {:d}\n", "mmf_output_energy_divide_by_nspins", parameters->output_energy_divide_by_nspins);
        config += fmt::format("{:<38} {:d}\n", "mmf_output_configuration_step",      parameters->output_configuration_step);
        config += fmt::format("{:<38} {:d}\n", "mmf_output_configuration_archive",   parameters->output_configuration_archive);
        config += fmt::format("{:<38} {:e}\n", "mmf_force_convergence",              parameters->force_convergence);
        config += fmt::format("{:<38} {}\n",   "mmf_n_iterations",                   parameters->n_iterations);
        config += fmt::format("{:<38} {}\n",   "mmf_n_iterations_log",               parameters->n_iterations_log);
        config += "############### End MMF Parameters ###############";
        Append_String_to_File(config, configFile);
    }// end Parameters_Method_MMF_to_Config

    void Hamiltonian_to_Config(const std::string configFile, const std::shared_ptr<Engine::Hamiltonian> hamiltonian, const std::shared_ptr<Data::Geometry> geometry)
    {
        std::string config = "";
        config += "################### Hamiltonian ##################\n";
        std::string name;
        if (hamiltonian->Name() == "Heisenberg (Pairs)") name = "heisenberg_pairs";
        else if (hamiltonian->Name() == "Heisenberg (Neighbours)") name = "heisenberg_neighbours";
        else if (hamiltonian->Name() == "Gaussian") name = "gaussian";
        config += fmt::format("{:<25} {}\n",       "hamiltonian", name);
        config += fmt::format("{:<25} {} {} {}\n", "boundary_conditions", hamiltonian->boundary_conditions[0], hamiltonian->boundary_conditions[1], hamiltonian->boundary_conditions[2]);
        Append_String_to_File(config, configFile);

        if (hamiltonian->Name() == "Heisenberg (Pairs)") Hamiltonian_Heisenberg_Pairs_to_Config(configFile, hamiltonian, geometry);
        else if (hamiltonian->Name() == "Heisenberg (Neighbours)") Hamiltonian_Heisenberg_Neighbours_to_Config(configFile, hamiltonian);
        else if (hamiltonian->Name() == "Gaussian") Hamiltonian_Gaussian_to_Config(configFile, hamiltonian);

        config = "################# End Hamiltonian ################";
        Append_String_to_File(config, configFile);
    }// end Hamiltonian_to_Config

    void Hamiltonian_Heisenberg_Neighbours_to_Config(const std::string configFile, const std::shared_ptr<Engine::Hamiltonian> hamiltonian)
    {
        std::string config = "";
        Engine::Hamiltonian_Heisenberg_Neighbours * ham = (Engine::Hamiltonian_Heisenberg_Neighbours *)hamiltonian.get();
        config += fmt::format("{:<25} {}\n", "external_field_magnitude", ham->external_field_magnitudes[0]/Constants::mu_B/ham->mu_s[0]);
        config += fmt::format("{:<25} {}\n", "external_field_normal",    ham->external_field_normals[0].transpose());
        config += fmt::format("{:<25} {}\n", "mu_s",                     ham->mu_s[0]);
        config += fmt::format("{:<25} {}\n", "anisotropy_magnitude",     ham->anisotropy_magnitudes[0]);
        config += fmt::format("{:<25} {}\n", "anisotropy_normal",        ham->anisotropy_normals[0].transpose());
        config += fmt::format("{:<25} {}\n", "n_neigh_shells",           ham->exchange_magnitudes.size());
        config += fmt::format("{:<25} {}\n", "jij",                      ham->exchange_magnitudes[0]);
        for (unsigned int i=1; i<ham->exchange_magnitudes.size(); ++i)
            config += fmt::format(" {}", ham->exchange_magnitudes[i]);
        config += "\n";
        config += "\n";
        config += fmt::format("{:<25} {}\n", "n_neigh_shells_dmi", ham->dmi_magnitudes.size());
        config += fmt::format("{:<25} {}\n", "dij",                ham->dmi_magnitudes[0]);
        for (unsigned int i=1; i<ham->dmi_magnitudes.size(); ++i)
            config += fmt::format(" {}", ham->dmi_magnitudes[i]);
        config += "\n";
        config += "\n";
        config += fmt::format("{:<25} {}\n", "dd_radius", ham->ddi_radius);
        Append_String_to_File(config, configFile);
    }// end Hamiltonian_Heisenberg_Neighbours_to_Config

    void Hamiltonian_Heisenberg_Pairs_to_Config(const std::string configFile, const std::shared_ptr<Engine::Hamiltonian> hamiltonian, const std::shared_ptr<Data::Geometry> geometry)
    {
        int n_cells_tot = geometry->n_cells[0]*geometry->n_cells[1]*geometry->n_cells[2];
        std::string config = "";
        Engine::Hamiltonian_Heisenberg_Pairs* ham = (Engine::Hamiltonian_Heisenberg_Pairs *)hamiltonian.get();
        
        // Magnetic moment
        config += "mu_s                     ";
        for (int i=0; i<geometry->n_spins_basic_domain; ++i)
            config += fmt::format(" {}", ham->mu_s[i]);
        config += "\n";

        // External Field
        config += "###    External Field:\n";
        scalar B = 0;
        Vector3 B_normal{ 0,0,0 };
        if (ham->external_field_indices.size() > 0)
        {
            B = ham->external_field_magnitudes[0] / ham->mu_s[0] / Constants::mu_B;
            B_normal = ham->external_field_normals[0];
        }
        config += fmt::format("{:<25} {}\n", "external_field_magnitude", B);
        config += fmt::format("{:<25} {}\n", "external_field_normal",    B_normal.transpose());
        
        // Anisotropy
        config += "###    Anisotropy:\n";
        scalar K = 0;
        Vector3 K_normal{ 0,0,0 };
        if (ham->anisotropy_indices.size() > 0)
        {
            K = ham->anisotropy_magnitudes[0];
            K_normal = ham->anisotropy_normals[0];
        }
        config += fmt::format("{:<25} {}\n", "anisotropy_magnitude", K);
        config += fmt::format("{:<25} {}\n", "anisotropy_normal", K_normal.transpose());
        
        config += "###    Interaction pairs:\n";
        config += fmt::format("n_interaction_pairs {}\n", ham->exchange_pairs.size() + ham->dmi_pairs.size());
        if (ham->exchange_pairs.size() + ham->dmi_pairs.size() > 0)
        {
            config += fmt::format("{:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15}    {:^15} {:^15} {:^15} {:^15}\n",
                "i", "j", "da", "db", "dc", "J", "Dij", "Dx", "Dy", "Dz");
            // Exchange
            for (unsigned int i=0; i<ham->exchange_pairs.size(); ++i)
            {
                config += fmt::format("{:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15.8f}    {:^15.8f} {:^15.8f} {:^15.8f} {:^15.8f}\n",
                    ham->exchange_pairs[i].i, ham->exchange_pairs[i].j,
                    ham->exchange_pairs[i].translations[0], ham->exchange_pairs[i].translations[1], ham->exchange_pairs[i].translations[2],
                    ham->exchange_magnitudes[i], 0.0, 0.0, 0.0, 0.0);
            }
            // DMI
            for (unsigned int i = 0; i<ham->dmi_pairs.size(); ++i)
            {
                config += fmt::format("{:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15.8f}    {:^15.8f} {:^15.8f} {:^15.8f} {:^15.8f}\n",
                    ham->dmi_pairs[i].i, ham->dmi_pairs[i].j,
                    ham->dmi_pairs[i].translations[0], ham->dmi_pairs[i].translations[1], ham->dmi_pairs[i].translations[2],
                    0.0, ham->dmi_magnitudes[i], ham->dmi_normals[i][0], ham->dmi_normals[i][1], ham->dmi_normals[i][2]);
            }
        }

        // Quadruplets
        config += "###    Quadruplets:\n";
        config += fmt::format("n_interaction_quadruplets {}\n", ham->quadruplets.size());
        if (ham->quadruplets.size() > 0)
        {
            config += fmt::format("{:^3} {:^3} {:^3} {:^3}    {:^3} {:^3} {:^3}    {:^3} {:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15}\n",
                "i", "j", "k", "l",   "da_j", "db_j", "dc_j",   "da_k", "db_k", "dc_k",   "da_l", "db_l", "dc_l",  "Q");
            for (unsigned int i=0; i<ham->quadruplets.size(); ++i)
            {
                config += fmt::format("{:^3} {:^3} {:^3} {:^3}    {:^3} {:^3} {:^3}    {:^3} {:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15.8f}\n",
                    ham->quadruplets[i].i, ham->quadruplets[i].j, ham->quadruplets[i].k, ham->quadruplets[i].l,
                    ham->quadruplets[i].d_j[0], ham->quadruplets[i].d_j[1], ham->quadruplets[i].d_j[2],
                    ham->quadruplets[i].d_k[0], ham->quadruplets[i].d_k[1], ham->quadruplets[i].d_k[2],
                    ham->quadruplets[i].d_l[0], ham->quadruplets[i].d_l[1], ham->quadruplets[i].d_l[2],
                    ham->quadruplet_magnitudes[i]);
            }
        }

        Append_String_to_File(config, configFile);
    }// end Hamiltonian_Heisenberg_Pairs_to_Config

    void Hamiltonian_Gaussian_to_Config(const std::string configFile, const std::shared_ptr<Engine::Hamiltonian> hamiltonian)
    {
        std::string config = "";
        Engine::Hamiltonian_Gaussian * ham_gaussian = (Engine::Hamiltonian_Gaussian *)hamiltonian.get();
        config += fmt::format("n_gaussians {}\n", ham_gaussian->n_gaussians);
        if (ham_gaussian->n_gaussians > 0)
        {
            config += "gaussians\n";
            for (int i=0; i< ham_gaussian->n_gaussians; ++i)
            {
                config +=fmt::format("{} {} {}\n", ham_gaussian->amplitude[i], ham_gaussian->width[i], ham_gaussian->center[i].transpose());
            }
        }
        Append_String_to_File(config, configFile);
    }// end Hamiltonian_Gaussian_to_Config
}// end namespace IO