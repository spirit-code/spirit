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
		config += "output_tag_time    " + std::to_string(Log.tag_time) + "\n";
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
		config += "log_to_file            " + std::to_string((int)Log.messages_to_file) + "\n";
		config += "log_file_level         " + std::to_string((int)Log.level_file) + "\n";
		config += "log_to_console         " + std::to_string((int)Log.messages_to_console) + "\n";
		config += "log_console_level      " + std::to_string((int)Log.level_console) + "\n";
		config += "log_input_save_initial " + std::to_string((int)Log.save_input_initial) + "\n";
		config += "log_input_save_final   " + std::to_string((int)Log.save_input_final) + "\n";
		config += "############# End Logging Parameters #############";
		Append_String_to_File(config, configFile);
	}// End Log_Levels_to_Config


	void Geometry_to_Config(const std::string configFile, const std::shared_ptr<Data::Geometry> geometry)
	{
		std::string config = "";
		config += "#################### Geometry ####################\n";
		config += "basis\n";
		config += fmt::format("{0}\n{1}\n{2}\n", geometry->basis[0], geometry->basis[1], geometry->basis[2]);
		config += std::to_string(geometry->n_spins_basic_domain) + "\n";
		for (int i=0; i<geometry->n_spins_basic_domain; ++i)
		{
			config += std::to_string(geometry->basis_atoms[i][0]) + " " + std::to_string(geometry->basis_atoms[i][1]) + " " + std::to_string(geometry->basis_atoms[i][2]) + "\n";
		}
		config += "translation_vectors\n";
		config += std::to_string(geometry->translation_vectors[0][0]) + " " + std::to_string(geometry->translation_vectors[0][1]) + " " + std::to_string(geometry->translation_vectors[0][2]) + " " + std::to_string(geometry->n_cells[0]) + "\n";
		config += std::to_string(geometry->translation_vectors[1][0]) + " " + std::to_string(geometry->translation_vectors[1][1]) + " " + std::to_string(geometry->translation_vectors[1][2]) + " " + std::to_string(geometry->n_cells[1]) + "\n";
		config += std::to_string(geometry->translation_vectors[2][0]) + " " + std::to_string(geometry->translation_vectors[2][1]) + " " + std::to_string(geometry->translation_vectors[2][2]) + " " + std::to_string(geometry->n_cells[2]) + "\n";
		config += "################## End Geometry ##################";
		Append_String_to_File(config, configFile);
	}// end Geometry_to_Config


	void Parameters_Method_LLG_to_Config(const std::string configFile, const std::shared_ptr<Data::Parameters_Method_LLG> parameters)
	{
		std::string config = "";
		config += "################# LLG Parameters #################\n";
		config += "llg_output_any            " + std::to_string(parameters->output_any) + "\n";
		config += "llg_output_initial        " + std::to_string(parameters->output_initial) + "\n";
		config += "llg_output_final          " + std::to_string(parameters->output_final) + "\n";
		config += "llg_output_energy_step             " + std::to_string(parameters->output_energy_step) + "\n";
		config += "llg_output_energy_archive          " + std::to_string(parameters->output_energy_archive) + "\n";
		config += "llg_output_energy_spin_resolved    " + std::to_string(parameters->output_energy_spin_resolved) + "\n";
		config += "llg_output_energy_divide_by_nspins " + std::to_string(parameters->output_energy_divide_by_nspins) + "\n";
		config += "llg_output_configuration_step      " + std::to_string(parameters->output_configuration_step) + "\n";
		config += "llg_output_configuration_archive   " + std::to_string(parameters->output_configuration_archive) + "\n";
		config += "llg_force_convergence          " + fmt::format("{:e}", parameters->force_convergence) + "\n";
		config += "llg_n_iterations               " + std::to_string(parameters->n_iterations) + "\n";
		config += "llg_n_iterations_log           " + std::to_string(parameters->n_iterations_log) + "\n";
		config += "llg_seed                       " + std::to_string(parameters->rng_seed) + "\n";
		config += "llg_temperature                " + std::to_string(parameters->temperature) + "\n";
		config += "llg_damping                    " + std::to_string(parameters->damping) + "\n";
		config += "llg_dt                         " + std::to_string(parameters->dt/std::pow(10, -12) * Constants::mu_B/1.760859644/std::pow(10, 11)) + "\n";
		config += "llg_stt_magnitude              " + std::to_string(parameters->stt_magnitude) + "\n";
		config += "llg_stt_polarisation_normal    " + std::to_string(parameters->stt_polarisation_normal[0]) + " " + std::to_string(parameters->stt_polarisation_normal[1]) + " " + std::to_string(parameters->stt_polarisation_normal[2]) + "\n";
		config += "############### End LLG Parameters ###############";
		Append_String_to_File(config, configFile);
	}// end Parameters_Method_LLG_to_Config

	void Parameters_Method_MC_to_Config(const std::string configFile, const std::shared_ptr<Data::Parameters_Method_MC> parameters)
	{
		std::string config = "";
		config += "################# MC Parameters ##################\n";
		config += "mc_output_any            " + std::to_string(parameters->output_any) + "\n";
		config += "mc_output_initial        " + std::to_string(parameters->output_initial) + "\n";
		config += "mc_output_final          " + std::to_string(parameters->output_final) + "\n";
		config += "mc_output_energy_step             " + std::to_string(parameters->output_energy_step) + "\n";
		config += "mc_output_energy_archive          " + std::to_string(parameters->output_energy_archive) + "\n";
		config += "mc_output_energy_spin_resolved    " + std::to_string(parameters->output_energy_spin_resolved) + "\n";
		config += "mc_output_energy_divide_by_nspins " + std::to_string(parameters->output_energy_divide_by_nspins) + "\n";
		config += "mc_output_configuration_step      " + std::to_string(parameters->output_configuration_step) + "\n";
		config += "mc_output_configuration_archive   " + std::to_string(parameters->output_configuration_archive) + "\n";
		config += "mc_n_iterations               " + std::to_string(parameters->n_iterations) + "\n";
		config += "mc_n_iterations_log           " + std::to_string(parameters->n_iterations_log) + "\n";
		config += "mc_seed                       " + std::to_string(parameters->rng_seed) + "\n";
		config += "mc_temperature                " + std::to_string(parameters->temperature) + "\n";
		config += "mc_acceptance_ratio           " + std::to_string(parameters->acceptance_ratio_target) + "\n";
		config += "############### End MC Parameters ################";
		Append_String_to_File(config, configFile);
	}// end Parameters_Method_LLG_to_Config

	void Parameters_Method_GNEB_to_Config(const std::string configFile, const std::shared_ptr<Data::Parameters_Method_GNEB> parameters)
	{
		std::string config = "";
		config += "################# GNEB Parameters ################\n";
		config += "gneb_output_any           " + std::to_string(parameters->output_any) + "\n";
		config += "gneb_output_initial       " + std::to_string(parameters->output_initial) + "\n";
		config += "gneb_output_final         " + std::to_string(parameters->output_final) + "\n";
		config += "gneb_output_energies_step " + std::to_string(parameters->output_energies_step) + "\n";
		config += "gneb_output_energies_interpolated     " + std::to_string(parameters->output_energies_interpolated) + "\n";
		config += "gneb_output_energies_divide_by_nspins " + std::to_string(parameters->output_energies_divide_by_nspins) + "\n";
		config += "gneb_output_chain_step    " + std::to_string(parameters->output_chain_step) + "\n";
		config += "gneb_force_convergence         " + fmt::format("{:e}", parameters->force_convergence) + "\n";
		config += "gneb_n_iterations              " + std::to_string(parameters->n_iterations) + "\n";
		config += "gneb_n_iterations_log          " + std::to_string(parameters->n_iterations_log) + "\n";
		// config += "gneb_renorm                    " + std::to_string(parameters->renorm_gneb) + "\n";
		config += "gneb_spring_constant           " + std::to_string(parameters->spring_constant) + "\n";
		config += "gneb_n_energy_interpolations   " + std::to_string(parameters->n_E_interpolations) + "\n";
		config += "############### End GNEB Parameters ##############";
		Append_String_to_File(config, configFile);
	}// end Parameters_Method_LLG_from_Config

	void Parameters_Method_MMF_to_Config(const std::string configFile, const std::shared_ptr<Data::Parameters_Method_MMF> parameters)
	{
		std::string config = "";
		config += "################# MMF Parameters #################\n";
		config += "mmf_output_any            " + std::to_string(parameters->output_any) + "\n";
		config += "mmf_output_initial        " + std::to_string(parameters->output_initial) + "\n";
		config += "mmf_output_final          " + std::to_string(parameters->output_final) + "\n";
		config += "mmf_output_energy_step             " + std::to_string(parameters->output_energy_step) + "\n";
		config += "mmf_output_energy_archive          " + std::to_string(parameters->output_energy_archive) + "\n";
		config += "mmf_output_energy_divide_by_nspins " + std::to_string(parameters->output_energy_divide_by_nspins) + "\n";
		config += "mmf_output_configuration_step      " + std::to_string(parameters->output_configuration_step) + "\n";
		config += "mmf_output_configuration_archive   " + std::to_string(parameters->output_configuration_archive) + "\n";
		config += "mmf_force_convergence          " + fmt::format("{:e}", parameters->force_convergence) + "\n";
		config += "mmf_n_iterations               " + std::to_string(parameters->n_iterations) + "\n";
		config += "mmf_n_iterations_log           " + std::to_string(parameters->n_iterations_log) + "\n";
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
		config += "hamiltonian              " + name + "\n";
		config += "boundary_conditions      " + std::to_string((int)hamiltonian->boundary_conditions[0]) + " " + std::to_string((int)hamiltonian->boundary_conditions[1]) + " " + std::to_string((int)hamiltonian->boundary_conditions[2]) + "\n";
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
		config += "external_field_magnitude " + std::to_string(ham->external_field_magnitudes[0]/Constants::mu_B/ham->mu_s[0]) + "\n";
		config += "external_field_normal    " + std::to_string(ham->external_field_normals[0][0]) + " " + std::to_string(ham->external_field_normals[0][1]) + " " + std::to_string(ham->external_field_normals[0][2]) + "\n";
		config += "mu_s                     " + std::to_string(ham->mu_s[0]) + "\n";
		config += "anisotropy_magnitude     " + std::to_string(ham->anisotropy_magnitudes[0]) + "\n";
		config += "anisotropy_normal        " + std::to_string(ham->anisotropy_normals[0][0]) + " " + std::to_string(ham->anisotropy_normals[0][1]) + " " + std::to_string(ham->anisotropy_normals[0][2]) + "\n";
		config += "n_neigh_shells  			" + std::to_string(ham->exchange_magnitudes.size()) + "\n";
		config += "jij                      " + std::to_string(ham->exchange_magnitudes[0]);
		for (unsigned int i=1; i<ham->exchange_magnitudes.size(); ++i) config += " " + std::to_string(ham->exchange_magnitudes[i]);
		config += "\n";
		config += "\n";
		config += "n_neigh_shells_dmi 		" + std::to_string(ham->dmi_magnitudes.size()) + "\n";
		config += "dij                      " + std::to_string(ham->dmi_magnitudes[0]);
		for (unsigned int i=1; i<ham->dmi_magnitudes.size(); ++i) config += " " + std::to_string(ham->dmi_magnitudes[i]);
		config += "\n";
		config += "\n";
		config += "dd_radius                " + std::to_string(ham->ddi_radius) + "\n";
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
			config += std::to_string(ham->mu_s[i]);
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
		config += "external_field_magnitude " + std::to_string(B) + "\n";
		config += "external_field_normal    "
			+ std::to_string(B_normal[0]) + " " + std::to_string(B_normal[1]) + " " + std::to_string(B_normal[2]) + "\n";
		
		// Anisotropy
		config += "###    Anisotropy:\n";
		scalar K = 0;
		Vector3 K_normal{ 0,0,0 };
		if (ham->anisotropy_indices.size() > 0)
		{
			K = ham->anisotropy_magnitudes[0];
			K_normal = ham->anisotropy_normals[0];
		}
		config += "anisotropy_magnitude " + std::to_string(K) + "\n";
		config += "anisotropy_normal    "
			+ std::to_string(K_normal[0]) + " " + std::to_string(K_normal[1]) + " " + std::to_string(K_normal[2]) + "\n";
		
		config += "###    Interaction pairs:\n";
		config += "n_interaction_pairs " + std::to_string(ham->exchange_pairs.size() + ham->dmi_pairs.size()) + "\n";
		config += "  i   j     da   db   dc     J     Dij   Dx   Dy   Dz\n";
		// Exchange
		for (unsigned int i=0; i<ham->exchange_pairs.size(); ++i)
		{
			config += " " + std::to_string(ham->exchange_pairs[i].i) + " " + std::to_string(ham->exchange_pairs[i].j) + "   "
						+ std::to_string(ham->exchange_pairs[i].translations[0]) + " " + std::to_string(ham->exchange_pairs[i].translations[1]) + " " + std::to_string(ham->exchange_pairs[i].translations[2]) + "   "
						+ std::to_string(ham->exchange_magnitudes[i]) + "   "
						+ std::to_string(0.0f) + " " + std::to_string(0.0f) + " " + std::to_string(0.0f) + " " + std::to_string(0.0f) + "\n";
		}
		// DMI
		for (unsigned int i = 0; i<ham->dmi_pairs.size(); ++i)
		{
			config += " " + std::to_string(ham->dmi_pairs[i].i) + " " + std::to_string(ham->dmi_pairs[i].j) + "   "
				+ std::to_string(ham->dmi_pairs[i].translations[0]) + " " + std::to_string(ham->dmi_pairs[i].translations[1]) + " " + std::to_string(ham->dmi_pairs[i].translations[2]) + "   "
				+ std::to_string(0.0f) + "   "
				+ std::to_string(ham->dmi_magnitudes[i]) + " "
				+ std::to_string(ham->dmi_normals[i][0]) + " " + std::to_string(ham->dmi_normals[i][1]) + " " + std::to_string(ham->dmi_normals[i][2]) + "\n";
		}

		// Quadruplets
		config += "###    Quadruplets:\n";
		config += "n_interaction_quadruplets " + std::to_string(ham->quadruplets.size()) + "\n";
		config += " i   j   k   l     da_j  db_j  dc_j     k da_k db_k dc_k     l da_l db_l dc_l     Q\n";
		for (unsigned int i=0; i<ham->quadruplets.size(); ++i)
		{
			config += " " + std::to_string(ham->quadruplets[i].i) + "   " + std::to_string(ham->quadruplets[i].j) + "   " + std::to_string(ham->quadruplets[i].k) + "   " + std::to_string(ham->quadruplets[i].l) + "    "
						+ std::to_string(ham->quadruplets[i].d_j[0]) + "   " + std::to_string(ham->quadruplets[i].d_j[1]) + "   " + std::to_string(ham->quadruplets[i].d_j[2])+ "    "
						+ std::to_string(ham->quadruplets[i].d_k[0]) + "   " + std::to_string(ham->quadruplets[i].d_k[1]) + "   " + std::to_string(ham->quadruplets[i].d_k[2])+ "    "
						+ std::to_string(ham->quadruplets[i].d_l[0]) + "   " + std::to_string(ham->quadruplets[i].d_l[1]) + "   " + std::to_string(ham->quadruplets[i].d_l[2])+ "    "
						+ std::to_string(ham->quadruplet_magnitudes[i]) + "\n";
		}

		Append_String_to_File(config, configFile);
	}// end Hamiltonian_Heisenberg_Pairs_to_Config
	
	void Hamiltonian_Gaussian_to_Config(const std::string configFile, const std::shared_ptr<Engine::Hamiltonian> hamiltonian)
	{
		std::string config = "";
		Engine::Hamiltonian_Gaussian * ham_gaussian = (Engine::Hamiltonian_Gaussian *)hamiltonian.get();
		config += "n_gaussians          " + std::to_string(ham_gaussian->n_gaussians) + "\n";
		config += "gaussians\n";
		for (int i=0; i< ham_gaussian->n_gaussians; ++i)
		{
			config += std::to_string(ham_gaussian->amplitude[i]) + " " + std::to_string(ham_gaussian->width[i]) + " "
					+ std::to_string(ham_gaussian->center[i][0]) + " " + std::to_string(ham_gaussian->center[i][1]) + " " + std::to_string(ham_gaussian->center[i][2]) + "\n";
		}
		Append_String_to_File(config, configFile);
	}// end Hamiltonian_Gaussian_to_Config
}// end namespace IO