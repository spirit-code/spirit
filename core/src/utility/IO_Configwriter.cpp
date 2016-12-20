#include <utility/IO.hpp>
#include <utility/IO_Filter_File_Handle.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Neighbours.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

namespace Utility
{
	namespace IO
	{
		void Folders_to_Config(const std::string configFile,
				std::shared_ptr<Data::Parameters_Method_LLG> parameters_llg,
				std::shared_ptr<Data::Parameters_Method_GNEB> parameters_gneb,
				std::shared_ptr<Data::Parameters_Method_MMF> parameters_mmf)
		{
			std::string config = "";
			config += "################# Output Folders #################\n";
			config += "log_output_folder  " + Log.output_folder + "\n";
			config += "llg_output_folder  " + parameters_llg->output_folder + "\n";
			config += "gneb_output_folder " + parameters_gneb->output_folder + "\n";
			config += "mmf_output_folder  " + parameters_mmf->output_folder + "\n";
			config += "############### End Output Folders ###############";
			Append_String_to_File(config, configFile);
		}// End Folders_to_Config


		void Log_Levels_to_Config(const std::string configFile)
		{
			std::string config = "";
			config += "############### Logging Parameters ###############\n";
			config += "log_print  " + std::to_string((int)Log.print_level) + "\n";
			config += "log_accept " + std::to_string((int)Log.accept_level) + "\n";
			config += "############# End Logging Parameters #############";
			Append_String_to_File(config, configFile);
		}// End Log_Levels_to_Config


		void Geometry_to_Config(const std::string configFile, std::shared_ptr<Data::Geometry> geometry)
		{
			std::string config = "";
			config += "#################### Geometry ####################\n";
			config += "basis\n";
			config += std::to_string(geometry->basis[0][0]) + " " + std::to_string(geometry->basis[0][1]) + " " + std::to_string(geometry->basis[0][2]) + "\n";
			config += std::to_string(geometry->basis[1][0]) + " " + std::to_string(geometry->basis[1][1]) + " " + std::to_string(geometry->basis[1][2]) + "\n";
			config += std::to_string(geometry->basis[2][0]) + " " + std::to_string(geometry->basis[2][1]) + " " + std::to_string(geometry->basis[2][2]) + "\n";
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


		void Parameters_Method_LLG_to_Config(const std::string configFile, std::shared_ptr<Data::Parameters_Method_LLG> parameters)
		{
			std::string config = "";
			config += "################# LLG Parameters #################\n";
			config += "llg_force_convergence          " + center(parameters->force_convergence, 14, 16) + "\n";
			config += "llg_n_iterations               " + std::to_string(parameters->n_iterations) + "\n";
			config += "llg_n_iterations_log           " + std::to_string(parameters->n_iterations_log) + "\n";
			config += "llg_renorm                     " + std::to_string(parameters->renorm_sd) + "\n";
			config += "llg_save_single_configurations " + std::to_string(parameters->save_single_configurations) + "\n";
			config += "llg_seed                       " + std::to_string(parameters->seed) + "\n";
			config += "llg_temperature                " + std::to_string(parameters->temperature) + "\n";
			config += "llg_damping                    " + std::to_string(parameters->damping) + "\n";
			config += "llg_dt                         " + std::to_string(parameters->dt) + "\n";
			config += "llg_stt_magnitude              " + std::to_string(parameters->stt_magnitude) + "\n";
			config += "llg_stt_polarisation_normal    " + std::to_string(parameters->stt_polarisation_normal[0]) + " " + std::to_string(parameters->stt_polarisation_normal[1]) + " " + std::to_string(parameters->stt_polarisation_normal[2]) + "\n";
			config += "############### End LLG Parameters ###############";
			Append_String_to_File(config, configFile);
		}// end Parameters_Method_LLG_to_Config

		void Parameters_Method_GNEB_to_Config(const std::string configFile, std::shared_ptr<Data::Parameters_Method_GNEB> parameters)
		{
			std::string config = "";
			config += "################# GNEB Parameters ################\n";
			config += "gneb_force_convergence         " + center(parameters->force_convergence, 14, 16) + "\n";
			config += "gneb_n_iterations              " + std::to_string(parameters->n_iterations) + "\n";
			config += "gneb_n_iterations_log          " + std::to_string(parameters->n_iterations_log) + "\n";
			// config += "gneb_renorm                    " + std::to_string(parameters->renorm_gneb) + "\n";
			config += "gneb_spring_constant           " + std::to_string(parameters->spring_constant) + "\n";
			config += "gneb_n_energy_interpolations   " + std::to_string(parameters->n_E_interpolations) + "\n";
			config += "############### End GNEB Parameters ##############";
			Append_String_to_File(config, configFile);
		}// end Parameters_Method_LLG_from_Config

		void Parameters_Method_MMF_to_Config(const std::string configFile, std::shared_ptr<Data::Parameters_Method_MMF> parameters)
		{
			std::string config = "";
			config += "################# MMF Parameters #################\n";
			config += "mmf_force_convergence          " + center(parameters->force_convergence, 14, 16) + "\n";
			config += "mmf_n_iterations               " + std::to_string(parameters->n_iterations) + "\n";
			config += "mmf_n_iterations_log           " + std::to_string(parameters->n_iterations_log) + "\n";
			config += "############### End MMF Parameters ###############";
			Append_String_to_File(config, configFile);
		}// end Parameters_Method_MMF_to_Config

		void Hamiltonian_to_Config(const std::string configFile, std::shared_ptr<Engine::Hamiltonian> hamiltonian, std::shared_ptr<Data::Geometry> geometry)
		{
			std::string config = "";
			config += "################### Hamiltonian ##################\n";
			std::string name;
			if (hamiltonian->Name() == "Anisotropic Heisenberg") name = "anisotropic";
			else if (hamiltonian->Name() == "Isotropic Heisenberg") name = "isotropic";
			else if (hamiltonian->Name() == "Gaussian") name = "gaussian";
			config += "hamiltonian              " + name + "\n";
			config += "boundary_conditions      " + std::to_string((int)hamiltonian->boundary_conditions[0]) + " " + std::to_string((int)hamiltonian->boundary_conditions[1]) + " " + std::to_string((int)hamiltonian->boundary_conditions[2]) + "\n";
			Append_String_to_File(config, configFile);

			if (hamiltonian->Name() == "Anisotropic Heisenberg") Hamiltonian_Anisotropic_to_Config(configFile, hamiltonian, geometry);
			else if (hamiltonian->Name() == "Isotropic Heisenberg") Hamiltonian_Isotropic_to_Config(configFile, hamiltonian);
			else if (hamiltonian->Name() == "Gaussian") Hamiltonian_Gaussian_to_Config(configFile, hamiltonian);

			config = "################# End Hamiltonian ################";
			Append_String_to_File(config, configFile);
		}// end Hamiltonian_to_Config

		void Hamiltonian_Isotropic_to_Config(const std::string configFile, std::shared_ptr<Engine::Hamiltonian> hamiltonian)
		{
			std::string config = "";
			Engine::Hamiltonian_Isotropic * ham_iso = (Engine::Hamiltonian_Isotropic *)hamiltonian.get();
			config += "external_field_magnitude " + std::to_string(ham_iso->external_field_magnitude) + "\n";
			config += "external_field_normal    " + std::to_string(ham_iso->external_field_normal[0]) + " " + std::to_string(ham_iso->external_field_normal[1]) + " " + std::to_string(ham_iso->external_field_normal[2]) + "\n";
			config += "mu_s                     " + std::to_string(ham_iso->mu_s) + "\n";
			config += "anisotropy_magnitude     " + std::to_string(ham_iso->anisotropy_magnitude) + "\n";
			config += "anisotropy_normal        " + std::to_string(ham_iso->anisotropy_normal[0]) + " " + std::to_string(ham_iso->anisotropy_normal[1]) + " " + std::to_string(ham_iso->anisotropy_normal[2]) + "\n";
			config += "n_neigh_shells  			" + std::to_string(ham_iso->n_neigh_shells) + "\n";
			config += "jij                      " + std::to_string(ham_iso->jij[0]);
			for (int i=1; i<ham_iso->n_neigh_shells; ++i) config += " " + std::to_string(ham_iso->jij[i]);
			config += "\n";
			config += "dij                      " + std::to_string(ham_iso->dij) + "\n";
			config += "bij                      " + std::to_string(ham_iso->bij) + "\n";
			config += "kijkl                    " + std::to_string(ham_iso->kijkl) + "\n";
			config += "dd_radius                " + std::to_string(ham_iso->dd_radius) + "\n";
			Append_String_to_File(config, configFile);
		}// end Hamiltonian_Isotropic_to_Config

		void Hamiltonian_Anisotropic_to_Config(const std::string configFile, std::shared_ptr<Engine::Hamiltonian> hamiltonian, std::shared_ptr<Data::Geometry> geometry)
		{
			int n_cells_tot = geometry->n_cells[0]*geometry->n_cells[1]*geometry->n_cells[2];
			std::string config = "";
			Engine::Hamiltonian_Anisotropic* ham_aniso = (Engine::Hamiltonian_Anisotropic *)hamiltonian.get();
			config += "###\n### Note the pairs and quadruplets are not yet logged here!\n###\n";
			config += "### The following can be used as input if you remove the '#'s\n";
			
			// Magnetic moment
			config += "mu_s                     ";
			for (unsigned int i=0; i<geometry->n_spins_basic_domain; ++i)
				config += std::to_string(ham_aniso->mu_s[i]);
			config += "\n";

			// External Field
			config += "###    External Field:\n";
			config += "#  i    H     Hx   Hy   Hz\n";
			for (unsigned int i=0; i<ham_aniso->external_field_index.size()/n_cells_tot; ++i)
			{
				config += "# " + std::to_string(ham_aniso->external_field_index[i]) + " " + std::to_string(ham_aniso->external_field_magnitude[i]) + " "
							+ std::to_string(ham_aniso->external_field_normal[i][0]) + " " + std::to_string(ham_aniso->external_field_normal[i][1]) + " " + std::to_string(ham_aniso->external_field_normal[i][2]) + "\n";
			}

			// Anisotropy
			config += "###    Anisotropy:\n";
			config += "#  i    K     Kx   Ky   Kz\n";
			for (unsigned int i=0; i<ham_aniso->anisotropy_index.size()/n_cells_tot; ++i)
			{
				config += "# " + std::to_string(ham_aniso->anisotropy_index[i]) + " " + std::to_string(ham_aniso->anisotropy_magnitude[i]) + " "
							+ std::to_string(ham_aniso->anisotropy_normal[i][0]) + " " + std::to_string(ham_aniso->anisotropy_normal[i][1]) + " " + std::to_string(ham_aniso->anisotropy_normal[i][2]) + "\n";
			}

			// TODO: how to only log the pairs and quadruplets that were given as input?
			//		 (in contrast to those generated by translations)

			// // Exchange
			// config += "###    Exchange:\n";
			// config += "#  i    j     Jij\n";
			// for (unsigned int i=0; i<ham_aniso->Exchange_indices.size(); ++i)
			// {
			// 	config += "# " + std::to_string(ham_aniso->Exchange_indices[i][0]) + " " + std::to_string(ham_aniso->Exchange_indices[i][1]) + " "
			// 				+ std::to_string(ham_aniso->anisotropy_normal[i][0]) + " " + std::to_string(ham_aniso->anisotropy_normal[i][1]) + " " + std::to_string(ham_aniso->anisotropy_normal[i][2]) + "\n";
			// }

			Append_String_to_File(config, configFile);
		}// end Hamiltonian_Anisotropic_to_Config
		
		void Hamiltonian_Gaussian_to_Config(const std::string configFile, std::shared_ptr<Engine::Hamiltonian> hamiltonian)
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
}// end namespace Utility