#include <utility/IO.hpp>
#include <utility/IO_Filter_File_Handle.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Neighbours.hpp>
#include <utility/Constants.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>

namespace Utility
{
	namespace IO
	{
		void Log_from_Config(const std::string configFile, bool force_quiet)
		{
			// Verbosity and Reject Level are read as integers
			int i_level_file = 5, i_level_console = 5;
			std::string output_folder = ".";
			bool tag_time = true, messages_to_file = true, messages_to_console = true, save_input_initial = false, save_input_final = false;

			// "Quiet" settings
			if (force_quiet)
			{
				// Don't save the Log to file
				Log.messages_to_file = false;
				// Don't print the Log to console
				Log.messages_to_console = false;
				// Don't save input configs
				Log.save_input_initial = false;
				Log.save_input_final = false;
				// Don't print messages, except Error & Severe
				Log.level_file = Utility::Log_Level::Error;
				Log.level_console = Utility::Log_Level::Error;
			}

			//------------------------------- Parser --------------------------------
			if (configFile != "")
			{
				try
				{
					Log(Log_Level::Info, Log_Sender::IO, "Building Log");
					IO::Filter_File_Handle myfile(configFile);

					// Time tag
					myfile.Read_Single(tag_time, "output_tag_time");

					// Output folder
					myfile.Read_Single(output_folder, "log_output_folder");
					
					// Save Output (Log Messages) to file
					myfile.Read_Single(messages_to_file, "log_to_file");
					// File Accept Level
					myfile.Read_Single(i_level_file, "log_file_level");

					// Print Output (Log Messages) to console
					myfile.Read_Single(messages_to_console, "log_to_console");
					// File Accept Level
					myfile.Read_Single(i_level_console, "log_console_level");

					// Save Input (parameters from config file and defaults)
					//    on State Setup
					myfile.Read_Single(save_input_initial, "log_input_save_initial");
					// Save Input (parameters from config file and defaults)
					//    on State Delete
					myfile.Read_Single(save_input_final, "log_input_save_final");

				}// end try
				catch (Exception ex) {
					if (ex == Exception::File_not_Found) {
						Log(Log_Level::Error, Log_Sender::IO, "Log_Levels: Unable to open Config File " + configFile + " Leaving values at default.");
					}
					else throw ex;
				}// end catch
			}

			// Log the parameters
			Log(Log_Level::Parameter, Log_Sender::IO, "Tag time on output     = " + std::to_string(tag_time));
			Log(Log_Level::Parameter, Log_Sender::IO, "Log output folder      = " + output_folder);
			Log(Log_Level::Parameter, Log_Sender::IO, "Log to file            = " + std::to_string(messages_to_file));
			Log(Log_Level::Parameter, Log_Sender::IO, "Log file accept level  = " + std::to_string(i_level_file));
			Log(Log_Level::Parameter, Log_Sender::IO, "Log to console         = " + std::to_string(messages_to_console));
			Log(Log_Level::Parameter, Log_Sender::IO, "Log print accept level = " + std::to_string(i_level_console));
			Log(Log_Level::Parameter, Log_Sender::IO, "Log input save initial = " + std::to_string(save_input_initial));
			Log(Log_Level::Parameter, Log_Sender::IO, "Log input save final   = " + std::to_string(save_input_final));
			
			// Update the Log
			if (!force_quiet)
			{
				Log.level_file    = Log_Level(i_level_file);
				Log.level_console = Log_Level(i_level_console);

				Log.messages_to_file    = messages_to_file;
				Log.messages_to_console = messages_to_console;
				Log.save_input_initial  = save_input_initial;
				Log.save_input_final    = save_input_final;
			}

			Log.tag_time      = tag_time;
			Log.output_folder = output_folder;
			if (tag_time)
				Log.fileName = "Log_" + Utility::Timing::CurrentDateTime() + ".txt";
			else
				Log.fileName = "Log.txt";

		}// End Log_Levels_from_Config


		std::unique_ptr<Data::Spin_System> Spin_System_from_Config(const std::string configFile)
		{
			Log(Log_Level::Info, Log_Sender::IO, "-------------- Initialising Spin System ------------");
			// ----------------------------------------------------------------------------------------------
			// Geometry
			auto geometry = Geometry_from_Config(configFile);
			// Pinning configuration
			auto pinning = Pinning_from_Config(configFile, geometry);
			// LLG Parameters
			auto llg_params = Parameters_Method_LLG_from_Config(configFile, pinning);
			// MC Parameters
			auto mc_params = Parameters_Method_MC_from_Config(configFile, pinning);
			// Hamiltonian
			auto hamiltonian = std::move(Hamiltonian_from_Config(configFile, geometry));
			// Spin System
			auto system = std::unique_ptr<Data::Spin_System>(new Data::Spin_System(std::move(hamiltonian), std::move(geometry), std::move(llg_params), std::move(mc_params), false));
			// ----------------------------------------------------------------------------------------------
			Log(Log_Level::Info, Log_Sender::IO, "-------------- Spin System Initialised -------------");

			// Return
			return system;
		}// End Spin_System_from_Config		


		void Basis_from_Config(const std::string configFile, std::vector<Vector3> & basis, std::vector<Vector3> & basis_atoms, scalar & lattice_constant)
		{
			// ---------- Default values
			// Lattice constant [Angtrom]
			lattice_constant = 1.0;
			// Basis: vector {a, b, c}
			basis = { Vector3{1,0,0}, Vector3{0,1,0}, Vector3{0,0,1} };
			// Atoms in the basis [dim][n_basis_atoms]
			basis_atoms = { Vector3{0,0,0} };
			// NoS in the basic domain (= unit cell for periodic lattices)
			int n_spins_basic_domain = 0;
			
			Log(Log_Level::Info, Log_Sender::IO, "Basis: building");

			if (configFile != "")
			{
				try {
					IO::Filter_File_Handle myfile(configFile);

					myfile.Read_Single(lattice_constant, "lattice_constant");

					// Utility 1D array to build vectors and use Vectormath
					Vector3 build_array = { 0, 0, 0 };

					if (myfile.Find("basis"))
					{
						// Read the basis vectors a, b, c
						myfile.GetLine();
						myfile.iss >> basis[0][0] >> basis[0][1] >> basis[0][2];
						myfile.GetLine();
						myfile.iss >> basis[1][0] >> basis[1][1] >> basis[1][2];
						myfile.GetLine();
						myfile.iss >> basis[2][0] >> basis[2][1] >> basis[2][2];

						// Read no_spins_basic_domain and atoms in basis
						myfile.GetLine();
						myfile.iss >> n_spins_basic_domain;
						basis_atoms = std::vector<Vector3>(n_spins_basic_domain);

						// Read spins per basic domain
						for (int iatom = 0; iatom < n_spins_basic_domain; ++iatom)
						{
							myfile.GetLine();
							myfile.iss >> basis_atoms[iatom][0] >> basis_atoms[iatom][1] >> basis_atoms[iatom][2];
							// Get x,y,z of component of spin_pos in unit of length (instead of in units of a,b,c)
							build_array = basis[0] * basis_atoms[iatom][0] + basis[1] * basis_atoms[iatom][1] + basis[2] * basis_atoms[iatom][2];
							basis_atoms[iatom] = lattice_constant * build_array;
						}// endfor iatom

					}// end find "basis"
					else {
						Log(Log_Level::Error, Log_Sender::IO, "Keyword 'basis' not found. Using Default (sc)");
					}
				}// end try
				catch (Exception ex) {
					if (ex == Exception::File_not_Found)
					{
						Log(Log_Level::Error, Log_Sender::IO, "Basis: Unable to open Config File " + configFile + " Leaving values at default.");
						throw Exception::System_not_Initialized;
					}
					else throw ex;
				}// end catch
			}
			else Log(Log_Level::Warning, Log_Sender::IO, "Basis: No config file specified. Leaving values at default.");
			
			// Log the parameters
			Log(Log_Level::Parameter, Log_Sender::IO, "Lattice constant = " + std::to_string(lattice_constant) + " angstrom");
			Log(Log_Level::Debug, Log_Sender::IO, "Basis: vectors in units of lattice constant");
			Log(Log_Level::Debug, Log_Sender::IO, "        a = " + std::to_string(basis[0][0]/lattice_constant) + " " + std::to_string(basis[0][1]/lattice_constant) + " " + std::to_string(basis[0][2]/lattice_constant));
			Log(Log_Level::Debug, Log_Sender::IO, "        b = " + std::to_string(basis[1][0]/lattice_constant) + " " + std::to_string(basis[1][1]/lattice_constant) + " " + std::to_string(basis[1][2]/lattice_constant));
			Log(Log_Level::Debug, Log_Sender::IO, "        c = " + std::to_string(basis[2][0]/lattice_constant) + " " + std::to_string(basis[2][1]/lattice_constant) + " " + std::to_string(basis[2][2]/lattice_constant));
			Log(Log_Level::Parameter, Log_Sender::IO, "Basis: vectors");
			Log(Log_Level::Parameter, Log_Sender::IO, "        a = " + std::to_string(basis[0][0]) + " " + std::to_string(basis[0][1]) + " " + std::to_string(basis[0][2]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        b = " + std::to_string(basis[1][0]) + " " + std::to_string(basis[1][1]) + " " + std::to_string(basis[1][2]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        c = " + std::to_string(basis[2][0]) + " " + std::to_string(basis[2][1]) + " " + std::to_string(basis[2][2]));
			Log(Log_Level::Parameter, Log_Sender::IO, "Basis: " + std::to_string(n_spins_basic_domain) + " atom(s) at the following positions:");
			for (int iatom = 0; iatom < n_spins_basic_domain; ++iatom)
			{
				Log(Log_Level::Parameter, Log_Sender::IO, "            " + std::to_string(iatom) + " = " + std::to_string(basis_atoms[iatom][0]) + " " + std::to_string(basis_atoms[iatom][1]) + " " + std::to_string(basis_atoms[iatom][2]));
			}
			Log(Log_Level::Info, Log_Sender::IO, "Basis: built");
		}// End Basis_from_Config

		std::shared_ptr<Data::Geometry> Geometry_from_Config(const std::string configFile)
		{
			//-------------- Insert default values here -----------------------------
			// Basis from separate file?
			std::string basis_file = "";
			// Basis: vector {a, b, c}
			std::vector<Vector3> basis = { Vector3{1,0,0}, Vector3{0,1,0}, Vector3{0,0,1} };
			// Atoms in the basis [dim][n_basis_atoms]
			std::vector<Vector3> basis_atoms = { Vector3{0,0,0} };
			// Lattice Constant [Angstrom]
			scalar lattice_constant = 1;
			// Translation vectors [dim][nov]
			std::vector<Vector3> translation_vectors = { Vector3{1,0,0}, Vector3{0,1,0}, Vector3{0,0,1} };
			// Number of translations nT for each basis direction
			intfield n_cells = { 100, 100, 1 };
			// Number of Spins
			int nos;
			vectorfield spin_pos;

			// Utility 1D array to build vectors and use Vectormath
			Vector3 build_array = { 0, 0, 0 };

			Log(Log_Level::Info, Log_Sender::IO, "Geometry: building");
			//------------------------------- Parser --------------------------------
			// iteration variables
			int iatom = 0, dim = 0;
			if (configFile != "")
			{
				try {
					Log(Log_Level::Info, Log_Sender::IO, "Reading Geometry Parameters");
					IO::Filter_File_Handle myfile(configFile);

					// Read Shape of spins in term of the basis
					if (myfile.Find("translation_vectors"))
					{
						// Read translation vectors into translation_vectors & nTa, nTb, nTc
						myfile.GetLine();
						myfile.iss >> translation_vectors[0][0] >> translation_vectors[0][1] >> translation_vectors[0][2] >> n_cells[0];
						myfile.GetLine();
						myfile.iss >> translation_vectors[1][0] >> translation_vectors[1][1] >> translation_vectors[1][2] >> n_cells[1];
						myfile.GetLine();
						myfile.iss >> translation_vectors[2][0] >> translation_vectors[2][1] >> translation_vectors[2][2] >> n_cells[2];
					}// finish Reading Shape in terms of basis
					else {
						Log(Log_Level::Warning, Log_Sender::IO, "Keyword 'translation_vectors' not found. Using default. (sc 30x30x0)");
					}
					// Read Basis
						
					if (myfile.Find("basis_from_config"))
					{
						myfile.iss >> basis_file;
						Basis_from_Config(basis_file, basis, basis_atoms, lattice_constant);
					}
					else if (myfile.Find("basis"))
					{
						Basis_from_Config(configFile, basis, basis_atoms, lattice_constant);
					}
					else {
						Log(Log_Level::Warning, Log_Sender::IO, "Neither Keyword 'basis_from_config', nor Keyword 'basis' found. Using Default (sc)");
					}// end Basis
				}// end try
				catch (Exception ex)
				{
					if (ex == Exception::File_not_Found)
					{
						Log(Log_Level::Error, Log_Sender::IO, "Geometry: Unable to open Config File " + configFile + " Leaving values at default.");
					}
					else throw ex;
				}// end catch
			}// end if file=""
			else Log(Log_Level::Warning, Log_Sender::IO, "Geometry: Using default configuration!");

			Log(Log_Level::Parameter, Log_Sender::IO, "Translation: vectors");
			Log(Log_Level::Parameter, Log_Sender::IO, "        a = " + std::to_string(translation_vectors[0][0]) + " " + std::to_string(translation_vectors[1][0]) + " " + std::to_string(translation_vectors[2][0]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        b = " + std::to_string(translation_vectors[0][1]) + " " + std::to_string(translation_vectors[1][1]) + " " + std::to_string(translation_vectors[2][1]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        c = " + std::to_string(translation_vectors[0][2]) + " " + std::to_string(translation_vectors[1][2]) + " " + std::to_string(translation_vectors[2][2]));

			// Get x,y,z of component of translation_vectors in unit of length (instead of in units of a,b,c)
			for (dim = 0; dim < 3; ++dim)
			{
				for (int i = 0; i < 3; ++i)
				{
					build_array[i] = basis[0][i] * translation_vectors[dim][0] + basis[1][i] * translation_vectors[dim][1] + basis[2][i] * translation_vectors[dim][2];
				}
				translation_vectors[dim] = build_array;
			}
			// Calculate NOS
			nos = basis_atoms.size() * n_cells[0] * n_cells[1] * n_cells[2];

			// Spin Positions
			spin_pos = vectorfield(nos);
			Engine::Vectormath::Build_Spins(spin_pos, basis_atoms, translation_vectors, n_cells);
			
			// Log parameters
			Log(Log_Level::Parameter, Log_Sender::IO, "Translation: vectors transformed by basis");
			Log(Log_Level::Parameter, Log_Sender::IO, "        a = " + std::to_string(translation_vectors[0][0]) + " " + std::to_string(translation_vectors[0][1]) + " " + std::to_string(translation_vectors[0][2]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        b = " + std::to_string(translation_vectors[1][0]) + " " + std::to_string(translation_vectors[1][1]) + " " + std::to_string(translation_vectors[1][2]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        c = " + std::to_string(translation_vectors[2][0]) + " " + std::to_string(translation_vectors[2][1]) + " " + std::to_string(translation_vectors[2][2]));
			Log(Log_Level::Parameter, Log_Sender::IO, "Translation: n_cells");
			Log(Log_Level::Parameter, Log_Sender::IO, "        na = " + std::to_string(n_cells[0]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        nb = " + std::to_string(n_cells[1]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        nc = " + std::to_string(n_cells[2]));
			Log(Log_Level::Parameter, Log_Sender::IO, "Geometry: " + std::to_string(nos) + " spins");
			
			// Return geometry
			auto geometry = std::shared_ptr<Data::Geometry>(new Data::Geometry(basis, translation_vectors, n_cells, basis_atoms, lattice_constant, spin_pos));
			Log(Log_Level::Parameter, Log_Sender::IO, "Geometry is " + std::to_string(geometry->dimensionality) + "-dimensional"); 
			Log(Log_Level::Info, Log_Sender::IO, "Geometry: built");
			return geometry;
		}// end Geometry from Config

		std::shared_ptr<Data::Pinning> Pinning_from_Config(const std::string configFile, const std::shared_ptr<Data::Geometry> geometry)
		{
			//-------------- Insert default values here -----------------------------
			int na = 0, na_left = 0, na_right = 0;
			int nb = 0, nb_left = 0, nb_right = 0;
			int nc = 0, nc_left = 0, nc_right = 0;
			vectorfield pinned_cell(geometry->n_spins_basic_domain, Vector3{ 0,0,1 });

			// Utility 1D array to build vectors and use Vectormath
			Vector3 build_array = { 0, 0, 0 };

			Log(Log_Level::Info, Log_Sender::IO, "Reading Pinning Configuration");
			//------------------------------- Parser --------------------------------
			if (configFile != "")
			{
				try
				{
					IO::Filter_File_Handle myfile(configFile);

					// N_a
					myfile.Read_Single(na_left, "pin_na_left", false);
					myfile.Read_Single(na_right, "pin_na_right", false);
					myfile.Read_Single(na, "pin_na ", false);
					if (na > 0 && (na_left == 0 || na_right == 0))
					{
						na_left = na;
						na_right = na;
					}

					// N_b
					myfile.Read_Single(nb_left, "pin_nb_left", false);
					myfile.Read_Single(nb_right, "pin_nb_right", false);
					myfile.Read_Single(nb, "pin_nb ", false);
					if (nb > 0 && (nb_left == 0 || nb_right == 0))
					{
						nb_left = nb;
						nb_right = nb;
					}

					// N_c
					myfile.Read_Single(nc_left, "pin_nc_left", false);
					myfile.Read_Single(nc_right, "pin_nc_right", false);
					myfile.Read_Single(nc, "pin_nc ", false);
					if (nc > 0 && (nc_left == 0 || nc_right == 0))
					{
						nc_left = nc;
						nc_right = nc;
					}

					// How should the cells be pinned
					if (na_left > 0 || na_right > 0 ||
						nb_left > 0 || nb_right > 0 ||
						nc_left > 0 || nc_right > 0)
					{
						if (myfile.Find("pinning_cell"))
						{
							for (int i = 0; i < geometry->n_spins_basic_domain; ++i)
							{
								myfile.GetLine();
								myfile.iss >> pinned_cell[i][0] >> pinned_cell[i][1] >> pinned_cell[i][2];
							}
						}
						else
						{
							na_left = 0; na_right = 0;
							nb_left = 0; nb_right = 0;
							nc_left = 0; nc_right = 0;
							Log(Log_Level::Warning, Log_Sender::IO, "Pinning specified, but keyword 'pinning_cell' not found. Won't pin any spins!");
						}
					}
				}// end try
				catch (Exception ex)
				{
					if (ex == Exception::File_not_Found)
					{
						Log(Log_Level::Error, Log_Sender::IO, "Pinning: Unable to open Config File " + configFile + " Leaving values at default.");
					}
					else throw ex;
				}// end catch
			}// end if file=""
			else Log(Log_Level::Parameter, Log_Sender::IO, "No pinning");


			// Return Pinning
			Log(Log_Level::Parameter, Log_Sender::IO, "Pinning:");
			Log(Log_Level::Parameter, Log_Sender::IO, "        n_a (left, right) = " + std::to_string(na_left) + ", " + std::to_string(na_right));
			Log(Log_Level::Parameter, Log_Sender::IO, "        n_b (left, right) = " + std::to_string(nb_left) + ", " + std::to_string(nb_right));
			Log(Log_Level::Parameter, Log_Sender::IO, "        n_c (left, right) = " + std::to_string(nc_left) + ", " + std::to_string(nc_right));
			for (int i = 0; i < geometry->n_spins_basic_domain; ++i)
				Log(Log_Level::Parameter, Log_Sender::IO, "        cell atom[0]      = (" + std::to_string(pinned_cell[0][0]) + ", " + std::to_string(pinned_cell[0][1]) + ", " + std::to_string(pinned_cell[0][2]) + ")");
			auto pinning = std::shared_ptr<Data::Pinning>(new Data::Pinning( geometry,
				na_left, na_right,
				nb_left, nb_right,
				nc_left, nc_right,
				pinned_cell) );
			Log(Log_Level::Info, Log_Sender::IO, "Pinning: read");
			return pinning;
		}

		std::unique_ptr<Data::Parameters_Method_LLG> Parameters_Method_LLG_from_Config(const std::string configFile, const std::shared_ptr<Data::Pinning> pinning)
		{
			//-------------- Insert default values here -----------------------------
			// Output folder for results
			std::string output_folder = "output_llg";
			// Save output when logging
			bool output_tag_time = true, output_any = true, output_initial = true, output_final = true;
			bool output_energy_divide_by_nspins=true, output_energy_spin_resolved=true, output_energy_step=true, output_energy_archive=true;
			bool output_configuration_step = false, output_configuration_archive = false;
			// Maximum walltime in seconds
			long int max_walltime = 0;
			std::string str_max_walltime;
			// PRNG Seed
			std::srand((unsigned int)std::time(0));
			int seed = std::rand();
			// number of iterations carried out when pressing "play" or calling "iterate"
			long int n_iterations = (int)2E+6;
			// Number of iterations after which the system is logged to file
			long int n_iterations_log = 100;
			// Temperature in K
			scalar temperature = 0.0;
			// Damping constant
			scalar damping = 0.5;
			// iteration time step
			scalar dt = 1.0E-02;
			// Whether to renormalize spins after every SD iteration
			bool renorm_sd = 1;
			// spin transfer torque vector
			scalar stt_magnitude = 1.5;
			// spin_current polarisation normal vector
			Vector3 stt_polarisation_normal = { 1.0, -1.0, 0.0 };
			// Force convergence parameter
			scalar force_convergence = 10e-9;

			//------------------------------- Parser --------------------------------
			Log(Log_Level::Info, Log_Sender::IO, "Parameters LLG: building");
			if (configFile != "")
			{
				try {
					IO::Filter_File_Handle myfile(configFile);

					myfile.Read_Single(output_tag_time,"output_tag_time");
					myfile.Read_Single(output_folder,  "llg_output_folder");
					myfile.Read_Single(output_any,     "llg_output_any");
					myfile.Read_Single(output_initial, "llg_output_initial");
					myfile.Read_Single(output_final,   "llg_output_final");
					myfile.Read_Single(output_energy_spin_resolved,    "llg_output_energy_spin_resolved");
					myfile.Read_Single(output_energy_step,             "llg_output_energy_step");
					myfile.Read_Single(output_energy_archive,          "llg_output_energy_archive");
					myfile.Read_Single(output_energy_divide_by_nspins, "llg_output_energy_divide_by_nspins");
					myfile.Read_Single(output_configuration_step,    "llg_output_configuration_step");
					myfile.Read_Single(output_configuration_archive, "llg_output_configuration_archive");
					myfile.Read_Single(str_max_walltime, "llg_max_walltime");
					myfile.Read_Single(seed, "llg_seed");
					myfile.Read_Single(n_iterations, "llg_n_iterations");
					myfile.Read_Single(n_iterations_log, "llg_n_iterations_log");
					myfile.Read_Single(temperature, "llg_temperature");
					myfile.Read_Single(damping, "llg_damping");
					myfile.Read_Single(dt, "llg_dt");
					// dt = time_step [ps] * 10^-12 * gyromagnetic raio / mu_B  { / (1+damping^2)} <- not implemented
					dt = dt*std::pow(10, -12) / Constants::mu_B*1.760859644*std::pow(10, 11);
					myfile.Read_Single(renorm_sd, "llg_renorm");
					myfile.Read_Single(stt_magnitude, "llg_stt_magnitude");
					myfile.Read_Vector3(stt_polarisation_normal, "llg_stt_polarisation_normal");
					myfile.Read_Single(force_convergence, "llg_force_convergence");
				}// end try
				catch (Exception ex) {
					if (ex == Exception::File_not_Found)
					{
						Log(Log_Level::Error, Log_Sender::IO, "Parameters LLG: Unable to open Config File " + configFile + " Leaving values at default.");
					}
					else throw ex;
				}// end catch
			}
			else Log(Log_Level::Warning, Log_Sender::IO, "Parameters LLG: Using default configuration!");

			// Normalize vectors
			stt_polarisation_normal.normalize();

			// Return
			Log(Log_Level::Parameter, Log_Sender::IO, "Parameters LLG:");
			Log(Log_Level::Parameter, Log_Sender::IO, "        maximum walltime  = " + str_max_walltime);
			Log(Log_Level::Parameter, Log_Sender::IO, "        seed              = " + std::to_string(seed));
			Log(Log_Level::Parameter, Log_Sender::IO, "        temperature       = " + std::to_string(temperature));
			Log(Log_Level::Parameter, Log_Sender::IO, "        damping           = " + std::to_string(damping));
			Log(Log_Level::Parameter, Log_Sender::IO, "        time step         = " + std::to_string(dt));
			Log(Log_Level::Parameter, Log_Sender::IO, "        stt magnitude     = " + std::to_string(stt_magnitude));
			Log(Log_Level::Parameter, Log_Sender::IO, "        stt normal        = " + std::to_string(stt_polarisation_normal[0]) + " " + std::to_string(stt_polarisation_normal[1]) + " " + std::to_string(stt_polarisation_normal[2]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        force convergence = " + std::to_string(force_convergence));
			Log(Log_Level::Parameter, Log_Sender::IO, "        n_iterations      = " + std::to_string(n_iterations));
			Log(Log_Level::Parameter, Log_Sender::IO, "        n_iterations_log  = " + std::to_string(n_iterations_log));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_folder     = " + output_folder);
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_any        = " + std::to_string(output_any));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_initial    = " + std::to_string(output_initial));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_final      = " + std::to_string(output_final));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_energy_step             = " + std::to_string(output_energy_step));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_energy_archive          = " + std::to_string(output_energy_archive));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_energy_spin_resolved    = " + std::to_string(output_energy_spin_resolved));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_energy_divide_by_nspins = " + std::to_string(output_energy_divide_by_nspins));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_configuration_step      = " + std::to_string(output_configuration_step));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_configuration_archive   = " + std::to_string(output_configuration_archive));
			max_walltime = (long int)Utility::Timing::DurationFromString(str_max_walltime).count();
			auto llg_params = std::unique_ptr<Data::Parameters_Method_LLG>(new Data::Parameters_Method_LLG( output_folder, { output_tag_time, output_any, output_initial, output_final, output_energy_step, output_energy_archive, output_energy_spin_resolved,
				output_energy_divide_by_nspins, output_configuration_step, output_configuration_archive}, force_convergence, n_iterations, n_iterations_log, max_walltime, pinning, seed, temperature, damping, dt, renorm_sd, stt_magnitude, stt_polarisation_normal));
			Log(Log_Level::Info, Log_Sender::IO, "Parameters LLG: built");
			return llg_params;
		}// end Parameters_Method_LLG_from_Config

		std::unique_ptr<Data::Parameters_Method_MC> Parameters_Method_MC_from_Config(const std::string configFile, const std::shared_ptr<Data::Pinning> pinning)
		{
			//-------------- Insert default values here -----------------------------
			// Output folder for results
			std::string output_folder = "output_mc";
			// Save output when logging
			bool output_tag_time = true, output_any = true, output_initial = true, output_final = true;
			bool output_energy_divide_by_nspins = true, output_energy_spin_resolved = true, output_energy_step = true, output_energy_archive = true;
			bool output_configuration_step = false, output_configuration_archive = false;
			// Maximum walltime in seconds
			long int max_walltime = 0;
			std::string str_max_walltime;
			// PRNG Seed
			std::srand((int)std::time(0));
			int seed = std::rand();
			// number of iterations carried out when pressing "play" or calling "iterate"
			int n_iterations = (int)2E+6;
			// Number of iterations after which the system is logged to file
			int n_iterations_log = 100;
			// Temperature in K
			scalar temperature = 0.0;
			// Acceptance ratio
			scalar acceptance_ratio = 0.5;

			//------------------------------- Parser --------------------------------
			Log(Log_Level::Info, Log_Sender::IO, "Parameters MC: building");

			if (configFile != "")
			{
				try {
					IO::Filter_File_Handle myfile(configFile);

					myfile.Read_Single(output_tag_time, "output_tag_time");
					myfile.Read_Single(output_folder, "mc_output_folder");
					myfile.Read_Single(output_any, "mc_output_any");
					myfile.Read_Single(output_initial, "mc_output_initial");
					myfile.Read_Single(output_final, "mc_output_final");
					myfile.Read_Single(output_energy_spin_resolved, "mc_output_energy_spin_resolved");
					myfile.Read_Single(output_energy_step, "mc_output_energy_step");
					myfile.Read_Single(output_energy_archive, "mc_output_energy_archive");
					myfile.Read_Single(output_energy_divide_by_nspins, "mc_output_energy_divide_by_nspins");
					myfile.Read_Single(output_configuration_step, "mc_output_configuration_step");
					myfile.Read_Single(output_configuration_archive, "mc_output_configuration_archive");
					myfile.Read_Single(str_max_walltime, "mc_max_walltime");
					myfile.Read_Single(seed, "mc_seed");
					myfile.Read_Single(n_iterations, "mc_n_iterations");
					myfile.Read_Single(n_iterations_log, "mc_n_iterations_log");
					myfile.Read_Single(temperature, "mc_temperature");
					myfile.Read_Single(acceptance_ratio, "mc_acceptance_ratio");
				}// end try
				catch (Exception ex) {
					if (ex == Exception::File_not_Found)
					{
						Log(Log_Level::Error, Log_Sender::IO, "Parameters MC: Unable to open Config File " + configFile + " Leaving values at default.");
					}
					else throw ex;
				}// end catch
			}
			else Log(Log_Level::Warning, Log_Sender::IO, "Parameters LLG: Using default configuration!");

			// Return
			Log(Log_Level::Parameter, Log_Sender::IO, "Parameters MC:");
			Log(Log_Level::Parameter, Log_Sender::IO, "        maximum walltime  = " + str_max_walltime);
			Log(Log_Level::Parameter, Log_Sender::IO, "        seed              = " + std::to_string(seed));
			Log(Log_Level::Parameter, Log_Sender::IO, "        temperature       = " + std::to_string(temperature));
			Log(Log_Level::Parameter, Log_Sender::IO, "        acceptance_ratio  = " + std::to_string(acceptance_ratio));
			Log(Log_Level::Parameter, Log_Sender::IO, "        n_iterations      = " + std::to_string(n_iterations));
			Log(Log_Level::Parameter, Log_Sender::IO, "        n_iterations_log  = " + std::to_string(n_iterations_log));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_folder     = " + output_folder);
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_any        = " + std::to_string(output_any));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_initial    = " + std::to_string(output_initial));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_final      = " + std::to_string(output_final));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_energy_step             = " + std::to_string(output_energy_step));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_energy_archive          = " + std::to_string(output_energy_archive));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_energy_spin_resolved    = " + std::to_string(output_energy_spin_resolved));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_energy_divide_by_nspins = " + std::to_string(output_energy_divide_by_nspins));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_configuration_step      = " + std::to_string(output_configuration_step));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_configuration_archive   = " + std::to_string(output_configuration_archive));
			max_walltime = (long int)Utility::Timing::DurationFromString(str_max_walltime).count();
			auto mc_params = std::unique_ptr<Data::Parameters_Method_MC>(new Data::Parameters_Method_MC(output_folder, { output_tag_time, output_any, output_initial, output_final, output_energy_step, output_energy_archive, output_energy_spin_resolved,
				output_energy_divide_by_nspins, output_configuration_step, output_configuration_archive }, n_iterations, n_iterations_log, max_walltime, pinning, seed, temperature, acceptance_ratio));
			Log(Log_Level::Info, Log_Sender::IO, "Parameters LLG: built");
			return mc_params;
		}

		std::unique_ptr<Data::Parameters_Method_GNEB> Parameters_Method_GNEB_from_Config(const std::string configFile, const std::shared_ptr<Data::Pinning> pinning)
		{
			//-------------- Insert default values here -----------------------------
			// Output folder for results
			std::string output_folder = "output_gneb";
			// Save output when logging
			bool output_tag_time = true, output_any = true, output_initial = false, output_final = true, output_energies_step = false, output_energies_interpolated = true, output_energies_divide_by_nspins = true, output_chain_step = false;
			// Maximum walltime in seconds
			long int max_walltime = 0;
			std::string str_max_walltime;
			// Spring constant
			scalar spring_constant = 1.0;
			// Force convergence parameter
			scalar force_convergence = 10e-9;
			// number of iterations carried out when pressing "play" or calling "iterate"
			int n_iterations = (int)2E+6;
			// Number of iterations after which the system is logged to file
			int n_iterations_log = 100;
			// Number of Energy Interpolation points
			int n_E_interpolations = 10;
			//------------------------------- Parser --------------------------------
			Log(Log_Level::Info, Log_Sender::IO, "Parameters GNEB: building");
			if (configFile != "")
			{
				try {
					IO::Filter_File_Handle myfile(configFile);

					myfile.Read_Single(output_tag_time, "output_tag_time");
					myfile.Read_Single(output_folder, "gneb_output_folder");
					myfile.Read_Single(output_any, "gneb_output_any");
					myfile.Read_Single(output_initial, "gneb_output_initial");
					myfile.Read_Single(output_final, "gneb_output_final");
					myfile.Read_Single(output_energies_step, "gneb_output_energies_step");
					myfile.Read_Single(output_energies_interpolated, "gneb_output_energies_interpolated");
					myfile.Read_Single(output_energies_divide_by_nspins, "gneb_output_energies_divide_by_nspins");
					myfile.Read_Single(output_chain_step, "gneb_output_chain_step");
					myfile.Read_Single(str_max_walltime, "gneb_max_walltime");
					myfile.Read_Single(spring_constant, "gneb_spring_constant");
					myfile.Read_Single(force_convergence, "gneb_force_convergence");
					myfile.Read_Single(n_iterations, "gneb_n_iterations");
					myfile.Read_Single(n_iterations_log, "gneb_n_iterations_log");
					myfile.Read_Single(n_E_interpolations, "gneb_n_energy_interpolations");
				}// end try
				catch (Exception ex) {
					if (ex == Exception::File_not_Found)
					{
						Log(Log_Level::Error, Log_Sender::IO, "Parameters GNEB: Unable to open Config File " + configFile + " Leaving values at default.");
					}
					else throw ex;
				}// end catch
			}
			else Log(Log_Level::Warning, Log_Sender::IO, "Parameters GNEB: Using default configuration!");

			// Return
			Log(Log_Level::Parameter, Log_Sender::IO, "Parameters GNEB:");
			Log(Log_Level::Parameter, Log_Sender::IO, "        maximum walltime  = " + str_max_walltime);
			Log(Log_Level::Parameter, Log_Sender::IO, "        spring_constant      = " + std::to_string(spring_constant));
			Log(Log_Level::Parameter, Log_Sender::IO, "        force_convergence    = " + std::to_string(force_convergence));
			Log(Log_Level::Parameter, Log_Sender::IO, "        n_E_interpolations   = " + std::to_string(n_E_interpolations));
			Log(Log_Level::Parameter, Log_Sender::IO, "        n_iterations         = " + std::to_string(n_iterations));
			Log(Log_Level::Parameter, Log_Sender::IO, "        n_iterations_log     = " + std::to_string(n_iterations_log));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_folder        = " + output_folder);
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_any           = " + std::to_string(output_any));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_initial       = " + std::to_string(output_initial));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_final         = " + std::to_string(output_final));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_energies_step = " + std::to_string(output_energies_step));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_chain_step    = " + std::to_string(output_chain_step));
			max_walltime = (long int)Utility::Timing::DurationFromString(str_max_walltime).count();
			auto gneb_params = std::unique_ptr<Data::Parameters_Method_GNEB>(new Data::Parameters_Method_GNEB(output_folder, { output_tag_time, output_any, output_initial, output_final, output_energies_step, output_energies_interpolated, output_energies_divide_by_nspins, output_chain_step },
				force_convergence, n_iterations, n_iterations_log, max_walltime, pinning, spring_constant, n_E_interpolations));
			Log(Log_Level::Info, Log_Sender::IO, "Parameters GNEB: built");
			return gneb_params;
		}// end Parameters_Method_LLG_from_Config

		std::unique_ptr<Data::Parameters_Method_MMF> Parameters_Method_MMF_from_Config(const std::string configFile, const std::shared_ptr<Data::Pinning> pinning)
		{
			//-------------- Insert default values here -----------------------------
			// Output folder for results
			std::string output_folder = "output_mmf";
			// Save output when logging
			bool output_tag_time = true, output_any = true, output_initial = false, output_final = true, output_energy_step = false, output_energy_archive = true, output_energy_divide_by_nspins = true, output_configuration_step = false, output_configuration_archive = true;
			// Maximum walltime in seconds
			long int max_walltime = 0;
			std::string str_max_walltime;
			// Force convergence parameter
			scalar force_convergence = 10e-9;
			// Number of iterations carried out when pressing "play" or calling "iterate"
			int n_iterations = (int)2E+6;
			// Number of iterations after which the system is logged to file
			int n_iterations_log = 100;
			
			//------------------------------- Parser --------------------------------
			Log(Log_Level::Info, Log_Sender::IO, "Parameters MMF: building");
			if (configFile != "")
			{
				try {
					IO::Filter_File_Handle myfile(configFile);

					myfile.Read_Single(output_tag_time, "output_tag_time");
					myfile.Read_Single(output_folder, "mmf_output_folder");
					myfile.Read_Single(output_any, "mmf_output_any");
					myfile.Read_Single(output_initial, "mmf_output_initial");
					myfile.Read_Single(output_final, "mmf_output_final");
					myfile.Read_Single(output_energy_step, "mmf_output_energy_step");
					myfile.Read_Single(output_energy_archive, "mmf_output_energy_archive");
					myfile.Read_Single(output_energy_divide_by_nspins, "mmf_output_energy_divide_by_nspins");
					myfile.Read_Single(output_configuration_step, "mmf_output_configuration_step");
					myfile.Read_Single(output_configuration_archive, "mmf_output_configuration_archive");
					myfile.Read_Single(str_max_walltime, "mmf_max_walltime");
					myfile.Read_Single(force_convergence, "mmf_force_convergence");
					myfile.Read_Single(n_iterations, "mmf_n_iterations");
					myfile.Read_Single(n_iterations_log, "mmf_n_iterations_log");
				}// end try
				catch (Exception ex) {
					if (ex == Exception::File_not_Found)
					{
						Log(Log_Level::Error, Log_Sender::IO, "Parameters MMF: Unable to open Config File " + configFile + " Leaving values at default.");
					}
					else throw ex;
				}// end catch
			}
			else Log(Log_Level::Warning, Log_Sender::IO, "Parameters MMF: Using default configuration!");

			// Return
			Log(Log_Level::Parameter, Log_Sender::IO, "Parameters MMF:");
			Log(Log_Level::Parameter, Log_Sender::IO, "        maximum walltime  = " + str_max_walltime);
			Log(Log_Level::Parameter, Log_Sender::IO, "        force_convergence = " + std::to_string(force_convergence));
			Log(Log_Level::Parameter, Log_Sender::IO, "        n_iterations      = " + std::to_string(n_iterations));
			Log(Log_Level::Parameter, Log_Sender::IO, "        n_iterations_log  = " + std::to_string(n_iterations_log));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_folder     = " + output_folder);
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_any        = " + std::to_string(output_any));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_initial    = " + std::to_string(output_initial));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_final      = " + std::to_string(output_final));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_energy_step             = " + std::to_string(output_energy_step));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_energy_archive          = " + std::to_string(output_energy_archive));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_energy_divide_by_nspins = " + std::to_string(output_energy_divide_by_nspins));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_configuration_step      = " + std::to_string(output_configuration_step));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_configuration_archive   = " + std::to_string(output_configuration_archive));
			max_walltime = (long int)Utility::Timing::DurationFromString(str_max_walltime).count();
			auto mmf_params = std::unique_ptr<Data::Parameters_Method_MMF>(new Data::Parameters_Method_MMF(output_folder, { output_tag_time, output_any, output_initial, output_final, output_energy_step, output_energy_archive, output_energy_divide_by_nspins, output_configuration_step,output_configuration_archive },
				force_convergence, n_iterations, n_iterations_log, max_walltime, pinning));
			Log(Log_Level::Info, Log_Sender::IO, "Parameters MMF: built");
			return mmf_params;
		}

		std::unique_ptr<Engine::Hamiltonian> Hamiltonian_from_Config(const std::string configFile, std::shared_ptr<Data::Geometry> geometry)
		{
			//-------------- Insert default values here -----------------------------
			// The type of hamiltonian we will use
			std::string hamiltonian_type = "heisenberg_neighbours";

			//------------------------------- Parser --------------------------------
			Log(Log_Level::Info, Log_Sender::IO, "Hamiltonian: building");

			// Hamiltonian type
			if (configFile != "")
			{
				try {
					Log(Log_Level::Info, Log_Sender::IO, "Hamiltonian: deciding type");
					IO::Filter_File_Handle myfile(configFile);

					// What hamiltonian do we use?
					myfile.Read_Single(hamiltonian_type, "hamiltonian");
				}// end try
				catch (Exception ex) {
					if (ex == Exception::File_not_Found) {
						Log(Log_Level::Error, Log_Sender::IO, "Hamiltonian: Unable to open Config File " + configFile + " Using default Hamiltonian: " + hamiltonian_type);
					}
					else throw ex;
				}// end catch
			}
			else Log(Log_Level::Warning, Log_Sender::IO, "Hamiltonian: Using default Hamiltonian: " + hamiltonian_type);
			
			// Hamiltonian
			std::unique_ptr<Engine::Hamiltonian> hamiltonian;
			if (hamiltonian_type == "heisenberg_neighbours")
			{
				hamiltonian = Hamiltonian_Heisenberg_Neighbours_from_Config(configFile, geometry);
			}// endif isotropic
			else if (hamiltonian_type == "heisenberg_pairs")
			{
				// TODO: to std::move or not to std::move, that is the question...
				hamiltonian = std::move(Hamiltonian_Heisenberg_Pairs_from_Config(configFile, geometry));
			}// endif anisotropic
			else if (hamiltonian_type == "gaussian")
			{
				hamiltonian = std::move(Hamiltonian_Gaussian_from_Config(configFile, geometry));
			}
			else
			{
				Log(Log_Level::Error, Log_Sender::IO, "Hamiltonian: Invalid type: " + hamiltonian_type);
			}// endif neither
			
			// Return
			Log(Log_Level::Info, Log_Sender::IO, "Hamiltonian: built hamiltonian of type: " + hamiltonian_type);
			return hamiltonian;
		}

		std::unique_ptr<Engine::Hamiltonian_Heisenberg_Neighbours> Hamiltonian_Heisenberg_Neighbours_from_Config(const std::string configFile, std::shared_ptr<Data::Geometry> geometry)
		{
			//-------------- Insert default values here -----------------------------
			// Boundary conditions (a, b, c)
			std::vector<int> boundary_conditions_i = { 0, 0, 0 };
			intfield boundary_conditions = { false, false, false };

			// Spin moment
			scalarfield mu_s = scalarfield(geometry->nos, 2);	// [nos]
			// External Magnetic Field
			std::string external_field_file = "";
			scalar B = 24;
			Vector3 B_normal = { 0.0, 0.0, 1.0 };
			bool external_field_from_file = false;
			intfield    external_field_index(geometry->nos);				// [nos]
			scalarfield external_field_magnitude(geometry->nos, 0);	// [nos]
			vectorfield external_field_normal(geometry->nos, B_normal);	// [3][nos]
			// Fill in defaults
			B_normal.normalize();
			if (B != 0)
			{
				// Fill the arrays
				for (int i = 0; i < geometry->nos; ++i)
				{
					external_field_index[i] = i;
					external_field_magnitude[i] = B;
					external_field_normal[i] = B_normal;
				}
			}
			
			// Anisotropy
			std::string anisotropy_file = "";
			scalar K = 0;
			Vector3 K_normal = { 0.0, 0.0, 1.0 };
			bool anisotropy_from_file = false;
			intfield    anisotropy_index(geometry->nos);				// [nos]
			scalarfield anisotropy_magnitude(geometry->nos, 0.0);	// [nos]
			vectorfield anisotropy_normal(geometry->nos, K_normal);	// [nos][3]
			// Fill in defaults
			K_normal.normalize();
			if (K != 0)
			{
				// Fill the arrays
				for (int i = 0; i < geometry->nos; ++i)
				{
					anisotropy_index[i] = i;
					anisotropy_magnitude[i] = K;
					anisotropy_normal[i] = K_normal;
				}
			}

			// Number of shells in which we calculate neighbours
			// Jij
			scalarfield jij = { 10.0 };
			int n_neigh_shells_exchange = jij.size();
			// DM constant
			scalarfield dij = { 6.0 };
			int n_neigh_shells_dmi = dij.size();
			int dm_chirality = 1;
			// Dipole-Dipole interaction radius
			scalar dd_radius = 0.0;

			//------------------------------- Parser --------------------------------
			Log(Log_Level::Info, Log_Sender::IO, "Hamiltonian_Heisenberg_Neighbours: building");
			// iteration variables
			int iatom = 0;
			if (configFile != "")
			{
				try
				{
					IO::Filter_File_Handle myfile(configFile);

					myfile.Read_3Vector(boundary_conditions_i, "boundary_conditions");
					boundary_conditions[0] = (boundary_conditions_i[0] != 0);
					boundary_conditions[1] = (boundary_conditions_i[1] != 0);
					boundary_conditions[2] = (boundary_conditions_i[2] != 0);
				}// end try
				catch (Exception ex)
				{
					if (ex == Exception::File_not_Found)
						Log(Log_Level::Error, Log_Sender::IO, "Hamiltonian_Heisenberg_Neighbours: Unable to open Config File " + configFile + " Leaving values at default.");
					else throw ex;
				}

				try
				{
					IO::Filter_File_Handle myfile(configFile);

					// External Field
					if (myfile.Find("external_field_file")) myfile.iss >> external_field_file;
					if (external_field_file.length() > 0)
					{
						int n;
						// The file name should be valid so we try to read it
						External_Field_from_File(external_field_file, geometry, n,
							external_field_index, external_field_magnitude, external_field_normal);
						
						external_field_from_file = true;
						B = external_field_magnitude[0];
						B_normal = external_field_normal[0];
					}
					else
					{
						// Read parameters from config if available
						myfile.Read_Single(B, "external_field_magnitude");
						myfile.Read_Vector3(B_normal, "external_field_normal");
						B_normal.normalize();

						if (B != 0)
						{
							// Fill the arrays
							for (int i = 0; i < geometry->nos; ++i)
							{
								external_field_index[i] = i;
								external_field_magnitude[i] = B;
								external_field_normal[i] = B_normal;
							}
						}
						else
						{
							external_field_index = intfield(0);
							external_field_magnitude = scalarfield(0);
							external_field_normal = vectorfield(0);
						}
					}
				}// end try
				catch (Exception ex)
				{
					if (ex == Exception::File_not_Found)
						Log(Log_Level::Error, Log_Sender::IO, "Hamiltonian_Heisenberg_Neighbours: Unable to open Config File " + configFile + " Leaving values at default.");
					else throw ex;
				}

				try
				{
					IO::Filter_File_Handle myfile(configFile);

					// Anisotropy
					if (myfile.Find("anisotropy_file")) myfile.iss >> anisotropy_file;
					if (anisotropy_file.length() > 0)
					{
						int n;
						// The file name should be valid so we try to read it
						Anisotropy_from_File(anisotropy_file, geometry, n,
							anisotropy_index, anisotropy_magnitude, anisotropy_normal);

						anisotropy_from_file = true;
						K = anisotropy_magnitude[0];
						K_normal = anisotropy_normal[0];
					}
					else
					{
						// Read parameters from config
						myfile.Read_Single(K, "anisotropy_magnitude");
						myfile.Read_Vector3(K_normal, "anisotropy_normal");
						K_normal.normalize();

						if (K != 0)
						{
							// Fill the arrays
							for (int i = 0; i < geometry->nos; ++i)
							{
								anisotropy_index[i] = i;
								anisotropy_magnitude[i] = K;
								anisotropy_normal[i] = K_normal;
							}
						}
						else
						{
							anisotropy_index = intfield(0);
							anisotropy_magnitude = scalarfield(0);
							anisotropy_normal = vectorfield(0);
						}
					}
				}// end try
				catch (Exception ex)
				{
					if (ex == Exception::File_not_Found)
						Log(Log_Level::Error, Log_Sender::IO, "Hamiltonian_Heisenberg_Neighbours: Unable to open Config File " + configFile + " Leaving values at default.");
					else throw ex;
				}

				try
				{
					IO::Filter_File_Handle myfile(configFile);

					myfile.Read_Single(n_neigh_shells_exchange, "n_neigh_shells_exchange");
					if (jij.size() != n_neigh_shells_exchange)
						jij = scalarfield(n_neigh_shells_exchange);
					if (n_neigh_shells_exchange > 0)
					{
						if (myfile.Find("jij"))
						{
							for (iatom = 0; iatom < n_neigh_shells_exchange; ++iatom)
								myfile.iss >> jij[iatom];
						}
						else Log(Log_Level::Warning, Log_Sender::IO, "Hamiltonian_Heisenberg_Neighbours: Keyword 'jij' not found. Using Default:  { 10.0 }");
					}
				}// end try
				catch (Exception ex)
				{
					if (ex == Exception::File_not_Found)
						Log(Log_Level::Error, Log_Sender::IO, "Hamiltonian_Heisenberg_Neighbours: Unable to open Config File " + configFile + " Leaving values at default.");
					else throw ex;
				}

				try
				{
					IO::Filter_File_Handle myfile(configFile);

					myfile.Read_Single(n_neigh_shells_dmi, "n_neigh_shells_dmi");
					if (dij.size() != n_neigh_shells_dmi)
						dij = scalarfield(n_neigh_shells_dmi);
					if (n_neigh_shells_dmi > 0)
					{
						if (myfile.Find("dij"))
						{
							for (iatom = 0; iatom < n_neigh_shells_dmi; ++iatom)
								myfile.iss >> dij[iatom];
						}
						else Log(Log_Level::Warning, Log_Sender::IO, "Hamiltonian_Heisenberg_Neighbours: Keyword 'dij' not found. Using Default:  { 6.0 }");
					}
					myfile.Read_Single(dm_chirality, "dm_chirality");

				}// end try
				catch (Exception ex)
				{
					if (ex == Exception::File_not_Found)
						Log(Log_Level::Error, Log_Sender::IO, "Hamiltonian_Heisenberg_Neighbours: Unable to open Config File " + configFile + " Leaving values at default.");
					else throw ex;
				}

				try
				{
					IO::Filter_File_Handle myfile(configFile);

					myfile.Read_Single(dd_radius, "dd_radius");
				}// end try
				catch (Exception ex)
				{
					if (ex == Exception::File_not_Found)
						Log(Log_Level::Error, Log_Sender::IO, "Hamiltonian_Heisenberg_Neighbours: Unable to open Config File " + configFile + " Leaving values at default.");
					else throw ex;
				}
			}
			else Log(Log_Level::Warning, Log_Sender::IO, "Hamiltonian_Heisenberg_Neighbours: Using default configuration!");

			// Return
			Log(Log_Level::Parameter, Log_Sender::IO, "Hamiltonian_Heisenberg_Neighbours:");
			Log(Log_Level::Parameter, Log_Sender::IO, "        boundary conditions = " + std::to_string(boundary_conditions[0]) + " " + std::to_string(boundary_conditions[1]) + " " + std::to_string(boundary_conditions[2]));
			if (external_field_from_file)
				Log(Log_Level::Parameter, Log_Sender::IO, "        B                     from file");
			Log(Log_Level::Parameter, Log_Sender::IO, "        B[0]                = " + std::to_string(B));
			Log(Log_Level::Parameter, Log_Sender::IO, "        B_normal[0]         = " + std::to_string(B_normal[0]) + " " + std::to_string(B_normal[1]) + " " + std::to_string(B_normal[2]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        mu_s[0]             = " + std::to_string(mu_s[0]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        K[0]                = " + std::to_string(K));
			if (anisotropy_from_file)
				Log(Log_Level::Parameter, Log_Sender::IO, "        K                     from file");
			Log(Log_Level::Parameter, Log_Sender::IO, "        K_normal[0]         = " + std::to_string(K_normal[0]) + " " + std::to_string(K_normal[1]) + " " + std::to_string(K_normal[2]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        n_shells_exchange   = " + std::to_string(n_neigh_shells_exchange));
			if (n_neigh_shells_exchange > 0)
				Log(Log_Level::Parameter, Log_Sender::IO, "        J_ij[0]             = " + std::to_string(jij[0]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        n_shells_dmi        = " + std::to_string(n_neigh_shells_dmi));
			if (n_neigh_shells_dmi > 0)
				Log(Log_Level::Parameter, Log_Sender::IO, "        D_ij[0]             = " + std::to_string(dij[0]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        DM chirality        = " + std::to_string(dm_chirality));
			Log(Log_Level::Parameter, Log_Sender::IO, "        dd_radius           = " + std::to_string(dd_radius));
			auto hamiltonian = std::unique_ptr<Engine::Hamiltonian_Heisenberg_Neighbours>(new Engine::Hamiltonian_Heisenberg_Neighbours(
					mu_s, external_field_index, external_field_magnitude, external_field_normal,
					anisotropy_index, anisotropy_magnitude, anisotropy_normal,
					jij,
					dij, dm_chirality,
					dd_radius,
					geometry,
					boundary_conditions
				));
			Log(Log_Level::Info, Log_Sender::IO, "Hamiltonian_Heisenberg_Neighbours: built");
			return hamiltonian;
		}// end Hamiltonian_Heisenberg_Neighbours_from_Config


		
		std::unique_ptr<Engine::Hamiltonian_Heisenberg_Pairs> Hamiltonian_Heisenberg_Pairs_from_Config(const std::string configFile, std::shared_ptr<Data::Geometry> geometry)
		{
			//-------------- Insert default values here -----------------------------
			// Boundary conditions (a, b, c)
			std::vector<int> boundary_conditions_i = { 0, 0, 0 };
			intfield boundary_conditions = { false, false, false };
			// Spin moment
			scalarfield mu_s = scalarfield(geometry->nos, 2);	// [nos]
			// External Magnetic Field
			std::string external_field_file = "";
			scalar B = 0;
			Vector3 B_normal = { 0.0, 0.0, 1.0 };
			bool external_field_from_file = false;
			intfield    external_field_index(geometry->nos);				// [nos]
			scalarfield external_field_magnitude(geometry->nos, 0);	// [nos]
			vectorfield external_field_normal(geometry->nos, B_normal);	// [3][nos]
			
			// Anisotropy
			std::string anisotropy_file = "";
			scalar K = 0;
			Vector3 K_normal = { 0.0, 0.0, 1.0 };
			bool anisotropy_from_file = false;
			intfield    anisotropy_index(geometry->nos);				// [nos]
			scalarfield anisotropy_magnitude(geometry->nos, 0.0);	// [nos]
			vectorfield anisotropy_normal(geometry->nos, K_normal);	// [nos][3]

			// ------------ Pair Interactions ------------
			int n_pairs = 0;
			std::string interaction_pairs_file = "";
			bool interaction_pairs_from_file = false;
			pairfield exchange_pairs(0); scalarfield exchange_magnitudes(0);
			pairfield dmi_pairs(0); scalarfield dmi_magnitudes(0); vectorfield dmi_normals(0);
			scalar ddi_radius = 0.0;

			// ------------ Quadruplet Interactions ------------
			int n_quadruplets = 0;
			std::string quadruplets_file = "";
			bool quadruplets_from_file = false;
			quadrupletfield quadruplets(0); scalarfield quadruplet_magnitudes(0);

			//------------------------------- Parser --------------------------------
			Log(Log_Level::Info, Log_Sender::IO, "Hamiltonian_Heisenberg_Pairs: building");
			// iteration variables
			int iatom = 0;
			if (configFile != "")
			{
				try
				{
					IO::Filter_File_Handle myfile(configFile);

					// Boundary conditions
					myfile.Read_3Vector(boundary_conditions_i, "boundary_conditions");
					boundary_conditions[0] = (boundary_conditions_i[0] != 0);
					boundary_conditions[1] = (boundary_conditions_i[1] != 0);
					boundary_conditions[2] = (boundary_conditions_i[2] != 0);
				}// end try
				catch (Exception ex)
				{
					if (ex == Exception::File_not_Found)
						Log(Log_Level::Error, Log_Sender::IO, "Hamiltonian_Heisenberg_Pairs: Unable to open Config File " + configFile + " Leaving values at default.");
					else throw ex;
				}

				try
				{
					IO::Filter_File_Handle myfile(configFile);

					// Spin moment
					mu_s = scalarfield(geometry->nos, 2.0);
					if (myfile.Find("mu_s"))
					{
						for (iatom = 0; iatom < geometry->n_spins_basic_domain; ++iatom)
						{
							myfile.iss >> mu_s[iatom];
							for (int ispin = 0; ispin < geometry->nos / geometry->n_spins_basic_domain; ++ispin)
							{
								mu_s[ispin*geometry->n_spins_basic_domain + iatom] = mu_s[iatom];
							}
						}
					}
					else Log(Log_Level::Error, Log_Sender::IO, "Keyword 'mu_s' not found. Using Default: 2.0");
				}// end try
				catch (Exception ex)
				{
					if (ex == Exception::File_not_Found)
						Log(Log_Level::Error, Log_Sender::IO, "Hamiltonian_Heisenberg_Pairs: Unable to open Config File " + configFile + " Leaving values at default.");
					else throw ex;
				}

				try
				{
					IO::Filter_File_Handle myfile(configFile);

					// External Field
					if (myfile.Find("n_external_field"))
						external_field_file = configFile;
					else if (myfile.Find("external_field_file"))
						myfile.iss >> external_field_file;
					if (external_field_file.length() > 0)
					{
						// The file name should be valid so we try to read it
						External_Field_from_File(external_field_file, geometry, n_pairs,
							external_field_index, external_field_magnitude, external_field_normal);
						
						external_field_from_file = true;
						if (external_field_index.size() != 0)
						{
							B = external_field_magnitude[0];
							B_normal = external_field_normal[0];
						}
						else
						{
							B = 0;
							B_normal = { 0,0,0 };
						}
					}
					else
					{
						// Read parameters from config if available
						myfile.Read_Single(B, "external_field_magnitude");
						myfile.Read_Vector3(B_normal, "external_field_normal");
						B_normal.normalize();

						if (B != 0)
						{
							// Fill the arrays
							for (int i = 0; i < geometry->nos; ++i)
							{
								external_field_index[i] = i;
								external_field_magnitude[i] = B;
								external_field_normal[i] = B_normal;
							}
						}
						else
						{
							external_field_index = intfield(0);
							external_field_magnitude = scalarfield(0);
							external_field_normal = vectorfield(0);
						}
					}
				}// end try
				catch (Exception ex)
				{
					if (ex == Exception::File_not_Found)
						Log(Log_Level::Error, Log_Sender::IO, "Hamiltonian_Heisenberg_Pairs: Unable to open Config File " + configFile + " Leaving values at default.");
					else throw ex;
				}

				try
				{
					IO::Filter_File_Handle myfile(configFile);

					// Anisotropy
					if (myfile.Find("n_anisotropy"))
						anisotropy_file = configFile;
					else if (myfile.Find("anisotropy_file"))
						myfile.iss >> anisotropy_file;
					if (anisotropy_file.length() > 0)
					{
						// The file name should be valid so we try to read it
						Anisotropy_from_File(anisotropy_file, geometry, n_pairs,
							anisotropy_index, anisotropy_magnitude, anisotropy_normal);

						anisotropy_from_file = true;
						if (anisotropy_index.size() != 0)
						{
							K = anisotropy_magnitude[0];
							K_normal = anisotropy_normal[0];
						}
						else
						{
							K = 0;
							K_normal = { 0,0,0 };
						}
					}
					else
					{
						// Read parameters from config
						myfile.Read_Single(K, "anisotropy_magnitude");
						myfile.Read_Vector3(K_normal, "anisotropy_normal");
						K_normal.normalize();

						if (K != 0)
						{
							// Fill the arrays
							for (int i = 0; i < geometry->nos; ++i)
							{
								anisotropy_index[i] = i;
								anisotropy_magnitude[i] = K;
								anisotropy_normal[i] = K_normal;
							}
						}
						else
						{
							anisotropy_index = intfield(0);
							anisotropy_magnitude = scalarfield(0);
							anisotropy_normal = vectorfield(0);
						}
					}
				}// end try
				catch (Exception ex)
				{
					if (ex == Exception::File_not_Found)
						Log(Log_Level::Error, Log_Sender::IO, "Hamiltonian_Heisenberg_Pairs: Unable to open Config File " + configFile + " Leaving values at default.");
					else throw ex;
				}

				try
				{
					IO::Filter_File_Handle myfile(configFile);

					// Interaction Pairs
					if (myfile.Find("n_interaction_pairs"))
						interaction_pairs_file = configFile;
					else if (myfile.Find("interaction_pairs_file"))
						myfile.iss >> interaction_pairs_file;

					if (interaction_pairs_file.length() > 0)
					{
						// The file name should be valid so we try to read it
						Pairs_from_File(interaction_pairs_file, geometry, n_pairs,
							exchange_pairs, exchange_magnitudes,
							dmi_pairs, dmi_magnitudes, dmi_normals);
					}
					//else
					//{
					//	Log(Log_Level::Warning, Log_Sender::IO, "Hamiltonian_Heisenberg_Pairs: Default Interaction pairs have not been implemented yet.");
					//	throw Exception::System_not_Initialized;
					//	// Not implemented!
					//}
				}// end try
				catch (Exception ex)
				{
					if (ex == Exception::File_not_Found)
						Log(Log_Level::Error, Log_Sender::IO, "Hamiltonian_Heisenberg_Pairs: Unable to open Config File " + configFile + " Leaving values at default.");
					else throw ex;
				}

				try
				{
					IO::Filter_File_Handle myfile(configFile);

					//		Dipole-Dipole Pairs
					// Dipole Dipole radius
					myfile.Read_Single(ddi_radius, "dd_radius");
				}// end try
				catch (Exception ex)
				{
					if (ex == Exception::File_not_Found)
						Log(Log_Level::Error, Log_Sender::IO, "Hamiltonian_Heisenberg_Pairs: Unable to open Config File " + configFile + " Leaving values at default.");
					else throw ex;
				}

				try
				{
					IO::Filter_File_Handle myfile(configFile);

					// Interaction Quadruplets
					if (myfile.Find("n_interaction_quadruplets"))
						quadruplets_file = configFile;
					else if (myfile.Find("interaction_quadruplets_file"))
						myfile.iss >> quadruplets_file;

					if (quadruplets_file.length() > 0)
					{
						// The file name should be valid so we try to read it
						Quadruplets_from_File(quadruplets_file, geometry, n_quadruplets,
							quadruplets, quadruplet_magnitudes);
					}

				}// end try
				catch (Exception ex)
				{
					if (ex == Exception::File_not_Found)
					{
						Log(Log_Level::Error, Log_Sender::IO, "Hamiltonian_Heisenberg_Pairs: Unable to open Config File " + configFile + " Leaving values at default.");
					}
					else throw ex;
				}// end catch
			}
			else Log(Log_Level::Warning, Log_Sender::IO, "Hamiltonian_Heisenberg_Pairs: Using default configuration!");
			
			// Return
			Log(Log_Level::Parameter, Log_Sender::IO, "Hamiltonian_Heisenberg_Pairs:");
			Log(Log_Level::Parameter, Log_Sender::IO, "        boundary conditions = " + std::to_string(boundary_conditions[0]) + " " + std::to_string(boundary_conditions[1]) + " " + std::to_string(boundary_conditions[2]));
			if (external_field_from_file)
				Log(Log_Level::Parameter, Log_Sender::IO, "        B                     from file");
			Log(Log_Level::Parameter, Log_Sender::IO, "        B[0]                = " + std::to_string(B));
			Log(Log_Level::Parameter, Log_Sender::IO, "        B_normal[0]         = " + std::to_string(B_normal[0]) + " " + std::to_string(B_normal[1]) + " " + std::to_string(B_normal[2]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        mu_s[0]             = " + std::to_string(mu_s[0]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        K[0]                = " + std::to_string(K));
			if (anisotropy_from_file)
				Log(Log_Level::Parameter, Log_Sender::IO, "        K                     from file");
			Log(Log_Level::Parameter, Log_Sender::IO, "        K_normal[0]         = " + std::to_string(K_normal[0]) + " " + std::to_string(K_normal[1]) + " " + std::to_string(K_normal[2]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        dd_radius           = " + std::to_string(ddi_radius));
			auto hamiltonian = std::unique_ptr<Engine::Hamiltonian_Heisenberg_Pairs>(new Engine::Hamiltonian_Heisenberg_Pairs(
				mu_s,
				external_field_index, external_field_magnitude, external_field_normal,
				anisotropy_index, anisotropy_magnitude, anisotropy_normal,
				exchange_pairs, exchange_magnitudes,
				dmi_pairs, dmi_magnitudes, dmi_normals,
				ddi_radius,
				quadruplets, quadruplet_magnitudes,
				geometry,
				boundary_conditions
			));
			Log(Log_Level::Info, Log_Sender::IO, "Hamiltonian_Heisenberg_Pairs: built");
			return hamiltonian;
		}// end Hamiltonian_Heisenberg_Pairs_From_Config
		
		
		std::unique_ptr<Engine::Hamiltonian_Gaussian> Hamiltonian_Gaussian_from_Config(const std::string configFile, std::shared_ptr<Data::Geometry> geometry)
		{
			//-------------- Insert default values here -----------------------------
			// Number of Gaussians
			int n_gaussians = 1;
			// Amplitudes
			std::vector<scalar> amplitude = { 1 };
			// Widths
			std::vector<scalar> width = { 1 };
			// Centers
			std::vector<Vector3> center = { Vector3{ 0, 0, 1 } };

			//------------------------------- Parser --------------------------------
			Log(Log_Level::Info, Log_Sender::IO, "Hamiltonian_Gaussian: building");
			
			if (configFile != "")
			{
				try
				{
					IO::Filter_File_Handle myfile(configFile);

					// N
					myfile.Read_Single(n_gaussians, "n_gaussians");

					// Allocate arrays
					amplitude = std::vector<scalar>(n_gaussians, 1.0);
					width = std::vector<scalar>(n_gaussians, 1.0);
					center = std::vector<Vector3>(n_gaussians, Vector3{0, 0, 1});
					// Read arrays
					if (myfile.Find("gaussians"))
					{
						for (int i = 0; i < n_gaussians; ++i)
						{
							myfile.GetLine();
							myfile.iss >> amplitude[i];
							myfile.iss >> width[i];
							for (int j = 0; j < 3; ++j)
							{
								myfile.iss >> center[i][j];
							}
							center[i].normalize();
						}
					}
					else Log(Log_Level::Error, Log_Sender::IO, "Hamiltonian_Gaussian: Keyword 'gaussians' not found. Using Default: {0, 0, 1}");
				}// end try
				catch (Exception ex) {
					if (ex == Exception::File_not_Found)
					{
						Log(Log_Level::Error, Log_Sender::IO, "Hamiltonian_Gaussian: Unable to open Config File " + configFile + " Leaving values at default.");
					}
					else throw ex;
				}// end catch
			}
			else Log(Log_Level::Warning, Log_Sender::IO, "Hamiltonian_Gaussian: Using default configuration!");


			// Return
			Log(Log_Level::Parameter, Log_Sender::IO, "Hamiltonian_Gaussian:");
			Log(Log_Level::Parameter, Log_Sender::IO, "        n_gaussians  = " + std::to_string(n_gaussians));
			Log(Log_Level::Parameter, Log_Sender::IO, "        amplitude[0] = " + std::to_string(amplitude[0]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        width[0]     = " + std::to_string(width[0]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        center[0]    = " + std::to_string(center[0][0]) + " " + std::to_string(center[0][1]) + " " + std::to_string(center[0][2]));
			auto hamiltonian = std::unique_ptr<Engine::Hamiltonian_Gaussian>(new Engine::Hamiltonian_Gaussian(
				amplitude, width, center
			));
			Log(Log_Level::Info, Log_Sender::IO, "Hamiltonian_Gaussian: built");
			return hamiltonian;
		}
	}// end namespace IO
}// end namespace Utility