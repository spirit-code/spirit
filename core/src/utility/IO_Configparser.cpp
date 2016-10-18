#include "IO.hpp"
#include "IO_Filter_File_Handle.hpp"
#include "Vectormath.hpp"
#include "Neighbours.hpp"
#include "Logging.hpp"
#include "Exception.hpp"

#include <iostream>
#include <fstream>
#include <thread>
#include <string>
#include <sstream>

namespace Utility
{
	namespace IO
	{
		void Log_Levels_from_Config(const std::string configFile)
		{
			// Verbosity and Reject Level are read as integers
			int i_print_level = 5, i_accept_level = 5;
			std::string output_folder = ".";

			//------------------------------- Parser --------------------------------
			if (configFile != "")
			{
				try {
					Log(Log_Level::Info, Log_Sender::IO, "Building Log_Levels");
					IO::Filter_File_Handle myfile(configFile);

					// Accept Level
					if (myfile.Find("log_accept")) myfile.iss >> i_accept_level;
					else Log(Log_Level::Error, Log_Sender::IO, "Keyword 'log_accept' not found. Using Default.");

					// Print level
					if (myfile.Find("log_print")) myfile.iss >> i_print_level;
					else Log(Log_Level::Error, Log_Sender::IO, "Keyword 'log_print' not found. Using Default.");

					// Output folder
					if (myfile.Find("log_output_folder")) myfile.iss >> output_folder;
					else Log(Log_Level::Error, Log_Sender::IO, "Keyword 'log_output_folder' not found. Using Default.");

				}// end try
				catch (Exception ex) {
					if (ex == Exception::File_not_Found) {
						Log(Log_Level::Error, Log_Sender::IO, "Log_Levels: Unable to open Config File " + configFile + " Leaving values at default.");
					}
					else throw ex;
				}// end catch
			}
			// Log the parameters
			Log(Log_Level::Parameter, Log_Sender::IO, "Log accept level  = " + std::to_string(i_accept_level));
			Log(Log_Level::Parameter, Log_Sender::IO, "Log print level   = " + std::to_string(i_print_level));
			Log(Log_Level::Parameter, Log_Sender::IO, "Log output folder = " + output_folder);
			// Update the Log
			Log.accept_level  = Log_Level(i_accept_level);
			Log.print_level   = Log_Level(i_print_level);
			Log.output_folder = output_folder;
		}// End Log_Levels_from_Config


		std::unique_ptr<Data::Spin_System> Spin_System_from_Config(const std::string configFile)
		{
			Log(Log_Level::Info, Log_Sender::IO, "-------------- Initialising Spin System ------------");
			// ----------------------------------------------------------------------------------------------
			// Geometry
			auto geometry = Geometry_from_Config(configFile);
			// LLG Parameters
			auto llg_params = Parameters_Method_LLG_from_Config(configFile);
			// Hamiltonian
			auto hamiltonian = std::move(Hamiltonian_from_Config(configFile, *geometry));
			// Spin System
			auto system = std::unique_ptr<Data::Spin_System>(new Data::Spin_System(std::move(hamiltonian), std::move(geometry), std::move(llg_params), false));
			// ----------------------------------------------------------------------------------------------
			Log(Log_Level::Info, Log_Sender::IO, "-------------- Spin System Initialised -------------");

			// Return
			return system;
		}// End Spin_System_from_Config		


		void Basis_from_Config(const std::string configFile, std::vector<std::vector<double>> & basis, std::vector<std::vector<double>> & basis_atoms,
			int & no_spins_basic_domain)
		{
			// ---------- Default values
			// Lattice constant [Angtrom]
			double lattice_constant = 1.0;
			// Basis: vector {a, b, c}
			basis = { std::vector<double>{1,0,0}, std::vector<double>{0,1,0}, std::vector<double>{0,0,1} };
			// Atoms in the basis [dim][n_basis_atoms]
			basis_atoms = { std::vector<double>{0}, std::vector<double>{0}, std::vector<double>{0} };
			// NoS in the basic domain (= unit cell for periodic lattices)
			no_spins_basic_domain = basis_atoms[0].size();
			
			Log(Log_Level::Info, Log_Sender::IO, "Basis: building");

			if (configFile != "")
			{
				try {
					IO::Filter_File_Handle myfile(configFile);

					myfile.Read_Single(lattice_constant, "lattice_constant");

					// Utility 1D array to build vectors and use Vectormath
					std::vector<double> build_array = { 0, 0, 0 };

					if (myfile.Find("basis"))
					{
						// Read the basis vectors a, b, c
						myfile.GetLine();
						myfile.iss >> basis[0][0] >> basis[1][0] >> basis[2][0];
						myfile.GetLine();
						myfile.iss >> basis[0][1] >> basis[1][1] >> basis[2][1];
						myfile.GetLine();
						myfile.iss >> basis[0][2] >> basis[1][2] >> basis[2][2];

						// Read no_spins_basic_domain and atoms in basis
						myfile.GetLine();
						myfile.iss >> no_spins_basic_domain;
						basis_atoms = std::vector<std::vector<double>>(3, std::vector<double>(no_spins_basic_domain));

						// Read spins per basic domain
						for (int iatom = 0; iatom < no_spins_basic_domain; ++iatom)
						{
							myfile.GetLine();
							myfile.iss >> basis_atoms[0][iatom] >> basis_atoms[1][iatom] >> basis_atoms[2][iatom];
							// Get x,y,z of component of spin_pos in unit of length (instead of in units of a,b,c)
							for (int i = 0; i < 3; ++i)
							{
								build_array[i] = basis[i][0] * basis_atoms[0][iatom] + basis[i][1] * basis_atoms[1][iatom] + basis[i][2] * basis_atoms[2][iatom];
							}
							for (int i=0; i<3; ++i) basis_atoms[i][iatom] = lattice_constant * build_array[i];
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
			Log(Log_Level::Debug, Log_Sender::IO, "        a = " + std::to_string(basis[0][0]/lattice_constant) + " " + std::to_string(basis[1][0]/lattice_constant) + " " + std::to_string(basis[2][0]/lattice_constant));
			Log(Log_Level::Debug, Log_Sender::IO, "        b = " + std::to_string(basis[0][1]/lattice_constant) + " " + std::to_string(basis[1][1]/lattice_constant) + " " + std::to_string(basis[2][1]/lattice_constant));
			Log(Log_Level::Debug, Log_Sender::IO, "        c = " + std::to_string(basis[0][2]/lattice_constant) + " " + std::to_string(basis[1][2]/lattice_constant) + " " + std::to_string(basis[2][2]/lattice_constant));
			Log(Log_Level::Parameter, Log_Sender::IO, "Basis: vectors");
			Log(Log_Level::Parameter, Log_Sender::IO, "        a = " + std::to_string(basis[0][0]) + " " + std::to_string(basis[1][0]) + " " + std::to_string(basis[2][0]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        b = " + std::to_string(basis[0][1]) + " " + std::to_string(basis[1][1]) + " " + std::to_string(basis[2][1]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        c = " + std::to_string(basis[0][2]) + " " + std::to_string(basis[1][2]) + " " + std::to_string(basis[2][2]));
			Log(Log_Level::Parameter, Log_Sender::IO, "Basis: " + std::to_string(no_spins_basic_domain) + " atom(s) at the following positions:");
			for (int iatom = 0; iatom < no_spins_basic_domain; ++iatom)
			{
				Log(Log_Level::Parameter, Log_Sender::IO, "            " + std::to_string(iatom) + " = " + std::to_string(basis_atoms[0][iatom]) + " " + std::to_string(basis_atoms[1][iatom]) + " " + std::to_string(basis_atoms[2][iatom]));
			}
			Log(Log_Level::Info, Log_Sender::IO, "Basis: built");
		}// End Basis_from_Config

		std::unique_ptr<Data::Geometry> Geometry_from_Config(const std::string configFile)
		{
			//-------------- Insert default values here -----------------------------
			// Basis from separate file?
			std::string basis_file = "";
			// Basis: vector {a, b, c}
			std::vector<std::vector<double>> basis = { std::vector<double>{1,0,0}, std::vector<double>{0,1,0}, std::vector<double>{0,0,1} };
			// Atoms in the basis [dim][n_basis_atoms]
			std::vector<std::vector<double>> basis_atoms = { std::vector<double>{0}, std::vector<double>{0}, std::vector<double>{0} };
			// NoS in the basic domain (= unit cell for periodic lattices)
			int no_spins_basic_domain = basis_atoms[0].size();
			// Translation vectors [dim][nov]
			std::vector<std::vector<double>> translation_vectors = { std::vector<double>{1,0,0}, std::vector<double>{0,1,0}, std::vector<double>{0,0,1} };
			// Number of translations nT for each basis direction
			std::vector<int> n_cells = { 100, 100, 1 };
			// Number of Spins
			int nos;
			std::vector<double> spin_pos;

			// Utility 1D array to build vectors and use Vectormath
			std::vector<double> build_array = { 0, 0, 0 };

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
						myfile.iss >> translation_vectors[0][0] >> translation_vectors[1][0] >> translation_vectors[2][0] >> n_cells[0];
						myfile.GetLine();
						myfile.iss >> translation_vectors[0][1] >> translation_vectors[1][1] >> translation_vectors[2][1] >> n_cells[1];
						myfile.GetLine();
						myfile.iss >> translation_vectors[0][2] >> translation_vectors[1][2] >> translation_vectors[2][2] >> n_cells[2];
					}// finish Reading Shape in terms of basis
					else {
						Log(Log_Level::Error, Log_Sender::IO, "Keyword 'translation_vectors' not found. Using default. (sc 30x30x0)");
					}
					// Read Basis
						
					if (myfile.Find("basis_from_config"))
					{
						myfile.iss >> basis_file;
						Basis_from_Config(basis_file, basis, basis_atoms, no_spins_basic_domain);
					}
					else if (myfile.Find("basis"))
					{
						Basis_from_Config(configFile, basis, basis_atoms, no_spins_basic_domain);
					}
					else {
						Log(Log_Level::Error, Log_Sender::IO, "Neither Keyword 'basis_from_config', nor Keyword 'basis' found. Using Default (sc)");
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
			nos = no_spins_basic_domain * n_cells[0] * n_cells[1] * n_cells[2];

			// Spin Positions
			spin_pos = std::vector<double>(3*nos);
			Vectormath::Build_Spins(spin_pos, basis_atoms, translation_vectors, n_cells, no_spins_basic_domain);
			
			// Log parameters
			Log(Log_Level::Parameter, Log_Sender::IO, "Translation: vectors transformed by basis");
			Log(Log_Level::Parameter, Log_Sender::IO, "        a = " + std::to_string(translation_vectors[0][0]) + " " + std::to_string(translation_vectors[1][0]) + " " + std::to_string(translation_vectors[2][0]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        b = " + std::to_string(translation_vectors[0][1]) + " " + std::to_string(translation_vectors[1][1]) + " " + std::to_string(translation_vectors[2][1]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        c = " + std::to_string(translation_vectors[0][2]) + " " + std::to_string(translation_vectors[1][2]) + " " + std::to_string(translation_vectors[2][2]));
			Log(Log_Level::Parameter, Log_Sender::IO, "Translation: n_cells");
			Log(Log_Level::Parameter, Log_Sender::IO, "        na = " + std::to_string(n_cells[0]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        nb = " + std::to_string(n_cells[1]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        nc = " + std::to_string(n_cells[2]));
			Log(Log_Level::Parameter, Log_Sender::IO, "Geometry: " + std::to_string(nos) + " spins");

			// Return geometry
			auto geometry = std::unique_ptr<Data::Geometry>(new Data::Geometry(basis, translation_vectors, n_cells, no_spins_basic_domain, basis_atoms, spin_pos));
			Log(Log_Level::Info, Log_Sender::IO, "Geometry: built");
			return geometry;
		}// end Geometry from Config

		std::unique_ptr<Data::Parameters_Method_LLG> Parameters_Method_LLG_from_Config(const std::string configFile)
		{
			//-------------- Insert default values here -----------------------------
			// Output folder for results
			std::string output_folder = "output_llg";
			// PRNG Seed
			int seed = 0;
			// number of iterations carried out when pressing "play" or calling "iterate"
			int n_iterations = (int)2E+6;
			// Number of iterations after which the system is logged to file
			int n_iterations_log = 100;
			// Temperature in K
			double temperature = 0.0;
			// Damping constant
			double damping = 0.5;
			// iteration time step
			double dt = 1.0E-02;
			// Whether to renormalize spins after every SD iteration
			bool renorm_sd = 1;
			// Whether to save a single "spins"
			bool save_single_configurations = true;
			// spin transfer torque vector
			double stt_magnitude = 1.5;
			// spin_current polarisation normal vector
			std::vector<double> stt_polarisation_normal = { 1.0, -1.0, 0.0 };
			// Force convergence parameter
			double force_convergence = 10e-9;

			//------------------------------- Parser --------------------------------
			Log(Log_Level::Info, Log_Sender::IO, "Parameters LLG: building");
			if (configFile != "")
			{
				try {
					IO::Filter_File_Handle myfile(configFile);

					myfile.Read_Single(output_folder, "llg_output_folder");
					myfile.Read_Single(seed, "llg_seed");
					myfile.Read_Single(n_iterations, "llg_n_iterations");
					myfile.Read_Single(n_iterations_log, "llg_n_iterations_log");
					myfile.Read_Single(temperature, "llg_temperature");
					myfile.Read_Single(damping, "llg_damping");
					myfile.Read_Single(dt, "llg_dt");
					// dt = time_step [ps] * 10^-12 * gyromagnetic raio / mu_B  { / (1+damping^2)} <- not implemented
					dt = dt*std::pow(10, -12) / Vectormath::MuB()*1.760859644*std::pow(10, 11);
					myfile.Read_Single(renorm_sd, "llg_renorm");
					myfile.Read_Single(save_single_configurations, "llg_save_single_configurations");
					myfile.Read_Single(stt_magnitude, "llg_stt_magnitude");
					myfile.Read_3Vector(stt_polarisation_normal, "llg_stt_polarisation_normal");
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

			// Return
			Log(Log_Level::Parameter, Log_Sender::IO, "Parameters LLG:");
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
			auto llg_params = std::unique_ptr<Data::Parameters_Method_LLG>(new Data::Parameters_Method_LLG(output_folder, force_convergence, n_iterations, n_iterations_log, seed, temperature, damping, dt, renorm_sd, save_single_configurations, stt_magnitude, stt_polarisation_normal));
			Log(Log_Level::Info, Log_Sender::IO, "Parameters LLG: built");
			return llg_params;
		}// end Parameters_Method_LLG_from_Config

		std::unique_ptr<Data::Parameters_Method_GNEB> Parameters_Method_GNEB_from_Config(const std::string configFile)
		{
			//-------------- Insert default values here -----------------------------
			// Output folder for results
			std::string output_folder = "output_gneb";
			// Spring constant
			double spring_constant = 1.0;
			// Force convergence parameter
			double force_convergence = 10e-9;
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
					
					myfile.Read_Single(output_folder, "gneb_output_folder");
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
			Log(Log_Level::Parameter, Log_Sender::IO, "        spring_constant    = " + std::to_string(spring_constant));
			Log(Log_Level::Parameter, Log_Sender::IO, "        force_convergence  = " + std::to_string(force_convergence));
			Log(Log_Level::Parameter, Log_Sender::IO, "        n_E_interpolations = " + std::to_string(n_E_interpolations));
			Log(Log_Level::Parameter, Log_Sender::IO, "        n_iterations       = " + std::to_string(n_iterations));
			Log(Log_Level::Parameter, Log_Sender::IO, "        n_iterations_log   = " + std::to_string(n_iterations_log));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_folder      = " + output_folder);
			auto gneb_params = std::unique_ptr<Data::Parameters_Method_GNEB>(new Data::Parameters_Method_GNEB(output_folder, force_convergence, n_iterations, n_iterations_log, spring_constant, n_E_interpolations));
			Log(Log_Level::Info, Log_Sender::IO, "Parameters GNEB: built");
			return gneb_params;
		}// end Parameters_Method_LLG_from_Config

		std::unique_ptr<Data::Parameters_Method_MMF> Parameters_Method_MMF_from_Config(const std::string configFile)
		{
			//-------------- Insert default values here -----------------------------
			// Output folder for results
			std::string output_folder = "output_mmf";
			// Force convergence parameter
			double force_convergence = 10e-9;
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
					
					myfile.Read_Single(output_folder, "mmf_output_folder");
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
			Log(Log_Level::Parameter, Log_Sender::IO, "        force_convergence  = " + std::to_string(force_convergence));
			Log(Log_Level::Parameter, Log_Sender::IO, "        n_iterations       = " + std::to_string(n_iterations));
			Log(Log_Level::Parameter, Log_Sender::IO, "        n_iterations_log   = " + std::to_string(n_iterations_log));
			Log(Log_Level::Parameter, Log_Sender::IO, "        output_folder      = " + output_folder);
			auto mmf_params = std::unique_ptr<Data::Parameters_Method_MMF>(new Data::Parameters_Method_MMF(output_folder, force_convergence, n_iterations, n_iterations_log));
			Log(Log_Level::Info, Log_Sender::IO, "Parameters MMF: built");
			return mmf_params;
		}

		std::unique_ptr<Engine::Hamiltonian> Hamiltonian_from_Config(const std::string configFile, Data::Geometry geometry)
		{
			//-------------- Insert default values here -----------------------------
			// The type of hamiltonian we will use
			std::string hamiltonian_type = "isotropic";

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
			if (hamiltonian_type == "isotropic")
			{
				hamiltonian = Hamiltonian_Isotropic_from_Config(configFile, geometry);
			}// endif isotropic
			else if (hamiltonian_type == "anisotropic")
			{
				// TODO: to std::move or not to std::move, that is the question...
				hamiltonian = std::move(Hamiltonian_Anisotropic_from_Config(configFile, geometry));
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

		std::unique_ptr<Engine::Hamiltonian_Isotropic> Hamiltonian_Isotropic_from_Config(const std::string configFile, Data::Geometry geometry)
		{
			//-------------- Insert default values here -----------------------------
			// Boundary conditions (a, b, c)
			std::vector<int> boundary_conditions_i = { 0, 0, 0 };
			std::vector<bool> boundary_conditions = { false, false, false };
			// Magnetic field magnitude
			double external_field_magnitude = 25.0;
			// Magnetic field vector
			std::vector<double> external_field_normal = { 0.0, 0.0, 1.0 };
			// mu_spin
			double mu_s = 2.0;
			// Anisotropy constant
			double anisotropy_magnitude = 0.0;
			// Anisotropy vector
			std::vector<double> anisotropy_normal = { 0.0, 0.0, 1.0 };

			// Number of shells in which we calculate neighbours
			int n_neigh_shells = 4;
			// Jij
			std::vector<double> jij = { 10.0, 0.0, 0.0, 0.0 };
			// DM constant
			double dij = 6.0;
			// Biquidratic exchange constant
			double bij = 0.0;
			// 4 Spin Interaction constant
			double kijkl = 0.0;
			// Dipole-Dipole interaction radius
			double dd_radius = 0.0;

			//------------------------------- Parser --------------------------------
			Log(Log_Level::Info, Log_Sender::IO, "Hamiltonian_Isotropic: building");
			// iteration variables
			int iatom = 0;
			if (configFile != "")
			{
				try {
					IO::Filter_File_Handle myfile(configFile);

					myfile.Read_3Vector(boundary_conditions_i, "boundary_conditions");
					boundary_conditions[0] = (boundary_conditions_i[0] != 0);
					boundary_conditions[1] = (boundary_conditions_i[1] != 0);
					boundary_conditions[2] = (boundary_conditions_i[2] != 0);

					myfile.Read_Single(external_field_magnitude, "external_field_magnitude");
					myfile.Read_3Vector(external_field_normal, "external_field_normal");
					myfile.Read_Single(mu_s, "mu_s");
					myfile.Read_Single(anisotropy_magnitude, "anisotropy_magnitude");
					myfile.Read_3Vector(anisotropy_normal, "anisotropy_normal");
					myfile.Read_Single(n_neigh_shells, "n_neigh_shells");

					jij = std::vector<double>(n_neigh_shells);
					if (myfile.Find("jij"))
					{
						for (iatom = 0; iatom < n_neigh_shells; ++iatom) {
							myfile.iss >> jij[iatom];
						}						
					}
					else Log(Log_Level::Error, Log_Sender::IO, "Hamiltonian_Isotropic: Keyword 'jij' not found. Using Default:  { 10.0, 0.5, 0.0, 0.0 }");
					
					myfile.Read_Single(dij, "dij");
					myfile.Read_Single(bij, "bij");
					myfile.Read_Single(kijkl, "kijkl");
					myfile.Read_Single(dd_radius, "dd_radius");
				}// end try
				catch (Exception ex) {
					if (ex == Exception::File_not_Found)
					{
						Log(Log_Level::Error, Log_Sender::IO, "Hamiltonian_isotropic: Unable to open Config File " + configFile + " Leaving values at default.");
					}
					else throw ex;
				}// end catch
			}
			else Log(Log_Level::Warning, Log_Sender::IO, "Hamiltonian_Isotropic: Using default configuration!");
			
			// Return
			Log(Log_Level::Parameter, Log_Sender::IO, "Hamiltonian_Isotropic:");
			Log(Log_Level::Parameter, Log_Sender::IO, "        boundary conditions = " + std::to_string(boundary_conditions[0]) + " " + std::to_string(boundary_conditions[1]) + " " + std::to_string(boundary_conditions[2]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        B                   = " + std::to_string(external_field_magnitude));
			Log(Log_Level::Parameter, Log_Sender::IO, "        B_normal            = " + std::to_string(external_field_normal[0]) + " " + std::to_string(external_field_normal[1]) + " " + std::to_string(external_field_normal[2]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        mu_s                = " + std::to_string(mu_s));
			Log(Log_Level::Parameter, Log_Sender::IO, "        K                   = " + std::to_string(anisotropy_magnitude));
			Log(Log_Level::Parameter, Log_Sender::IO, "        K_normal            = " + std::to_string(anisotropy_normal[0]) + " " + std::to_string(anisotropy_normal[1]) + " " + std::to_string(anisotropy_normal[2]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        n_neigh_shells      = " + std::to_string(n_neigh_shells));
			Log(Log_Level::Parameter, Log_Sender::IO, "        J_ij[0]             = " + std::to_string(jij[0]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        D_ij                = " + std::to_string(dij));
			Log(Log_Level::Parameter, Log_Sender::IO, "        B_ij                = " + std::to_string(bij));
			Log(Log_Level::Parameter, Log_Sender::IO, "        K_ijkl              = " + std::to_string(kijkl));
			Log(Log_Level::Parameter, Log_Sender::IO, "        dd_radius           = " + std::to_string(dd_radius));
			auto hamiltonian = std::unique_ptr<Engine::Hamiltonian_Isotropic>(new Engine::Hamiltonian_Isotropic(boundary_conditions, external_field_magnitude,
					external_field_normal, mu_s, anisotropy_magnitude, anisotropy_normal,
					n_neigh_shells, jij, dij, bij, kijkl, dd_radius, geometry));
			Log(Log_Level::Info, Log_Sender::IO, "Hamiltonian_Isotropic: built");
			return hamiltonian;
		}// end Hamiltonian_Isotropic_from_Config


		
		std::unique_ptr<Engine::Hamiltonian_Anisotropic> Hamiltonian_Anisotropic_from_Config(const std::string configFile, Data::Geometry geometry)
		{
			//-------------- Insert default values here -----------------------------
			// Boundary conditions (a, b, c)
			std::vector<int> boundary_conditions_i = { 0, 0, 0 };
			std::vector<bool> boundary_conditions = { false, false, false };
			// Spin moment
			std::vector<double> mu_s = std::vector<double>(geometry.nos, 2.0);	// [nos]
			// External Magnetic Field
			std::string external_field_file = "";
			double B = 0;
			std::vector<double> B_normal = { 0.0, 0.0, 1.0 };
			std::vector<double> external_field_magnitude = std::vector<double>(geometry.nos, 0.0);	// [nos]
			std::vector<std::vector<double>> external_field_normal(3, std::vector<double>(geometry.nos, 0.0));	// [3][nos]
			
			// Anisotropy
			std::string anisotropy_file = "";
			double K = 0;
			std::vector<double> K_normal = { 0.0, 0.0, 1.0 };
			bool anisotropy_from_file = false;
			std::vector<int> anisotropy_index(geometry.nos);				// [nos]
			std::vector<double> anisotropy_magnitude(geometry.nos, 0.0);	// [nos]
			std::vector<std::vector<double>> anisotropy_normal(geometry.nos, { 0.0, 0.0, 0.0 });	// [3][nos]

			// ------------ Two Spin Interactions ------------
			int n_pairs = 0;
			std::string interaction_pairs_file = "";
			bool interaction_pairs_from_file = false;
			std::vector<std::vector<std::vector<int>>> Exchange_indices(8); std::vector<std::vector<double>> Exchange_magnitude(8);
			std::vector<std::vector<std::vector<int>>> DMI_indices(8); std::vector<std::vector<double>> DMI_magnitude(8); std::vector<std::vector<std::vector<double>>> DMI_normal(8);
			std::vector<std::vector<std::vector<int>>> BQC_indices(8); std::vector<std::vector<double>> BQC_magnitude(8);
			std::vector<std::vector<std::vector<int>>> DD_indices(8); std::vector<std::vector<double>> DD_magnitude(8); std::vector<std::vector<std::vector<double>>> DD_normal(8);

			double dd_radius = 0.0;

			//------------------------------- Parser --------------------------------
			Log(Log_Level::Info, Log_Sender::IO, "Hamiltonian_Anisotropic: building");
			// iteration variables
			int iatom = 0;
			if (configFile != "")
			{
				try {
					IO::Filter_File_Handle myfile(configFile);

					// Boundary conditions
					myfile.Read_3Vector(boundary_conditions_i, "boundary_conditions");
					boundary_conditions[0] = (boundary_conditions_i[0] != 0);
					boundary_conditions[1] = (boundary_conditions_i[1] != 0);
					boundary_conditions[2] = (boundary_conditions_i[2] != 0);

					// Spin moment
					mu_s = std::vector<double>(geometry.nos, 2.0);
					if (myfile.Find("mu_s"))
					{
						for (iatom = 0; iatom < geometry.n_spins_basic_domain; ++iatom)
						{
							myfile.iss >> mu_s[iatom];
							for (int ispin = 0; ispin < geometry.nos / geometry.n_spins_basic_domain; ++ispin)
							{
								mu_s[ispin*geometry.n_spins_basic_domain + iatom] = mu_s[iatom];
							}
						}
					}
					else Log(Log_Level::Error, Log_Sender::IO, "Keyword 'mu_s' not found. Using Default: 2.0");

					// External Field
					if (myfile.Find("external_field_file")) myfile.iss >> external_field_file;
					if (external_field_file.length() > 0)
					{
						Log(Log_Level::Warning, Log_Sender::IO, "Hamiltonian_anisotropic: Read external field file has not been implemented yet. Using 0 field for now.");
						// The file name should be valid so we try to read it
						// Not yet implemented!
					}
					else 
					{
						// Read parameters from config if available
						myfile.Read_Single(B, "external_field_magnitude");
						myfile.Read_3Vector(B_normal, "external_field_normal");

						// Fill the arrays
						for (int i = 0; i < geometry.nos; ++i)
						{
							external_field_magnitude[i] = B;
							for (int dim = 0; dim < 3; ++dim)
							{
								external_field_normal[dim][i] = B_normal[dim];
							}
						}
					}

					// Anisotropy
					if (myfile.Find("anisotropy_file")) myfile.iss >> anisotropy_file;
					if (anisotropy_file.length() > 0)
					{
						// The file name should be valid so we try to read it
						Anisotropy_from_File(anisotropy_file, geometry, n_pairs,
							anisotropy_index, anisotropy_magnitude, anisotropy_normal);
					}
					else
					{
						// Read parameters from config
						myfile.Read_Single(K, "anisotropy_magnitude");
						myfile.Read_3Vector(K_normal, "anisotropy_normal");
						
						// Fill the arrays
						for (int i = 0; i < geometry.nos; ++i)
						{
							anisotropy_index[i] = i;
							anisotropy_magnitude[i] = K;
							for (int dim = 0; dim < 3; ++dim)
							{
								anisotropy_normal[i][dim] = K_normal[dim];
							}
						}
					}

					// Interaction Pairs
					if (myfile.Find("interaction_pairs_file")) myfile.iss >> interaction_pairs_file;
					if (interaction_pairs_file.length() > 0)
					{
						// The file name should be valid so we try to read it
						Pairs_from_File(interaction_pairs_file, geometry, n_pairs,
							Exchange_indices, Exchange_magnitude,
							DMI_indices, DMI_magnitude, DMI_normal,
							BQC_indices, BQC_magnitude);
					}
					//else
					//{
					//	Log(Log_Level::Warning, Log_Sender::IO, "Hamiltonian_anisotropic: Default Interaction pairs have not been implemented yet.");
					//	throw Exception::System_not_Initialized;
					//	// Not implemented!
					//}
					
					//		Dipole-Dipole Pairs
					// Dipole Dipole radius
					myfile.Read_Single(dd_radius, "dd_radius");
					// if (dd_radius >0 ) Log(Log_Level::Error, Log_Sender::IO, "Hamiltonian_anisotropic: Dipole-Dipole energy is not correctly implemented, but you chose a radius > 0! -- r=" + std::to_string(dd_radius));
					// Dipole Dipole neighbours of each spin neigh_dd[nos][max_n]
					// std::vector<std::vector<int>> dd_neigh;
					// // Dipole Dipole neighbour positions of each spin neigh_dd[dim][nos][max_n]
					// std::vector<std::vector<std::vector<double>>> dd_neigh_pos;
					// // Dipole Dipole normal vectors [dim][nos][max_n]
					// std::vector<std::vector<std::vector<double>>> dd_normal;
					// // Dipole Dipole distance [nos][max_n]
					// std::vector<std::vector<double>> dd_distance;
					// // Create the DD neighbours
					// Engine::Neighbours::Create_Dipole_Neighbours(geometry, std::vector<bool>{ true, true, true }, dd_radius, dd_neigh, dd_neigh_pos, dd_normal, dd_distance);
					// // Get the DD pairs from the neighbours
					// Engine::Neighbours::Create_DD_Pairs_from_Neighbours(geometry, dd_neigh, dd_neigh_pos, dd_distance, dd_normal, DD_indices, DD_magnitude, DD_normal);
					
					
					Engine::Neighbours::Create_Dipole_Pairs(geometry, dd_radius, DD_indices, DD_magnitude, DD_normal);

				}// end try
				catch (Exception ex) {
					if (ex == Exception::File_not_Found)
					{
						Log(Log_Level::Error, Log_Sender::IO, "Hamiltonian_anisotropic: Unable to open Config File " + configFile + " Leaving values at default.");
					}
					else throw ex;
				}// end catch
			}
			else Log(Log_Level::Warning, Log_Sender::IO, "Hamiltonian_Anisotropic: Using default configuration!");
			
			// Return
			Log(Log_Level::Parameter, Log_Sender::IO, "Hamiltonian_Anisotropic:");
			Log(Log_Level::Parameter, Log_Sender::IO, "        boundary conditions = " + std::to_string(boundary_conditions[0]) + " " + std::to_string(boundary_conditions[1]) + " " + std::to_string(boundary_conditions[2]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        B[0]                = " + std::to_string(external_field_magnitude[0]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        B_normal[0]         = " + std::to_string(external_field_normal[0][0]) + " " + std::to_string(external_field_normal[1][0]) + " " + std::to_string(external_field_normal[2][0]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        mu_s[0]             = " + std::to_string(mu_s[0]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        K[0]                = " + std::to_string(anisotropy_magnitude[0]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        K_normal[0]         = " + std::to_string(anisotropy_normal[0][0]) + " " + std::to_string(anisotropy_normal[0][1]) + " " + std::to_string(anisotropy_normal[0][2]));
			Log(Log_Level::Parameter, Log_Sender::IO, "        dd_radius           = " + std::to_string(dd_radius));
			auto hamiltonian = std::unique_ptr<Engine::Hamiltonian_Anisotropic>(new Engine::Hamiltonian_Anisotropic(
				mu_s,
				external_field_magnitude, external_field_normal,
				anisotropy_index, anisotropy_magnitude, anisotropy_normal,
				Exchange_indices, Exchange_magnitude,
				DMI_indices, DMI_magnitude, DMI_normal,
				BQC_indices, BQC_magnitude,
				DD_indices, DD_magnitude, DD_normal,
				boundary_conditions
			));
			Log(Log_Level::Info, Log_Sender::IO, "Hamiltonian_Anisotropic: built");
			return hamiltonian;
		}// end Hamiltonian_Anisotropic_From_Config
		
		
		std::unique_ptr<Engine::Hamiltonian_Gaussian> Hamiltonian_Gaussian_from_Config(const std::string configFile, Data::Geometry geometry)
		{
			//-------------- Insert default values here -----------------------------
			// Number of Gaussians
			int n_gaussians = 1;
			// Amplitudes
			std::vector<double> amplitude = { 1 };
			// Widths
			std::vector<double> width = { 1 };
			// Centers
			std::vector<std::vector<double>> center = { std::vector<double>{ 0, 0, 1 } };

			//------------------------------- Parser --------------------------------
			Log(Log_Level::Info, Log_Sender::IO, "Hamiltonian_Gaussian: building");
			
			if (configFile != "")
			{
				try {
					IO::Filter_File_Handle myfile(configFile);

					// N
					myfile.Read_Single(n_gaussians, "gaussian_n");

					// Amplitude
					amplitude = std::vector<double>(n_gaussians, 1.0);
					if (myfile.Find("gaussian_amplitudes"))
					{
						for (int i = 0; i < n_gaussians; ++i) myfile.iss >> amplitude[i];
					}
					else Log(Log_Level::Error, Log_Sender::IO, "Hamiltonian_Gaussian: Keyword 'gaussian_amplitudes' not found. Using Default: 1.0");

					// Width
					width = std::vector<double>(n_gaussians, 1.0);
					if (myfile.Find("gaussian_widths"))
					{
						for (int i = 0; i < n_gaussians; ++i) myfile.iss >> width[i];
					}
					else Log(Log_Level::Error, Log_Sender::IO, "Hamiltonian_Gaussian: Keyword 'gaussian_widths' not found. Using Default: 1.0");

					// Center
					center = std::vector<std::vector<double>>(n_gaussians, std::vector<double>{0, 0, 1});
					if (myfile.Find("gaussian_centers"))
					{
						for (int i = 0; i < n_gaussians; ++i)
						{
							myfile.GetLine();
							for (int j = 0; j < 3; ++j)
							{
								myfile.iss >> center[i][j];
							}
							Utility::Vectormath::Normalize(center[i]);
						}
					}
					else Log(Log_Level::Error, Log_Sender::IO, "Hamiltonian_Gaussian: Keyword 'gaussian_centers' not found. Using Default: {0, 0, 1}");
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