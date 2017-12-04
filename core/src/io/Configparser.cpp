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
#include <ctime>

#include <fmt/format.h>
#include <fmt/ostream.h>

using namespace Utility;

namespace IO
{
    void Log_from_Config(const std::string configFile, bool force_quiet)
    {
        // Verbosity and Reject Level are read as integers
        int i_level_file = 5, i_level_console = 5;
        std::string output_folder = ".";
        std::string file_tag = "";
        bool messages_to_file    = true, 
             messages_to_console = true, 
             save_input_initial  = false, 
             save_input_final    = false,
             save_positions_initial  = false,
             save_positions_final    = false,
             save_neighbours_initial = false,
             save_neighbours_final   = false;

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
            // Don't save positions
            Log.save_positions_initial = false;
            Log.save_positions_final = false;
            // Don't save neighbours
            Log.save_neighbours_initial = false;
            Log.save_neighbours_final = false;
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
                myfile.Read_Single(file_tag, "output_file_tag");

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

                // Save Input (parameters from config file and defaults) on State Setup
                myfile.Read_Single(save_input_initial, "save_input_initial");
                // Save Input (parameters from config file and defaults) on State Delete
                myfile.Read_Single(save_input_final, "save_input_final");
                
                // Save Input (parameters from config file and defaults) on State Setup
                myfile.Read_Single(save_positions_initial, "save_positions_initial");
                // Save Input (parameters from config file and defaults) on State Delete
                myfile.Read_Single(save_positions_final, "save_positions_final");
                
                // // Save Input (parameters from config file and defaults) on State Setup
                // myfile.Read_Single(save_neighbours_initial, "save_neighbours_initial");
                // // Save Input (parameters from config file and defaults) on State Delete
                // myfile.Read_Single(save_neighbours_final, "save_neighbours_final");

            }// end try
            catch( ... )
            {
                spirit_rethrow(	fmt::format("Failed to read Log Levels from file \"{}\". Leaving values at default.", configFile) );
            }
        }

        // Log the parameters
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("File tag on output     = {}", file_tag));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("Log output folder      = {0}", output_folder));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("Log to file            = {0}", messages_to_file));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("Log file accept level  = {0}", i_level_file));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("Log to console         = {0}", messages_to_console));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("Log print accept level = {0}", i_level_console));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("Log input save initial = {0}", save_input_initial));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("Log input save final   = {0}", save_input_final));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("Log positions save initial  = {0}", save_positions_initial));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("Log positions save final    = {0}", save_positions_final));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("Log neighbours save initial = {0}", save_neighbours_initial));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("Log neighbours save final   = {0}", save_neighbours_final));
        
        // Update the Log
        if (!force_quiet)
        {
            Log.level_file    = Log_Level(i_level_file);
            Log.level_console = Log_Level(i_level_console);

            Log.messages_to_file    = messages_to_file;
            Log.messages_to_console = messages_to_console;
            Log.save_input_initial  = save_input_initial;
            Log.save_input_final    = save_input_final;
            Log.save_positions_initial  = save_positions_initial;
            Log.save_positions_final    = save_positions_final;
            Log.save_neighbours_initial = save_neighbours_initial;
            Log.save_neighbours_final   = save_neighbours_final;
        }

        Log.file_tag      = file_tag;
        Log.output_folder = output_folder;
        
        if ( file_tag == "<time>" )
            Log.fileName = "Log_" + Utility::Timing::CurrentDateTime() + ".txt";
        else if ( file_tag != "" )
            Log.fileName = "Log_" + file_tag + ".txt";
        else
            Log.fileName = "Log.txt";

    }// End Log_Levels_from_Config


    std::unique_ptr<Data::Spin_System> Spin_System_from_Config(const std::string configFile)
    {
        try
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
        }
        catch (...)
        {
            spirit_handle_exception_core(fmt::format("Unable to initialize spin system from config file  \"{}\"", configFile));
        }

        return nullptr;
    }// End Spin_System_from_Config		


    void Bravais_Vectors_from_Config(const std::string configFile, std::vector<Vector3> & bravais_vectors, Data::BravaisLatticeType & bravais_lattice_type)
    {
        try
        {
            std::string bravais_lattice = "sc";
            // Manually specified bravais vectors/matrix?
            bool irregular = true;

            IO::Filter_File_Handle myfile(configFile);
            // Bravais lattice type or manually specified vectors/matrix
            if (myfile.Find("bravais_lattice"))
            {
                myfile.iss >> bravais_lattice;

                if (bravais_lattice == "sc")
                {
                    Log(Log_Level::Parameter, Log_Sender::IO, "Bravais lattice type: simple cubic");
                    bravais_lattice_type = Data::BravaisLatticeType::SC;
                    bravais_vectors = Data::Geometry::BravaisVectorsSC();
                }
                else if (bravais_lattice == "fcc")
                {
                    Log(Log_Level::Parameter, Log_Sender::IO, "Bravais lattice type: face-centered cubic");
                    bravais_lattice_type = Data::BravaisLatticeType::FCC;
                    bravais_vectors = Data::Geometry::BravaisVectorsFCC();
                }
                else if (bravais_lattice == "bcc")
                {
                    Log(Log_Level::Parameter, Log_Sender::IO, "Bravais lattice type: body-centered cubic");
                    bravais_lattice_type = Data::BravaisLatticeType::BCC;
                    bravais_vectors = Data::Geometry::BravaisVectorsBCC();
                }
                else if (bravais_lattice == "hex2D60")
                {
                    Log(Log_Level::Parameter, Log_Sender::IO, "Bravais lattice type: hexagonal 2D 60deg angle");
                    bravais_lattice_type = Data::BravaisLatticeType::Hex2D;
                    bravais_vectors = Data::Geometry::BravaisVectorsHex2D60();
                }
                else if (bravais_lattice == "hex2D120")
                {
                    Log(Log_Level::Parameter, Log_Sender::IO, "Bravais lattice type: hexagonal 2D 120deg angle");
                    bravais_lattice_type = Data::BravaisLatticeType::Hex2D;
                    bravais_vectors = Data::Geometry::BravaisVectorsHex2D120();
                }
                else
                {
                }
            }
            else if (myfile.Find("bravais_vectors"))
            {
                Log(Log_Level::Parameter, Log_Sender::IO, "Bravais lattice type: irregular");
                bravais_lattice_type = Data::BravaisLatticeType::Irregular;
                myfile.GetLine();
                myfile.iss >> bravais_vectors[0][0] >> bravais_vectors[0][1] >> bravais_vectors[0][2];
                myfile.GetLine();
                myfile.iss >> bravais_vectors[1][0] >> bravais_vectors[1][1] >> bravais_vectors[1][2];
                myfile.GetLine();
                myfile.iss >> bravais_vectors[2][0] >> bravais_vectors[2][1] >> bravais_vectors[2][2];

            }
            else if (myfile.Find("bravais_matrix"))
            {
                Log(Log_Level::Parameter, Log_Sender::IO, "Bravais lattice type: irregular");
                bravais_lattice_type = Data::BravaisLatticeType::Irregular;
                myfile.GetLine();
                myfile.iss >> bravais_vectors[0][0] >> bravais_vectors[1][0] >> bravais_vectors[2][0];
                myfile.GetLine();
                myfile.iss >> bravais_vectors[0][1] >> bravais_vectors[1][1] >> bravais_vectors[2][1];
                myfile.GetLine();
                myfile.iss >> bravais_vectors[0][2] >> bravais_vectors[1][2] >> bravais_vectors[2][2];
            }
        }
        catch( ... )
        {
            spirit_rethrow(	fmt::format("Unable to parse bravais vectors from config file \"{}\"", configFile) );
        }
    }// End Basis_from_Config

    std::shared_ptr<Data::Geometry> Geometry_from_Config(const std::string configFile)
    {
        try
        {
            //-------------- Insert default values here -----------------------------
            // Basis from separate file?
            std::string basis_file = "";
            // Bravais lattice type
            std::string bravais_lattice = "sc";
            // Bravais vectors {a, b, c}
            Data::BravaisLatticeType bravais_lattice_type = Data::BravaisLatticeType::SC;
            std::vector<Vector3> bravais_vectors = { Vector3{1,0,0}, Vector3{0,1,0}, Vector3{0,0,1} };
            // Atoms in the basis
            std::vector<Vector3> cell_atoms = { Vector3{0,0,0} };
            int n_cell_atoms = cell_atoms.size();
            // Lattice Constant [Angstrom]
            scalar lattice_constant = 1;
            // Number of translations nT for each basis direction
            intfield n_cells = { 100, 100, 1 };
            // Atom types
            intfield atom_types;
            intfield defect_indices(0);
            intfield defects(0);

            // Utility 1D array to build vectors and use Vectormath
            Vector3 build_array = { 0, 0, 0 };

            Log(Log_Level::Info, Log_Sender::IO, "Geometry: building");
            //------------------------------- Parser --------------------------------
            // iteration variables
            int iatom = 0, dim = 0;
            if (configFile != "")
            {
                try
                {
                    Log(Log_Level::Info, Log_Sender::IO, "Reading Geometry Parameters");
                    IO::Filter_File_Handle myfile(configFile);

                    // Lattice constant
                    myfile.Read_Single(lattice_constant, "lattice_constant");

                    // Get the bravais lattice type and vectors
                    Bravais_Vectors_from_Config(configFile, bravais_vectors, bravais_lattice_type);

                    // Read basis cell
                    if (myfile.Find("basis"))
                    {
                        // Read number of atoms in the basis cell
                        myfile.GetLine();
                        myfile.iss >> n_cell_atoms;
                        cell_atoms = std::vector<Vector3>(n_cell_atoms);

                        // Read atom positions
                        for (int iatom = 0; iatom < n_cell_atoms; ++iatom)
                        {
                            myfile.GetLine();
                            myfile.iss >> cell_atoms[iatom][0] >> cell_atoms[iatom][1] >> cell_atoms[iatom][2];
                        }// endfor iatom
                    }

                    // Read number of basis cells
                    myfile.Read_3Vector(n_cells, "n_basis_cells");

                    // Defects
                    #ifdef SPIRIT_ENABLE_DEFECTS
                    int n_defects = 0;

                    std::string defectsFile = "";
                    if (myfile.Find("n_defects"))
                        defectsFile = configFile;
                    else if (myfile.Find("defects_from_file"))
                        myfile.iss >> defectsFile;

                    if (defectsFile.length() > 0)
                    {
                        // The file name should be valid so we try to read it
                        Defects_from_File(defectsFile, n_defects,
                            defect_indices, defects);
                    }
                    #endif
                }// end try
                catch( ... )
                {
                    spirit_handle_exception_core(fmt::format("Failed to read Geometry parameters from file \"{}\". Leaving values at default.", configFile));
                }
            }// end if file=""
            else Log(Log_Level::Warning, Log_Sender::IO, "Geometry: Using default configuration!");

            // Log the parameters
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("Lattice constant = {} angstrom", lattice_constant));
            Log(Log_Level::Debug, Log_Sender::IO, "Bravais vectors in units of lattice constant");
            Log(Log_Level::Debug, Log_Sender::IO, fmt::format("        a = {}", bravais_vectors[0].transpose() / lattice_constant));
            Log(Log_Level::Debug, Log_Sender::IO, fmt::format("        b = {}", bravais_vectors[1].transpose() / lattice_constant));
            Log(Log_Level::Debug, Log_Sender::IO, fmt::format("        c = {}", bravais_vectors[2].transpose() / lattice_constant));
            Log(Log_Level::Parameter, Log_Sender::IO, "Bravais vectors");
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        a = {}", bravais_vectors[0].transpose()));
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        b = {}", bravais_vectors[1].transpose()));
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        c = {}", bravais_vectors[2].transpose()));
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("Basis: {}  atom(s) at the following positions:", n_cell_atoms));
            for (int iatom = 0; iatom < n_cell_atoms; ++iatom)
                Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        atom {0} at {1}", iatom, cell_atoms[iatom].transpose()));
            Log(Log_Level::Info, Log_Sender::IO, "Basis: built");


            // Atom types (default: type 0, vacancy: < 0)
            atom_types = intfield(cell_atoms.size(), 0);

            // Defects
            #ifdef SPIRIT_ENABLE_DEFECTS
            int n_defects = defect_indices.size();
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("Geometry: {} defects. Printing the first 10 indices:", n_defects));
            for (int i = 0; i < n_defects; ++i)
            {
                if (i < 10) Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("  defect[{0}]: ispin={1}, type=", i, defect_indices[i], defects[i]));
            }
            #endif

            // Log parameters
            Log(Log_Level::Parameter, Log_Sender::IO, "Lattice: n_basis_cells");
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("       na = {}", n_cells[0]));
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("       nb = {}", n_cells[1]));
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("       nc = {}", n_cells[2]));
            
            // Return geometry
            auto geometry = std::shared_ptr<Data::Geometry>(new Data::Geometry(bravais_vectors, n_cells, cell_atoms, atom_types, lattice_constant));

            #ifdef SPIRIT_ENABLE_DEFECTS
            for (int i = 0; i < n_defects; ++i)
                geometry->atom_types[defect_indices[i]] = defects[i]; // TODO: maybe a function instead of a for-loop?
            #endif

            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("Geometry: {} spins", geometry->nos));
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("Geometry is {}-dimensional", geometry->dimensionality));
            Log(Log_Level::Info, Log_Sender::IO, "Geometry: built");
            return geometry;
        }
        catch( ... )
        {
            spirit_rethrow(	fmt::format("Unable to parse geometry from config file  \"{}\"", configFile) );
        }

        return nullptr;
    }// end Geometry from Config

    std::shared_ptr<Data::Pinning> Pinning_from_Config(const std::string configFile, const std::shared_ptr<Data::Geometry> geometry)
    {
        //-------------- Insert default values here -----------------------------
        int na = 0, na_left = 0, na_right = 0;
        int nb = 0, nb_left = 0, nb_right = 0;
        int nc = 0, nc_left = 0, nc_right = 0;
        vectorfield pinned_cell(geometry->n_cell_atoms, Vector3{ 0,0,1 });
        // Additional pinned sites
        intfield pinned_indices(0);
        vectorfield pinned_spins(0);
        int n_pinned = 0;

        // Utility 1D array to build vectors and use Vectormath
        Vector3 build_array = { 0, 0, 0 };

        #ifdef SPIRIT_ENABLE_PINNING
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
                    myfile.Read_Single(nb, "pin_nb ",  false);
                    if (nb > 0 && (nb_left == 0 || nb_right == 0))
                    {
                        nb_left = nb;
                        nb_right = nb;
                    }

                    // N_c
                    myfile.Read_Single(nc_left, "pin_nc_left", false);
                    myfile.Read_Single(nc_right, "pin_nc_right", false);
                    myfile.Read_Single(nc, "pin_nc ",  false);
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
                            for (int i = 0; i < geometry->n_cell_atoms; ++i)
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

                    // Additional pinned sites
                    std::string pinnedFile = "";
                    if (myfile.Find("n_pinned"))
                        pinnedFile = configFile;
                    else if (myfile.Find("pinned_from_file"))
                        myfile.iss >> pinnedFile;

                    if (pinnedFile.length() > 0)
                    {
                        // The file name should be valid so we try to read it
                        Pinned_from_File(pinnedFile, n_pinned,
                            pinned_indices, pinned_spins);
                    }
                }// end try
                catch (...)
                {
                    spirit_handle_exception_core(fmt::format("Failed to read Pinning from file \"{}\". Leaving values at default.", configFile));
                }
                
            }// end if file=""
            else Log(Log_Level::Parameter, Log_Sender::IO, "No pinning");

            // Create Pinning
            auto pinning = std::shared_ptr<Data::Pinning>(new Data::Pinning( geometry,
                na_left, na_right,
                nb_left, nb_right,
                nc_left, nc_right,
                pinned_cell) );

            // Apply additional pinned sites
            for (int i = 0; i < n_pinned; ++i)
            {
                int idx = pinned_indices[i];
                pinning->mask_unpinned[idx] = 0;
                pinning->mask_pinned_cells[idx] = pinned_spins[i];
            }

            // Return Pinning
            Log(Log_Level::Parameter, Log_Sender::IO, "Pinning:");
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        n_a: left={0}, right={1}", na_left, na_right));
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        n_b: left={0}, right={1}", nb_left, nb_right));
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        n_c: left={0}, right={1}", nc_left, nc_right));
            for (int i = 0; i < geometry->n_cell_atoms; ++i)
                Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        cell atom[0]      = ({0})", pinned_cell[0].transpose()));
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {} additional pinned sites: ", n_pinned));
            for (int i = 0; i < n_pinned; ++i)
                Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        pinned site[0]           = ({0})", pinned_spins[0].transpose()));
            Log(Log_Level::Info, Log_Sender::IO, "Pinning: read");
            return pinning;
        #else // SPIRIT_ENABLE_PINNING
            Log(Log_Level::Info, Log_Sender::IO, "Pinning is disabled");
            if (configFile != "")
            {
                try
                {
                    IO::Filter_File_Handle myfile(configFile);
                    if (myfile.Find("pinning_cell"))
                        Log(Log_Level::Warning, Log_Sender::IO, "You specified a pinning cell even though pinning is disabled!");
                }
                catch( ... )
                {
                    spirit_handle_exception_core(fmt::format("Failed to read pinning parameters from file \"{}\". Leaving values at default.", configFile));
                }
            }

            auto pinning = std::shared_ptr<Data::Pinning>(new Data::Pinning(geometry,
                intfield(geometry->nos, 1),
                vectorfield(0)));
            return pinning;
        #endif // SPIRIT_ENABLE_PINNING
    }

    std::unique_ptr<Data::Parameters_Method_LLG> Parameters_Method_LLG_from_Config(const std::string configFile, const std::shared_ptr<Data::Pinning> pinning)
    {
        //-------------- Insert default values here -----------------------------
        // Output folder for results
        std::string output_folder = "output_llg";
        // Save output when logging
        std::string output_file_tag = ""; 
        bool output_any = true, 
                output_initial = true, 
                output_final = true;
        bool output_energy_divide_by_nspins=true, 
                output_energy_spin_resolved=true, 
                output_energy_step=true, 
                output_energy_archive=true;
        bool output_configuration_step = false, 
                output_configuration_archive = false;
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
        // non-adiabatic parameter
        scalar beta = 0.0;
        // iteration time step
        scalar dt = 1.0E-02;
        // Whether to renormalize spins after every SD iteration
        bool renorm_sd = 1;
        // use the gradient method for stt
        bool stt_use_gradient = false;
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
            try
            {
                IO::Filter_File_Handle myfile(configFile);

                myfile.Read_Single(output_file_tag,"output_file_tag");
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
                myfile.Read_Single(dt, "llg_dt");
                myfile.Read_Single(temperature, "llg_temperature");
                myfile.Read_Single(damping, "llg_damping");
                myfile.Read_Single(beta, "llg_beta");
                myfile.Read_Single(renorm_sd, "llg_renorm");
                myfile.Read_Single(stt_use_gradient, "llg_stt_use_gradient");
                myfile.Read_Single(stt_magnitude, "llg_stt_magnitude");
                myfile.Read_Vector3(stt_polarisation_normal, "llg_stt_polarisation_normal");
                myfile.Read_Single(force_convergence, "llg_force_convergence");
            }// end try
            catch (...)
            {
                spirit_handle_exception_core(fmt::format("Unable to parse LLG parameters from config file \"{}\"", configFile));
            }
        }
        else Log(Log_Level::Warning, Log_Sender::IO, "Parameters LLG: Using default configuration!");

        // Normalize vectors
        stt_polarisation_normal.normalize();

        // Return
        Log(Log_Level::Parameter, Log_Sender::IO, "Parameters LLG:");
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1}", "seed", seed));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1}", "time step [ps]", dt));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1}", "temperature", temperature));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1}", "damping", damping));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1}", "beta", beta));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1}", "stt use gradient", stt_use_gradient));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1}", "stt magnitude", stt_magnitude));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1}", "stt normal", stt_polarisation_normal.transpose()));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1:e}", "force convergence", force_convergence));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1}", "maximum walltime", str_max_walltime));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1}", "n_iterations", n_iterations));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1}", "n_iterations_log", n_iterations_log));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1}", "output_folder", output_folder));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1}", "output_any", output_any));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1}", "output_initial", output_initial));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1}", "output_final", output_final));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<30} = {1}", "output_energy_step", output_energy_step));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<30} = {1}", "output_energy_archive", output_energy_archive));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<30} = {1}", "output_energy_spin_resolved", output_energy_spin_resolved));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<30} = {1}", "output_energy_divide_by_nspins", output_energy_divide_by_nspins));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<30} = {1}", "output_configuration_step", output_configuration_step));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<30} = {1}", "output_configuration_archive", output_configuration_archive));

        max_walltime = (long int)Utility::Timing::DurationFromString(str_max_walltime).count();
        auto llg_params = std::unique_ptr<Data::Parameters_Method_LLG>(new Data::Parameters_Method_LLG( output_folder, output_file_tag, { output_any, output_initial, output_final, output_energy_step, output_energy_archive, output_energy_spin_resolved,
            output_energy_divide_by_nspins, output_configuration_step, output_configuration_archive}, force_convergence, n_iterations, n_iterations_log, max_walltime, pinning, seed, temperature, damping, beta, dt, renorm_sd,
            stt_use_gradient, stt_magnitude, stt_polarisation_normal));
        Log(Log_Level::Info, Log_Sender::IO, "Parameters LLG: built");
        return llg_params;
    }// end Parameters_Method_LLG_from_Config

    std::unique_ptr<Data::Parameters_Method_MC> Parameters_Method_MC_from_Config(const std::string configFile, const std::shared_ptr<Data::Pinning> pinning)
    {
        //-------------- Insert default values here -----------------------------
        // Output folder for results
        std::string output_folder = "output_mc";
        // Save output when logging
        std::string output_file_tag; 
        bool output_any = true, 
                output_initial = true, 
                output_final = true;
        bool output_energy_divide_by_nspins = true, 
                output_energy_spin_resolved = true, 
                output_energy_step = true, 
                output_energy_archive = true;
        bool output_configuration_step = false,
                output_configuration_archive = false;
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
            try
            {
                IO::Filter_File_Handle myfile(configFile);

                myfile.Read_Single(output_file_tag, "output_file_tag");
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
            catch (...)
            {
                spirit_handle_exception_core(fmt::format("Unable to parse MC parameters from config file \"{}\"", configFile));
            }
        }
        else Log(Log_Level::Warning, Log_Sender::IO, "Parameters MC: Using default configuration!");

        // Return
        Log(Log_Level::Parameter, Log_Sender::IO, "Parameters MC:");
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1}", "seed", seed));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1}", "temperature", temperature));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1}", "acceptance_ratio", acceptance_ratio));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1}", "maximum walltime", str_max_walltime));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1}", "n_iterations", n_iterations));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1}", "n_iterations_log", n_iterations_log));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1}", "output_folder", output_folder));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1}", "output_any", output_any));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1}", "output_initial", output_initial));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1}", "output_final", output_final));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<30} = {1}", "output_energy_step", output_energy_step));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<30} = {1}", "output_energy_archive", output_energy_archive));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<30} = {1}", "output_energy_spin_resolved", output_energy_spin_resolved));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<30} = {1}", "output_energy_divide_by_nspins", output_energy_divide_by_nspins));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<30} = {1}", "output_configuration_step", output_configuration_step));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<30} = {1}", "output_configuration_archive", output_configuration_archive));
        max_walltime = (long int)Utility::Timing::DurationFromString(str_max_walltime).count();
        auto mc_params = std::unique_ptr<Data::Parameters_Method_MC>(new Data::Parameters_Method_MC(output_folder, output_file_tag, { output_any, output_initial, output_final, output_energy_step, output_energy_archive, output_energy_spin_resolved,
            output_energy_divide_by_nspins, output_configuration_step, output_configuration_archive }, n_iterations, n_iterations_log, max_walltime, pinning, seed, temperature, acceptance_ratio));
        Log(Log_Level::Info, Log_Sender::IO, "Parameters MC: built");
        return mc_params;
    }

    std::unique_ptr<Data::Parameters_Method_GNEB> Parameters_Method_GNEB_from_Config(const std::string configFile, const std::shared_ptr<Data::Pinning> pinning)
    {
        //-------------- Insert default values here -----------------------------
        // Output folder for results
        std::string output_folder = "output_gneb";
        // Save output when logging
        std::string output_file_tag = ""; 
        bool output_any = true, 
                output_initial = false, 
                output_final = true, 
                output_energies_step = false, 
                output_energies_interpolated = true, 
                output_energies_divide_by_nspins = true, 
                output_chain_step = false;
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
            try
            {
                IO::Filter_File_Handle myfile(configFile);

                myfile.Read_Single(output_file_tag, "output_file_tag");
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
            catch (...)
            {
                spirit_handle_exception_core(fmt::format("Unable to parse GNEB parameters from config file \"{}\"", configFile));
            }
        }
        else Log(Log_Level::Warning, Log_Sender::IO, "Parameters GNEB: Using default configuration!");

        // Return
        Log(Log_Level::Parameter, Log_Sender::IO, "Parameters GNEB:");
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<18} = {1}", "spring_constant", spring_constant));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<18} = {1}", "n_E_interpolations", n_E_interpolations));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<18} = {1:e}", "force convergence", force_convergence));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<18} = {1}", "maximum walltime", str_max_walltime));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<18} = {1}", "n_iterations", n_iterations));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<18} = {1}", "n_iterations_log", n_iterations_log));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<18} = {1}", "output_folder", output_folder));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<18} = {1}", "output_any", output_any));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<18} = {1}", "output_initial", output_initial));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<18} = {1}", "output_final", output_final));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<18} = {1}", "output_energies_step", output_energies_step));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<18} = {1}", "output_chain_step", output_chain_step));

        max_walltime = (long int)Utility::Timing::DurationFromString(str_max_walltime).count();
        auto gneb_params = std::unique_ptr<Data::Parameters_Method_GNEB>(new Data::Parameters_Method_GNEB(output_folder, output_file_tag, { output_any, output_initial, output_final, output_energies_step, output_energies_interpolated, output_energies_divide_by_nspins, output_chain_step },
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
        std::string output_file_tag = "";
        bool output_any = true, 
                output_initial = false, 
                output_final = true, 
                output_energy_step = false, 
                output_energy_archive = true, 
                output_energy_divide_by_nspins = true, 
                output_configuration_step = false, 
                output_configuration_archive = true;
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
            try
            {
                IO::Filter_File_Handle myfile(configFile);

                myfile.Read_Single(output_file_tag, "output_file_tag");
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
            catch (...)
            {
                spirit_handle_exception_core(fmt::format("Unable to parse MMF parameters from config file \"{}\"", configFile));
            }
        }
        else Log(Log_Level::Warning, Log_Sender::IO, "Parameters MMF: Using default configuration!");

        // Return
        Log(Log_Level::Parameter, Log_Sender::IO, "Parameters MMF:");
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1:e}", "force convergence", force_convergence));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1}", "maximum walltime", str_max_walltime));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1}", "n_iterations", n_iterations));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1}", "n_iterations_log", n_iterations_log));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1}", "output_folder", output_folder));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1}", "output_any", output_any));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1}", "output_initial", output_initial));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<17} = {1}", "output_final", output_final));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<30} = {1}", "output_energy_step", output_energy_step));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<30} = {1}", "output_energy_archive", output_energy_archive));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<30} = {1}", "output_energy_divide_by_nspins", output_energy_divide_by_nspins));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<30} = {1}", "output_configuration_step", output_configuration_step));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<30} = {1}", "output_configuration_archive", output_configuration_archive));
        max_walltime = (long int)Utility::Timing::DurationFromString(str_max_walltime).count();
        auto mmf_params = std::unique_ptr<Data::Parameters_Method_MMF>(new Data::Parameters_Method_MMF(output_folder, output_file_tag, {output_any, output_initial, output_final, output_energy_step, output_energy_archive, output_energy_divide_by_nspins, output_configuration_step,output_configuration_archive },
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
            try
            {
                Log(Log_Level::Info, Log_Sender::IO, "Hamiltonian: deciding type");
                IO::Filter_File_Handle myfile(configFile);

                // What hamiltonian do we use?
                myfile.Read_Single(hamiltonian_type, "hamiltonian");
            }// end try
            catch (...)
            {
                spirit_handle_exception_core(fmt::format("Unable to read Hamiltonian type from config file  \"{}\". Using default.", configFile));
                hamiltonian_type = "heisenberg_neighbours";
            }
        }
        else Log(Log_Level::Warning, Log_Sender::IO, "Hamiltonian: Using default Hamiltonian: " + hamiltonian_type);

        // Hamiltonian
        std::unique_ptr<Engine::Hamiltonian> hamiltonian;
        try
        {
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
                spirit_throw(Exception_Classifier::System_not_Initialized, Log_Level::Severe, fmt::format("Hamiltonian: Invalid type \"{}\"", hamiltonian_type));
            }// endif neither
        }
        catch (...)
        {
            spirit_handle_exception_core(fmt::format("Unable to initialize Hamiltonian from config file  \"{}\"", configFile));
        }
        
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
        scalarfield mu_s = scalarfield(geometry->n_cell_atoms, 2);

        // External Magnetic Field
        scalar B = 0;
        Vector3 B_normal = { 0.0, 0.0, 1.0 };

        // Anisotropy
        std::string anisotropy_file = "";
        scalar K = 0;
        Vector3 K_normal = { 0.0, 0.0, 1.0 };
        bool anisotropy_from_file = false;
        intfield    anisotropy_index(geometry->n_cell_atoms);
        scalarfield anisotropy_magnitude(geometry->n_cell_atoms, 0.0);
        vectorfield anisotropy_normal(geometry->n_cell_atoms, K_normal);

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
            catch( ... )
            {
                spirit_handle_exception_core(fmt::format("Failed to read boundary_conditions from config file \"{}\"", configFile));
            }

            try
            {
                IO::Filter_File_Handle myfile(configFile);

                // External Field
                // Read parameters from config if available
                myfile.Read_Single(B, "external_field_magnitude");
                myfile.Read_Vector3(B_normal, "external_field_normal");
                B_normal.normalize();
                if (B_normal.norm() < 1e-8)
                {
                    B_normal = { 0,0,1 };
                    Log(Log_Level::Warning, Log_Sender::IO, "Input for 'external_field_normal' had norm zero and has been set to (0,0,1)");
                }
            }// end try
            catch( ... )
            {
                spirit_handle_exception_core(fmt::format("Failed to read external field from config file \"{}\"", configFile));
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
                        for (int i = 0; i < anisotropy_index.size(); ++i)
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
            catch( ... )
            {
                spirit_handle_exception_core(fmt::format("Failed to read anisotropy from config file \"{}\"", configFile));
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
            catch( ... )
            {
                spirit_handle_exception_core(fmt::format("Failed to read exchange parameters from config file \"{}\"", configFile));
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
            catch( ... )
            {
                spirit_handle_exception_core(fmt::format("Failed to read DMI parameters from config file \"{}\"", configFile));
            }

            try
            {
                IO::Filter_File_Handle myfile(configFile);

                myfile.Read_Single(dd_radius, "dd_radius");
            }// end try
            catch( ... )
            {
                spirit_handle_exception_core(fmt::format("Failed to read dd_radius from config file \"{}\"", configFile));
            }
        }
        else Log(Log_Level::Warning, Log_Sender::IO, "Hamiltonian_Heisenberg_Neighbours: Using default configuration!");

        // Return
        Log(Log_Level::Parameter, Log_Sender::IO, "Hamiltonian_Heisenberg_Neighbours:");
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        boundary conditions = {0} {1} {2}", boundary_conditions[0], boundary_conditions[1], boundary_conditions[2]));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<19} = {1}", "B[0]", B));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<19} = {1}", "B_normal[0]", B_normal.transpose()));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<19} = {1}", "mu_s[0]", mu_s[0]));
        if (anisotropy_from_file)
            Log(Log_Level::Parameter, Log_Sender::IO, "        K                     from file");
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<19} = {1}", "K[0]", K));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<19} = {1}", "K_normal[0]", K_normal.transpose()));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<19} = {1}", "n_shells_exchange", n_neigh_shells_exchange));
        if (n_neigh_shells_exchange > 0)
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<19} = {1}", "J_ij[0]", jij[0]));
        if (n_neigh_shells_dmi > 0)
            Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<19} = {1}", "n_shells_dmi", n_neigh_shells_dmi));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<19} = {1}", "D_ij[0]", dij[0]));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<19} = {1}", "DM chirality", dm_chirality));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<19} = {1}", "dd_radius", dd_radius));
        auto hamiltonian = std::unique_ptr<Engine::Hamiltonian_Heisenberg_Neighbours>(new Engine::Hamiltonian_Heisenberg_Neighbours(
                mu_s, B, B_normal,
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
        scalarfield mu_s = scalarfield(geometry->n_cell_atoms, 2);

        // External Magnetic Field
        scalar B = 0;
        Vector3 B_normal = { 0.0, 0.0, 1.0 };
        
        // Anisotropy
        std::string anisotropy_file = "";
        scalar K = 0;
        Vector3 K_normal = { 0.0, 0.0, 1.0 };
        bool anisotropy_from_file = false;
        intfield    anisotropy_index(geometry->n_cell_atoms);
        scalarfield anisotropy_magnitude(geometry->n_cell_atoms, 0.0);
        vectorfield anisotropy_normal(geometry->n_cell_atoms, K_normal);

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
            catch( ... )
            {
                spirit_handle_exception_core(fmt::format("Unable to read boundary conditions from config file  \"{}\"", configFile));
            }

            try
            {
                IO::Filter_File_Handle myfile(configFile);

                // Spin moment
                if (myfile.Find("mu_s"))
                {
                    for (iatom = 0; iatom < geometry->n_cell_atoms; ++iatom)
                    {
                        myfile.iss >> mu_s[iatom];  // TODO: need to catch when user has not input enough mu_s values
                    }
                }
                else Log(Log_Level::Error, Log_Sender::IO, "Keyword 'mu_s' not found. Using Default: 2.0");
            }// end try
            catch( ... )
            {
                spirit_handle_exception_core(fmt::format("Unable to read mu_s from config file  \"{}\"", configFile));
            }

            try
            {
                IO::Filter_File_Handle myfile(configFile);

                // Read parameters from config if available
                myfile.Read_Single(B, "external_field_magnitude");
                myfile.Read_Vector3(B_normal, "external_field_normal");
                B_normal.normalize();
                if (B_normal.norm() < 1e-8)
                {
                    B_normal = { 0,0,1 };
                    Log(Log_Level::Warning, Log_Sender::IO, "Input for 'external_field_normal' had norm zero and has been set to (0,0,1)");
                }
            }// end try
            catch( ... )
            {
                spirit_handle_exception_core(fmt::format("Unable to read external field from config file  \"{}\"", configFile));
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
                        for (int i = 0; i < anisotropy_index.size(); ++i)
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
            catch( ... )
            {
                spirit_handle_exception_core(fmt::format("Unable to read anisotropy from config file  \"{}\"", configFile));
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
            catch( ... )
            {
                spirit_handle_exception_core(fmt::format("Unable to read interaction pairs from config file  \"{}\"", configFile));
            }
            
            try
            {
                IO::Filter_File_Handle myfile(configFile);

                //		Dipole-Dipole Pairs
                // Dipole Dipole radius
                myfile.Read_Single(ddi_radius, "dd_radius");
            }// end try
            catch( ... )
            {
                spirit_handle_exception_core(fmt::format("Unable to read DDI radius from config file  \"{}\"", configFile));
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
            catch( ... )
            {
                spirit_handle_exception_core(fmt::format("Unable to read interaction quadruplets from config file  \"{}\"", configFile));
            }
        }
        else Log(Log_Level::Warning, Log_Sender::IO, "Hamiltonian_Heisenberg_Pairs: Using default configuration!");
        
        // Return
        Log(Log_Level::Parameter, Log_Sender::IO, "Hamiltonian_Heisenberg_Pairs:");
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<19} = {1} {2} {3}", "boundary conditions", boundary_conditions[0], boundary_conditions[1], boundary_conditions[2]));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<19} = {1}", "B[0]", B));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<19} = {1}", "B_normal[0]", B_normal.transpose()));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<19} = {1}", "mu_s[0]", mu_s[0]));
        if (anisotropy_from_file)
            Log(Log_Level::Parameter, Log_Sender::IO, "        K                     from file");
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<19} = {1}", "K[0]", K));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<19} = {1}", "K_normal[0]", K_normal.transpose()));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<19} = {1}", "dd_radius", ddi_radius));
        auto hamiltonian = std::unique_ptr<Engine::Hamiltonian_Heisenberg_Pairs>(new Engine::Hamiltonian_Heisenberg_Pairs(
            mu_s,
            B, B_normal,
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
            catch( ... )
            {
                spirit_handle_exception_core(fmt::format("Unable to read Hamiltonian_Gaussian parameters from config file  \"{}\"", configFile));
            }
        }
        else Log(Log_Level::Warning, Log_Sender::IO, "Hamiltonian_Gaussian: Using default configuration!");


        // Return
        Log(Log_Level::Parameter, Log_Sender::IO, "Hamiltonian_Gaussian:");
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<12} = {1}", "n_gaussians", n_gaussians));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<12} = {1}", "amplitude[0]", amplitude[0]));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<12} = {1}", "width[0]", width[0]));
        Log(Log_Level::Parameter, Log_Sender::IO, fmt::format("        {0:<12} = {1}", "center[0]", center[0].transpose()));
        auto hamiltonian = std::unique_ptr<Engine::Hamiltonian_Gaussian>(new Engine::Hamiltonian_Gaussian(
            amplitude, width, center
        ));
        Log(Log_Level::Info, Log_Sender::IO, "Hamiltonian_Gaussian: built");
        return hamiltonian;
    }
}// end namespace IO