#include <engine/Neighbours.hpp>
#include <engine/Vectormath.hpp>
#include <io/Configparser.hpp>
#include <io/Dataparser.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/IO.hpp>
#include <io/hamiltonian/Hamiltonian.hpp>
#include <utility/Constants.hpp>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <algorithm>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

// using namespace Utility;
using Utility::Log_Level;
using Utility::Log_Sender;

namespace IO
{

void Log_from_Config( const std::string & config_file_name, bool force_quiet )
try
{
    // Verbosity and Reject Level are read as integers
    int i_level_file = 5, i_level_console = 5;
    std::string output_folder = ".";
    std::string file_tag      = "";
    bool messages_to_file = true, messages_to_console = true, save_input_initial = false, save_input_final = false,
         save_positions_initial = false, save_positions_final = false, save_neighbours_initial = false,
         save_neighbours_final = false;

    // "Quiet" settings
    if( force_quiet )
    {
        // Don't save the Log to file
        Log.messages_to_file = false;
        // Don't print the Log to console
        Log.messages_to_console = false;
        // Don't save input configs
        Log.save_input_initial = false;
        Log.save_input_final   = false;
        // Don't save positions
        Log.save_positions_initial = false;
        Log.save_positions_final   = false;
        // Don't save neighbours
        Log.save_neighbours_initial = false;
        Log.save_neighbours_final   = false;
        // Don't print messages, except Error & Severe
        Log.level_file    = Utility::Log_Level::Error;
        Log.level_console = Utility::Log_Level::Error;
    }

    //------------------------------- Parser --------------------------------
    if( !config_file_name.empty() )
    {
        try
        {
            Log( Log_Level::Debug, Log_Sender::IO, "Building Log" );
            IO::Filter_File_Handle config_file_handle( config_file_name );

            // Time tag
            config_file_handle.Read_Single( file_tag, "output_file_tag" );

            // Output folder
            config_file_handle.Read_Single( output_folder, "log_output_folder" );

            // Save Output (Log Messages) to file
            config_file_handle.Read_Single( messages_to_file, "log_to_file" );
            // File Accept Level
            config_file_handle.Read_Single( i_level_file, "log_file_level" );

            // Print Output (Log Messages) to console
            config_file_handle.Read_Single( messages_to_console, "log_to_console" );
            // File Accept Level
            config_file_handle.Read_Single( i_level_console, "log_console_level" );

            // Save Input (parameters from config file and defaults) on State Setup
            config_file_handle.Read_Single( save_input_initial, "save_input_initial" );
            // Save Input (parameters from config file and defaults) on State Delete
            config_file_handle.Read_Single( save_input_final, "save_input_final" );

            // Save Input (parameters from config file and defaults) on State Setup
            config_file_handle.Read_Single( save_positions_initial, "save_positions_initial" );
            // Save Input (parameters from config file and defaults) on State Delete
            config_file_handle.Read_Single( save_positions_final, "save_positions_final" );

            // Save Input (parameters from config file and defaults) on State Setup
            config_file_handle.Read_Single( save_neighbours_initial, "save_neighbours_initial" );
            // Save Input (parameters from config file and defaults) on State Delete
            config_file_handle.Read_Single( save_neighbours_final, "save_neighbours_final" );
        }
        catch( ... )
        {
            spirit_rethrow( fmt::format(
                "Failed to read log levels from file \"{}\". Leaving values at default.", config_file_name ) );
        }
    }

    // Log the parameters
    std::vector<std::string> block;
    block.emplace_back( "Logging parameters" );
    block.emplace_back( fmt::format( "    file tag on output = \"{}\"", file_tag ) );
    block.emplace_back( fmt::format( "    output folder      = \"{}\"", output_folder ) );
    block.emplace_back( fmt::format( "    to file            = {}", messages_to_file ) );
    block.emplace_back( fmt::format( "    file accept level  = {}", i_level_file ) );
    block.emplace_back( fmt::format( "    to console         = {}", messages_to_console ) );
    block.emplace_back( fmt::format( "    print accept level = {}", i_level_console ) );
    block.emplace_back( fmt::format( "    input save initial = {}", save_input_initial ) );
    block.emplace_back( fmt::format( "    input save final   = {}", save_input_final ) );
    block.emplace_back( fmt::format( "    positions save initial  = {}", save_positions_initial ) );
    block.emplace_back( fmt::format( "    positions save final    = {}", save_positions_final ) );
    block.emplace_back( fmt::format( "    neighbours save initial = {}", save_neighbours_initial ) );
    block.emplace_back( fmt::format( "    neighbours save final   = {}", save_neighbours_final ) );
    Log( Log_Level::Parameter, Log_Sender::IO, block );

    // Update the Log
    if( !force_quiet )
    {
        Log.level_file    = Log_Level( i_level_file );
        Log.level_console = Log_Level( i_level_console );

        Log.messages_to_file        = messages_to_file;
        Log.messages_to_console     = messages_to_console;
        Log.save_input_initial      = save_input_initial;
        Log.save_input_final        = save_input_final;
        Log.save_positions_initial  = save_positions_initial;
        Log.save_positions_final    = save_positions_final;
        Log.save_neighbours_initial = save_neighbours_initial;
        Log.save_neighbours_final   = save_neighbours_final;
    }

    Log.file_tag      = file_tag;
    Log.output_folder = output_folder;

    if( file_tag == "<time>" )
        Log.file_name = "Log_" + Utility::Timing::CurrentDateTime() + ".txt";
    else if( !file_tag.empty() )
        Log.file_name = "Log_" + file_tag + ".txt";
    else
        Log.file_name = "Log.txt";
}
catch( ... )
{
    spirit_handle_exception_core(
        fmt::format( "Unable to read logging parameters from config file \"{}\"", config_file_name ) );
} // End Log_from_Config

std::unique_ptr<State::system_t> Spin_System_from_Config( const std::string & config_file_name )
try
{
    Log( Log_Level::Info, Log_Sender::IO, "-------------- Initialising Spin System ------------" );

    // Geometry
    auto geometry = Geometry_from_Config( config_file_name );
    // Boundary conditions
    auto boundary_conditions = Boundary_Conditions_from_Config( config_file_name );
    // LLG Parameters
    auto llg_params = Parameters_Method_LLG_from_Config( config_file_name );
    // MC Parameters
    auto mc_params = Parameters_Method_MC_from_Config( config_file_name );
    // EMA Parameters
    auto ema_params = Parameters_Method_EMA_from_Config( config_file_name );
    // MMF Parameters
    auto mmf_params = Parameters_Method_MMF_from_Config( config_file_name );
    // Hamiltonian
    auto hamiltonian = Hamiltonian_from_Config( config_file_name, std::move( geometry ), std::move( boundary_conditions ) );
    // Spin System
    auto system = std::make_unique<State::system_t>(
        std::move( hamiltonian ), std::move( llg_params ), std::move( mc_params ), std::move( ema_params ),
        std::move( mmf_params ), false );

    Log( Log_Level::Info, Log_Sender::IO, "-------------- Spin System Initialised -------------" );

    return system;
}
catch( ... )
{
    spirit_handle_exception_core(
        fmt::format( "Unable to initialize spin system from config file \"{}\"", config_file_name ) );
    return nullptr;
} // End Spin_System_from_Config

void Bravais_Vectors_from_Config(
    const std::string & config_file_name, std::vector<Vector3> & bravais_vectors,
    Data::BravaisLatticeType & bravais_lattice_type, std::string & bravais_lattice_type_str )
try
{
    std::string bravais_lattice = "sc";

    IO::Filter_File_Handle config_file_handle( config_file_name );
    // Bravais lattice type or manually specified vectors/matrix
    if( config_file_handle.Find( "bravais_lattice" ) )
    {
        config_file_handle >> bravais_lattice;
        std::transform( bravais_lattice.begin(), bravais_lattice.end(), bravais_lattice.begin(), ::tolower );

        if( bravais_lattice == "sc" )
        {
            bravais_lattice_type     = Data::BravaisLatticeType::SC;
            bravais_vectors          = Data::Geometry::BravaisVectorsSC();
            bravais_lattice_type_str = "simple cubic";
        }
        else if( bravais_lattice == "fcc" )
        {
            bravais_lattice_type     = Data::BravaisLatticeType::FCC;
            bravais_vectors          = Data::Geometry::BravaisVectorsFCC();
            bravais_lattice_type_str = "face-centered cubic";
        }
        else if( bravais_lattice == "bcc" )
        {
            bravais_lattice_type     = Data::BravaisLatticeType::BCC;
            bravais_vectors          = Data::Geometry::BravaisVectorsBCC();
            bravais_lattice_type_str = "body-centered cubic";
        }
        else if( bravais_lattice == "hex2d" )
        {
            bravais_lattice_type     = Data::BravaisLatticeType::Hex2D;
            bravais_vectors          = Data::Geometry::BravaisVectorsHex2D60();
            bravais_lattice_type_str = "hexagonal 2D (default: 60deg angle)";
        }
        else if( bravais_lattice == "hex2d60" )
        {
            bravais_lattice_type     = Data::BravaisLatticeType::Hex2D;
            bravais_vectors          = Data::Geometry::BravaisVectorsHex2D60();
            bravais_lattice_type_str = "hexagonal 2D 60deg angle";
        }
        else if( bravais_lattice == "hex2d120" )
        {
            bravais_lattice_type     = Data::BravaisLatticeType::Hex2D;
            bravais_vectors          = Data::Geometry::BravaisVectorsHex2D120();
            bravais_lattice_type_str = "hexagonal 2D 120deg angle";
        }
        else
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format( "Bravais lattice \"{}\" unknown. Using simple cubic...", bravais_lattice ) );
    }
    else if( config_file_handle.Find( "bravais_vectors" ) )
    {
        Log( Log_Level::Parameter, Log_Sender::IO, "Bravais lattice type: irregular" );
        bravais_lattice_type = Data::BravaisLatticeType::Irregular;
        config_file_handle.GetLine();
        config_file_handle >> bravais_vectors[0][0] >> bravais_vectors[0][1] >> bravais_vectors[0][2];
        config_file_handle.GetLine();
        config_file_handle >> bravais_vectors[1][0] >> bravais_vectors[1][1] >> bravais_vectors[1][2];
        config_file_handle.GetLine();
        config_file_handle >> bravais_vectors[2][0] >> bravais_vectors[2][1] >> bravais_vectors[2][2];
    }
    else if( config_file_handle.Find( "bravais_matrix" ) )
    {
        Log( Log_Level::Parameter, Log_Sender::IO, "Bravais lattice type: irregular" );
        bravais_lattice_type = Data::BravaisLatticeType::Irregular;
        config_file_handle.GetLine();
        config_file_handle >> bravais_vectors[0][0] >> bravais_vectors[1][0] >> bravais_vectors[2][0];
        config_file_handle.GetLine();
        config_file_handle >> bravais_vectors[0][1] >> bravais_vectors[1][1] >> bravais_vectors[2][1];
        config_file_handle.GetLine();
        config_file_handle >> bravais_vectors[0][2] >> bravais_vectors[1][2] >> bravais_vectors[2][2];
    }
    else
        Log( Log_Level::Parameter, Log_Sender::IO, "Bravais lattice not specified. Using simple cubic..." );
}
catch( ... )
{
    spirit_rethrow( fmt::format( "Unable to parse bravais vectors from config file \"{}\"", config_file_name ) );
} // End Basis_from_Config

Data::Geometry Geometry_from_Config( const std::string & config_file_name )
{
    //-------------- Insert default values here -----------------------------
    // Basis from separate file?
    std::string basis_file = "";
    // Bravais lattice type
    std::string bravais_lattice = "sc";
    // Bravais vectors {a, b, c}
    Data::BravaisLatticeType bravais_lattice_type = Data::BravaisLatticeType::SC;
    std::string bravais_lattice_type_str;
    std::vector<Vector3> bravais_vectors = { Vector3{ 1, 0, 0 }, Vector3{ 0, 1, 0 }, Vector3{ 0, 0, 1 } };
    // Atoms in the basis
    std::vector<Vector3> cell_atoms = { Vector3{ 0, 0, 0 } };
    std::size_t n_cell_atoms        = cell_atoms.size();
    // Basis cell composition information (atom types, magnetic moments, ...)
    Data::Basis_Cell_Composition cell_composition{ false, { 0 }, { 0 }, { 1 }, {} };
    // Lattice Constant [Angstrom]
    scalar lattice_constant = 1;
    // Number of translations nT for each basis direction
    intfield n_cells = { 100, 100, 1 };
    // Atom types
    field<Site> defect_sites( 0 );
    intfield defect_types( 0 );
    int n_atom_types = 0;

    // Utility 1D array to build vectors and use Vectormath
    Vector3 build_array = { 0, 0, 0 };

    try
    {
        //------------------------------- Parser --------------------------------
        Log( Log_Level::Debug, Log_Sender::IO, "Geometry: building" );
        if( !config_file_name.empty() )
        {
            try
            {
                IO::Filter_File_Handle config_file_handle( config_file_name );

                // Lattice constant
                config_file_handle.Read_Single( lattice_constant, "lattice_constant" );

                // Get the bravais lattice type and vectors
                Bravais_Vectors_from_Config(
                    config_file_name, bravais_vectors, bravais_lattice_type, bravais_lattice_type_str );

                // Read number of basis cells
                config_file_handle.Read_3Vector( n_cells, "n_basis_cells" );

                // Basis
                if( config_file_handle.Find( "basis_file" ) )
                {
                    config_file_handle >> basis_file;
                }
                else if( config_file_handle.Find( "basis" ) )
                {
                    basis_file = config_file_name;
                }

                if( !basis_file.empty() )
                {
                    Basis_from_File( basis_file, cell_composition, cell_atoms, n_cell_atoms );
                }
            }
            catch( ... )
            {
                spirit_handle_exception_core( fmt::format(
                    "Failed to read Geometry parameters from file \"{}\". Leaving values at default.",
                    config_file_name ) );
            }

            try
            {
                IO::Filter_File_Handle config_file_handle( config_file_name );

                // Spin moment
                if( !config_file_handle.Find( "atom_types" ) )
                {
                    if( config_file_handle.Find( "mu_s" ) )
                    {
                        for( std::size_t iatom = 0; iatom < n_cell_atoms; ++iatom )
                        {
                            if( !( config_file_handle >> cell_composition.mu_s[iatom] ) )
                            {
                                Log( Log_Level::Warning, Log_Sender::IO,
                                     fmt::format(
                                         "Not enough values specified after 'mu_s'. Expected {}. Using "
                                         "mu_s[{}]=mu_s[0]={}",
                                         n_cell_atoms, iatom, cell_composition.mu_s[0] ) );
                                cell_composition.mu_s[iatom] = cell_composition.mu_s[0];
                            }
                        }
                    }
                    else
                        Log( Log_Level::Warning, Log_Sender::IO,
                             fmt::format( "Keyword 'mu_s' not found. Using Default: {}", cell_composition.mu_s[0] ) );
                }
                // else
                // {
                //     cell_composition.mu_s = std::vector<scalar>(n_atom_types, 1);
                //     if( config_file_handle.Find("mu_s") )
                //     {
                //         for (int itype = 0; itype < n_atom_types; ++itype)
                //         {
                //             config_file_handle >> cell_composition.mu_s[itype];
                //             // config_file_handle.GetLine();
                //             // config_file_handle >> cell_composition.iatom[itype];
                //             // config_file_handle >> cell_composition.atom_type[itype];
                //             // config_file_handle >> cell_composition.concentration[itype];
                //         }
                //     }
                //     else Log(Log_Level::Error, Log_Sender::IO, fmt::format("Keyword 'mu_s' not found. Using Default:
                //     {}", cell_composition.mu_s[0]));
                // }
            }
            catch( ... )
            {
                spirit_handle_exception_core(
                    fmt::format( "Unable to read mu_s from config file \"{}\"", config_file_name ) );
            }

            // Defects
#ifdef SPIRIT_ENABLE_DEFECTS
            try
            {
                IO::Filter_File_Handle config_file_handle( config_file_name );

                int n_defects = 0;

                std::string defects_file = "";
                if( config_file_handle.Find( "n_defects" ) )
                    defects_file = config_file_name;
                else if( config_file_handle.Find( "defects_from_file" ) )
                    config_file_handle >> defects_file;

                if( !defects_file.empty() )
                {
                    // The file name should be valid so we try to read it
                    Defects_from_File( defects_file, n_defects, defect_sites, defect_types );
                }

                // Disorder
                if( config_file_handle.Find( "atom_types" ) )
                {
                    config_file_handle >> n_atom_types;
                    cell_composition.disordered = true;
                    cell_composition.iatom.resize( n_atom_types );
                    cell_composition.atom_type.resize( n_atom_types );
                    cell_composition.mu_s.resize( n_atom_types );
                    cell_composition.concentration.resize( n_atom_types );
                    for( int itype = 0; itype < n_atom_types; ++itype )
                    {
                        config_file_handle.GetLine();
                        config_file_handle >> cell_composition.iatom[itype];
                        config_file_handle >> cell_composition.atom_type[itype];
                        config_file_handle >> cell_composition.mu_s[itype];
                        config_file_handle >> cell_composition.concentration[itype];
                        // if ( !(config_file_handle >> mu_s[itype]) )
                        // {
                        //     Log(Log_Level::Warning, Log_Sender::IO,
                        //         fmt::format("Not enough values specified after 'mu_s'. Expected {}. Using
                        //         mu_s[{}]=mu_s[0]={}", n_cell_atoms, iatom, mu_s[0]));
                        //     mu_s[iatom] = mu_s[0];
                        // }
                    }
                    Log( Log_Level::Warning, Log_Sender::IO,
                         fmt::format(
                             "{} atom types, iatom={} atom type={} concentration={}", n_atom_types,
                             cell_composition.iatom[0], cell_composition.atom_type[0],
                             cell_composition.concentration[0] ) );
                }
            }
            catch( ... )
            {
                spirit_handle_exception_core( fmt::format(
                    "Failed to read defect parameters from file \"{}\". Leaving values at default.",
                    config_file_name ) );
            }
#else
            Log( Log_Level::Parameter, Log_Sender::IO, "Disorder is disabled" );
#endif

        } // end if file=""
        else
            Log( Log_Level::Parameter, Log_Sender::IO, "Geometry: Using default configuration!" );

        // Pinning configuration
        auto pinning = Pinning_from_Config( config_file_name, cell_atoms.size() );

        // Log the parameters
        std::vector<std::string> parameter_log;
        parameter_log.emplace_back( "Geometry:" );
        parameter_log.emplace_back( fmt::format( "    lattice constant = {} Angstrom", lattice_constant ) );
        parameter_log.emplace_back( fmt::format( "    Bravais lattice type: {}", bravais_lattice_type_str ) );
        Log( Log_Level::Debug, Log_Sender::IO,
             {
                 "    Bravais vectors in units of lattice constant",
                 fmt::format( "        a = {}", bravais_vectors[0].transpose() / lattice_constant ),
                 fmt::format( "        b = {}", bravais_vectors[1].transpose() / lattice_constant ),
                 fmt::format( "        c = {}", bravais_vectors[2].transpose() / lattice_constant ),
             } );
        parameter_log.emplace_back( "    Bravais vectors" );
        parameter_log.emplace_back( fmt::format( "        a = {}", bravais_vectors[0].transpose() ) );
        parameter_log.emplace_back( fmt::format( "        b = {}", bravais_vectors[1].transpose() ) );
        parameter_log.emplace_back( fmt::format( "        c = {}", bravais_vectors[2].transpose() ) );
        parameter_log.emplace_back( fmt::format( "    basis cell: {} atom(s)", n_cell_atoms ) );
        parameter_log.emplace_back( "    relative positions (first 10):" );
        for( std::size_t iatom = 0; iatom < n_cell_atoms && iatom < 10; ++iatom )
            parameter_log.emplace_back( fmt::format(
                "        atom {} at ({}), mu_s={}", iatom, cell_atoms[iatom].transpose(),
                cell_composition.mu_s[iatom] ) );

        parameter_log.emplace_back( "    absolute atom positions (first 10):" );
        for( std::size_t iatom = 0; iatom < n_cell_atoms && iatom < 10; ++iatom )
        {
            Vector3 cell_atom
                = lattice_constant
                  * ( bravais_vectors[0] * cell_atoms[iatom][0] + bravais_vectors[1] * cell_atoms[iatom][1]
                      + bravais_vectors[2] * cell_atoms[iatom][2] );
            parameter_log.emplace_back( fmt::format( "        atom {} at ({})", iatom, cell_atom.transpose() ) );
        }

        if( cell_composition.disordered )
            parameter_log.emplace_back( "    note: the lattice has some disorder!" );

#ifdef SPIRIT_ENABLE_PINNING
        // Log pinning
        auto n_pinned_cell_sites = pinning.sites.size();
        if( n_pinned_cell_sites == 0 && pinning.na_left == 0 && pinning.na_right == 0 && pinning.nb_left == 0
            && pinning.nb_right == 0 && pinning.nc_left == 0 && pinning.nc_right == 0 )
        {
            parameter_log.emplace_back( "    no pinned spins" );
        }
        else
        {
            parameter_log.emplace_back( "    pinning of boundary cells:" );
            parameter_log.emplace_back(
                fmt::format( "        n_a: left={}, right={}", pinning.na_left, pinning.na_right ) );
            parameter_log.emplace_back(
                fmt::format( "        n_b: left={}, right={}", pinning.nb_left, pinning.nb_right ) );
            parameter_log.emplace_back(
                fmt::format( "        n_c: left={}, right={}", pinning.nc_left, pinning.nc_right ) );

            parameter_log.emplace_back( "        pinned to (showing first 10 sites):" );
            for( std::size_t i = 0; i < std::min( pinning.pinned_cell.size(), static_cast<std::size_t>( 10 ) ); ++i )
            {
                parameter_log.emplace_back(
                    fmt::format( "          cell atom[{}] = ({})", i, pinning.pinned_cell[i].transpose() ) );
            }
            if( n_pinned_cell_sites == 0 )
                parameter_log.emplace_back( "    no individually pinned sites" );
            else
            {
                parameter_log.emplace_back(
                    fmt::format( "    {} individually pinned sites. Showing the first 10:", n_pinned_cell_sites ) );
                for( std::size_t i = 0; i < std::min( pinning.sites.size(), static_cast<std::size_t>( 10 ) ); ++i )
                {
                    parameter_log.emplace_back( fmt::format(
                        "        pinned site[{}]: {} at ({} {} {}) = ({})", i, pinning.sites[i].i,
                        pinning.sites[i].translations[0], pinning.sites[i].translations[1],
                        pinning.sites[i].translations[2], pinning.spins[i].transpose() ) );
                }
            }
        }
#endif

        // Defects
#ifdef SPIRIT_ENABLE_DEFECTS
        if( defect_sites.empty() )
            parameter_log.emplace_back( "    no defects" );
        else
        {
            parameter_log.emplace_back(
                fmt::format( "    {} defects (showing first 10 sites):", defect_sites.size() ) );
            for( std::size_t i = 0; i < std::min( defect_sites.size(), static_cast<std::size_t>( 10 ) ); ++i )
            {
                parameter_log.emplace_back( fmt::format(
                    "        defect[{}]: translations=({} {} {}), type=", i, defect_sites[i].translations[0],
                    defect_sites[i].translations[1], defect_sites[i].translations[2], defect_types[i] ) );
            }
        }
#endif

        // Log parameters
        parameter_log.emplace_back( "    lattice: n_basis_cells" );
        parameter_log.emplace_back( fmt::format( "        na = {}", n_cells[0] ) );
        parameter_log.emplace_back( fmt::format( "        nb = {}", n_cells[1] ) );
        parameter_log.emplace_back( fmt::format( "        nc = {}", n_cells[2] ) );

        // Return geometry
        auto geometry = Data::Geometry(
            bravais_vectors, n_cells, cell_atoms, cell_composition, lattice_constant, pinning,
            Data::Defects{ defect_sites, defect_types } );

        parameter_log.emplace_back( fmt::format( "    {} spins", geometry.nos ) );
        parameter_log.emplace_back( fmt::format( "    the geometry is {}-dimensional", geometry.dimensionality ) );

        Log( Log_Level::Parameter, Log_Sender::IO, parameter_log );

        Log( Log_Level::Debug, Log_Sender::IO, "Geometry: built" );
        return geometry;
    }
    catch( ... )
    {
        spirit_rethrow( fmt::format( "Unable to parse geometry from config file \"{}\"", config_file_name ) );
        return Data::Geometry(
            bravais_vectors, n_cells, cell_atoms, cell_composition, lattice_constant, Data::Pinning(),
            Data::Defects{ defect_sites, defect_types } );
    }
} // End Geometry from Config

intfield Boundary_Conditions_from_Config( const std::string & config_file_name )
try
{
    // Boundary conditions (a, b, c)
    std::vector<int> boundary_conditions_i = { 0, 0, 0 };
    intfield boundary_conditions           = { false, false, false };

    if( !config_file_name.empty() )
    {
        IO::Filter_File_Handle config_file_handle( config_file_name );

        // Boundary conditions
        config_file_handle.Read_3Vector( boundary_conditions_i, "boundary_conditions" );
        boundary_conditions[0] = static_cast<int>( boundary_conditions_i[0] != 0 );
        boundary_conditions[1] = static_cast<int>( boundary_conditions_i[1] != 0 );
        boundary_conditions[2] = static_cast<int>( boundary_conditions_i[2] != 0 );
    }

    return boundary_conditions;
}
catch( ... )
{
    spirit_rethrow( fmt::format( "Unable to parse boundary conditions from config file \"{}\"", config_file_name ) );
    return { 0, 0, 0 };
} // End boundary_conditions from Config

Data::Pinning Pinning_from_Config( const std::string & config_file_name, std::size_t n_cell_atoms )
{
    //-------------- Insert default values here -----------------------------
    int na = 0, na_left = 0, na_right = 0;
    int nb = 0, nb_left = 0, nb_right = 0;
    int nc = 0, nc_left = 0, nc_right = 0;
    vectorfield pinned_cell( n_cell_atoms, Vector3{ 0, 0, 1 } );
    // Additional pinned sites
    field<Site> pinned_sites( 0 );
    vectorfield pinned_spins( 0 );
    int n_pinned = 0;

    // Utility 1D array to build vectors and use Vectormath
    Vector3 build_array = { 0, 0, 0 };

#ifdef SPIRIT_ENABLE_PINNING
    //------------------------------- Parser --------------------------------
    Log( Log_Level::Debug, Log_Sender::IO, "going to read pinning" );
    if( !config_file_name.empty() )
    {
        try
        {
            IO::Filter_File_Handle config_file_handle( config_file_name );

            // N_a
            config_file_handle.Read_Single( na_left, "pin_na_left", false );
            config_file_handle.Read_Single( na_right, "pin_na_right", false );
            config_file_handle.Read_Single( na, "pin_na ", false );
            if( na > 0 && ( na_left == 0 || na_right == 0 ) )
            {
                na_left  = na;
                na_right = na;
            }

            // N_b
            config_file_handle.Read_Single( nb_left, "pin_nb_left", false );
            config_file_handle.Read_Single( nb_right, "pin_nb_right", false );
            config_file_handle.Read_Single( nb, "pin_nb ", false );
            if( nb > 0 && ( nb_left == 0 || nb_right == 0 ) )
            {
                nb_left  = nb;
                nb_right = nb;
            }

            // N_c
            config_file_handle.Read_Single( nc_left, "pin_nc_left", false );
            config_file_handle.Read_Single( nc_right, "pin_nc_right", false );
            config_file_handle.Read_Single( nc, "pin_nc ", false );
            if( nc > 0 && ( nc_left == 0 || nc_right == 0 ) )
            {
                nc_left  = nc;
                nc_right = nc;
            }

            // How should the cells be pinned
            if( na_left > 0 || na_right > 0 || nb_left > 0 || nb_right > 0 || nc_left > 0 || nc_right > 0 )
            {
                if( config_file_handle.Find( "pinning_cell" ) )
                {
                    for( std::size_t i = 0; i < n_cell_atoms; ++i )
                    {
                        config_file_handle.GetLine();
                        config_file_handle >> pinned_cell[i][0] >> pinned_cell[i][1] >> pinned_cell[i][2];
                    }
                }
                else
                {
                    na_left  = 0;
                    na_right = 0;
                    nb_left  = 0;
                    nb_right = 0;
                    nc_left  = 0;
                    nc_right = 0;
                    Log( Log_Level::Warning, Log_Sender::IO,
                         "Pinning specified, but keyword 'pinning_cell' not found. Won't pin any spins!" );
                }
            }

            // Additional pinned sites
            std::string pinned_file = "";
            if( config_file_handle.Find( "n_pinned" ) )
                pinned_file = config_file_name;
            else if( config_file_handle.Find( "pinned_from_file" ) )
                config_file_handle >> pinned_file;

            if( !pinned_file.empty() )
            {
                // The file name should be valid so we try to read it
                Pinned_from_File( pinned_file, n_pinned, pinned_sites, pinned_spins );
            }
        }
        catch( ... )
        {
            spirit_handle_exception_core( fmt::format(
                "Failed to read Pinning from file \"{}\". Leaving values at default.", config_file_name ) );
        }
    }

    // Create Pinning
    auto pinning = Data::Pinning{
        na_left, na_right, nb_left, nb_right, nc_left, nc_right, pinned_cell, pinned_sites, pinned_spins,
    };

    // Return Pinning
    Log( Log_Level::Debug, Log_Sender::IO, "pinning has been read" );
    return pinning;
#else  // SPIRIT_ENABLE_PINNING
    Log( Log_Level::Parameter, Log_Sender::IO, "Pinning is disabled" );
    if( !config_file_name.empty() )
    {
        try
        {
            IO::Filter_File_Handle config_file_handle( config_file_name );
            if( config_file_handle.Find( "pinning_cell" ) )
                Log( Log_Level::Warning, Log_Sender::IO,
                     "You specified a pinning cell even though pinning is disabled!" );
        }
        catch( ... )
        {
            spirit_handle_exception_core( fmt::format(
                "Failed to read pinning parameters from file \"{}\". Leaving values at default.", config_file_name ) );
        }
    }

    return Data::Pinning{ 0, 0, 0, 0, 0, 0, vectorfield( 0 ), field<Site>( 0 ), vectorfield( 0 ) };
#endif // SPIRIT_ENABLE_PINNING
}

std::unique_ptr<Data::Parameters_Method_LLG> Parameters_Method_LLG_from_Config( const std::string & config_file_name )
{
    // Default parameters
    auto parameters = std::make_unique<Data::Parameters_Method_LLG>();

    // PRNG Seed
    std::random_device random;
    parameters->rng_seed = random();
    parameters->prng     = std::mt19937( parameters->rng_seed );

    // Maximum wall time
    std::string str_max_walltime = "0";

    // Configuration output filetype
    int output_configuration_filetype = (int)parameters->output_vf_filetype;

    // Parse
    Log( Log_Level::Debug, Log_Sender::IO, "Parameters LLG: building" );
    if( !config_file_name.empty() )
    {
        try
        {
            IO::Filter_File_Handle config_file_handle( config_file_name );

            // Output parameters
            config_file_handle.Read_Single( parameters->output_file_tag, "output_file_tag" );
            config_file_handle.Read_Single( parameters->output_folder, "llg_output_folder" );
            config_file_handle.Read_Single( parameters->output_any, "llg_output_any" );
            config_file_handle.Read_Single( parameters->output_initial, "llg_output_initial" );
            config_file_handle.Read_Single( parameters->output_final, "llg_output_final" );
            config_file_handle.Read_Single(
                parameters->output_energy_spin_resolved, "llg_output_energy_spin_resolved" );
            config_file_handle.Read_Single( parameters->output_energy_step, "llg_output_energy_step" );
            config_file_handle.Read_Single( parameters->output_energy_archive, "llg_output_energy_archive" );
            config_file_handle.Read_Single(
                parameters->output_energy_divide_by_nspins, "llg_output_energy_divide_by_nspins" );
            config_file_handle.Read_Single(
                parameters->output_energy_add_readability_lines, "llg_output_energy_add_readability_lines" );
            config_file_handle.Read_Single( parameters->output_configuration_step, "llg_output_configuration_step" );
            config_file_handle.Read_Single(
                parameters->output_configuration_archive, "llg_output_configuration_archive" );
            config_file_handle.Read_Single( output_configuration_filetype, "llg_output_configuration_filetype" );
            parameters->output_vf_filetype = IO::VF_FileFormat( output_configuration_filetype );
            // Method parameters
            config_file_handle.Read_Single( str_max_walltime, "llg_max_walltime" );
            parameters->max_walltime_sec = (long int)Utility::Timing::DurationFromString( str_max_walltime ).count();
            config_file_handle.Read_Single( parameters->rng_seed, "llg_seed" );
            parameters->prng = std::mt19937( parameters->rng_seed );
            config_file_handle.Read_Single( parameters->n_iterations, "llg_n_iterations" );
            config_file_handle.Read_Single( parameters->n_iterations_log, "llg_n_iterations_log" );
            config_file_handle.Read_Single( parameters->n_iterations_amortize, "llg_n_iterations_amortize" );
            config_file_handle.Read_Single( parameters->dt, "llg_dt" );
            config_file_handle.Read_Single( parameters->temperature, "llg_temperature" );
            config_file_handle.Read_Vector3(
                parameters->temperature_gradient_direction, "llg_temperature_gradient_direction" );
            parameters->temperature_gradient_direction.normalize();
            config_file_handle.Read_Single(
                parameters->temperature_gradient_inclination, "llg_temperature_gradient_inclination" );
            config_file_handle.Read_Single( parameters->damping, "llg_damping" );
            config_file_handle.Read_Single( parameters->beta, "llg_beta" );
            // config_file_handle.Read_Single(parameters->renorm_sd, "llg_renorm");
            config_file_handle.Read_Single( parameters->stt_use_gradient, "llg_stt_use_gradient" );
            config_file_handle.Read_Single( parameters->stt_magnitude, "llg_stt_magnitude" );
            config_file_handle.Read_Vector3( parameters->stt_polarisation_normal, "llg_stt_polarisation_normal" );
            parameters->stt_polarisation_normal.normalize();
            config_file_handle.Read_Single( parameters->force_convergence, "llg_force_convergence" );
        }
        catch( ... )
        {
            spirit_handle_exception_core(
                fmt::format( "Unable to parse LLG parameters from config file \"{}\"", config_file_name ) );
        }
    }
    else
        Log( Log_Level::Parameter, Log_Sender::IO, "Parameters LLG: Using default configuration!" );

    // Return
    std::vector<std::string> parameter_log;
    parameter_log.emplace_back( "Parameters LLG:" );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "seed", parameters->rng_seed ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "time step [ps]", parameters->dt ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "temperature [K]", parameters->temperature ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<17} = {}", "temperature gradient direction", parameters->temperature_gradient_direction.transpose() ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<17} = {}", "temperature gradient inclination", parameters->temperature_gradient_inclination ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "damping", parameters->damping ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "beta", parameters->beta ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "stt use gradient", parameters->stt_use_gradient ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "stt magnitude", parameters->stt_magnitude ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<17} = {}", "stt normal", parameters->stt_polarisation_normal.transpose() ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<17} = {:e}", "force convergence", parameters->force_convergence ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "maximum walltime", str_max_walltime ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "n_iterations", parameters->n_iterations ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "n_iterations_log", parameters->n_iterations_log ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<17} = {}", "n_iterations_amortize", parameters->n_iterations_amortize ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = \"{}\"", "output_folder", parameters->output_folder ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output_any", parameters->output_any ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output_initial", parameters->output_initial ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output_final", parameters->output_final ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_energy_step", parameters->output_energy_step ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_energy_archive", parameters->output_energy_archive ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_energy_spin_resolved", parameters->output_energy_spin_resolved ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<30} = {}", "output_energy_divide_by_nspins", parameters->output_energy_divide_by_nspins ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<30} = {}", "output_energy_add_readability_lines", parameters->output_energy_add_readability_lines ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_configuration_step", parameters->output_configuration_step ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_configuration_archive", parameters->output_configuration_archive ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_configuration_filetype", (int)parameters->output_vf_filetype ) );
    Log( Log_Level::Parameter, Log_Sender::IO, parameter_log );

    Log( Log_Level::Debug, Log_Sender::IO, "Parameters LLG: built" );
    return parameters;
} // end Parameters_Method_LLG_from_Config

std::unique_ptr<Data::Parameters_Method_EMA> Parameters_Method_EMA_from_Config( const std::string & config_file_name )
{
    // Default parameters
    auto parameters = std::make_unique<Data::Parameters_Method_EMA>();

    // Maximum wall time
    std::string str_max_walltime = "0";

    // Parse
    Log( Log_Level::Debug, Log_Sender::IO, "Parameters EMA: building" );
    if( !config_file_name.empty() )
    {
        try
        {
            IO::Filter_File_Handle config_file_handle( config_file_name );

            // Output parameters
            config_file_handle.Read_Single( parameters->output_folder, "ema_output_folder" );
            config_file_handle.Read_Single( parameters->output_file_tag, "output_file_tag" );
            config_file_handle.Read_Single( parameters->output_any, "ema_output_any" );
            config_file_handle.Read_Single( parameters->output_initial, "ema_output_initial" );
            config_file_handle.Read_Single( parameters->output_final, "ema_output_final" );
            config_file_handle.Read_Single(
                parameters->output_energy_divide_by_nspins, "ema_output_energy_divide_by_nspins" );
            config_file_handle.Read_Single(
                parameters->output_energy_spin_resolved, "ema_output_energy_spin_resolved" );
            config_file_handle.Read_Single( parameters->output_energy_step, "ema_output_energy_step" );
            config_file_handle.Read_Single( parameters->output_energy_archive, "ema_output_energy_archive" );
            config_file_handle.Read_Single( parameters->output_configuration_step, "ema_output_configuration_step" );
            config_file_handle.Read_Single(
                parameters->output_configuration_archive, "ema_output_configuration_archive" );
            // Method parameters
            config_file_handle.Read_Single( str_max_walltime, "ema_max_walltime" );
            parameters->max_walltime_sec = (long int)Utility::Timing::DurationFromString( str_max_walltime ).count();
            config_file_handle.Read_Single( parameters->n_iterations, "ema_n_iterations" );
            config_file_handle.Read_Single( parameters->n_iterations_log, "ema_n_iterations_log" );
            config_file_handle.Read_Single( parameters->n_modes, "ema_n_modes" );
            config_file_handle.Read_Single( parameters->n_mode_follow, "ema_n_mode_follow" );
            config_file_handle.Read_Single( parameters->frequency, "ema_frequency" );
            config_file_handle.Read_Single( parameters->amplitude, "ema_amplitude" );
            config_file_handle.Read_Single( parameters->sparse, "ema_sparse" );
        }
        catch( ... )
        {
            spirit_handle_exception_core(
                fmt::format( "Unable to parse EMA parameters from config file \"{}\"", config_file_name ) );
        }
    }
    else
        Log( Log_Level::Parameter, Log_Sender::IO, "Parameters EMA: Using default configuration!" );

    // Return
    std::vector<std::string> parameter_log;
    parameter_log.emplace_back( "Parameters EMA:" );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "n_modes", parameters->n_modes ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "n_mode_follow", parameters->n_mode_follow ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "frequency", parameters->frequency ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "amplitude", parameters->amplitude ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "n_iterations_log", parameters->n_iterations_log ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "n_iterations", parameters->n_iterations ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "sparse", parameters->sparse ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "maximum walltime", str_max_walltime ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_configuration_archive", parameters->output_configuration_archive ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_configuration_step", parameters->output_configuration_step ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_energy_archive", parameters->output_energy_archive ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_energy_step", parameters->output_energy_step ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_energy_spin_resolved", parameters->output_energy_spin_resolved ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<30} = {}", "output_energy_divide_by_nspins", parameters->output_energy_divide_by_nspins ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output_final", parameters->output_final ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output_initial", parameters->output_initial ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output_any", parameters->output_any ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = \"{}\"", "output_folder", parameters->output_folder ) );
    Log( Log_Level::Parameter, Log_Sender::IO, parameter_log );

    Log( Log_Level::Debug, Log_Sender::IO, "Parameters EMA: built" );
    return parameters;
}

std::unique_ptr<Data::Parameters_Method_MC> Parameters_Method_MC_from_Config( const std::string & config_file_name )
{
    // Default parameters
    auto parameters = std::make_unique<Data::Parameters_Method_MC>();

    // PRNG Seed
    std::random_device random;
    parameters->rng_seed = random();
    parameters->prng     = std::mt19937( parameters->rng_seed );

    // Maximum wall time
    std::string str_max_walltime = "0";

    // Configuration output filetype
    int output_configuration_filetype = (int)parameters->output_vf_filetype;

    // Parse
    Log( Log_Level::Debug, Log_Sender::IO, "Parameters MC: building" );
    if( !config_file_name.empty() )
    {
        try
        {
            IO::Filter_File_Handle config_file_handle( config_file_name );

            // Output parameters
            config_file_handle.Read_Single( parameters->output_file_tag, "output_file_tag" );
            config_file_handle.Read_Single( parameters->output_folder, "mc_output_folder" );
            config_file_handle.Read_Single( parameters->output_any, "mc_output_any" );
            config_file_handle.Read_Single( parameters->output_initial, "mc_output_initial" );
            config_file_handle.Read_Single( parameters->output_final, "mc_output_final" );
            config_file_handle.Read_Single( parameters->output_energy_spin_resolved, "mc_output_energy_spin_resolved" );
            config_file_handle.Read_Single( parameters->output_energy_step, "mc_output_energy_step" );
            config_file_handle.Read_Single( parameters->output_energy_archive, "mc_output_energy_archive" );
            config_file_handle.Read_Single(
                parameters->output_energy_divide_by_nspins, "mc_output_energy_divide_by_nspins" );
            config_file_handle.Read_Single(
                parameters->output_energy_add_readability_lines, "mc_output_energy_add_readability_lines" );
            config_file_handle.Read_Single( parameters->output_configuration_step, "mc_output_configuration_step" );
            config_file_handle.Read_Single(
                parameters->output_configuration_archive, "mc_output_configuration_archive" );
            config_file_handle.Read_Single( output_configuration_filetype, "mc_output_configuration_filetype" );
            parameters->output_vf_filetype = IO::VF_FileFormat( output_configuration_filetype );
            // Method parameters
            config_file_handle.Read_Single( str_max_walltime, "mc_max_walltime" );
            parameters->max_walltime_sec = (long int)Utility::Timing::DurationFromString( str_max_walltime ).count();
            config_file_handle.Read_Single( parameters->rng_seed, "mc_seed" );
            parameters->prng = std::mt19937( parameters->rng_seed );
            config_file_handle.Read_Single( parameters->n_iterations, "mc_n_iterations" );
            config_file_handle.Read_Single( parameters->n_iterations_log, "mc_n_iterations_log" );
            config_file_handle.Read_Single( parameters->n_iterations_amortize, "mc_n_iterations_amortize" );
            config_file_handle.Read_Single( parameters->temperature, "mc_temperature" );
            config_file_handle.Read_Single( parameters->acceptance_ratio_target, "mc_acceptance_ratio" );
        }
        catch( ... )
        {
            spirit_handle_exception_core(
                fmt::format( "Unable to parse MC parameters from config file \"{}\"", config_file_name ) );
        }
    }
    else
        Log( Log_Level::Parameter, Log_Sender::IO, "Parameters MC: Using default configuration!" );

    // Return
    std::vector<std::string> parameter_log;
    parameter_log.emplace_back( "Parameters MC:" );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "seed", parameters->rng_seed ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "temperature", parameters->temperature ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<17} = {}", "acceptance_ratio", parameters->acceptance_ratio_target ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "maximum walltime", str_max_walltime ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "n_iterations", parameters->n_iterations ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "n_iterations_log", parameters->n_iterations_log ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<17} = {}", "n_iterations_amortize", parameters->n_iterations_amortize ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = \"{}\"", "output_folder", parameters->output_folder ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output_any", parameters->output_any ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output_initial", parameters->output_initial ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output_final", parameters->output_final ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_energy_step", parameters->output_energy_step ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_energy_archive", parameters->output_energy_archive ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_energy_spin_resolved", parameters->output_energy_spin_resolved ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<30} = {}", "output_energy_divide_by_nspins", parameters->output_energy_divide_by_nspins ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<30} = {}", "output_energy_add_readability_lines", parameters->output_energy_add_readability_lines ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_configuration_step", parameters->output_configuration_step ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_configuration_archive", parameters->output_configuration_archive ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_configuration_filetype", (int)parameters->output_vf_filetype ) );
    Log( Log_Level::Parameter, Log_Sender::IO, parameter_log );

    Log( Log_Level::Debug, Log_Sender::IO, "Parameters MC: built" );
    return parameters;
}

std::unique_ptr<Data::Parameters_Method_GNEB> Parameters_Method_GNEB_from_Config( const std::string & config_file_name )
{
    // Default parameters
    auto parameters = std::make_unique<Data::Parameters_Method_GNEB>();

    // Maximum wall time
    std::string str_max_walltime = "0";

    // Chain output filetype
    int output_chain_filetype = (int)parameters->output_vf_filetype;

    // Parse
    Log( Log_Level::Debug, Log_Sender::IO, "Parameters GNEB: building" );
    if( !config_file_name.empty() )
    {
        try
        {
            IO::Filter_File_Handle config_file_handle( config_file_name );

            // Output parameters
            config_file_handle.Read_Single( parameters->output_file_tag, "output_file_tag" );
            config_file_handle.Read_Single( parameters->output_folder, "gneb_output_folder" );
            config_file_handle.Read_Single( parameters->output_any, "gneb_output_any" );
            config_file_handle.Read_Single( parameters->output_initial, "gneb_output_initial" );
            config_file_handle.Read_Single( parameters->output_final, "gneb_output_final" );
            config_file_handle.Read_Single( parameters->output_energies_step, "gneb_output_energies_step" );
            config_file_handle.Read_Single(
                parameters->output_energies_add_readability_lines, "gneb_output_energies_add_readability_lines" );
            config_file_handle.Read_Single(
                parameters->output_energies_interpolated, "gneb_output_energies_interpolated" );
            config_file_handle.Read_Single(
                parameters->output_energies_divide_by_nspins, "gneb_output_energies_divide_by_nspins" );
            config_file_handle.Read_Single( parameters->output_chain_step, "gneb_output_chain_step" );
            config_file_handle.Read_Single( output_chain_filetype, "gneb_output_chain_filetype" );
            parameters->output_vf_filetype = IO::VF_FileFormat( output_chain_filetype );
            // Method parameters
            config_file_handle.Read_Single( str_max_walltime, "gneb_max_walltime" );
            parameters->max_walltime_sec = (long int)Utility::Timing::DurationFromString( str_max_walltime ).count();
            config_file_handle.Read_Single( parameters->spring_constant, "gneb_spring_constant" );
            config_file_handle.Read_Single( parameters->force_convergence, "gneb_force_convergence" );
            config_file_handle.Read_Single( parameters->n_iterations, "gneb_n_iterations" );
            config_file_handle.Read_Single( parameters->n_iterations_log, "gneb_n_iterations_log" );
            config_file_handle.Read_Single( parameters->n_iterations_amortize, "gneb_n_iterations_amortize" );
            config_file_handle.Read_Single( parameters->n_E_interpolations, "gneb_n_energy_interpolations" );
            config_file_handle.Read_Single( parameters->moving_endpoints, "gneb_moving_endpoints" );
            config_file_handle.Read_Single( parameters->equilibrium_delta_Rx_left, "gneb_equilibrium_delta_Rx_left" );
            config_file_handle.Read_Single( parameters->equilibrium_delta_Rx_right, "gneb_equilibrium_delta_Rx_right" );
            config_file_handle.Read_Single( parameters->translating_endpoints, "gneb_translating_endpoints" );
        }
        catch( ... )
        {
            spirit_handle_exception_core(
                fmt::format( "Unable to parse GNEB parameters from config file \"{}\"", config_file_name ) );
        }
    }
    else
        Log( Log_Level::Parameter, Log_Sender::IO, "Parameters GNEB: Using default configuration!" );

    // Return
    std::vector<std::string> parameter_log;
    parameter_log.emplace_back( "Parameters GNEB:" );
    parameter_log.emplace_back( fmt::format( "    {:<18} = {}", "spring_constant", parameters->spring_constant ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<18} = {}", "n_E_interpolations", parameters->n_E_interpolations ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<18} = {:e}", "force convergence", parameters->force_convergence ) );
    parameter_log.emplace_back( fmt::format( "    {:<18} = {}", "maximum walltime", str_max_walltime ) );
    parameter_log.emplace_back( fmt::format( "    {:<18} = {}", "n_iterations", parameters->n_iterations ) );
    parameter_log.emplace_back( fmt::format( "    {:<18} = {}", "n_iterations_log", parameters->n_iterations_log ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<18} = {}", "n_iterations_amortize", parameters->n_iterations_amortize ) );
    parameter_log.emplace_back( fmt::format( "    {:<18} = {}", "moving_endpoints", parameters->moving_endpoints ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<18} = {}", "equilibrium_delta_Rx_left", parameters->equilibrium_delta_Rx_left ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<18} = {}", "equilibrium_delta_Rx_right", parameters->equilibrium_delta_Rx_right ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<18} = {}", "translating_endpoints", parameters->translating_endpoints ) );
    parameter_log.emplace_back( fmt::format( "    {:<18} = \"{}\"", "output_folder", parameters->output_folder ) );
    parameter_log.emplace_back( fmt::format( "    {:<18} = {}", "output_any", parameters->output_any ) );
    parameter_log.emplace_back( fmt::format( "    {:<18} = {}", "output_initial", parameters->output_initial ) );
    parameter_log.emplace_back( fmt::format( "    {:<18} = {}", "output_final", parameters->output_final ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<18} = {}", "output_energies_step", parameters->output_energies_step ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<18} = {}", "output_energies_add_readability_lines",
        parameters->output_energies_add_readability_lines ) );
    parameter_log.emplace_back( fmt::format( "    {:<18} = {}", "output_chain_step", parameters->output_chain_step ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<18} = {}", "output_chain_filetype", (int)parameters->output_vf_filetype ) );
    Log( Log_Level::Parameter, Log_Sender::IO, parameter_log );

    Log( Log_Level::Debug, Log_Sender::IO, "Parameters GNEB: built" );
    return parameters;
} // end Parameters_Method_LLG_from_Config

std::unique_ptr<Data::Parameters_Method_MMF> Parameters_Method_MMF_from_Config( const std::string & config_file_name )
{
    // Default parameters
    auto parameters = std::make_unique<Data::Parameters_Method_MMF>();

    // Maximum wall time
    std::string str_max_walltime = "0";

    // Configuration output filetype
    int output_configuration_filetype = (int)parameters->output_vf_filetype;

    // Parse
    Log( Log_Level::Debug, Log_Sender::IO, "Parameters MMF: building" );
    if( !config_file_name.empty() )
    {
        try
        {
            IO::Filter_File_Handle config_file_handle( config_file_name );

            // Output parameters
            config_file_handle.Read_Single( parameters->output_file_tag, "output_file_tag" );
            config_file_handle.Read_Single( parameters->output_folder, "mmf_output_folder" );
            config_file_handle.Read_Single( parameters->output_any, "mmf_output_any" );
            config_file_handle.Read_Single( parameters->output_initial, "mmf_output_initial" );
            config_file_handle.Read_Single( parameters->output_final, "mmf_output_final" );
            config_file_handle.Read_Single( parameters->output_energy_step, "mmf_output_energy_step" );
            config_file_handle.Read_Single( parameters->output_energy_archive, "mmf_output_energy_archive" );
            config_file_handle.Read_Single(
                parameters->output_energy_divide_by_nspins, "mmf_output_energy_divide_by_nspins" );
            config_file_handle.Read_Single(
                parameters->output_energy_add_readability_lines, "mmf_output_energy_add_readability_lines" );
            config_file_handle.Read_Single( parameters->output_configuration_step, "mmf_output_configuration_step" );
            config_file_handle.Read_Single(
                parameters->output_configuration_archive, "mmf_output_configuration_archive" );
            config_file_handle.Read_Single( output_configuration_filetype, "mmf_output_configuration_filetype" );
            parameters->output_vf_filetype = IO::VF_FileFormat( output_configuration_filetype );
            // Method parameters
            config_file_handle.Read_Single( str_max_walltime, "mmf_max_walltime" );
            parameters->max_walltime_sec = (long int)Utility::Timing::DurationFromString( str_max_walltime ).count();
            config_file_handle.Read_Single( parameters->force_convergence, "mmf_force_convergence" );
            config_file_handle.Read_Single( parameters->n_iterations, "mmf_n_iterations" );
            config_file_handle.Read_Single( parameters->n_iterations_log, "mmf_n_iterations_log" );
            config_file_handle.Read_Single( parameters->n_iterations_amortize, "mmf_n_iterations_amortize" );
            config_file_handle.Read_Single( parameters->n_modes, "mmf_n_modes" );
            config_file_handle.Read_Single( parameters->n_mode_follow, "mmf_n_mode_follow" );
        }
        catch( ... )
        {
            spirit_handle_exception_core(
                fmt::format( "Unable to parse MMF parameters from config file \"{}\"", config_file_name ) );
        }
    }
    else
        Log( Log_Level::Parameter, Log_Sender::IO, "Parameters MMF: Using default configuration!" );

    // Return
    std::vector<std::string> parameter_log;
    parameter_log.emplace_back( "Parameters MMF:" );
    parameter_log.emplace_back(
        fmt::format( "    {:<17} = {:e}", "force convergence", parameters->force_convergence ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "maximum walltime", str_max_walltime ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "n_iterations", parameters->n_iterations ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "n_iterations_log", parameters->n_iterations_log ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<17} = {}", "n_iterations_amortize", parameters->n_iterations_amortize ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = \"{}\"", "output_folder", parameters->output_folder ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output_any", parameters->output_any ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output_initial", parameters->output_initial ) );
    parameter_log.emplace_back( fmt::format( "    {:<17} = {}", "output_final", parameters->output_final ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_energy_step", parameters->output_energy_step ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_energy_archive", parameters->output_energy_archive ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<30} = {}", "output_energy_divide_by_nspins", parameters->output_energy_divide_by_nspins ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<30} = {}", "output_energy_add_readability_lines", parameters->output_energy_add_readability_lines ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_configuration_step", parameters->output_configuration_step ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_configuration_archive", parameters->output_configuration_archive ) );
    parameter_log.emplace_back(
        fmt::format( "    {:<30} = {}", "output_configuration_filetype", (int)parameters->output_vf_filetype ) );
    Log( Log_Level::Parameter, Log_Sender::IO, parameter_log );

    Log( Log_Level::Debug, Log_Sender::IO, "Parameters MMF: built" );
    return parameters;
}

} // namespace IO
