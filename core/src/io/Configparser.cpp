#include <engine/Neighbours.hpp>
#include <engine/Vectormath.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/IO.hpp>
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

void Log_from_Config( const std::string & config_file, bool force_quiet )
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
    if( !config_file.empty() )
    {
        try
        {
            Log( Log_Level::Debug, Log_Sender::IO, "Building Log" );
            IO::Filter_File_Handle myfile( config_file );

            // Time tag
            myfile.Read_Single( file_tag, "output_file_tag" );

            // Output folder
            myfile.Read_Single( output_folder, "log_output_folder" );

            // Save Output (Log Messages) to file
            myfile.Read_Single( messages_to_file, "log_to_file" );
            // File Accept Level
            myfile.Read_Single( i_level_file, "log_file_level" );

            // Print Output (Log Messages) to console
            myfile.Read_Single( messages_to_console, "log_to_console" );
            // File Accept Level
            myfile.Read_Single( i_level_console, "log_console_level" );

            // Save Input (parameters from config file and defaults) on State Setup
            myfile.Read_Single( save_input_initial, "save_input_initial" );
            // Save Input (parameters from config file and defaults) on State Delete
            myfile.Read_Single( save_input_final, "save_input_final" );

            // Save Input (parameters from config file and defaults) on State Setup
            myfile.Read_Single( save_positions_initial, "save_positions_initial" );
            // Save Input (parameters from config file and defaults) on State Delete
            myfile.Read_Single( save_positions_final, "save_positions_final" );

            // Save Input (parameters from config file and defaults) on State Setup
            myfile.Read_Single( save_neighbours_initial, "save_neighbours_initial" );
            // Save Input (parameters from config file and defaults) on State Delete
            myfile.Read_Single( save_neighbours_final, "save_neighbours_final" );
        }
        catch( ... )
        {
            spirit_rethrow(
                fmt::format( "Failed to read log levels from file \"{}\". Leaving values at default.", config_file ) );
        }
    }

    // Log the parameters
    std::vector<std::string> block;
    block.push_back( "Logging parameters" );
    block.push_back( fmt::format( "    file tag on output = \"{}\"", file_tag ) );
    block.push_back( fmt::format( "    output folder      = \"{}\"", output_folder ) );
    block.push_back( fmt::format( "    to file            = {}", messages_to_file ) );
    block.push_back( fmt::format( "    file accept level  = {}", i_level_file ) );
    block.push_back( fmt::format( "    to console         = {}", messages_to_console ) );
    block.push_back( fmt::format( "    print accept level = {}", i_level_console ) );
    block.push_back( fmt::format( "    input save initial = {}", save_input_initial ) );
    block.push_back( fmt::format( "    input save final   = {}", save_input_final ) );
    block.push_back( fmt::format( "    positions save initial  = {}", save_positions_initial ) );
    block.push_back( fmt::format( "    positions save final    = {}", save_positions_final ) );
    block.push_back( fmt::format( "    neighbours save initial = {}", save_neighbours_initial ) );
    block.push_back( fmt::format( "    neighbours save final   = {}", save_neighbours_final ) );
    Log.SendBlock( Log_Level::Parameter, Log_Sender::IO, block );

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
        fmt::format( "Unable to read logging parameters from config file \"{}\"", config_file ) );
} // End Log_from_Config

std::unique_ptr<Data::Spin_System> Spin_System_from_Config( const std::string & config_file )
try
{
    Log( Log_Level::Info, Log_Sender::IO, "-------------- Initialising Spin System ------------" );

    // Geometry
    auto geometry = Geometry_from_Config( config_file );
    // LLG Parameters
    auto llg_params = Parameters_Method_LLG_from_Config( config_file );
    // MC Parameters
    auto mc_params = Parameters_Method_MC_from_Config( config_file );
    // EMA Parameters
    auto ema_params = Parameters_Method_EMA_from_Config( config_file );
    // MMF Parameters
    auto mmf_params = Parameters_Method_MMF_from_Config( config_file );
    // Hamiltonian
    auto hamiltonian = Hamiltonian_from_Config( config_file, geometry );
    // Spin System
    auto system = std::make_unique<Data::Spin_System>(
        std::move( hamiltonian ), std::move( geometry ), std::move( llg_params ), std::move( mc_params ),
        std::move( ema_params ), std::move( mmf_params ), false );

    Log( Log_Level::Info, Log_Sender::IO, "-------------- Spin System Initialised -------------" );

    return system;
}
catch( ... )
{
    spirit_handle_exception_core(
        fmt::format( "Unable to initialize spin system from config file \"{}\"", config_file ) );
    return nullptr;
} // End Spin_System_from_Config

void Bravais_Vectors_from_Config(
    const std::string & config_file, std::vector<Vector3> & bravais_vectors,
    Data::BravaisLatticeType & bravais_lattice_type, std::string & bravais_lattice_type_str )
try
{
    std::string bravais_lattice = "sc";
    // Manually specified bravais vectors/matrix?
    bool irregular = true;

    IO::Filter_File_Handle myfile( config_file );
    // Bravais lattice type or manually specified vectors/matrix
    if( myfile.Find( "bravais_lattice" ) )
    {
        myfile.iss >> bravais_lattice;
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
    else if( myfile.Find( "bravais_vectors" ) )
    {
        Log( Log_Level::Parameter, Log_Sender::IO, "Bravais lattice type: irregular" );
        bravais_lattice_type = Data::BravaisLatticeType::Irregular;
        myfile.GetLine();
        myfile.iss >> bravais_vectors[0][0] >> bravais_vectors[0][1] >> bravais_vectors[0][2];
        myfile.GetLine();
        myfile.iss >> bravais_vectors[1][0] >> bravais_vectors[1][1] >> bravais_vectors[1][2];
        myfile.GetLine();
        myfile.iss >> bravais_vectors[2][0] >> bravais_vectors[2][1] >> bravais_vectors[2][2];
    }
    else if( myfile.Find( "bravais_matrix" ) )
    {
        Log( Log_Level::Parameter, Log_Sender::IO, "Bravais lattice type: irregular" );
        bravais_lattice_type = Data::BravaisLatticeType::Irregular;
        myfile.GetLine();
        myfile.iss >> bravais_vectors[0][0] >> bravais_vectors[1][0] >> bravais_vectors[2][0];
        myfile.GetLine();
        myfile.iss >> bravais_vectors[0][1] >> bravais_vectors[1][1] >> bravais_vectors[2][1];
        myfile.GetLine();
        myfile.iss >> bravais_vectors[0][2] >> bravais_vectors[1][2] >> bravais_vectors[2][2];
    }
    else
        Log( Log_Level::Parameter, Log_Sender::IO, "Bravais lattice not specified. Using simple cubic..." );
}
catch( ... )
{
    spirit_rethrow( fmt::format( "Unable to parse bravais vectors from config file \"{}\"", config_file ) );
} // End Basis_from_Config

std::shared_ptr<Data::Geometry> Geometry_from_Config( const std::string & config_file )
try
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
    int n_cell_atoms                = cell_atoms.size();
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

    //------------------------------- Parser --------------------------------
    Log( Log_Level::Debug, Log_Sender::IO, "Geometry: building" );
    if( !config_file.empty() )
    {
        try
        {
            IO::Filter_File_Handle myfile( config_file );

            // Lattice constant
            myfile.Read_Single( lattice_constant, "lattice_constant" );

            // Get the bravais lattice type and vectors
            Bravais_Vectors_from_Config( config_file, bravais_vectors, bravais_lattice_type, bravais_lattice_type_str );

            // Read basis cell
            if( myfile.Find( "basis" ) )
            {
                // Read number of atoms in the basis cell
                myfile.GetLine();
                myfile.iss >> n_cell_atoms;
                cell_atoms = std::vector<Vector3>( n_cell_atoms );
                cell_composition.iatom.resize( n_cell_atoms );
                cell_composition.atom_type = std::vector<int>( n_cell_atoms, 0 );
                cell_composition.mu_s      = std::vector<scalar>( n_cell_atoms, 1 );

                // Read atom positions
                for( int iatom = 0; iatom < n_cell_atoms; ++iatom )
                {
                    myfile.GetLine();
                    myfile.iss >> cell_atoms[iatom][0] >> cell_atoms[iatom][1] >> cell_atoms[iatom][2];
                    cell_composition.iatom[iatom] = iatom;
                }
            }

            // Read number of basis cells
            myfile.Read_3Vector( n_cells, "n_basis_cells" );

// Defects
#ifdef SPIRIT_ENABLE_DEFECTS
            int n_defects = 0;

            std::string defects_file = "";
            if( myfile.Find( "n_defects" ) )
                defects_file = config_file;
            else if( myfile.Find( "defects_from_file" ) )
                myfile.iss >> defects_file;

            if( defects_file.length() > 0 )
            {
                // The file name should be valid so we try to read it
                Defects_from_File( defects_file, n_defects, defect_sites, defect_types );
            }

            // Disorder
            if( myfile.Find( "atom_types" ) )
            {
                myfile.iss >> n_atom_types;
                cell_composition.disordered = true;
                cell_composition.iatom.resize( n_atom_types );
                cell_composition.atom_type.resize( n_atom_types );
                cell_composition.mu_s.resize( n_atom_types );
                cell_composition.concentration.resize( n_atom_types );
                for( int itype = 0; itype < n_atom_types; ++itype )
                {
                    myfile.GetLine();
                    myfile.iss >> cell_composition.iatom[itype];
                    myfile.iss >> cell_composition.atom_type[itype];
                    myfile.iss >> cell_composition.mu_s[itype];
                    myfile.iss >> cell_composition.concentration[itype];
                    // if ( !(myfile.iss >> mu_s[itype]) )
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
#else
            Log( Log_Level::Parameter, Log_Sender::IO, "Disorder is disabled" );
#endif
        }
        catch( ... )
        {
            spirit_handle_exception_core( fmt::format(
                "Failed to read Geometry parameters from file \"{}\". Leaving values at default.", config_file ) );
        }

        try
        {
            IO::Filter_File_Handle myfile( config_file );

            // Spin moment
            if( !myfile.Find( "atom_types" ) )
            {
                if( myfile.Find( "mu_s" ) )
                {
                    for( int iatom = 0; iatom < n_cell_atoms; ++iatom )
                    {
                        if( !( myfile.iss >> cell_composition.mu_s[iatom] ) )
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
                    Log( Log_Level::Error, Log_Sender::IO,
                         fmt::format( "Keyword 'mu_s' not found. Using Default: {}", cell_composition.mu_s[0] ) );
            }
            // else
            // {
            //     cell_composition.mu_s = std::vector<scalar>(n_atom_types, 1);
            //     if( myfile.Find("mu_s") )
            //     {
            //         for (int itype = 0; itype < n_atom_types; ++itype)
            //         {
            //             myfile.iss >> cell_composition.mu_s[itype];
            //             // myfile.GetLine();
            //             // myfile.iss >> cell_composition.iatom[itype];
            //             // myfile.iss >> cell_composition.atom_type[itype];
            //             // myfile.iss >> cell_composition.concentration[itype];
            //         }
            //     }
            //     else Log(Log_Level::Error, Log_Sender::IO, fmt::format("Keyword 'mu_s' not found. Using Default:
            //     {}", cell_composition.mu_s[0]));
            // }
        }
        catch( ... )
        {
            spirit_handle_exception_core( fmt::format( "Unable to read mu_s from config file \"{}\"", config_file ) );
        }
    } // end if file=""
    else
        Log( Log_Level::Parameter, Log_Sender::IO, "Geometry: Using default configuration!" );

    // Log the parameters
    std::vector<std::string> parameter_log;
    parameter_log.push_back( "Geometry:" );
    parameter_log.push_back( fmt::format( "    lattice constant = {} Angstrom", lattice_constant ) );
    parameter_log.push_back( fmt::format( "    Bravais lattice type: {}", bravais_lattice_type_str ) );
    Log( Log_Level::Debug, Log_Sender::IO, "    Bravais vectors in units of lattice constant" );
    Log( Log_Level::Debug, Log_Sender::IO,
         fmt::format( "        a = {}", bravais_vectors[0].transpose() / lattice_constant ) );
    Log( Log_Level::Debug, Log_Sender::IO,
         fmt::format( "        b = {}", bravais_vectors[1].transpose() / lattice_constant ) );
    Log( Log_Level::Debug, Log_Sender::IO,
         fmt::format( "        c = {}", bravais_vectors[2].transpose() / lattice_constant ) );
    parameter_log.push_back( "    Bravais vectors" );
    parameter_log.push_back( fmt::format( "        a = {}", bravais_vectors[0].transpose() ) );
    parameter_log.push_back( fmt::format( "        b = {}", bravais_vectors[1].transpose() ) );
    parameter_log.push_back( fmt::format( "        c = {}", bravais_vectors[2].transpose() ) );
    parameter_log.push_back( fmt::format( "    basis cell: {} atom(s)", n_cell_atoms ) );
    parameter_log.push_back( "    relative positions (first 10):" );
    for( int iatom = 0; iatom < n_cell_atoms && iatom < 10; ++iatom )
        parameter_log.push_back( fmt::format(
            "        atom {} at ({}), mu_s={}", iatom, cell_atoms[iatom].transpose(), cell_composition.mu_s[iatom] ) );

    parameter_log.push_back( "    absolute atom positions (first 10):" );
    for( int iatom = 0; iatom < n_cell_atoms && iatom < 10; ++iatom )
    {
        Vector3 cell_atom = lattice_constant
                            * ( bravais_vectors[0] * cell_atoms[iatom][0] + bravais_vectors[1] * cell_atoms[iatom][1]
                                + bravais_vectors[2] * cell_atoms[iatom][2] );
        parameter_log.push_back( fmt::format( "        atom {} at ({})", iatom, cell_atom.transpose() ) );
    }

    if( cell_composition.disordered )
        parameter_log.push_back( "    note: the lattice has some disorder!" );

// Defects
#ifdef SPIRIT_ENABLE_DEFECTS
    parameter_log.push_back( fmt::format( "    {} defects. Printing the first 10:", defect_sites.size() ) );
    for( int i = 0; i < defect_sites.size(); ++i )
        if( i < 10 )
            parameter_log.push_back( fmt::format(
                "  defect[{}]: translations=({} {} {}), type=", i, defect_sites[i].translations[0],
                defect_sites[i].translations[1], defect_sites[i].translations[2], defect_types[i] ) );
#endif

    // Log parameters
    parameter_log.push_back( "    lattice: n_basis_cells" );
    parameter_log.push_back( fmt::format( "        na = {}", n_cells[0] ) );
    parameter_log.push_back( fmt::format( "        nb = {}", n_cells[1] ) );
    parameter_log.push_back( fmt::format( "        nc = {}", n_cells[2] ) );

    // Pinning configuration
    auto pinning = Pinning_from_Config( config_file, cell_atoms.size() );

    // Return geometry
    auto geometry = std::make_shared<Data::Geometry>(
        bravais_vectors, n_cells, cell_atoms, cell_composition, lattice_constant, pinning,
        Data::Defects{ defect_sites, defect_types } );

    parameter_log.push_back( fmt::format( "    {} spins", geometry->nos ) );
    parameter_log.push_back( fmt::format( "    the geometry is {}-dimensional", geometry->dimensionality ) );

    Log.SendBlock( Log_Level::Parameter, Log_Sender::IO, parameter_log );

    Log( Log_Level::Debug, Log_Sender::IO, "Geometry: built" );
    return geometry;
}
catch( ... )
{
    spirit_rethrow( fmt::format( "Unable to parse geometry from config file \"{}\"", config_file ) );
    return nullptr;
} // End Geometry from Config

Data::Pinning Pinning_from_Config( const std::string & config_file, std::size_t n_cell_atoms )
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
    Log( Log_Level::Parameter, Log_Sender::IO, "Reading Pinning Configuration" );
    //------------------------------- Parser --------------------------------
    if( !config_file.empty() )
    {
        try
        {
            IO::Filter_File_Handle myfile( config_file );

            // N_a
            myfile.Read_Single( na_left, "pin_na_left", false );
            myfile.Read_Single( na_right, "pin_na_right", false );
            myfile.Read_Single( na, "pin_na ", false );
            if( na > 0 && ( na_left == 0 || na_right == 0 ) )
            {
                na_left  = na;
                na_right = na;
            }

            // N_b
            myfile.Read_Single( nb_left, "pin_nb_left", false );
            myfile.Read_Single( nb_right, "pin_nb_right", false );
            myfile.Read_Single( nb, "pin_nb ", false );
            if( nb > 0 && ( nb_left == 0 || nb_right == 0 ) )
            {
                nb_left  = nb;
                nb_right = nb;
            }

            // N_c
            myfile.Read_Single( nc_left, "pin_nc_left", false );
            myfile.Read_Single( nc_right, "pin_nc_right", false );
            myfile.Read_Single( nc, "pin_nc ", false );
            if( nc > 0 && ( nc_left == 0 || nc_right == 0 ) )
            {
                nc_left  = nc;
                nc_right = nc;
            }

            // How should the cells be pinned
            if( na_left > 0 || na_right > 0 || nb_left > 0 || nb_right > 0 || nc_left > 0 || nc_right > 0 )
            {
                if( myfile.Find( "pinning_cell" ) )
                {
                    for( int i = 0; i < n_cell_atoms; ++i )
                    {
                        myfile.GetLine();
                        myfile.iss >> pinned_cell[i][0] >> pinned_cell[i][1] >> pinned_cell[i][2];
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
            if( myfile.Find( "n_pinned" ) )
                pinned_file = config_file;
            else if( myfile.Find( "pinned_from_file" ) )
                myfile.iss >> pinned_file;

            if( !pinned_file.empty() )
            {
                // The file name should be valid so we try to read it
                Pinned_from_File( pinned_file, n_pinned, pinned_sites, pinned_spins );
            }
            else
                Log( Log_Level::Parameter, Log_Sender::IO, "wtf no pinnedFile" );
        }
        catch( ... )
        {
            spirit_handle_exception_core(
                fmt::format( "Failed to read Pinning from file \"{}\". Leaving values at default.", config_file ) );
        }

    } // end if file=""
    else
        Log( Log_Level::Parameter, Log_Sender::IO, "No pinning" );

    // Create Pinning
    auto pinning = Data::Pinning{ na_left,  na_right,    nb_left,      nb_right,    nc_left,
                                  nc_right, pinned_cell, pinned_sites, pinned_spins };

    // Return Pinning
    Log( Log_Level::Parameter, Log_Sender::IO, "Pinning:" );
    Log( Log_Level::Parameter, Log_Sender::IO, fmt::format( "    n_a: left={}, right={}", na_left, na_right ) );
    Log( Log_Level::Parameter, Log_Sender::IO, fmt::format( "    n_b: left={}, right={}", nb_left, nb_right ) );
    Log( Log_Level::Parameter, Log_Sender::IO, fmt::format( "    n_c: left={}, right={}", nc_left, nc_right ) );
    for( int i = 0; i < n_cell_atoms; ++i )
        Log( Log_Level::Parameter, Log_Sender::IO,
             fmt::format( "    cell atom[{}] = ({})", i, pinned_cell[0].transpose() ) );
    Log( Log_Level::Parameter, Log_Sender::IO,
         fmt::format( "    {} additional pinned sites. Showing the first 10:", n_pinned ) );
    for( int i = 0; i < n_pinned; ++i )
    {
        if( i < 10 )
            Log( Log_Level::Parameter, Log_Sender::IO,
                 fmt::format(
                     "         pinned site[{}]: {} at ({} {} {}) = ({})", i, pinned_sites[i].i,
                     pinned_sites[i].translations[0], pinned_sites[i].translations[1], pinned_sites[i].translations[2],
                     pinned_spins[0].transpose() ) );
    }
    Log( Log_Level::Parameter, Log_Sender::IO, "Pinning: read" );
    return pinning;
#else  // SPIRIT_ENABLE_PINNING
    Log( Log_Level::Parameter, Log_Sender::IO, "Pinning is disabled" );
    if( !config_file.empty() )
    {
        try
        {
            IO::Filter_File_Handle myfile( config_file );
            if( myfile.Find( "pinning_cell" ) )
                Log( Log_Level::Warning, Log_Sender::IO,
                     "You specified a pinning cell even though pinning is disabled!" );
        }
        catch( ... )
        {
            spirit_handle_exception_core( fmt::format(
                "Failed to read pinning parameters from file \"{}\". Leaving values at default.", config_file ) );
        }
    }

    return Data::Pinning{ 0, 0, 0, 0, 0, 0, vectorfield( 0 ), field<Site>( 0 ), vectorfield( 0 ) };
#endif // SPIRIT_ENABLE_PINNING
}

std::unique_ptr<Data::Parameters_Method_LLG> Parameters_Method_LLG_from_Config( const std::string & config_file )
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
    if( !config_file.empty() )
    {
        try
        {
            IO::Filter_File_Handle myfile( config_file );

            // Output parameters
            myfile.Read_Single( parameters->output_file_tag, "output_file_tag" );
            myfile.Read_Single( parameters->output_folder, "llg_output_folder" );
            myfile.Read_Single( parameters->output_any, "llg_output_any" );
            myfile.Read_Single( parameters->output_initial, "llg_output_initial" );
            myfile.Read_Single( parameters->output_final, "llg_output_final" );
            myfile.Read_Single( parameters->output_energy_spin_resolved, "llg_output_energy_spin_resolved" );
            myfile.Read_Single( parameters->output_energy_step, "llg_output_energy_step" );
            myfile.Read_Single( parameters->output_energy_archive, "llg_output_energy_archive" );
            myfile.Read_Single( parameters->output_energy_divide_by_nspins, "llg_output_energy_divide_by_nspins" );
            myfile.Read_Single(
                parameters->output_energy_add_readability_lines, "llg_output_energy_add_readability_lines" );
            myfile.Read_Single( parameters->output_configuration_step, "llg_output_configuration_step" );
            myfile.Read_Single( parameters->output_configuration_archive, "llg_output_configuration_archive" );
            myfile.Read_Single( output_configuration_filetype, "llg_output_configuration_filetype" );
            parameters->output_vf_filetype = IO::VF_FileFormat( output_configuration_filetype );
            // Method parameters
            myfile.Read_Single( str_max_walltime, "llg_max_walltime" );
            parameters->max_walltime_sec = (long int)Utility::Timing::DurationFromString( str_max_walltime ).count();
            myfile.Read_Single( parameters->rng_seed, "llg_seed" );
            parameters->prng = std::mt19937( parameters->rng_seed );
            myfile.Read_Single( parameters->n_iterations, "llg_n_iterations" );
            myfile.Read_Single( parameters->n_iterations_log, "llg_n_iterations_log" );
            myfile.Read_Single( parameters->dt, "llg_dt" );
            myfile.Read_Single( parameters->temperature, "llg_temperature" );
            myfile.Read_Vector3( parameters->temperature_gradient_direction, "llg_temperature_gradient_direction" );
            parameters->temperature_gradient_direction.normalize();
            myfile.Read_Single( parameters->temperature_gradient_inclination, "llg_temperature_gradient_inclination" );
            myfile.Read_Single( parameters->damping, "llg_damping" );
            myfile.Read_Single( parameters->beta, "llg_beta" );
            // myfile.Read_Single(parameters->renorm_sd, "llg_renorm");
            myfile.Read_Single( parameters->stt_use_gradient, "llg_stt_use_gradient" );
            myfile.Read_Single( parameters->stt_magnitude, "llg_stt_magnitude" );
            myfile.Read_Vector3( parameters->stt_polarisation_normal, "llg_stt_polarisation_normal" );
            parameters->stt_polarisation_normal.normalize();
            myfile.Read_Single( parameters->force_convergence, "llg_force_convergence" );
        }
        catch( ... )
        {
            spirit_handle_exception_core(
                fmt::format( "Unable to parse LLG parameters from config file \"{}\"", config_file ) );
        }
    }
    else
        Log( Log_Level::Parameter, Log_Sender::IO, "Parameters LLG: Using default configuration!" );

    // Return
    std::vector<std::string> parameter_log;
    parameter_log.push_back( "Parameters LLG:" );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "seed", parameters->rng_seed ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "time step [ps]", parameters->dt ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "temperature [K]", parameters->temperature ) );
    parameter_log.push_back( fmt::format(
        "    {:<17} = {}", "temperature gradient direction", parameters->temperature_gradient_direction.transpose() ) );
    parameter_log.push_back( fmt::format(
        "    {:<17} = {}", "temperature gradient inclination", parameters->temperature_gradient_inclination ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "damping", parameters->damping ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "beta", parameters->beta ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "stt use gradient", parameters->stt_use_gradient ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "stt magnitude", parameters->stt_magnitude ) );
    parameter_log.push_back(
        fmt::format( "    {:<17} = {}", "stt normal", parameters->stt_polarisation_normal.transpose() ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {:e}", "force convergence", parameters->force_convergence ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "maximum walltime", str_max_walltime ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "n_iterations", parameters->n_iterations ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "n_iterations_log", parameters->n_iterations_log ) );
    parameter_log.push_back( fmt::format( "    {:<17} = \"{}\"", "output_folder", parameters->output_folder ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "output_any", parameters->output_any ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "output_initial", parameters->output_initial ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "output_final", parameters->output_final ) );
    parameter_log.push_back( fmt::format( "    {:<30} = {}", "output_energy_step", parameters->output_energy_step ) );
    parameter_log.push_back(
        fmt::format( "    {:<30} = {}", "output_energy_archive", parameters->output_energy_archive ) );
    parameter_log.push_back(
        fmt::format( "    {:<30} = {}", "output_energy_spin_resolved", parameters->output_energy_spin_resolved ) );
    parameter_log.push_back( fmt::format(
        "    {:<30} = {}", "output_energy_divide_by_nspins", parameters->output_energy_divide_by_nspins ) );
    parameter_log.push_back( fmt::format(
        "    {:<30} = {}", "output_energy_add_readability_lines", parameters->output_energy_add_readability_lines ) );
    parameter_log.push_back(
        fmt::format( "    {:<30} = {}", "output_configuration_step", parameters->output_configuration_step ) );
    parameter_log.push_back(
        fmt::format( "    {:<30} = {}", "output_configuration_archive", parameters->output_configuration_archive ) );
    parameter_log.push_back(
        fmt::format( "    {:<30} = {}", "output_configuration_filetype", (int)parameters->output_vf_filetype ) );
    Log.SendBlock( Log_Level::Parameter, Log_Sender::IO, parameter_log );

    Log( Log_Level::Debug, Log_Sender::IO, "Parameters LLG: built" );
    return parameters;
} // end Parameters_Method_LLG_from_Config

std::unique_ptr<Data::Parameters_Method_EMA> Parameters_Method_EMA_from_Config( const std::string & config_file )
{
    // Default parameters
    auto parameters = std::make_unique<Data::Parameters_Method_EMA>();

    // Maximum wall time
    std::string str_max_walltime = "0";

    // Parse
    Log( Log_Level::Debug, Log_Sender::IO, "Parameters EMA: building" );
    if( !config_file.empty() )
    {
        try
        {
            IO::Filter_File_Handle myfile( config_file );

            // Output parameters
            myfile.Read_Single( parameters->output_folder, "ema_output_folder" );
            myfile.Read_Single( parameters->output_file_tag, "output_file_tag" );
            myfile.Read_Single( parameters->output_any, "ema_output_any" );
            myfile.Read_Single( parameters->output_initial, "ema_output_initial" );
            myfile.Read_Single( parameters->output_final, "ema_output_final" );
            myfile.Read_Single( parameters->output_energy_divide_by_nspins, "ema_output_energy_divide_by_nspins" );
            myfile.Read_Single( parameters->output_energy_spin_resolved, "ema_output_energy_spin_resolved" );
            myfile.Read_Single( parameters->output_energy_step, "ema_output_energy_step" );
            myfile.Read_Single( parameters->output_energy_archive, "ema_output_energy_archive" );
            myfile.Read_Single( parameters->output_configuration_step, "ema_output_configuration_step" );
            myfile.Read_Single( parameters->output_configuration_archive, "ema_output_configuration_archive" );
            // Method parameters
            myfile.Read_Single( str_max_walltime, "ema_max_walltime" );
            parameters->max_walltime_sec = (long int)Utility::Timing::DurationFromString( str_max_walltime ).count();
            myfile.Read_Single( parameters->n_iterations, "ema_n_iterations" );
            myfile.Read_Single( parameters->n_iterations_log, "ema_n_iterations_log" );
            myfile.Read_Single( parameters->n_modes, "ema_n_modes" );
            myfile.Read_Single( parameters->n_mode_follow, "ema_n_mode_follow" );
            myfile.Read_Single( parameters->frequency, "ema_frequency" );
            myfile.Read_Single( parameters->amplitude, "ema_amplitude" );
        }
        catch( ... )
        {
            spirit_handle_exception_core(
                fmt::format( "Unable to parse EMA parameters from config file \"{}\"", config_file ) );
        }
    }
    else
        Log( Log_Level::Parameter, Log_Sender::IO, "Parameters EMA: Using default configuration!" );

    // Return
    std::vector<std::string> parameter_log;
    parameter_log.push_back( "Parameters EMA:" );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "n_modes", parameters->n_modes ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "n_mode_follow", parameters->n_mode_follow ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "frequency", parameters->frequency ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "amplitude", parameters->amplitude ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "n_iterations_log", parameters->n_iterations_log ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "n_iterations", parameters->n_iterations ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "maximum walltime", str_max_walltime ) );
    parameter_log.push_back(
        fmt::format( "    {:<30} = {}", "output_configuration_archive", parameters->output_configuration_archive ) );
    parameter_log.push_back(
        fmt::format( "    {:<30} = {}", "output_configuration_step", parameters->output_configuration_step ) );
    parameter_log.push_back(
        fmt::format( "    {:<30} = {}", "output_energy_archive", parameters->output_energy_archive ) );
    parameter_log.push_back( fmt::format( "    {:<30} = {}", "output_energy_step", parameters->output_energy_step ) );
    parameter_log.push_back(
        fmt::format( "    {:<30} = {}", "output_energy_spin_resolved", parameters->output_energy_spin_resolved ) );
    parameter_log.push_back( fmt::format(
        "    {:<30} = {}", "output_energy_divide_by_nspins", parameters->output_energy_divide_by_nspins ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "output_final", parameters->output_final ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "output_initial", parameters->output_initial ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "output_any", parameters->output_any ) );
    parameter_log.push_back( fmt::format( "    {:<17} = \"{}\"", "output_folder", parameters->output_folder ) );
    Log.SendBlock( Log_Level::Parameter, Log_Sender::IO, parameter_log );

    Log( Log_Level::Debug, Log_Sender::IO, "Parameters EMA: built" );
    return parameters;
}

std::unique_ptr<Data::Parameters_Method_MC> Parameters_Method_MC_from_Config( const std::string & config_file )
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
    if( !config_file.empty() )
    {
        try
        {
            IO::Filter_File_Handle myfile( config_file );

            // Output parameters
            myfile.Read_Single( parameters->output_file_tag, "output_file_tag" );
            myfile.Read_Single( parameters->output_folder, "mc_output_folder" );
            myfile.Read_Single( parameters->output_any, "mc_output_any" );
            myfile.Read_Single( parameters->output_initial, "mc_output_initial" );
            myfile.Read_Single( parameters->output_final, "mc_output_final" );
            myfile.Read_Single( parameters->output_energy_spin_resolved, "mc_output_energy_spin_resolved" );
            myfile.Read_Single( parameters->output_energy_step, "mc_output_energy_step" );
            myfile.Read_Single( parameters->output_energy_archive, "mc_output_energy_archive" );
            myfile.Read_Single( parameters->output_energy_divide_by_nspins, "mc_output_energy_divide_by_nspins" );
            myfile.Read_Single(
                parameters->output_energy_add_readability_lines, "mc_output_energy_add_readability_lines" );
            myfile.Read_Single( parameters->output_configuration_step, "mc_output_configuration_step" );
            myfile.Read_Single( parameters->output_configuration_archive, "mc_output_configuration_archive" );
            myfile.Read_Single( output_configuration_filetype, "mc_output_configuration_filetype" );
            parameters->output_vf_filetype = IO::VF_FileFormat( output_configuration_filetype );
            // Method parameters
            myfile.Read_Single( str_max_walltime, "mc_max_walltime" );
            parameters->max_walltime_sec = (long int)Utility::Timing::DurationFromString( str_max_walltime ).count();
            myfile.Read_Single( parameters->rng_seed, "mc_seed" );
            parameters->prng = std::mt19937( parameters->rng_seed );
            myfile.Read_Single( parameters->n_iterations, "mc_n_iterations" );
            myfile.Read_Single( parameters->n_iterations_log, "mc_n_iterations_log" );
            myfile.Read_Single( parameters->temperature, "mc_temperature" );
            myfile.Read_Single( parameters->acceptance_ratio_target, "mc_acceptance_ratio" );
        }
        catch( ... )
        {
            spirit_handle_exception_core(
                fmt::format( "Unable to parse MC parameters from config file \"{}\"", config_file ) );
        }
    }
    else
        Log( Log_Level::Parameter, Log_Sender::IO, "Parameters MC: Using default configuration!" );

    // Return
    std::vector<std::string> parameter_log;
    parameter_log.push_back( "Parameters MC:" );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "seed", parameters->rng_seed ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "temperature", parameters->temperature ) );
    parameter_log.push_back(
        fmt::format( "    {:<17} = {}", "acceptance_ratio", parameters->acceptance_ratio_target ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "maximum walltime", str_max_walltime ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "n_iterations", parameters->n_iterations ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "n_iterations_log", parameters->n_iterations_log ) );
    parameter_log.push_back( fmt::format( "    {:<17} = \"{}\"", "output_folder", parameters->output_folder ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "output_any", parameters->output_any ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "output_initial", parameters->output_initial ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "output_final", parameters->output_final ) );
    parameter_log.push_back( fmt::format( "    {:<30} = {}", "output_energy_step", parameters->output_energy_step ) );
    parameter_log.push_back(
        fmt::format( "    {:<30} = {}", "output_energy_archive", parameters->output_energy_archive ) );
    parameter_log.push_back(
        fmt::format( "    {:<30} = {}", "output_energy_spin_resolved", parameters->output_energy_spin_resolved ) );
    parameter_log.push_back( fmt::format(
        "    {:<30} = {}", "output_energy_divide_by_nspins", parameters->output_energy_divide_by_nspins ) );
    parameter_log.push_back( fmt::format(
        "    {:<30} = {}", "output_energy_add_readability_lines", parameters->output_energy_add_readability_lines ) );
    parameter_log.push_back(
        fmt::format( "    {:<30} = {}", "output_configuration_step", parameters->output_configuration_step ) );
    parameter_log.push_back(
        fmt::format( "    {:<30} = {}", "output_configuration_archive", parameters->output_configuration_archive ) );
    parameter_log.push_back(
        fmt::format( "    {:<30} = {}", "output_configuration_filetype", (int)parameters->output_vf_filetype ) );
    Log.SendBlock( Log_Level::Parameter, Log_Sender::IO, parameter_log );

    Log( Log_Level::Debug, Log_Sender::IO, "Parameters MC: built" );
    return parameters;
}

std::unique_ptr<Data::Parameters_Method_GNEB> Parameters_Method_GNEB_from_Config( const std::string & config_file )
{
    // Default parameters
    auto parameters = std::make_unique<Data::Parameters_Method_GNEB>();

    // Maximum wall time
    std::string str_max_walltime = "0";

    // Chain output filetype
    int output_chain_filetype = (int)parameters->output_vf_filetype;

    // Parse
    Log( Log_Level::Debug, Log_Sender::IO, "Parameters GNEB: building" );
    if( !config_file.empty() )
    {
        try
        {
            IO::Filter_File_Handle myfile( config_file );

            // Output parameters
            myfile.Read_Single( parameters->output_file_tag, "output_file_tag" );
            myfile.Read_Single( parameters->output_folder, "gneb_output_folder" );
            myfile.Read_Single( parameters->output_any, "gneb_output_any" );
            myfile.Read_Single( parameters->output_initial, "gneb_output_initial" );
            myfile.Read_Single( parameters->output_final, "gneb_output_final" );
            myfile.Read_Single( parameters->output_energies_step, "gneb_output_energies_step" );
            myfile.Read_Single(
                parameters->output_energies_add_readability_lines, "gneb_output_energies_add_readability_lines" );
            myfile.Read_Single( parameters->output_energies_interpolated, "gneb_output_energies_interpolated" );
            myfile.Read_Single( parameters->output_energies_divide_by_nspins, "gneb_output_energies_divide_by_nspins" );
            myfile.Read_Single( parameters->output_chain_step, "gneb_output_chain_step" );
            myfile.Read_Single( output_chain_filetype, "gneb_output_chain_filetype" );
            parameters->output_vf_filetype = IO::VF_FileFormat( output_chain_filetype );
            // Method parameters
            myfile.Read_Single( str_max_walltime, "gneb_max_walltime" );
            parameters->max_walltime_sec = (long int)Utility::Timing::DurationFromString( str_max_walltime ).count();
            myfile.Read_Single( parameters->spring_constant, "gneb_spring_constant" );
            myfile.Read_Single( parameters->force_convergence, "gneb_force_convergence" );
            myfile.Read_Single( parameters->n_iterations, "gneb_n_iterations" );
            myfile.Read_Single( parameters->n_iterations_log, "gneb_n_iterations_log" );
            myfile.Read_Single( parameters->n_E_interpolations, "gneb_n_energy_interpolations" );
        }
        catch( ... )
        {
            spirit_handle_exception_core(
                fmt::format( "Unable to parse GNEB parameters from config file \"{}\"", config_file ) );
        }
    }
    else
        Log( Log_Level::Parameter, Log_Sender::IO, "Parameters GNEB: Using default configuration!" );

    // Return
    std::vector<std::string> parameter_log;
    parameter_log.push_back( "Parameters GNEB:" );
    parameter_log.push_back( fmt::format( "    {:<18} = {}", "spring_constant", parameters->spring_constant ) );
    parameter_log.push_back( fmt::format( "    {:<18} = {}", "n_E_interpolations", parameters->n_E_interpolations ) );
    parameter_log.push_back( fmt::format( "    {:<18} = {:e}", "force convergence", parameters->force_convergence ) );
    parameter_log.push_back( fmt::format( "    {:<18} = {}", "maximum walltime", str_max_walltime ) );
    parameter_log.push_back( fmt::format( "    {:<18} = {}", "n_iterations", parameters->n_iterations ) );
    parameter_log.push_back( fmt::format( "    {:<18} = {}", "n_iterations_log", parameters->n_iterations_log ) );
    parameter_log.push_back( fmt::format( "    {:<18} = \"{}\"", "output_folder", parameters->output_folder ) );
    parameter_log.push_back( fmt::format( "    {:<18} = {}", "output_any", parameters->output_any ) );
    parameter_log.push_back( fmt::format( "    {:<18} = {}", "output_initial", parameters->output_initial ) );
    parameter_log.push_back( fmt::format( "    {:<18} = {}", "output_final", parameters->output_final ) );
    parameter_log.push_back(
        fmt::format( "    {:<18} = {}", "output_energies_step", parameters->output_energies_step ) );
    parameter_log.push_back( fmt::format(
        "    {:<18} = {}", "output_energies_add_readability_lines",
        parameters->output_energies_add_readability_lines ) );
    parameter_log.push_back( fmt::format( "    {:<18} = {}", "output_chain_step", parameters->output_chain_step ) );
    parameter_log.push_back(
        fmt::format( "    {:<18} = {}", "output_chain_filetype", (int)parameters->output_vf_filetype ) );
    Log.SendBlock( Log_Level::Parameter, Log_Sender::IO, parameter_log );

    Log( Log_Level::Debug, Log_Sender::IO, "Parameters GNEB: built" );
    return parameters;
} // end Parameters_Method_LLG_from_Config

std::unique_ptr<Data::Parameters_Method_MMF> Parameters_Method_MMF_from_Config( const std::string & config_file )
{
    // Default parameters
    auto parameters = std::make_unique<Data::Parameters_Method_MMF>();

    // Maximum wall time
    std::string str_max_walltime = "0";

    // Configuration output filetype
    int output_configuration_filetype = (int)parameters->output_vf_filetype;

    // Parse
    Log( Log_Level::Debug, Log_Sender::IO, "Parameters MMF: building" );
    if( !config_file.empty() )
    {
        try
        {
            IO::Filter_File_Handle myfile( config_file );

            // Output parameters
            myfile.Read_Single( parameters->output_file_tag, "output_file_tag" );
            myfile.Read_Single( parameters->output_folder, "mmf_output_folder" );
            myfile.Read_Single( parameters->output_any, "mmf_output_any" );
            myfile.Read_Single( parameters->output_initial, "mmf_output_initial" );
            myfile.Read_Single( parameters->output_final, "mmf_output_final" );
            myfile.Read_Single( parameters->output_energy_step, "mmf_output_energy_step" );
            myfile.Read_Single( parameters->output_energy_archive, "mmf_output_energy_archive" );
            myfile.Read_Single( parameters->output_energy_divide_by_nspins, "mmf_output_energy_divide_by_nspins" );
            myfile.Read_Single(
                parameters->output_energy_add_readability_lines, "mmf_output_energy_add_readability_lines" );
            myfile.Read_Single( parameters->output_configuration_step, "mmf_output_configuration_step" );
            myfile.Read_Single( parameters->output_configuration_archive, "mmf_output_configuration_archive" );
            myfile.Read_Single( output_configuration_filetype, "mmf_output_configuration_filetype" );
            parameters->output_vf_filetype = IO::VF_FileFormat( output_configuration_filetype );
            // Method parameters
            myfile.Read_Single( str_max_walltime, "mmf_max_walltime" );
            parameters->max_walltime_sec = (long int)Utility::Timing::DurationFromString( str_max_walltime ).count();
            myfile.Read_Single( parameters->force_convergence, "mmf_force_convergence" );
            myfile.Read_Single( parameters->n_iterations, "mmf_n_iterations" );
            myfile.Read_Single( parameters->n_iterations_log, "mmf_n_iterations_log" );
            myfile.Read_Single( parameters->n_modes, "mmf_n_modes" );
            myfile.Read_Single( parameters->n_mode_follow, "mmf_n_mode_follow" );
        }
        catch( ... )
        {
            spirit_handle_exception_core(
                fmt::format( "Unable to parse MMF parameters from config file \"{}\"", config_file ) );
        }
    }
    else
        Log( Log_Level::Parameter, Log_Sender::IO, "Parameters MMF: Using default configuration!" );

    // Return
    std::vector<std::string> parameter_log;
    parameter_log.push_back( "Parameters MMF:" );
    parameter_log.push_back( fmt::format( "    {:<17} = {:e}", "force convergence", parameters->force_convergence ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "maximum walltime", str_max_walltime ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "n_iterations", parameters->n_iterations ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "n_iterations_log", parameters->n_iterations_log ) );
    parameter_log.push_back( fmt::format( "    {:<17} = \"{}\"", "output_folder", parameters->output_folder ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "output_any", parameters->output_any ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "output_initial", parameters->output_initial ) );
    parameter_log.push_back( fmt::format( "    {:<17} = {}", "output_final", parameters->output_final ) );
    parameter_log.push_back( fmt::format( "    {:<30} = {}", "output_energy_step", parameters->output_energy_step ) );
    parameter_log.push_back(
        fmt::format( "    {:<30} = {}", "output_energy_archive", parameters->output_energy_archive ) );
    parameter_log.push_back( fmt::format(
        "    {:<30} = {}", "output_energy_divide_by_nspins", parameters->output_energy_divide_by_nspins ) );
    parameter_log.push_back( fmt::format(
        "    {:<30} = {}", "output_energy_add_readability_lines", parameters->output_energy_add_readability_lines ) );
    parameter_log.push_back(
        fmt::format( "    {:<30} = {}", "output_configuration_step", parameters->output_configuration_step ) );
    parameter_log.push_back(
        fmt::format( "    {:<30} = {}", "output_configuration_archive", parameters->output_configuration_archive ) );
    parameter_log.push_back(
        fmt::format( "    {:<30} = {}", "output_configuration_filetype", (int)parameters->output_vf_filetype ) );
    Log.SendBlock( Log_Level::Parameter, Log_Sender::IO, parameter_log );

    Log( Log_Level::Debug, Log_Sender::IO, "Parameters MMF: built" );
    return parameters;
}

std::unique_ptr<Engine::Hamiltonian>
Hamiltonian_from_Config( const std::string & config_file, std::shared_ptr<Data::Geometry> geometry )
{
    //-------------- Insert default values here -----------------------------
    // The type of hamiltonian we will use
    std::string hamiltonian_type = "heisenberg_neighbours";

    //------------------------------- Parser --------------------------------
    Log( Log_Level::Debug, Log_Sender::IO, "Hamiltonian: building" );

    // Hamiltonian type
    if( !config_file.empty() )
    {
        try
        {
            Log( Log_Level::Debug, Log_Sender::IO, "Hamiltonian: deciding type" );
            IO::Filter_File_Handle myfile( config_file );

            // What hamiltonian do we use?
            myfile.Read_Single( hamiltonian_type, "hamiltonian" );
        }
        catch( ... )
        {
            spirit_handle_exception_core( fmt::format(
                "Unable to read Hamiltonian type from config file  \"{}\". Using default.", config_file ) );
            hamiltonian_type = "heisenberg_neighbours";
        }
    }
    else
        Log( Log_Level::Parameter, Log_Sender::IO, "Hamiltonian: Using default Hamiltonian: " + hamiltonian_type );

    // Hamiltonian
    std::unique_ptr<Engine::Hamiltonian> hamiltonian;
    try
    {
        if( hamiltonian_type == "heisenberg_neighbours" || hamiltonian_type == "heisenberg_pairs" )
        {
            hamiltonian = Hamiltonian_Heisenberg_from_Config( config_file, geometry, hamiltonian_type );
        }
        else if( hamiltonian_type == "gaussian" )
        {
            hamiltonian = std::move( Hamiltonian_Gaussian_from_Config( config_file, geometry ) );
        }
        else
        {
            spirit_throw(
                Utility::Exception_Classifier::System_not_Initialized, Log_Level::Severe,
                fmt::format( "Hamiltonian: Invalid type \"{}\"", hamiltonian_type ) );
        }
    }
    catch( ... )
    {
        spirit_handle_exception_core(
            fmt::format( "Unable to initialize Hamiltonian from config file \"{}\"", config_file ) );
    }

    // Return
    Log( Log_Level::Debug, Log_Sender::IO, "Hamiltonian: built hamiltonian of type: " + hamiltonian_type );
    return hamiltonian;
}

std::unique_ptr<Engine::Hamiltonian_Heisenberg> Hamiltonian_Heisenberg_from_Config(
    const std::string & config_file, std::shared_ptr<Data::Geometry> geometry, const std::string & hamiltonian_type )
{
    //-------------- Insert default values here -----------------------------
    // Boundary conditions (a, b, c)
    std::vector<int> boundary_conditions_i = { 0, 0, 0 };
    intfield boundary_conditions           = { false, false, false };

    // External Magnetic Field
    scalar B         = 0;
    Vector3 B_normal = { 0.0, 0.0, 1.0 };

    // Anisotropy
    std::string anisotropy_file = "";
    scalar K                    = 0;
    Vector3 K_normal            = { 0.0, 0.0, 1.0 };
    bool anisotropy_from_file   = false;
    intfield anisotropy_index( geometry->n_cell_atoms );
    scalarfield anisotropy_magnitude( geometry->n_cell_atoms, 0.0 );
    vectorfield anisotropy_normal( geometry->n_cell_atoms, K_normal );

    // ------------ Pair Interactions ------------
    int n_pairs                        = 0;
    std::string interaction_pairs_file = "";
    bool interaction_pairs_from_file   = false;
    pairfield exchange_pairs( 0 );
    scalarfield exchange_magnitudes( 0 );
    pairfield dmi_pairs( 0 );
    scalarfield dmi_magnitudes( 0 );
    vectorfield dmi_normals( 0 );

    // Number of shells in which we calculate neighbours
    int n_shells_exchange = exchange_magnitudes.size();
    // DM constant
    int n_shells_dmi = dmi_magnitudes.size();
    int dm_chirality = 1;

    std::string ddi_method_str     = "none";
    auto ddi_method                = Engine::DDI_Method::None;
    intfield ddi_n_periodic_images = { 4, 4, 4 };
    scalar ddi_radius              = 0.0;
    bool ddi_pb_zero_padding       = true;

    // ------------ Quadruplet Interactions ------------
    int n_quadruplets            = 0;
    std::string quadruplets_file = "";
    bool quadruplets_from_file   = false;
    quadrupletfield quadruplets( 0 );
    scalarfield quadruplet_magnitudes( 0 );

    //------------------------------- Parser --------------------------------
    Log( Log_Level::Debug, Log_Sender::IO, "Hamiltonian_Heisenberg: building" );
    // Iteration variables
    int iatom = 0;
    if( !config_file.empty() )
    {
        try
        {
            IO::Filter_File_Handle myfile( config_file );

            // Boundary conditions
            myfile.Read_3Vector( boundary_conditions_i, "boundary_conditions" );
            boundary_conditions[0] = ( boundary_conditions_i[0] != 0 );
            boundary_conditions[1] = ( boundary_conditions_i[1] != 0 );
            boundary_conditions[2] = ( boundary_conditions_i[2] != 0 );
        }
        catch( ... )
        {
            spirit_handle_exception_core(
                fmt::format( "Unable to read boundary conditions from config file \"{}\"", config_file ) );
        }

        try
        {
            IO::Filter_File_Handle myfile( config_file );

            // Read parameters from config if available
            myfile.Read_Single( B, "external_field_magnitude" );
            myfile.Read_Vector3( B_normal, "external_field_normal" );
            B_normal.normalize();
            if( B_normal.norm() < 1e-8 )
            {
                B_normal = { 0, 0, 1 };
                Log( Log_Level::Warning, Log_Sender::IO,
                     "Input for 'external_field_normal' had norm zero and has been set to (0,0,1)" );
            }
        }
        catch( ... )
        {
            spirit_handle_exception_core(
                fmt::format( "Unable to read external field from config file \"{}\"", config_file ) );
        }

        try
        {
            IO::Filter_File_Handle myfile( config_file );

            // Anisotropy
            if( myfile.Find( "n_anisotropy" ) )
                anisotropy_file = config_file;
            else if( myfile.Find( "anisotropy_file" ) )
                myfile.iss >> anisotropy_file;
            if( anisotropy_file.length() > 0 )
            {
                // The file name should be valid so we try to read it
                Anisotropy_from_File(
                    anisotropy_file, geometry, n_pairs, anisotropy_index, anisotropy_magnitude, anisotropy_normal );

                anisotropy_from_file = true;
                if( !anisotropy_index.empty() )
                {
                    K        = anisotropy_magnitude[0];
                    K_normal = anisotropy_normal[0];
                }
                else
                {
                    K        = 0;
                    K_normal = { 0, 0, 0 };
                }
            }
            else
            {
                // Read parameters from config
                myfile.Read_Single( K, "anisotropy_magnitude" );
                myfile.Read_Vector3( K_normal, "anisotropy_normal" );
                K_normal.normalize();

                if( K != 0 )
                {
                    // Fill the arrays
                    for( int i = 0; i < anisotropy_index.size(); ++i )
                    {
                        anisotropy_index[i]     = i;
                        anisotropy_magnitude[i] = K;
                        anisotropy_normal[i]    = K_normal;
                    }
                }
                else
                {
                    anisotropy_index     = intfield( 0 );
                    anisotropy_magnitude = scalarfield( 0 );
                    anisotropy_normal    = vectorfield( 0 );
                }
            }
        }
        catch( ... )
        {
            spirit_handle_exception_core(
                fmt::format( "Unable to read anisotropy from config file \"{}\"", config_file ) );
        }

        if( hamiltonian_type == "heisenberg_pairs" )
        {
            try
            {
                IO::Filter_File_Handle myfile( config_file );

                // Interaction Pairs
                if( myfile.Find( "n_interaction_pairs" ) )
                    interaction_pairs_file = config_file;
                else if( myfile.Find( "interaction_pairs_file" ) )
                    myfile.iss >> interaction_pairs_file;

                if( interaction_pairs_file.length() > 0 )
                {
                    // The file name should be valid so we try to read it
                    Pairs_from_File(
                        interaction_pairs_file, geometry, n_pairs, exchange_pairs, exchange_magnitudes, dmi_pairs,
                        dmi_magnitudes, dmi_normals );
                }
                // else
                //{
                //	Log(Log_Level::Warning, Log_Sender::IO, "Hamiltonian_Heisenberg: Default Interaction pairs have not
                // been implemented yet."); 	throw Exception::System_not_Initialized;
                //	// Not implemented!
                //}
            }
            catch( ... )
            {
                spirit_handle_exception_core(
                    fmt::format( "Unable to read interaction pairs from config file \"{}\"", config_file ) );
            }
        }
        else
        {
            try
            {
                IO::Filter_File_Handle myfile( config_file );

                myfile.Read_Single( n_shells_exchange, "n_shells_exchange" );
                if( exchange_magnitudes.size() != n_shells_exchange )
                    exchange_magnitudes = scalarfield( n_shells_exchange );
                if( n_shells_exchange > 0 )
                {
                    if( myfile.Find( "jij" ) )
                    {
                        for( int ishell = 0; ishell < n_shells_exchange; ++ishell )
                            myfile.iss >> exchange_magnitudes[ishell];
                    }
                    else
                        Log( Log_Level::Warning, Log_Sender::IO,
                             fmt::format(
                                 "Hamiltonian_Heisenberg: Keyword 'jij' not found. Using Default: {}",
                                 exchange_magnitudes[0] ) );
                }
            }
            catch( ... )
            {
                spirit_handle_exception_core(
                    fmt::format( "Failed to read exchange parameters from config file \"{}\"", config_file ) );
            }

            try
            {
                IO::Filter_File_Handle myfile( config_file );

                myfile.Read_Single( n_shells_dmi, "n_shells_dmi" );
                if( dmi_magnitudes.size() != n_shells_dmi )
                    dmi_magnitudes = scalarfield( n_shells_dmi );
                if( n_shells_dmi > 0 )
                {
                    if( myfile.Find( "dij" ) )
                    {
                        for( int ishell = 0; ishell < n_shells_dmi; ++ishell )
                            myfile.iss >> dmi_magnitudes[ishell];
                    }
                    else
                        Log( Log_Level::Warning, Log_Sender::IO,
                             fmt::format(
                                 "Hamiltonian_Heisenberg: Keyword 'dij' not found. Using Default: {}",
                                 dmi_magnitudes[0] ) );
                }
                myfile.Read_Single( dm_chirality, "dm_chirality" );
            }
            catch( ... )
            {
                spirit_handle_exception_core(
                    fmt::format( "Failed to read DMI parameters from config file \"{}\"", config_file ) );
            }
        }

        try
        {
            IO::Filter_File_Handle myfile( config_file );

            // DDI method
            myfile.Read_String( ddi_method_str, "ddi_method" );
            if( ddi_method_str == "none" )
                ddi_method = Engine::DDI_Method::None;
            else if( ddi_method_str == "fft" )
                ddi_method = Engine::DDI_Method::FFT;
            else if( ddi_method_str == "fmm" )
                ddi_method = Engine::DDI_Method::FMM;
            else if( ddi_method_str == "cutoff" )
                ddi_method = Engine::DDI_Method::Cutoff;
            else
            {
                Log( Log_Level::Warning, Log_Sender::IO,
                     fmt::format(
                         "Hamiltonian_Heisenberg: Keyword 'ddi_method' got passed invalid method \"{}\". Setting to "
                         "\"none\".",
                         ddi_method_str ) );
                ddi_method_str = "none";
            }

            // Number of periodical images
            myfile.Read_3Vector( ddi_n_periodic_images, "ddi_n_periodic_images" );
            myfile.Read_Single( ddi_pb_zero_padding, "ddi_pb_zero_padding" );

            // Dipole-dipole cutoff radius
            myfile.Read_Single( ddi_radius, "ddi_radius" );
        }
        catch( ... )
        {
            spirit_handle_exception_core(
                fmt::format( "Unable to read DDI radius from config file \"{}\"", config_file ) );
        }

        try
        {
            IO::Filter_File_Handle myfile( config_file );

            // Interaction Quadruplets
            if( myfile.Find( "n_interaction_quadruplets" ) )
                quadruplets_file = config_file;
            else if( myfile.Find( "interaction_quadruplets_file" ) )
                myfile.iss >> quadruplets_file;

            if( quadruplets_file.length() > 0 )
            {
                // The file name should be valid so we try to read it
                Quadruplets_from_File( quadruplets_file, geometry, n_quadruplets, quadruplets, quadruplet_magnitudes );
            }
        }
        catch( ... )
        {
            spirit_handle_exception_core(
                fmt::format( "Unable to read interaction quadruplets from config file \"{}\"", config_file ) );
        }
    }
    else
        Log( Log_Level::Parameter, Log_Sender::IO, "Hamiltonian_Heisenberg: Using default configuration!" );

    // Return
    std::vector<std::string> parameter_log;
    parameter_log.push_back( "Hamiltonian Heisenberg:" );
    parameter_log.push_back( fmt::format(
        "    {:<21} = {} {} {}", "boundary conditions", boundary_conditions[0], boundary_conditions[1],
        boundary_conditions[2] ) );
    parameter_log.push_back( fmt::format( "    {:<21} = {}", "external field", B ) );
    parameter_log.push_back( fmt::format( "    {:<21} = {}", "field_normal", B_normal.transpose() ) );
    if( anisotropy_from_file )
        parameter_log.push_back( fmt::format( "    K from file \"{}\"", anisotropy_file ) );
    parameter_log.push_back( fmt::format( "    {:<21} = {}", "anisotropy[0]", K ) );
    parameter_log.push_back( fmt::format( "    {:<21} = {}", "anisotropy_normal[0]", K_normal.transpose() ) );
    if( hamiltonian_type == "heisenberg_neighbours" )
    {
        parameter_log.push_back( fmt::format( "    {:<21} = {}", "n_shells_exchange", n_shells_exchange ) );
        if( n_shells_exchange > 0 )
            parameter_log.push_back( fmt::format( "    {:<21} = {}", "J_ij[0]", exchange_magnitudes[0] ) );
        parameter_log.push_back( fmt::format( "    {:<21} = {}", "n_shells_dmi", n_shells_dmi ) );
        if( n_shells_dmi > 0 )
            parameter_log.push_back( fmt::format( "    {:<21} = {}", "D_ij[0]", dmi_magnitudes[0] ) );
        parameter_log.push_back( fmt::format( "    {:<21} = {}", "DM chirality", dm_chirality ) );
    }
    parameter_log.push_back( fmt::format( "    {:<21} = {}", "ddi_method", ddi_method_str ) );
    parameter_log.push_back( fmt::format(
        "    {:<21} = ({} {} {})", "ddi_n_periodic_images", ddi_n_periodic_images[0], ddi_n_periodic_images[1],
        ddi_n_periodic_images[2] ) );
    parameter_log.push_back( fmt::format( "    {:<21} = {}", "ddi_radius", ddi_radius ) );
    parameter_log.push_back( fmt::format( "    {:<21} = {}", "ddi_pb_zero_padding", ddi_pb_zero_padding ) );
    Log.SendBlock( Log_Level::Parameter, Log_Sender::IO, parameter_log );

    std::unique_ptr<Engine::Hamiltonian_Heisenberg> hamiltonian;

    if( hamiltonian_type == "heisenberg_neighbours" )
    {
        hamiltonian = std::make_unique<Engine::Hamiltonian_Heisenberg>(
            B, B_normal, anisotropy_index, anisotropy_magnitude, anisotropy_normal, exchange_magnitudes, dmi_magnitudes,
            dm_chirality, ddi_method, ddi_n_periodic_images, ddi_pb_zero_padding, ddi_radius, quadruplets,
            quadruplet_magnitudes, geometry, boundary_conditions );
    }
    else
    {
        hamiltonian = std::make_unique<Engine::Hamiltonian_Heisenberg>(
            B, B_normal, anisotropy_index, anisotropy_magnitude, anisotropy_normal, exchange_pairs, exchange_magnitudes,
            dmi_pairs, dmi_magnitudes, dmi_normals, ddi_method, ddi_n_periodic_images, ddi_pb_zero_padding, ddi_radius,
            quadruplets, quadruplet_magnitudes, geometry, boundary_conditions );
    }
    Log( Log_Level::Debug, Log_Sender::IO, "Hamiltonian_Heisenberg: built" );
    return hamiltonian;
} // end Hamiltonian_Heisenberg_From_Config

std::unique_ptr<Engine::Hamiltonian_Gaussian>
Hamiltonian_Gaussian_from_Config( const std::string & config_file, std::shared_ptr<Data::Geometry> geometry )
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
    Log( Log_Level::Debug, Log_Sender::IO, "Hamiltonian_Gaussian: building" );

    if( !config_file.empty() )
    {
        try
        {
            IO::Filter_File_Handle myfile( config_file );

            // N
            myfile.Read_Single( n_gaussians, "n_gaussians" );

            // Allocate arrays
            amplitude = std::vector<scalar>( n_gaussians, 1.0 );
            width     = std::vector<scalar>( n_gaussians, 1.0 );
            center    = std::vector<Vector3>( n_gaussians, Vector3{ 0, 0, 1 } );
            // Read arrays
            if( myfile.Find( "gaussians" ) )
            {
                for( int i = 0; i < n_gaussians; ++i )
                {
                    myfile.GetLine();
                    myfile.iss >> amplitude[i];
                    myfile.iss >> width[i];
                    for( int j = 0; j < 3; ++j )
                    {
                        myfile.iss >> center[i][j];
                    }
                    center[i].normalize();
                }
            }
            else
                Log( Log_Level::Error, Log_Sender::IO,
                     "Hamiltonian_Gaussian: Keyword 'gaussians' not found. Using Default: {0, 0, 1}" );
        }
        catch( ... )
        {
            spirit_handle_exception_core(
                fmt::format( "Unable to read Hamiltonian_Gaussian parameters from config file  \"{}\"", config_file ) );
        }
    }
    else
        Log( Log_Level::Parameter, Log_Sender::IO, "Hamiltonian_Gaussian: Using default configuration!" );

    // Return
    std::vector<std::string> parameter_log;
    parameter_log.push_back( "Hamiltonian Gaussian:" );
    parameter_log.push_back( fmt::format( "    {0:<12} = {1}", "n_gaussians", n_gaussians ) );
    parameter_log.push_back( fmt::format( "    {0:<12} = {1}", "amplitude[0]", amplitude[0] ) );
    parameter_log.push_back( fmt::format( "    {0:<12} = {1}", "width[0]", width[0] ) );
    parameter_log.push_back( fmt::format( "    {0:<12} = {1}", "center[0]", center[0].transpose() ) );
    Log.SendBlock( Log_Level::Parameter, Log_Sender::IO, parameter_log );
    auto hamiltonian = std::make_unique<Engine::Hamiltonian_Gaussian>( amplitude, width, center );
    Log( Log_Level::Debug, Log_Sender::IO, "Hamiltonian_Gaussian: built" );
    return hamiltonian;
}

} // namespace IO