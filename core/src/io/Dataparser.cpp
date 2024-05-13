#include <engine/Vectormath.hpp>
#include <io/Dataparser.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/IO.hpp>
#include <io/OVF_File.hpp>
#include <io/Tableparser.hpp>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include <Eigen/Core>
#include <Eigen/Dense>

using Utility::Log_Level;
using Utility::Log_Sender;

namespace IO
{

// Reads a non-OVF spins file with plain text and discarding any headers starting with '#'
void Read_NonOVF_Spin_Configuration(
    vectorfield & spins, Data::Geometry & geometry, const int nos, const int idx_image_infile,
    const std::string & file )
{
    IO::Filter_File_Handle file_handle( file, "#" );

    // Jump to the specified image in the file
    for( int i = 0; i < ( nos * idx_image_infile ); i++ )
        file_handle.GetLine();

    for( int i = 0; i < nos && file_handle.GetLine( "," ); i++ )
    {
        file_handle >> spins[i][0];
        file_handle >> spins[i][1];
        file_handle >> spins[i][2];

        if( spins[i].norm() < 1e-5 )
        {
            spins[i] = { 0, 0, 1 };
            // In case of spin vector close to zero we have a vacancy
#ifdef SPIRIT_ENABLE_DEFECTS
            geometry.atom_types[i] = -1;
#endif
        }
    }

    // normalize read in spins
    Engine::Vectormath::normalize_vectors( spins );
}

void Check_NonOVF_Chain_Configuration(
    std::shared_ptr<State::chain_t> chain, const std::string & file, int start_image_infile,
    int end_image_infile, const int insert_idx, int & noi_to_add, int & noi_to_read, const int idx_chain )
{
    IO::Filter_File_Handle file_handle( file, "#" );

    int nol = file_handle.Get_N_Non_Comment_Lines();
    int noi = chain->noi;
    int nos = chain->images[0]->nos;

    int noi_infile = nol / nos;
    int remainder  = nol % nos;

    if( remainder != 0 )
    {
        Log( Utility::Log_Level::Warning, Utility::Log_Sender::IO,
             fmt::format( "Calculated number of images in the nonOVF file is not integer" ), insert_idx, idx_chain );
    }

    // Check if the ending image is valid otherwise set it to the last image infile
    if( end_image_infile < start_image_infile || end_image_infile >= noi_infile )
    {
        end_image_infile = noi_infile - 1;
        Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
             fmt::format( "Invalid end_image_infile. Value was set to the last image "
                          "of the file" ),
             insert_idx, idx_chain );
    }

    // If the idx of the starting image is valid
    if( start_image_infile < noi_infile )
    {
        noi_to_read = end_image_infile - start_image_infile + 1;

        noi_to_add = noi_to_read - ( noi - insert_idx );
    }
    else
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::IO,
             fmt::format( "Invalid starting_idx. File {} has {} noi", file, noi_infile ), insert_idx, idx_chain );
    }
}

// Read Basis from file
void Basis_from_File(
    const std::string & basis_file, Data::Basis_Cell_Composition & cell_composition, std::vector<Vector3> & cell_atoms,
    std::size_t & n_cell_atoms ) noexcept
{

    Log( Log_Level::Info, Log_Sender::IO, "Reading basis from file " + basis_file );

    Filter_File_Handle basis_file_handle( basis_file );

    // Read basis cell
    if( basis_file_handle.Find( "basis" ) )
    {
        // Read number of atoms in the basis cell
        basis_file_handle.GetLine();
        basis_file_handle >> n_cell_atoms;
        cell_atoms = std::vector<Vector3>( n_cell_atoms );
        cell_composition.iatom.resize( n_cell_atoms );
        cell_composition.atom_type = std::vector<int>( n_cell_atoms, 0 );
        cell_composition.mu_s      = std::vector<scalar>( n_cell_atoms, 1 );

        // Read atom positions
        for( std::size_t iatom = 0; iatom < n_cell_atoms; ++iatom )
        {
            basis_file_handle.GetLine();
            basis_file_handle >> cell_atoms[iatom][0] >> cell_atoms[iatom][1] >> cell_atoms[iatom][2];
            cell_composition.iatom[iatom] = static_cast<int>( iatom );
        }
    }
}

void Defects_from_File(
    const std::string & defects_file, int & n_defects, field<Site> & defect_sites, intfield & defect_types ) noexcept
try
{
    n_defects    = 0;
    defect_sites = field<Site>( 0 );
    defect_types = intfield( 0 );

    Log( Log_Level::Debug, Log_Sender::IO, fmt::format( "Reading defects from file \"{}\"", defects_file ) );
    Filter_File_Handle myfile( defects_file );
    int nod = 0;

    if( myfile.Find( "n_defects" ) )
    {
        // Read n interaction pairs
        myfile >> nod;
        Log( Log_Level::Debug, Log_Sender::IO, fmt::format( "File \"{}\" should have {} defects", defects_file, nod ) );
    }
    else
    {
        // Read the whole file
        nod = (int)1e8;
        // First line should contain the columns
        myfile.To_Start();
        Log( Log_Level::Debug, Log_Sender::IO,
             fmt::format( "Trying to parse defects from top of file \"{}\"", defects_file ) );
    }

    while( myfile.GetLine() && n_defects < nod )
    {
        Site site{};
        int type{ 0 };
        myfile >> site.i >> site.translations[0] >> site.translations[1] >> site.translations[2] >> type;
        defect_sites.push_back( site );
        defect_types.push_back( type );
        ++n_defects;
    }

    Log( Log_Level::Parameter, Log_Sender::IO,
         fmt::format( "Done reading {} defects from file \"{}\"", n_defects, defects_file ) );
}
catch( ... )
{
    spirit_rethrow( fmt::format( "Could not read defects file \"{}\"", defects_file ) );
}

void Pinned_from_File(
    const std::string & pinned_file, int & n_pinned, field<Site> & pinned_sites, vectorfield & pinned_spins ) noexcept
try
{
    int nop      = 0;
    n_pinned     = 0;
    pinned_sites = field<Site>( 0 );
    pinned_spins = vectorfield( 0 );

    Log( Log_Level::Debug, Log_Sender::IO, fmt::format( "Reading pinned sites from file \"{}\"", pinned_file ) );
    Filter_File_Handle myfile( pinned_file );

    if( myfile.Find( "n_pinned" ) )
    {
        // Read n interaction pairs
        myfile >> nop;
        Log( Log_Level::Debug, Log_Sender::IO,
             fmt::format( "File \"{}\" should have {} pinned sites", pinned_file, nop ) );
    }
    else
    {
        // Read the whole file
        nop = (int)1e8;
        // First line should contain the columns
        myfile.To_Start();
        Log( Log_Level::Debug, Log_Sender::IO,
             fmt::format( "Trying to parse pinned sites from top of file \"{}\"", pinned_file ) );
    }

    while( myfile.GetLine() && n_pinned < nop )
    {
        Site site{};
        Vector3 orientation{};
        myfile >> site.i >> site.translations[0] >> site.translations[1] >> site.translations[2] >> orientation.x()
            >> orientation.y() >> orientation.z();
        pinned_sites.push_back( site );
        pinned_spins.push_back( orientation );
        ++n_pinned;
    }

    Log( Log_Level::Parameter, Log_Sender::IO,
         fmt::format( "Done reading {} pinned sites from file \"{}\"", n_pinned, pinned_file ) );
}
catch( ... )
{
    spirit_rethrow( fmt::format( "Could not read pinned sites file  \"{}\"", pinned_file ) );
}

} // namespace IO
