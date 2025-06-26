#include <engine/StateType.hpp>
#include <engine/Vectormath.hpp>
#include <io/Dataparser.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/IO.hpp>
#include <io/OVF_File.hpp>
#include <io/Tableparser.hpp>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

#include <iostream>
#include <string>

#include <Eigen/Core>
#include <Eigen/Dense>

using Utility::Log_Level;
using Utility::Log_Sender;

namespace IO
{

namespace Spin
{

// Reads a non-OVF spins file with plain text and discarding any headers starting with '#'
void Read_NonOVF_System_Configuration(
    StateType & state, Data::Geometry & geometry, const int nos, const int idx_image_infile, const std::string & file )
{
    IO::Filter_File_Handle file_handle( file, "#" );

    // Jump to the specified image in the file
    for( int i = 0; i < ( nos * idx_image_infile ); i++ )
        file_handle.GetLine();

    for( int i = 0; i < nos && file_handle.GetLine( "," ); i++ )
    {
        file_handle >> state.spin[i][0];
        file_handle >> state.spin[i][1];
        file_handle >> state.spin[i][2];

        if( state.spin[i].norm() < 1e-5 )
        {
            state.spin[i] = { 0, 0, 1 };
            // In case of spin vector close to zero we have a vacancy
#ifdef SPIRIT_ENABLE_DEFECTS
            geometry.atom_types[i] = -1;
#endif
        }
    }

    // normalize read in spins
    Engine::Vectormath::normalize_vectors( state.spin );
}

} // namespace Spin

void Check_NonOVF_Chain_Configuration(
    std::shared_ptr<::State::chain_t> chain, const std::string & file, int start_image_infile, int end_image_infile,
    const int insert_idx, int & noi_to_add, int & noi_to_read, const int idx_chain )
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
auto Basis_from_File( Filter_File_Handle & basis_file ) noexcept -> std::vector<Vector3>
{
    Log( Log_Level::Info, Log_Sender::IO, fmt::format( "Reading basis from {}", basis_file.filename() ) );

    // Read basis cell
    if( basis_file.Find( "basis" ) )
    {
        std::size_t n_cell_atoms = 0;
        // Read number of atoms in the basis cell
        basis_file.GetLine();
        basis_file >> n_cell_atoms;

        // Read atom positions
        std::vector<Vector3> cell_atoms( n_cell_atoms );
        for( std::size_t iatom = 0; iatom < n_cell_atoms; ++iatom )
        {
            basis_file.GetLine();
            basis_file >> cell_atoms[iatom][0] >> cell_atoms[iatom][1] >> cell_atoms[iatom][2];
        }
        return cell_atoms;
    }

    return { { 0, 0, 0 } };
}

auto Defects_from_File( Filter_File_Handle & defects_file ) noexcept -> Data::Defects
{
    auto defect_sites = field<Site>( 0 );
    auto defect_types = intfield( 0 );
#ifdef SPIRIT_ENABLE_DEFECTS
    int n_defects = 0;

    Log( Log_Level::Debug, Log_Sender::IO, fmt::format( "Reading defects from {}", defects_file.filename() ) );
    int nod = 0;

    if( defects_file.Find( "n_defects" ) )
    {
        // Read n interaction pairs
        defects_file >> nod;
        Log( Log_Level::Debug, Log_Sender::IO,
             fmt::format( "File \"{}\" should have {} defects", defects_file.filename(), nod ) );
    }
    else
    {
        // Read the whole file
        nod = (int)1e8;
        // First line should contain the columns
        defects_file.To_Start();
        Log( Log_Level::Debug, Log_Sender::IO,
             fmt::format( "Trying to parse defects from top of \"{}\"", defects_file.filename() ) );
    }

    while( defects_file.GetLine() && n_defects < nod )
    {
        Site site{};
        int type{ 0 };
        defects_file >> site.i >> site.translations[0] >> site.translations[1] >> site.translations[2] >> type;
        defect_sites.push_back( site );
        defect_types.push_back( type );
        ++n_defects;
    }

    Log( Log_Level::Parameter, Log_Sender::IO,
         fmt::format( "Done reading {} defects from file \"{}\"", n_defects, defects_file.filename() ) );

#else
    Log( Log_Level::Parameter, Log_Sender::IO, "Disorder is disabled" );
#endif
    return Data::Defects{ defect_sites, defect_types };
}

void Pinned_from_File(
    Filter_File_Handle & pinned_file, int & n_pinned, field<Site> & pinned_sites, vectorfield & pinned_spins ) noexcept
try
{
    int nop      = 0;
    n_pinned     = 0;
    pinned_sites = field<Site>( 0 );
    pinned_spins = vectorfield( 0 );

    Log( Log_Level::Debug, Log_Sender::IO,
         fmt::format( "Reading pinned sites from file \"{}\"", pinned_file.filename() ) );

    if( pinned_file.Find( "n_pinned" ) )
    {
        // Read n interaction pairs
        pinned_file >> nop;
        Log( Log_Level::Debug, Log_Sender::IO,
             fmt::format( "File \"{}\" should have {} pinned sites", pinned_file.filename(), nop ) );
    }
    else
    {
        // Read the whole file
        nop = (int)1e8;
        // First line should contain the columns
        pinned_file.To_Start();
        Log( Log_Level::Debug, Log_Sender::IO,
             fmt::format( "Trying to parse pinned sites from top of file \"{}\"", pinned_file.filename() ) );
    }

    while( pinned_file.GetLine() && n_pinned < nop )
    {
        Site site{};
        Vector3 orientation{};
        pinned_file >> site.i >> site.translations[0] >> site.translations[1] >> site.translations[2] >> orientation.x()
            >> orientation.y() >> orientation.z();
        pinned_sites.push_back( site );
        pinned_spins.push_back( orientation );
        ++n_pinned;
    }

    Log( Log_Level::Parameter, Log_Sender::IO,
         fmt::format( "Done reading {} pinned sites from file \"{}\"", n_pinned, pinned_file.filename() ) );
}
catch( ... )
{
    spirit_rethrow( fmt::format( "Could not read pinned sites file  \"{}\"", pinned_file.filename() ) );
}

} // namespace IO
