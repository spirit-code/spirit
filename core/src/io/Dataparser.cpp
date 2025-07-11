#include <engine/StateType.hpp>
#include <engine/Vectormath.hpp>
#include <io/Dataparser.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/IO.hpp>
#include <io/OVF_File.hpp>
#include <io/Tableparser.hpp>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

#include <sstream>
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

    // Get first line after 'basis' keyword or first line
    if( !basis_file.Find( "basis" ) )
        basis_file.To_Start();
    basis_file.GetLine();

    // check specification type
    std::string first_line_str = std::string{ basis_file.CurrentLine() };
    std::istringstream iss{ first_line_str };
    auto first_line = std::vector<scalar>( std::istream_iterator<scalar>( iss ), std::istream_iterator<scalar>() );

    if( !first_line.empty() && first_line.size() != 2 )
    {
        std::vector<Vector3> cell_atoms{};
        std::size_t n_cell_atoms = 0;
        if( first_line.size() == 1 )
        {
            // This is the old format, starting with a single number to indicate the count.
            basis_file >> n_cell_atoms;
            cell_atoms.reserve( n_cell_atoms );
        }
        else if( first_line.size() >= 3 )
        {
            // If the list starts with data this is also fine, we just parse the whole file in that case.
            n_cell_atoms = static_cast<std::size_t>( 1e8 );
            cell_atoms.reserve( n_cell_atoms );
            cell_atoms.emplace_back( first_line[0], first_line[1], first_line[2] );
        }

        for( std::size_t iatom = 0; iatom < n_cell_atoms; ++iatom )
        {
            if( !basis_file.GetLine() )
                break;

            Vector3 pos = Vector3::Zero();
            basis_file >> pos[0] >> pos[1] >> pos[2];
            cell_atoms.emplace_back( pos );
        }
        cell_atoms.shrink_to_fit();
        return cell_atoms;
    }
    else
    {
        Log( Log_Level::Warning, Log_Sender::IO,
             fmt::format( "No basis vectors found in '{}', using default (0 0 0)", basis_file.filename() ) );
        return { { 0, 0, 0 } };
    }
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

    using DefectsParser = TableParser<int, int, int, int, int>;
    DefectsParser parser( { "i", "da", "db", "dc", "type" } );

    auto data = parser.parse( defects_file, "n_defects", 5 );
    defect_sites.reserve( data.size() );
    defect_types.reserve( data.size() );
    for( auto [i, da, db, dc, type] : data )
    {
        defect_sites.emplace_back( Site{ i, { da, db, dc } } );
        defect_types.emplace_back( type );
    }

    Log( Log_Level::Parameter, Log_Sender::IO,
         fmt::format( "Done reading {} defects from file \"{}\"", defect_sites.size(), defects_file.filename() ) );
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

    using DefectsParser = TableParser<int, int, int, int, scalar, scalar, scalar>;
    DefectsParser parser( { "i", "da", "db", "dc", "x", "y", "z" } );

    auto data = parser.parse( pinned_file, "n_defects", 5 );
    n_pinned  = data.size();

    pinned_sites.reserve( data.size() );
    pinned_spins.reserve( data.size() );
    for( auto [i, da, db, dc, x, y, z] : data )
    {
        pinned_sites.emplace_back( Site{ i, { da, db, dc } } );
        pinned_spins.emplace_back( Vector3{ x, y, z } );
    }

    Log( Log_Level::Parameter, Log_Sender::IO,
         fmt::format( "Done reading {} pinned sites from file \"{}\"", pinned_sites.size(), pinned_file.filename() ) );
}
catch( ... )
{
    spirit_rethrow( fmt::format( "Could not read pinned sites file  \"{}\"", pinned_file.filename() ) );
}

} // namespace IO
