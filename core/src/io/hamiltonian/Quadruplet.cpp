#include <engine/Vectormath_Defines.hpp>
#include <io/Dataparser.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/Tableparser.hpp>

#include <vector>

using Utility::Log_Level, Utility::Log_Sender;

namespace IO
{

namespace
{

// Read from Quadruplet file
void Quadruplets_from_File(
    const std::string & quadruplets_file, const Data::Geometry &, int & noq, quadrupletfield & quadruplets,
    scalarfield & quadruplet_magnitudes ) noexcept
try
{
    Log( Log_Level::Debug, Log_Sender::IO,
         fmt::format( "Reading spin quadruplets from file \"{}\"", quadruplets_file ) );

    // parser initialization
    using QuadrupletTableParser = TableParser<int, int, int, int, int, int, int, int, int, int, int, int, int, scalar>;
    const QuadrupletTableParser parser(
        { "i", "j", "da_j", "db_j", "dc_j", "k", "da_k", "db_k", "dc_k", "l", "da_l", "db_l", "dc_l", "q" } );

    // factory function for creating a lambda that transforms the row that is read
    auto transform_factory = [&quadruplets_file]( const std::map<std::string_view, int> & idx )
    {
        if( idx.at( "q" ) < 0 )
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format( "No interactions could be found in header of quadruplets file ", quadruplets_file ) );

        return []( const QuadrupletTableParser::read_row_t & row ) -> std::tuple<Quadruplet, scalar>
        {
            const auto & [i, j, da_j, db_j, dc_j, k, da_k, db_k, dc_k, l, da_l, db_l, dc_l, Q] = row;
            return std::make_tuple(
                Quadruplet{ i, j, k, l, { da_j, db_j, dc_j }, { da_k, db_k, dc_k }, { da_l, db_l, dc_l } }, Q );
        };
    };
    const auto data = parser.parse( quadruplets_file, "n_interaction_quadruplets", 20, transform_factory );
    noq             = data.size();

    for( const auto & [quadruplet, magnitude] : data )
    {
        if( magnitude != 0 )
        {
            quadruplets.push_back( quadruplet );
            quadruplet_magnitudes.push_back( magnitude );
        }
    }
}
catch( ... )
{
    spirit_rethrow( fmt::format( "Could not read quadruplets from file  \"{}\"", quadruplets_file ) );
}

} // namespace

void Quadruplets_from_Config(
    const std::string & config_file_name, const Data::Geometry & geometry, std::vector<std::string> & parameter_log,
    quadrupletfield & quadruplets, scalarfield & quadruplet_magnitudes )
{
    std::string quadruplets_file{};
    int n_quadruplets = 0;

    try
    {
        IO::Filter_File_Handle config_file_handle( config_file_name );

        // Interaction Quadruplets
        if( config_file_handle.Find( "n_interaction_quadruplets" ) )
            quadruplets_file = config_file_name;
        else if( config_file_handle.Find( "interaction_quadruplets_file" ) )
            config_file_handle >> quadruplets_file;

        if( quadruplets_file.length() > 0 )
        {
            // The file name should be valid so we try to read it
            Quadruplets_from_File( quadruplets_file, geometry, n_quadruplets, quadruplets, quadruplet_magnitudes );
        }
    }
    catch( ... )
    {
        spirit_handle_exception_core(
            fmt::format( "Unable to read interaction quadruplets from config file \"{}\"", config_file_name ) );
    }
}

} // namespace IO
