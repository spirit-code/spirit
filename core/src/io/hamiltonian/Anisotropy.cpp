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

// Read from Anisotropy file
void Anisotropy_from_File(
    const std::string & anisotropy_file, const Data::Geometry & geometry, int & n_indices, intfield & anisotropy_index,
    scalarfield & anisotropy_magnitude, vectorfield & anisotropy_normal, intfield & cubic_anisotropy_index,
    scalarfield & cubic_anisotropy_magnitude ) noexcept
try
{
    Log( Log_Level::Debug, Log_Sender::IO, "Reading anisotropy from file " + anisotropy_file );

    // parser initialization
    using AnisotropyTableParser = TableParser<int, scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar>;
    const AnisotropyTableParser parser( { "i", "k", "kx", "ky", "kz", "ka", "kb", "kc", "k4" } );

    // factory function for creating a lambda that transforms the row that is read
    auto transform_factory = [&anisotropy_file, &geometry]( const std::map<std::string_view, int> & idx )
    {
        bool K_xyz = false, K_abc = false, K_magnitude = false;

        if( idx.at( "kx" ) >= 0 && idx.at( "ky" ) >= 0 && idx.at( "kz" ) >= 0 )
            K_xyz = true;
        if( idx.at( "ka" ) >= 0 && idx.at( "kb" ) >= 0 && idx.at( "kc" ) >= 0 )
            K_abc = true;
        if( idx.at( "k" ) >= 0 )
            K_magnitude = true;

        if( !K_xyz && !K_abc )
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format( "No anisotropy data could be found in header of file \"{}\"", anisotropy_file ) );

        return [K_xyz, K_abc, K_magnitude,
                &geometry]( const AnisotropyTableParser::read_row_t & row ) -> std::tuple<int, scalar, Vector3, scalar>
        {
            auto [i, k, kx, ky, kz, ka, kb, kc, k4] = row;

            Vector3 K_temp;
            if( K_xyz )
                K_temp = { kx, ky, kz };
            // Anisotropy vector orientation
            if( K_abc )
            {
                K_temp = { ka, kb, kc };
                K_temp = { K_temp.dot( geometry.lattice_constant * geometry.bravais_vectors[0] ),
                           K_temp.dot( geometry.lattice_constant * geometry.bravais_vectors[1] ),
                           K_temp.dot( geometry.lattice_constant * geometry.bravais_vectors[2] ) };
            }

            // Anisotropy vector normalisation
            if( K_magnitude )
            {
                K_temp.normalize();
                if( K_temp.norm() == 0 )
                    K_temp = Vector3{ 0, 0, 1 };
            }
            else
            {
                k = K_temp.norm();
                if( k != 0 )
                    K_temp.normalize();
            }

            return std::make_tuple( i, k, K_temp, k4 );
        };
    };

    const std::string anisotropy_size_id = "n_anisotropy";
    const auto data                      = parser.parse( anisotropy_file, anisotropy_size_id, 6ul, transform_factory );
    n_indices                            = data.size();

    // Arrays
    anisotropy_index           = intfield( 0 );
    anisotropy_magnitude       = scalarfield( 0 );
    anisotropy_normal          = vectorfield( 0 );
    cubic_anisotropy_index     = intfield( 0 );
    cubic_anisotropy_magnitude = scalarfield( 0 );

    for( const auto & [i, k, k_vec, k4] : data )
    {
        if( k != 0 )
        {
            anisotropy_index.push_back( i );
            anisotropy_magnitude.push_back( k );
            anisotropy_normal.push_back( k_vec );
        }
        if( k4 != 0 )
        {
            Log( Log_Level::Debug, Log_Sender::IO, fmt::format( "appending spin K4\"{}\"", k4 ) );
            cubic_anisotropy_index.push_back( i );
            cubic_anisotropy_magnitude.push_back( k4 );
        }
    }
}
catch( ... )
{
    spirit_rethrow( fmt::format( "Could not read anisotropies from file \"{}\"", anisotropy_file ) );
}

} // namespace

void Anisotropy_from_Config(
    const std::string & config_file_name, const Data::Geometry & geometry, std::vector<std::string> & parameter_log,
    intfield & uniaxial_indices, scalarfield & uniaxial_magnitudes, vectorfield & uniaxial_normals,
    intfield & cubic_indices, scalarfield & cubic_magnitudes )
{
    std::string anisotropy_file{};
    bool anisotropy_from_file = false;
    int n_pairs               = 0;
    scalar K = 0, K4 = 0;
    Vector3 K_normal = { 0, 0, 0 };

    try
    {

        IO::Filter_File_Handle config_file_handle( config_file_name );

        // Anisotropy
        if( config_file_handle.Find( "n_anisotropy" ) )
            anisotropy_file = config_file_name;
        else if( config_file_handle.Find( "anisotropy_file" ) )
            config_file_handle >> anisotropy_file;

        if( !anisotropy_file.empty() )
        {
            // The file name should be valid so we try to read it
            Anisotropy_from_File(
                anisotropy_file, geometry, n_pairs, uniaxial_indices, uniaxial_magnitudes, uniaxial_normals,
                cubic_indices, cubic_magnitudes );

            anisotropy_from_file = true;
            if( !uniaxial_indices.empty() )
            {
                K        = uniaxial_magnitudes[0];
                K_normal = uniaxial_normals[0];
            }
            else
            {
                K        = 0;
                K_normal = { 0, 0, 0 };
            }
            if( !cubic_indices.empty() )
                K4 = cubic_magnitudes[0];
            else
                K4 = 0;
        }
        else
        {
            // Read parameters from config
            config_file_handle.Read_Single( K, "anisotropy_magnitude" );
            config_file_handle.Read_Vector3( K_normal, "anisotropy_normal" );
            K_normal.normalize();

            config_file_handle.Read_Single( K4, "cubic_anisotropy_magnitude" );

            if( K != 0 )
            {
                // Fill the arrays
                for( std::size_t i = 0; i < uniaxial_indices.size(); ++i )
                {
                    uniaxial_indices[i]    = static_cast<int>( i );
                    uniaxial_magnitudes[i] = K;
                    uniaxial_normals[i]    = K_normal;
                }
            }
            else
            {
                uniaxial_indices    = intfield( 0 );
                uniaxial_magnitudes = scalarfield( 0 );
                uniaxial_normals    = vectorfield( 0 );
            }
            if( K4 != 0 )
            {
                // Fill the arrays
                for( std::size_t i = 0; i < cubic_indices.size(); ++i )
                {
                    cubic_indices[i]    = static_cast<int>( i );
                    cubic_magnitudes[i] = K4;
                }
            }
            else
            {
                cubic_indices    = intfield( 0 );
                cubic_magnitudes = scalarfield( 0 );
            }
        }
    }
    catch( ... )
    {
        spirit_handle_exception_core(
            fmt::format( "Unable to read anisotropy from config file \"{}\"", config_file_name ) );
    }

    if( anisotropy_from_file )
        parameter_log.emplace_back( fmt::format( "    K from file \"{}\"", anisotropy_file ) );
    parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "anisotropy[0]", K ) );
    parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "anisotropy_normal[0]", K_normal.transpose() ) );
    parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "cubic_anisotropy_magnitude[0]", K4 ) );
}

} // namespace IO
