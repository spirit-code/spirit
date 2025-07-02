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
    Filter_File_Handle & anisotropy_file, const Data::Geometry & geometry, intfield & anisotropy_index,
    scalarfield & anisotropy_magnitude, vectorfield & anisotropy_normal, intfield & cubic_anisotropy_index,
    scalarfield & cubic_anisotropy_magnitude ) noexcept
try
{
    Log( Log_Level::Debug, Log_Sender::IO, fmt::format( "Reading anisotropy from {}", anisotropy_file.filename() ) );

    // parser initialization
    using AnisotropyTableParser = TableParserInit<std::array<int, 1>, std::array<scalar, 8>>;
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
                 fmt::format(
                     "No anisotropy data could be found in header of file \"{}\"", anisotropy_file.filename() ) );

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

    const auto reset = [size = data.size()]( auto & container ) { container.clear(), container.reserve( size ); };
    reset( anisotropy_index );
    reset( anisotropy_normal );
    reset( anisotropy_magnitude );
    reset( cubic_anisotropy_index );
    reset( cubic_anisotropy_magnitude );

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
    spirit_rethrow( fmt::format( "Could not read anisotropies from file \"{}\"", anisotropy_file.filename() ) );
}

} // namespace

auto Anisotropy_from_TOML(
    const toml::table & tbl, const Data::Geometry & geometry,
    std::vector<std::string> & parameter_log ) -> Anisotropy_from_TOML_result_t
{
    Anisotropy_from_TOML_result_t result{};
    auto & [anisotropy, cubic_anisotropy] = result;

    scalar K = 0, K4 = 0;
    Vector3 K_normal = { 0, 0, 0 };

    if( auto anisotropy_table = tbl["anisotropy"].as_string() )
    {
        auto file_handle = Filter_File_Handle::from_string( anisotropy_table->get() );
        Anisotropy_from_File(
            file_handle, geometry, anisotropy.indices, anisotropy.magnitudes, anisotropy.normals,
            cubic_anisotropy.indices, cubic_anisotropy.magnitudes );

        if( !anisotropy.indices.empty() )
        {
            K        = anisotropy.magnitudes[0];
            K_normal = anisotropy.normals[0];
        }
        if( !cubic_anisotropy.indices.empty() )
            K4 = cubic_anisotropy.magnitudes[0];

        parameter_log.emplace_back( fmt::format( "    K from table \"{}\"", file_handle.filename() ) );
    }
    else
    {
        K  = static_cast<scalar>( tbl["anisotropy_magnitude"].value_or( 0.0 ) );
        K4 = static_cast<scalar>( tbl["cubic_anisotropy_magnitude"].value_or( 0.0 ) );
        if( auto normal = tbl["anisotropy_normal"].as_array() )
        {
            if( auto parsed_k = toml_transform<Vector3>( *normal ) )
                K_normal = parsed_k->normalized();
        }

        if( K != 0 && K_normal.norm() > 1e-8 )
        {
            anisotropy.magnitudes = scalarfield( geometry.n_cell_atoms, K );
            anisotropy.normals    = vectorfield( geometry.n_cell_atoms, K_normal );
            anisotropy.indices    = intfield( geometry.n_cell_atoms );
            std::iota( anisotropy.indices.begin(), anisotropy.indices.end(), 0 );
        }

        if( K4 != 0 )
        {
            cubic_anisotropy.magnitudes = scalarfield( geometry.n_cell_atoms, K4 );
            cubic_anisotropy.indices    = intfield( geometry.n_cell_atoms );
            std::iota( cubic_anisotropy.indices.begin(), cubic_anisotropy.indices.end(), 0 );
        }
    }

    parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "anisotropy[0]", K ) );
    parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "anisotropy_normal[0]", K_normal.transpose() ) );
    parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "cubic_anisotropy_magnitude[0]", K4 ) );

    return result;
}

auto Anisotropy_to_TOML(
    const Engine::Spin::Interaction::Anisotropy::Data * uniaxial,
    const Engine::Spin::Interaction::Cubic_Anisotropy::Data * cubic ) -> toml::table
{
    struct Row
    {
        Vector3 K_normal = Vector3::Zero();
        scalar K = 0, K4 = 0;
    };

    const auto n_uniaxial = uniaxial ? uniaxial->indices.size() : 0;
    const auto n_cubic    = cubic ? cubic->indices.size() : 0;
    std::unordered_map<int, Row> output{};
    output.reserve( n_uniaxial + n_cubic ); // reserve for the worst case of no overlap
    for( unsigned int i = 0; i < n_uniaxial; ++i )
    {
        auto & row   = output[uniaxial->indices[i]];
        row.K        = uniaxial->magnitudes[i];
        row.K_normal = uniaxial->normals[i];
    };
    for( unsigned int i = 0; i < n_cubic; ++i )
    {
        output[cubic->indices[i]].K4 = cubic->magnitudes[i];
    }

    if( !output.empty() )
    {
        std::ostringstream oss;
        oss << '\n'
            << fmt::format( "{:^3}   {:^15} {:^15} {:^15}  {:^15}  {:^15}\n", "i", "K1x", "K1y", "K1z", "K", "K4" );
        for( auto & [idx, row] : output )
        {
            oss << fmt::format(
                "{:^3}   {:^15.8f} {:^15.8f} {:^15.8f}  {:^15.8f}  {:^15.8f}\n", idx, row.K_normal[0], row.K_normal[1],
                row.K_normal[2], row.K, row.K4 );
        }
        return toml::table{ { "anisotropy", oss.str() } };
    }
    else
    {
        return toml::table{};
    }
}

} // namespace IO
