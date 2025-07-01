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
auto Quadruplets_from_File( Filter_File_Handle & quadruplets_file, const Data::Geometry & ) noexcept
    -> Engine::Spin::Interaction::Quadruplet::Data
try
{
    Log( Log_Level::Debug, Log_Sender::IO,
         fmt::format( "Reading spin quadruplets from \"{}\"", quadruplets_file.filename() ) );

    // parser initialization
    using QuadrupletTableParser = TableParserInit<std::array<int, 13>, std::array<scalar, 1>>;
    const QuadrupletTableParser parser(
        { "i", "j", "da_j", "db_j", "dc_j", "k", "da_k", "db_k", "dc_k", "l", "da_l", "db_l", "dc_l", "q" } );

    // factory function for creating a lambda that transforms the row that is read
    auto transform_factory = [&quadruplets_file]( const std::map<std::string_view, int> & idx )
    {
        if( idx.at( "q" ) < 0 )
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format(
                     "No interactions could be found in header of quadruplets file ", quadruplets_file.filename() ) );

        return []( const QuadrupletTableParser::read_row_t & row ) -> std::tuple<Quadruplet, scalar>
        {
            const auto & [i, j, da_j, db_j, dc_j, k, da_k, db_k, dc_k, l, da_l, db_l, dc_l, Q] = row;
            return std::make_tuple(
                Quadruplet{ i, j, k, l, { da_j, db_j, dc_j }, { da_k, db_k, dc_k }, { da_l, db_l, dc_l } }, Q );
        };
    };
    const auto data = parser.parse( quadruplets_file, "n_interaction_quadruplets", 20, transform_factory );

    auto quadruplets = Engine::Spin::Interaction::Quadruplet::Data{};
    quadruplets.quadruplets.reserve( data.size() );
    quadruplets.magnitudes.reserve( data.size() );

    for( const auto & [quadruplet, magnitude] : data )
    {
        if( magnitude != 0 )
        {
            quadruplets.quadruplets.push_back( quadruplet );
            quadruplets.magnitudes.push_back( magnitude );
        }
    }

    return quadruplets;
}
catch( ... )
{
    spirit_handle_exception_core(
        fmt::format( "Could not read quadruplets from file  \"{}\"", quadruplets_file.filename() ) );
    return Engine::Spin::Interaction::Quadruplet::Data{};
}

} // namespace

auto Quadruplets_from_TOML(
    const toml::table & tbl, const Data::Geometry & geometry,
    std::vector<std::string> & parameter_log ) -> Engine::Spin::Interaction::Quadruplet::Data
{
    auto quadruplets = tbl["quadruplets"].as_string();
    if( !quadruplets )
        return {};

    auto quadruplets_handle = Filter_File_Handle::from_string( quadruplets->get() );
    const auto result       = Quadruplets_from_File( quadruplets_handle, geometry );

    parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "n_quadruplets", result.quadruplets.size() ) );

    return result;
}

auto Quadruplets_to_TOML( const Engine::Spin::Interaction::Quadruplet::Data * data ) -> toml::table
{
    if( !data )
        return toml::table{};

    const auto & quadruplets           = data->quadruplets;
    const auto & quadruplet_magnitudes = data->magnitudes;

    std::ostringstream oss;
    if( !quadruplets.empty() )
    {
        oss << fmt::format(
            "{:^3} {:^3} {:^3} {:^3}    {:^3} {:^3} {:^3}    {:^3} {:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15}\n", "i",
            "j", "k", "l", "da_j", "db_j", "dc_j", "da_k", "db_k", "dc_k", "da_l", "db_l", "dc_l", "Q" );
        for( unsigned int i = 0; i < quadruplets.size(); ++i )
        {
            // clang-format off
            oss << fmt::format(
                "{:^3} {:^3} {:^3} {:^3}    {:^3} {:^3} {:^3}    {:^3} {:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15.8f}\n",
                quadruplets[i].i, quadruplets[i].j, quadruplets[i].k, quadruplets[i].l,
                quadruplets[i].d_j[0], quadruplets[i].d_j[1], quadruplets[i].d_j[2],
                quadruplets[i].d_k[0], quadruplets[i].d_k[1], quadruplets[i].d_k[2],
                quadruplets[i].d_l[0], quadruplets[i].d_l[1], quadruplets[i].d_l[2],
                quadruplet_magnitudes[i] );
            // clang-format on
        }
    }
    return toml::table{ { "quadruplets", oss.str() } };
}

} // namespace IO
