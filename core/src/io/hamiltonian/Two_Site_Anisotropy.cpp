#include <engine/Vectormath_Defines.hpp>
#include <io/Dataparser.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/Tableparser.hpp>

#include <bitset>
#include <unordered_set>
#include <vector>

using Utility::Log_Level, Utility::Log_Sender;

namespace IO
{

namespace
{

// Read from table file or string
auto Two_Site_Anisotropy_from_File( Filter_File_Handle & table_file, const Data::Geometry & )
    -> Engine::Spin::Interaction::Two_Site_Anisotropy::Data
{
    Log( Log_Level::Debug, Log_Sender::IO,
         fmt::format( "Reading \"Two_Site_Anisotropy\" table from \"{}\"", table_file.filename() ) );

    using CustomTableParser = TableParserInit<std::array<int, 5>, std::array<scalar, 6>>;
    CustomTableParser parser( { "i", "j", "da", "db", "dc", "kijxx", "kijyy", "kijzz", "kijyz", "kijxz", "kijxy" } );

    auto transform_factory = [&table_file]( const std::map<std::string_view, int> & idx )
    {
        const std::bitset<3> Kdiag = [&idx]
        {
            std::bitset<3> flags{};
            flags[0] = idx.at( "kijxx" ) >= 0;
            flags[1] = idx.at( "kijyy" ) >= 0;
            flags[2] = idx.at( "kijzz" ) >= 0;
            return flags;
        }();

        if( idx.at( "j" ) < 0 && Kdiag.count() == 1 )
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format(
                     "No interactions could be found in two-site anisotropy file \"{}\"", table_file.filename() ) );

        if( Kdiag.count() == 3 )
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format(
                     "Found three diagonal elements for two-site anisotropy. "
                     "This could interfere with the Heisenberg exchange interaction! "
                     "file: \"{}\"! ",
                     table_file.filename() ) );

        return [Kdiag]( const CustomTableParser::read_row_t & row )
        {
            struct RowData
            {
                int i, j, da, db, dc;
                scalar Kijxx, Kijyy, Kijzz, Kijyz, Kijxz, Kijxy;
            };

            auto data = IO::make_from_tuple<RowData>( row );

            std::pair<Pair, std::optional<std::array<scalar, 6>>> result
                = { Pair{ data.i, data.j, { data.da, data.db, data.dc } }, std::nullopt };
            if( Kdiag.count() < 2 )
                return result;
            else if(
                data.Kijxx == 0 && data.Kijxy == 0 && data.Kijxz == 0 && data.Kijyy == 0 && data.Kijyz == 0
                && data.Kijzz == 0 )
                return result;
            else if( Kdiag.count() == 2 )
            {
                if( !Kdiag[0] )
                    data.Kijxx = -( data.Kijyy + data.Kijzz );
                else if( !Kdiag[1] )
                    data.Kijyy = -( data.Kijxx + data.Kijzz );
                else
                    data.Kijzz = -( data.Kijxx + data.Kijyy );
            }

            result.second.emplace(
                std::array{ data.Kijxx, data.Kijyy, data.Kijzz, data.Kijyz, data.Kijxz, data.Kijxy } );
            return result;
        };
    };

    const auto data = parser.parse( table_file, 11, transform_factory );

    auto result = Engine::Spin::Interaction::Two_Site_Anisotropy::Data{};
    result.pairs.reserve( data.size() );
    result.coefficients.reserve( data.size() );

    std::unordered_set<Pair, equiv_hash<Pair>> seen_pairs;
    for( const auto & [pair, coeff] : data )
    {
        if( coeff && seen_pairs.insert( pair ).second )
        {
            result.pairs.push_back( pair );
            std::copy( coeff->begin(), coeff->end(), std::back_inserter( result.coefficients ) );
        }
    }

    result.pairs.shrink_to_fit();
    result.coefficients.shrink_to_fit();

    return result;
}

} // namespace

auto Two_Site_Anisotropy_from_TOML(
    const toml::table & tbl, const Data::Geometry & geometry,
    std::vector<std::string> & parameter_log ) -> Engine::Spin::Interaction::Two_Site_Anisotropy::Data
{
    auto two_site_anisotropy = tbl["two_site_anisotropy"].as_string();
    if( !two_site_anisotropy )
        return {};

    auto table_handle = Filter_File_Handle::from_string( two_site_anisotropy->get() );
    const auto result = Two_Site_Anisotropy_from_File( table_handle, geometry );

    // TODO: provide some logging on what was read from the input file.
    return result;
}

auto Two_Site_Anisotropy_to_TOML( const Engine::Spin::Interaction::Two_Site_Anisotropy::Data * data ) -> toml::table
{
    if( !data )
        return toml::table{};

    std::ostringstream oss;
    if( data->pairs.empty() || data->coefficients.empty() )
    {
        oss << '\n'
            << fmt::format(
                   "{:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15} {:^15} {:^15} {:^15} {:^15} {:^15}\n", "i", "j", "da",
                   "db", "dc", "Kijxx", "Kijyy", "Kijzz", "Kijyz", "Kijxz", "Kijxy" );
        const auto & pairs = data->pairs;
        const auto & coeff = data->coefficients;
        for( unsigned int i = 0, k = 0; i < pairs.size() && k + 5 < coeff.size(); ++i, k += 6 )
        {
            oss << fmt::format(
                "{:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15.8f} {:^15.8f} {:^15.8f} {:^15.8f} {:^15.8f} {:^15.8f}\n",
                pairs[i].i, pairs[i].j, pairs[i].translations[0], pairs[i].translations[1], pairs[i].translations[2],
                coeff[k + 0], coeff[k + 1], coeff[k + 2], coeff[k + 3], coeff[k + 4], coeff[k + 5] );
        }
    }

    return toml::table{ { "two_site_anisotropy", oss.str() } };
}

} // namespace IO
