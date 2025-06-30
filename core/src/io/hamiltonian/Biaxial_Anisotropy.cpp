#include <engine/Vectormath_Defines.hpp>
#include <io/Dataparser.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/Tableparser.hpp>
#include <io/configparser/Converter.hpp>

#include <iso646.h>
#include <vector>

using Utility::Log_Level, Utility::Log_Sender;

namespace IO
{

namespace
{

void Biaxial_Anisotropy_Axes_from_File(
    Filter_File_Handle & file_handle, const Data::Geometry & geometry, int & n_axes,
    std::map<int, std::pair<Vector3, Vector3>> & anisotropy_axes ) noexcept
try
{
    // parser initialization
    using AnisotropyTableParser = TableParserInit<std::array<int, 1>, std::array<scalar, 12>>;
    const AnisotropyTableParser parser(
        { "i", "k1x", "k1y", "k1z", "k1a", "k1b", "k1c", "k2x", "k2y", "k2z", "k2a", "k2b", "k2c" } );

    // factory function for creating a lambda that transforms the row that is read
    auto transform_factory = [&file_handle, &geometry]( const std::map<std::string_view, int> & idx )
    {
        bool K1_xyz = ( idx.at( "k1x" ) >= 0 && idx.at( "k1y" ) >= 0 && idx.at( "k1z" ) >= 0 );
        bool K1_abc = ( idx.at( "k1a" ) >= 0 && idx.at( "k1b" ) >= 0 && idx.at( "k1c" ) >= 0 );
        bool K2_xyz = ( idx.at( "k2x" ) >= 0 && idx.at( "k2y" ) >= 0 && idx.at( "k2z" ) >= 0 );
        bool K2_abc = ( idx.at( "k2a" ) >= 0 && idx.at( "k2b" ) >= 0 && idx.at( "k2c" ) >= 0 );

        if( !( ( K1_xyz || K1_abc ) && ( K2_xyz || K2_abc ) ) )
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format( "No anisotropy data could be found in header of \"{}\"", file_handle.filename() ) );

        return [K1_xyz, K1_abc, K2_xyz, K2_abc, &geometry](
                   const AnisotropyTableParser::read_row_t & row ) -> std::pair<int, std::pair<Vector3, Vector3>>
        {
            auto [i, k1x, k1y, k1z, k1a, k1b, k1c, k2x, k2y, k2z, k2a, k2b, k2c] = row;

            Vector3 K1_temp, K2_temp;
            if( K1_xyz )
                K1_temp = { k1x, k1y, k1z };
            // Anisotropy vector orientation
            if( K1_abc )
            {
                K1_temp = { k1a, k1b, k1c };
                K1_temp = { K1_temp.dot( geometry.lattice_constant * geometry.bravais_vectors[0] ),
                            K1_temp.dot( geometry.lattice_constant * geometry.bravais_vectors[1] ),
                            K1_temp.dot( geometry.lattice_constant * geometry.bravais_vectors[2] ) };
            }
            K1_temp.normalize();

            if( K2_xyz )
                K2_temp = { k2x, k2y, k2z };
            // Anisotropy vector orientation
            if( K2_abc )
            {
                K2_temp = { k2a, k2b, k2c };
                K2_temp = { K2_temp.dot( geometry.lattice_constant * geometry.bravais_vectors[0] ),
                            K2_temp.dot( geometry.lattice_constant * geometry.bravais_vectors[1] ),
                            K2_temp.dot( geometry.lattice_constant * geometry.bravais_vectors[2] ) };
            }

            // orthogonalize and normalize
            K2_temp = K2_temp - K1_temp.dot( K2_temp ) * K1_temp;
            K2_temp.normalize();

            return std::pair( i, std::pair{ K1_temp, K2_temp } );
        };
    };

    const auto data = parser.parse( file_handle, "n_biaxial_anisotropy_axes", 7ul, transform_factory );
    n_axes          = data.size();

    anisotropy_axes = std::map( begin( data ), end( data ) );
}
catch( ... )
{
    spirit_rethrow( fmt::format( "Could not read anisotropy axes from \"{}\"", file_handle.filename() ) );
}

void Biaxial_Anisotropy_Terms_from_File(
    Filter_File_Handle & file_handle, const Data::Geometry &, int & n_terms,
    std::map<int, field<PolynomialTerm>> & anisotropy_terms ) noexcept
try
{
    // parser initialization
    using AnisotropyTableParser
        = TableParserInit<std::array<int, 1>, std::array<unsigned int, 3>, std::array<scalar, 1>>;
    const AnisotropyTableParser parser( { "i", "n1", "n2", "n3", "k" } );

    // factory function for creating a lambda that transforms the row that is read
    auto transform_factory = [&file_handle]( const std::map<std::string_view, int> & idx )
    {
        if( idx.at( "i" ) < 0 || idx.at( "k" ) < 0
            || ( idx.at( "n1" ) < 0 && idx.at( "n2" ) < 0 && idx.at( "n3" ) < 0 ) )
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format( "No anisotropy data could be found in header of \"{}\"", file_handle.filename() ) );

        return []( AnisotropyTableParser::read_row_t row ) -> std::pair<int, PolynomialTerm>
        {
            auto [i, n1, n2, n3, k] = row;
            return { i, PolynomialTerm{ k, n1, n2, n3 } };
        };
    };

    const auto data = parser.parse( file_handle, "n_biaxial_anisotropy_terms", 6ul, transform_factory );
    n_terms         = data.size();

    anisotropy_terms.clear();
    for( const auto & [i, term] : data )
        anisotropy_terms[i].push_back( term );
}
catch( ... )
{
    spirit_rethrow( fmt::format( "Could not read anisotropy terms from \"{}\"", file_handle.filename() ) );
}

void Biaxial_Anisotropy_from_File(
    Filter_File_Handle & anisotropy_axes_file, Filter_File_Handle & anisotropy_terms_file,
    const Data::Geometry & geometry, intfield & anisotropy_indices,
    field<PolynomialBasis> & anisotropy_polynomial_bases, field<unsigned int> & anisotropy_polynomial_site_p,
    field<PolynomialTerm> & anisotropy_polynomial_terms ) noexcept
try
{
    int n_axes = 0, n_terms = 0;
    auto anisotropy_axes  = std::map<int, std::pair<Vector3, Vector3>>();
    auto anisotropy_terms = std::map<int, field<PolynomialTerm>>();

    Log( Log_Level::Debug, Log_Sender::IO,
         fmt::format( "Reading anisotropy axes from {}", anisotropy_axes_file.filename() ) );
    Biaxial_Anisotropy_Axes_from_File( anisotropy_axes_file, geometry, n_axes, anisotropy_axes );

    Log( Log_Level::Debug, Log_Sender::IO,
         fmt::format( "Reading anisotropy terms from {}", anisotropy_terms_file.filename() ) );
    Biaxial_Anisotropy_Terms_from_File( anisotropy_terms_file, geometry, n_terms, anisotropy_terms );

    // Arrays
    anisotropy_indices           = intfield{};
    anisotropy_polynomial_bases  = field<PolynomialBasis>{};
    anisotropy_polynomial_site_p = field<unsigned int>{};
    anisotropy_polynomial_terms  = field<PolynomialTerm>{};

    if( n_terms > 0 )
    {
        anisotropy_polynomial_site_p.push_back( 0 );
        anisotropy_polynomial_terms.reserve( n_terms );
    }

    const scalar thresh = 1e-5;
    for( const auto & [i, axes] : anisotropy_axes )
    {
        if( axes.first.norm() > thresh && axes.second.norm() > thresh )
        {
            if( const auto & terms = anisotropy_terms[i]; !terms.empty() )
            {
                anisotropy_indices.push_back( i );
                anisotropy_polynomial_bases.push_back(
                    PolynomialBasis{ axes.first, axes.second, axes.first.cross( axes.second ).normalized() } );
                anisotropy_polynomial_site_p.push_back( anisotropy_polynomial_site_p.back() + terms.size() );
                std::copy( begin( terms ), end( terms ), std::back_inserter( anisotropy_polynomial_terms ) );
            }
            else
            {
                Log( Log_Level::Warning, Log_Sender::IO,
                     fmt::format( "Anisotropy axes specified at site i={} but no polynomial terms were found.", i ) );
            }
        }
        else
        {
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format(
                     "Discarding anisotropy axes at site i={} because they are smaller than threshold ({})", i,
                     thresh ) );
        }
    }

    if( int diff = anisotropy_terms.size() - anisotropy_axes.size(); diff > 0 )
    {
        Log( Log_Level::Warning, Log_Sender::IO,
             fmt::format( "There were polynomials specified without any matching axes at {} sites.", diff ) );
    }
}
catch( ... )
{
    spirit_rethrow( fmt::format(
        "Could not read anisotropies from files \"{}\" & \"{}\" ", anisotropy_axes_file.filename(),
        anisotropy_terms_file.filename() ) );
}

} // namespace

auto Biaxial_Anisotropy_from_TOML(
    const toml::table & tbl, const Data::Geometry & geometry,
    std::vector<std::string> & parameter_log ) -> Engine::Spin::Interaction::Biaxial_Anisotropy::Data
{
    Engine::Spin::Interaction::Biaxial_Anisotropy::Data data{};
    try
    {
        auto biaxial_anisotropy_axes_handle = [&tbl]() -> std::optional<Filter_File_Handle>
        {
            if( auto value = tbl["biaxial_anisotropy_axes"].value<std::string>() )
                return Filter_File_Handle::from_string_optional( *value );
            else
                return std::nullopt;
        }();

        auto biaxial_anisotropy_terms_handle = [&tbl]() -> std::optional<Filter_File_Handle>
        {
            if( auto value = tbl["biaxial_anisotropy_terms"].value<std::string>() )
                return Filter_File_Handle::from_string_optional( *value );
            else
                return std::nullopt;
        }();

        if( !biaxial_anisotropy_terms_handle xor !biaxial_anisotropy_axes_handle )
        {
            Log( Log_Level::Error, Log_Sender::IO,
                 fmt::format(
                     "Incomplete specification for biaxial anisotropy: missing or invalid specification for \"{}\"",
                     biaxial_anisotropy_axes_handle ? "axes" : "terms" ) );
        }
        else if( biaxial_anisotropy_terms_handle && biaxial_anisotropy_axes_handle )
        {
            Biaxial_Anisotropy_from_File(
                *biaxial_anisotropy_axes_handle, *biaxial_anisotropy_terms_handle, geometry, data.indices, data.bases,
                data.site_p, data.terms );
        }
    }
    catch( ... )
    {
        spirit_handle_exception_core( "Could not read biaxial anisotropy!" );
    }

    if( !data.bases.empty() )
    {
        const auto & p = data.bases[0];
        parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "biaxial_anisotropy[0].k1", p.k1.transpose() ) );
        parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "biaxial_anisotropy[0].k2", p.k2.transpose() ) );
        parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "biaxial_anisotropy[0].k3", p.k3.transpose() ) );
    }
    if( !data.terms.empty() )
        parameter_log.emplace_back( fmt::format( "    Read {} biaxial anisotropy terms!", data.terms.size() ) );

    return data;
}

auto Biaxial_Anisotropy_to_TOML( const Engine::Spin::Interaction::Biaxial_Anisotropy::Data * data ) -> toml::table
{
    if( !data )
        return toml::table{};

    const intfield & indices                        = data->indices;
    const field<PolynomialBasis> & polynomial_bases = data->bases;
    const field<unsigned int> & polynomial_site_p   = data->site_p;
    const field<PolynomialTerm> & polynomial_terms  = data->terms;

    const auto n_anisotropy_axes  = indices.size();
    const auto n_anisotropy_terms = polynomial_terms.size();

    assert( n_anisotropy_axes != 0 || n_anisotropy_terms == 0 );

    toml::table tbl;
    if( n_anisotropy_axes > 0 )
    {
        std::ostringstream oss;
        oss << fmt::format(
            "{:^3}   {:^15} {:^15} {:^15}  {:^15} {:^15} {:^15}\n", "i", "K1x", "K1y", "K1z", "K2x", "K2y", "K2z" );

        for( std::size_t i = 0; i < n_anisotropy_axes; ++i )
        {
            oss << fmt::format(
                "{:^3}   {:^15.8f} {:^15.8f} {:^15.8f}  {:^15.8f} {:^15.8f} {:^15.8f}\n", indices[i],
                polynomial_bases[i].k1[0], polynomial_bases[i].k1[1], polynomial_bases[i].k1[2],
                polynomial_bases[i].k2[0], polynomial_bases[i].k2[1], polynomial_bases[i].k2[2] );
        }
        tbl.insert( "biaxial_anisotropy_axes", oss.str() );
    }

    if( n_anisotropy_terms > 0 )
    {
        std::ostringstream oss;
        oss << fmt::format( "{:^3}  {:^3} {:^3} {:^3}  {:^15}\n", "i", "n1", "n2", "n3", "k" );
        for( std::size_t i = 0; i < n_anisotropy_terms; ++i )
        {
            for( std::size_t j = polynomial_site_p[i]; j < polynomial_site_p[i + 1]; ++j )
            {
                const auto & p = polynomial_terms[j];
                oss << fmt::format( "{:^3}  {:^3} {:^3} {:^3}  {:^15.8f}\n", i, p.n1, p.n2, p.n3, p.coefficient );
            }
        }
        tbl.insert( "biaxial_anisotropy_terms", oss.str() );
    }

    return tbl;
}

} // namespace IO
