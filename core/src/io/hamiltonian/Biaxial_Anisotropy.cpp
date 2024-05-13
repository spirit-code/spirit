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

void Biaxial_Anisotropy_Axes_from_File(
    const std::string & anisotropy_axes_file, const Data::Geometry & geometry, int & n_axes,
    std::map<int, std::pair<Vector3, Vector3>> & anisotropy_axes ) noexcept
try
{
    // parser initialization
    using AnisotropyTableParser = TableParser<
        int, scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar>;
    const AnisotropyTableParser parser(
        { "i", "k1x", "k1y", "k1z", "k1a", "k1b", "k1c", "k2x", "k2y", "k2z", "k2a", "k2b", "k2c" } );

    // factory function for creating a lambda that transforms the row that is read
    auto transform_factory = [&anisotropy_axes_file, &geometry]( const std::map<std::string_view, int> & idx )
    {
        bool K1_xyz = ( idx.at( "k1x" ) >= 0 && idx.at( "k1y" ) >= 0 && idx.at( "k1z" ) >= 0 );
        bool K1_abc = ( idx.at( "k1a" ) >= 0 && idx.at( "k1b" ) >= 0 && idx.at( "k1c" ) >= 0 );
        bool K2_xyz = ( idx.at( "k2x" ) >= 0 && idx.at( "k2y" ) >= 0 && idx.at( "k2z" ) >= 0 );
        bool K2_abc = ( idx.at( "k2a" ) >= 0 && idx.at( "k2b" ) >= 0 && idx.at( "k2c" ) >= 0 );

        if( !( ( K1_xyz || K1_abc ) && ( K2_xyz || K2_abc ) ) )
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format( "No anisotropy data could be found in header of file \"{}\"", anisotropy_axes_file ) );

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

    const auto data = parser.parse( anisotropy_axes_file, "n_biaxial_anisotropy_axes", 7ul, transform_factory );
    n_axes          = data.size();

    anisotropy_axes = std::map( begin( data ), end( data ) );
}
catch( ... )
{
    spirit_rethrow( fmt::format( "Could not read anisotropy axes from file \"{}\"", anisotropy_axes_file ) );
}

void Biaxial_Anisotropy_Terms_from_File(
    const std::string & anisotropy_terms_file, const Data::Geometry &, int & n_terms,
    std::map<int, field<PolynomialTerm>> & anisotropy_terms ) noexcept
try
{
    // parser initialization
    using AnisotropyTableParser = TableParser<int, unsigned int, unsigned int, unsigned int, scalar>;
    const AnisotropyTableParser parser( { "i", "n1", "n2", "n3", "k" } );

    // factory function for creating a lambda that transforms the row that is read
    auto transform_factory = [&anisotropy_terms_file]( const std::map<std::string_view, int> & idx )
    {
        if( idx.at( "i" ) < 0 || idx.at( "k" ) < 0
            || ( idx.at( "n1" ) < 0 && idx.at( "n2" ) < 0 && idx.at( "n3" ) < 0 ) )
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format( "No anisotropy data could be found in header of file \"{}\"", anisotropy_terms_file ) );

        return []( AnisotropyTableParser::read_row_t row ) -> std::pair<int, PolynomialTerm>
        {
            auto [i, n1, n2, n3, k] = row;
            return { i, PolynomialTerm{ k, n1, n2, n3 } };
        };
    };

    const auto data = parser.parse( anisotropy_terms_file, "n_biaxial_anisotropy_terms", 6ul, transform_factory );
    n_terms         = data.size();

    anisotropy_terms.clear();
    for( const auto & [i, term] : data )
        anisotropy_terms[i].push_back( term );
}
catch( ... )
{
    spirit_rethrow( fmt::format( "Could not read anisotropy terms from file \"{}\"", anisotropy_terms_file ) );
}

void Biaxial_Anisotropy_from_File(
    const std::string & anisotropy_axes_file, const std::string & anisotropy_terms_file,
    const Data::Geometry & geometry, int & n_indices, intfield & anisotropy_indices,
    field<PolynomialBasis> & anisotropy_polynomial_bases, field<unsigned int> & anisotropy_polynomial_site_p,
    field<PolynomialTerm> & anisotropy_polynomial_terms ) noexcept
try
{
    int n_axes = 0, n_terms = 0;
    auto anisotropy_axes  = std::map<int, std::pair<Vector3, Vector3>>();
    auto anisotropy_terms = std::map<int, field<PolynomialTerm>>();

    Log( Log_Level::Debug, Log_Sender::IO, "Reading anisotropy axes from file " + anisotropy_axes_file );
    Biaxial_Anisotropy_Axes_from_File( anisotropy_axes_file, geometry, n_axes, anisotropy_axes );

    Log( Log_Level::Debug, Log_Sender::IO, "Reading anisotropy terms from file " + anisotropy_terms_file );
    Biaxial_Anisotropy_Terms_from_File( anisotropy_terms_file, geometry, n_terms, anisotropy_terms );

    n_indices = n_axes + n_terms;

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
        "Could not read anisotropies from files \"{}\" & \"{}\" ", anisotropy_axes_file, anisotropy_terms_file ) );
}

} // namespace

void Biaxial_Anisotropy_from_Config(
    const std::string & config_file_name, const Data::Geometry & geometry, std::vector<std::string> & parameter_log,
    intfield & indices, field<PolynomialBasis> & polynomial_bases, field<unsigned int> & polynomial_site_p,
    field<PolynomialTerm> & polynomial_terms )
{
    std::string biaxial_anisotropy_axes_file  = "";
    std::string biaxial_anisotropy_terms_file = "";
    int n_biaxial_anisotropy                  = 0;
    try
    {
        IO::Filter_File_Handle config_file_handle( config_file_name );

        if( config_file_handle.Find( "n_biaxial_anisotropy_axes" ) )
            biaxial_anisotropy_axes_file = config_file_name;
        else if( config_file_handle.Find( "biaxial_anisotropy_axes_file" ) )
            config_file_handle >> biaxial_anisotropy_axes_file;

        if( config_file_handle.Find( "n_biaxial_anisotropy_terms" ) )
            biaxial_anisotropy_terms_file = config_file_name;
        else if( config_file_handle.Find( "biaxial_anisotropy_terms_file" ) )
            config_file_handle >> biaxial_anisotropy_terms_file;

        if( biaxial_anisotropy_terms_file.empty() xor biaxial_anisotropy_axes_file.empty() )
        {
            Log( Log_Level::Error, Log_Sender::IO,
                 fmt::format(
                     "Found incomplete specification for biaxial anisotropy: missing specification for \"{}\"",
                     biaxial_anisotropy_axes_file.empty() ? "axes" : "terms" ) );
        }
        else if( !biaxial_anisotropy_terms_file.empty() && !biaxial_anisotropy_axes_file.empty() )
        {
            Biaxial_Anisotropy_from_File(
                biaxial_anisotropy_axes_file, biaxial_anisotropy_terms_file, geometry, n_biaxial_anisotropy, indices,
                polynomial_bases, polynomial_site_p, polynomial_terms );
        }
    }
    catch( ... )
    {
        spirit_handle_exception_core(
            fmt::format( "Could not read biaxial anisotropy from config \"{}\"", config_file_name ) );
    }

    if( !polynomial_bases.empty() )
    {
        const auto & p = polynomial_bases[0];
        parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "biaxial_anisotropy[0].k1", p.k1.transpose() ) );
        parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "biaxial_anisotropy[0].k2", p.k2.transpose() ) );
        parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "biaxial_anisotropy[0].k3", p.k3.transpose() ) );
    }
    if( !biaxial_anisotropy_terms_file.empty() )
        parameter_log.emplace_back(
            fmt::format( "    biaxial anisotropy terms from file \"{}\"", biaxial_anisotropy_terms_file ) );
}

} // namespace IO
