#include <engine/Vectormath_Defines.hpp>
#include <io/Dataparser.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/Tableparser.hpp>
#include <io/configparser/Converter.hpp>

#include <vector>

using Utility::Log_Level, Utility::Log_Sender;

namespace IO
{

void Gaussian_from_TOML(
    const toml::table & tbl, std::vector<std::string> & parameter_log, scalarfield & amplitude, scalarfield & width,
    vectorfield & center )
{
    auto gaussians = tbl["gaussians"].as_string();

    if( !gaussians )
        return;

    auto handle = Filter_File_Handle::from_string( gaussians->get() );

    const auto cap = handle.Get_N_Non_Comment_Lines();
    handle.To_Start();

    amplitude = scalarfield( 0 );
    width     = scalarfield( 0 );
    center    = vectorfield( 0 );

    amplitude.reserve( cap );
    width.reserve( cap );
    center.reserve( cap );

    while( handle.GetLine() )
    {
        std::array<scalar, 5> values;
        for( int i = 0; i < 5; ++i )
            handle >> values[i];

        amplitude.emplace_back( values[0] );
        width.emplace_back( values[1] );
        center.emplace_back( Vector3{ values[2], values[3], values[4] } );
    }

    const auto n_gaussians = amplitude.size();
    parameter_log.emplace_back( fmt::format( "    {0:<12} = {1}", "n_gaussians", n_gaussians ) );
    if( n_gaussians > 0 )
    {
        parameter_log.emplace_back( fmt::format( "    {0:<12} = {1}", "amplitude[0]", amplitude[0] ) );
        parameter_log.emplace_back( fmt::format( "    {0:<12} = {1}", "width[0]", width[0] ) );
        parameter_log.emplace_back( fmt::format( "    {0:<12} = {1}", "center[0]", center[0].transpose() ) );
    }
}

void Gaussian_from_Config(
    const std::string & config_file_name, std::vector<std::string> & parameter_log, scalarfield & amplitude,
    scalarfield & width, vectorfield & center )
{
    return Gaussian_from_TOML(
        convert::Interaction::Gaussian( config_file_name ), parameter_log, amplitude, width, center );
};

} // namespace IO
