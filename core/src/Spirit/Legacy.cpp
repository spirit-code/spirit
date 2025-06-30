#include <Spirit/Legacy.h>
#include <Spirit/Spirit_Defines.h>

#include <io/IO.hpp>
#include <io/configparser/Converter.hpp>

void Legacy_Convert_Config_to_TOML( const char * config, const char * toml_config ) noexcept
try
{
    IO::write_to_file( IO::convert::Config( config ), toml_config );
}
catch( ... )
{
    spirit_handle_exception_api( -1, -1 );
}
