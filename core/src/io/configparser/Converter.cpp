#include <io/Configparser.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/configparser/Converter.hpp>
#include <utility/Logging.hpp>

#include <toml++/toml.hpp>

#include <string>

namespace IO
{

namespace convert
{

using Utility::Log_Level;
using Utility::Log_Sender;

auto Logging( const std::string & config_file_name ) -> toml::table
{
    // Verbosity and Reject Level are read as integers
    int i_level_file = 5, i_level_console = 5;
    std::string output_folder = ".";
    std::string file_tag      = "";
    bool messages_to_file = true, messages_to_console = true, save_input_initial = false, save_input_final = false,
         save_positions_initial = false, save_positions_final = false, save_neighbours_initial = false,
         save_neighbours_final = false;
    try
    {
        if( !config_file_name.empty() )
        {
            try
            {
                Log( Log_Level::Debug, Log_Sender::IO, "Building Log" );
                IO::Filter_File_Handle config_file_handle( config_file_name );

                // Time tag
                config_file_handle.Read_Single( file_tag, "output_file_tag" );

                // Output folder
                config_file_handle.Read_Single( output_folder, "log_output_folder" );

                // Save Output (Log Messages) to file
                config_file_handle.Read_Single( messages_to_file, "log_to_file" );
                // File Accept Level
                config_file_handle.Read_Single( i_level_file, "log_file_level" );

                // Print Output (Log Messages) to console
                config_file_handle.Read_Single( messages_to_console, "log_to_console" );
                // File Accept Level
                config_file_handle.Read_Single( i_level_console, "log_console_level" );

                // Save Input (parameters from config file and defaults) on State Setup
                config_file_handle.Read_Single( save_input_initial, "save_input_initial" );
                // Save Input (parameters from config file and defaults) on State Delete
                config_file_handle.Read_Single( save_input_final, "save_input_final" );

                // Save Input (parameters from config file and defaults) on State Setup
                config_file_handle.Read_Single( save_positions_initial, "save_positions_initial" );
                // Save Input (parameters from config file and defaults) on State Delete
                config_file_handle.Read_Single( save_positions_final, "save_positions_final" );

                // Save Input (parameters from config file and defaults) on State Setup
                config_file_handle.Read_Single( save_neighbours_initial, "save_neighbours_initial" );
                // Save Input (parameters from config file and defaults) on State Delete
                config_file_handle.Read_Single( save_neighbours_final, "save_neighbours_final" );
            }
            catch( ... )
            {
                spirit_rethrow( fmt::format(
                    "Failed to read log levels from file \"{}\". Leaving values at default.", config_file_name ) );
            }
        }
    }
    catch( ... )
    {
        spirit_handle_exception_core(
            fmt::format( "Unable to read logging parameters from config file \"{}\"", config_file_name ) );
    }

    return toml::table{
        { "output_file_tag", file_tag },
        { "output_folder", output_folder },
        { "log_file_level", i_level_file },
        { "log_console_level", i_level_console },
        { "log_to_file", messages_to_file },
        { "log_to_console", messages_to_console },
        { "save_input_initial", save_input_initial },
        { "save_input_final", save_input_final },
        { "save_positions_initial", save_positions_initial },
        { "save_positions_final", save_positions_final },
        { "save_neighbours_initial", save_neighbours_initial },
        { "save_neighbours_final", save_neighbours_final },
    };
}

} // namespace convert

} // namespace IO
