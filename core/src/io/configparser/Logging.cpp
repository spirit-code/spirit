#include <io/Configparser.hpp>
#include <utility/Logging.hpp>
#include <utility/Timing.hpp>

#include <fmt/format.h>

#include <string>

// using namespace Utility;
using Utility::Log_Level;
using Utility::Log_Sender;

namespace IO
{

void Log_from_TOML( const toml::table & tbl, const Defaults & defaults, bool force_quiet )
try
{
    {
        std::string file_tag      = "";
        std::string output_folder = ".";

        read_value_with_default( tbl, "logging.output.file_tag", file_tag, defaults.output.file_tag, false );
        read_value_with_default( tbl, "logging.output.folder", output_folder, defaults.output.directory, false );
        Log.file_tag      = file_tag;
        Log.output_folder = output_folder;

        if( file_tag == "<time>" )
            Log.file_name = "Log_" + Utility::Timing::CurrentDateTime() + ".txt";
        else if( !file_tag.empty() )
            Log.file_name = "Log_" + file_tag + ".txt";
        else
            Log.file_name = "Log.txt";
    }

    const auto section = tbl["logging"];
    // "Quiet" settings
    if( force_quiet )
    {
        // Don't save the Log to file
        Log.messages_to_file = false;
        // Don't print the Log to console
        Log.messages_to_console = false;
        // Don't save input configs
        Log.save_input_initial = false;
        Log.save_input_final   = false;
        // Don't save positions
        Log.save_positions_initial = false;
        Log.save_positions_final   = false;
        // Don't save neighbours
        Log.save_neighbours_initial = false;
        Log.save_neighbours_final   = false;
        // Don't print messages, except Error & Severe
        Log.level_file    = Utility::Log_Level::Error;
        Log.level_console = Utility::Log_Level::Error;
    }
    else
    {

        Log.messages_to_file        = section["log_to_file"].value_or( true );
        Log.messages_to_console     = section["log_to_console"].value_or( true );
        Log.save_input_initial      = section["save_input_initial"].value_or( false );
        Log.save_input_final        = section["save_input_final"].value_or( false );
        Log.save_positions_initial  = section["save_positions_initial"].value_or( false );
        Log.save_positions_final    = section["save_positions_final"].value_or( false );
        Log.save_neighbours_initial = section["save_neighbours_initial"].value_or( false );
        Log.save_neighbours_final   = section["save_neighbours_final"].value_or( false );
        Log.level_file              = Log_Level( section["log_file_level"].value_or( 5 ) );
        Log.level_console           = Log_Level( section["log_console_level"].value_or( 5 ) );

        // Log the parameters
        Log( Log_Level::Debug, Log_Sender::IO, "Building Log" );
        if( !section )
            Log( Log_Level::Warning, Log_Sender::IO, "Missing config section: 'logging'. Using defaults..." );
        std::vector<std::string> block;
        block.emplace_back( "Logging parameters" );
        block.emplace_back( fmt::format( "    file tag on output = \"{}\"", Log.file_tag ) );
        block.emplace_back( fmt::format( "    output folder      = \"{}\"", Log.output_folder ) );
        block.emplace_back( fmt::format( "    to file            = {}", Log.messages_to_file ) );
        block.emplace_back( fmt::format( "    file accept level  = {}", (int)Log.level_file ) );
        block.emplace_back( fmt::format( "    to console         = {}", Log.messages_to_console ) );
        block.emplace_back( fmt::format( "    print accept level = {}", (int)Log.level_console ) );
        block.emplace_back( fmt::format( "    input save initial = {}", Log.save_input_initial ) );
        block.emplace_back( fmt::format( "    input save final   = {}", Log.save_input_final ) );
        block.emplace_back( fmt::format( "    positions save initial  = {}", Log.save_positions_initial ) );
        block.emplace_back( fmt::format( "    positions save final    = {}", Log.save_positions_final ) );
        block.emplace_back( fmt::format( "    neighbours save initial = {}", Log.save_neighbours_initial ) );
        block.emplace_back( fmt::format( "    neighbours save final   = {}", Log.save_neighbours_final ) );
        Log( Log_Level::Parameter, Log_Sender::IO, block );
    }
}
catch( ... )
{
    spirit_handle_exception_core( "Unable to read logging parameters" );
} // End Log_from_TOML

} // namespace IO
