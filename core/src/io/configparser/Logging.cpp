#include <io/configparser/Converter.hpp>
#include <utility/Logging.hpp>
#include <utility/Timing.hpp>

#include <fmt/format.h>

#include <string>

// using namespace Utility;
using Utility::Log_Level;
using Utility::Log_Sender;

namespace IO
{

void Log_from_TOML( const toml::table & tbl, bool force_quiet )
try
{

    std::string file_tag      = tbl["output_file_tag"].value_or( "" );
    std::string output_folder = tbl["output_folder"].value_or( "." );

    Log.file_tag      = file_tag;
    Log.output_folder = output_folder;

    if( file_tag == "<time>" )
        Log.file_name = "Log_" + Utility::Timing::CurrentDateTime() + ".txt";
    else if( !file_tag.empty() )
        Log.file_name = "Log_" + file_tag + ".txt";
    else
        Log.file_name = "Log.txt";

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
        auto i_level_file    = tbl["log_file_level"].value_or( 5 );
        auto i_level_console = tbl["log_console_level"].value_or( 5 );
        Log.level_file       = Log_Level( i_level_file );
        Log.level_console    = Log_Level( i_level_console );

        auto messages_to_file        = tbl["log_to_file"].value_or( true );
        auto messages_to_console     = tbl["log_to_console"].value_or( true );
        auto save_input_initial      = tbl["save_input_initial"].value_or( false );
        auto save_input_final        = tbl["save_input_final"].value_or( false );
        auto save_positions_initial  = tbl["save_positions_initial"].value_or( false );
        auto save_positions_final    = tbl["save_positions_final"].value_or( false );
        auto save_neighbours_initial = tbl["save_neighbours_initial"].value_or( false );
        auto save_neighbours_final   = tbl["save_neighbours_final"].value_or( false );

        Log.messages_to_file        = messages_to_file;
        Log.messages_to_console     = messages_to_console;
        Log.save_input_initial      = save_input_initial;
        Log.save_input_final        = save_input_final;
        Log.save_positions_initial  = save_positions_initial;
        Log.save_positions_final    = save_positions_final;
        Log.save_neighbours_initial = save_neighbours_initial;
        Log.save_neighbours_final   = save_neighbours_final;

        // Log the parameters
        Log( Log_Level::Debug, Log_Sender::IO, "Building Log" );
        std::vector<std::string> block;
        block.emplace_back( "Logging parameters" );
        block.emplace_back( fmt::format( "    file tag on output = \"{}\"", file_tag ) );
        block.emplace_back( fmt::format( "    output folder      = \"{}\"", output_folder ) );
        block.emplace_back( fmt::format( "    to file            = {}", messages_to_file ) );
        block.emplace_back( fmt::format( "    file accept level  = {}", i_level_file ) );
        block.emplace_back( fmt::format( "    to console         = {}", messages_to_console ) );
        block.emplace_back( fmt::format( "    print accept level = {}", i_level_console ) );
        block.emplace_back( fmt::format( "    input save initial = {}", save_input_initial ) );
        block.emplace_back( fmt::format( "    input save final   = {}", save_input_final ) );
        block.emplace_back( fmt::format( "    positions save initial  = {}", save_positions_initial ) );
        block.emplace_back( fmt::format( "    positions save final    = {}", save_positions_final ) );
        block.emplace_back( fmt::format( "    neighbours save initial = {}", save_neighbours_initial ) );
        block.emplace_back( fmt::format( "    neighbours save final   = {}", save_neighbours_final ) );
        Log( Log_Level::Parameter, Log_Sender::IO, block );
    }
}
catch( ... )
{
    spirit_handle_exception_core( "Unable to read logging parameters" );
} // End Log_from_TOML

} // namespace IO
