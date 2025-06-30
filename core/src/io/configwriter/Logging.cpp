#include <io/Configwriter.hpp>

namespace IO
{

auto Logging_to_TOML() -> toml::table
{
    return toml::table{
        { "log_file_level", static_cast<int>( Log.level_file ) },
        { "log_console_level", static_cast<int>( Log.level_console ) },
        { "log_to_file", Log.messages_to_file },
        { "log_to_console", Log.messages_to_console },
        { "save_input_initial", Log.save_input_initial },
        { "save_input_final", Log.save_input_final },
        { "save_positions_initial", Log.save_positions_initial },
        { "save_positions_final", Log.save_positions_final },
        { "save_neighbours_initial", Log.save_neighbours_initial },
        { "save_neighbours_final", Log.save_neighbours_final },
    };
}

} // namespace IO
