#pragma once
#ifndef SPIRIT_CORE_FORMATTERS_LOGGING_HPP
#define SPIRIT_CORE_FORMATTERS_LOGGING_HPP

#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

#include <fmt/chrono.h>
#include <fmt/color.h>
#include <fmt/ostream.h>

#include <iostream>
#include <type_traits>

template<>
struct fmt::formatter<Utility::Log_Sender> : formatter<string_view>
{
    template<typename FmtContext>
    auto format( Utility::Log_Sender sender, FmtContext & ctx )
    {
        using namespace Utility;
        string_view name = "???";

        if( sender == Log_Sender::All )
            name = "ALL";
        else if( sender == Log_Sender::IO )
            name = "IO";
        else if( sender == Log_Sender::API )
            name = "API";
        else if( sender == Log_Sender::GNEB )
            name = "GNEB";
        else if( sender == Log_Sender::HTST )
            name = "HTST";
        else if( sender == Log_Sender::LLG )
            name = "LLG";
        else if( sender == Log_Sender::MC )
            name = "MC";
        else if( sender == Log_Sender::MMF )
            name = "MMF";
        else if( sender == Log_Sender::UI )
            name = "UI";
        else if( sender == Log_Sender::EMA )
            name = "EMA";
        else
        {
            spirit_throw(
                Exception_Classifier::Not_Implemented, Log_Level::Severe,
                "Tried converting unknown Log_Sender to string" );
        }

        return formatter<string_view>::format( name, ctx );
    }
};

template<>
struct fmt::formatter<Utility::Log_Level> : formatter<string_view>
{
    template<typename FmtContext>
    auto format( Utility::Log_Level level, FmtContext & ctx )
    {
        using namespace Utility;

        string_view name = "unknown";
        if( level == Log_Level::All )
            name = "ALL";
        else if( level == Log_Level::Severe )
            name = "SEVERE";
        else if( level == Log_Level::Error )
            name = "ERROR";
        else if( level == Log_Level::Warning )
            name = "WARNING";
        else if( level == Log_Level::Parameter )
            name = "PARAM";
        else if( level == Log_Level::Info )
            name = "INFO";
        else if( level == Log_Level::Debug )
            name = "DEBUG";
        else
        {
            spirit_throw(
                Exception_Classifier::Not_Implemented, Log_Level::Severe,
                "Tried converting unknown Log_Level to string" );
        }

        return formatter<string_view>::format( name, ctx );
    }
};

template<>
struct fmt::formatter<Utility::LogEntry> : formatter<string_view>
{
    template<typename FmtContext>
    auto format( Utility::LogEntry entry, FmtContext & ctx )
    {
        using namespace Utility;

        auto index_to_string = []( int idx ) -> std::string
        {
            if( idx >= 0 )
                return fmt::format( "{:0>2}", idx + 1 );
            else
                return "--";
        };

        // First line includes the datetime tag etc.
        std::string format_str;
        if( Log.bracket_separators )
            format_str = "{}  [{:^7}] [{:^4}] [{}]  {}";
        else
            format_str = "{}   {:^7}   {:^4}   {}   {}";
        // Note, ctx.out() is an output iterator to write to.
        // Currently not using entry.idx_chain, as it is not actively used in the codebase
        auto out = fmt::format_to(
            ctx.out(), fmt::runtime( format_str ), entry.time, entry.level, entry.sender,
            index_to_string( entry.idx_image ), entry.message_lines[0] );

        // Rest of the block
        for( std::size_t i = 1; i < entry.message_lines.size(); ++i )
            out = fmt::format_to( ctx.out(), "\n{}{}", Log.tags_space, entry.message_lines[i] );

        return out;
    }
};

#endif
