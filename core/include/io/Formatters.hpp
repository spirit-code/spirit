#pragma once
#include <type_traits>
#ifndef SPIRIT_CORE_IO_TYPE_FORMATTERS_HPP
#define SPIRIT_CORE_IO_TYPE_FORMATTERS_HPP

#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

#include <Eigen/Core>

#include <fmt/ostream.h>

#include <iostream>

template<typename T>
struct fmt::formatter<T, std::enable_if_t<std::is_base_of_v<Eigen::DenseBase<T>, T>, char>> : ostream_formatter
{
};

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

#endif