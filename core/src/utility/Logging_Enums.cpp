#include "Logging_Enums.h"

#include "Timing.h"
#include "IO.h"

std::string Utility::LogEntryToString(LogEntry entry, bool braces_separators)
{
    // Format indices
    auto s_chain = IO::int_to_formatted_string(entry.idx_chain, 2);
    auto s_image = IO::int_to_formatted_string(entry.idx_image, 2);

    // Time
    std::string t = Timing::TimePointToString_Pretty(entry.time);
    std::string result = "";
    result.append(t);
    // Message Level
    if (braces_separators) result.append("  [");
    else result.append("   ");
    if      (entry.level == Log_Level::All)    	result.append("  ALL  ");
    else if (entry.level == Log_Level::Severe) 	result.append("SEVERE ");
    else if (entry.level == Log_Level::Error)	result.append(" ERROR ");
    else if (entry.level == Log_Level::Warning)	result.append("WARNING");
    else if (entry.level == Log_Level::Parameter) result.append(" PARAM ");
    else if (entry.level == Log_Level::Info)    	result.append(" INFO  ");
    else if (entry.level == Log_Level::Debug)   	result.append(" DEBUG ");
    // Sender
    if (braces_separators) result.append("] [");
    else result.append("  ");
    if     (entry.sender == Log_Sender::All)  result.append("ALL ");
    else if(entry.sender == Log_Sender::IO)   result.append("IO  ");
    else if(entry.sender == Log_Sender::API)  result.append("API ");
    else if(entry.sender == Log_Sender::GNEB) result.append("GNEB");
    else if(entry.sender == Log_Sender::LLG)  result.append("LLG ");
    else if(entry.sender == Log_Sender::MMF)  result.append("MMF ");
    else if(entry.sender == Log_Sender::UI)   result.append("UI  ");
    // Chain Index
    if (braces_separators) result.append("] [");
    else result.append("  ");
    if (entry.idx_chain >= 0) result.append(s_chain);
    else result.append("--");
    // Image Index
    if (braces_separators) result.append("] [");
    else result.append("  ");
    if (entry.idx_image >= 0) result.append(s_image);
    else result.append("--");
    if (braces_separators) result.append("]  ");
    else result.append("   ");
    // Message string
    result.append(entry.message);
    // Return
    return result;
}