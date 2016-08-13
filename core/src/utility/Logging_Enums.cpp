#include "Logging_Enums.h"

#include "Timing.h"

std::string Utility::LogEntryToString(LogEntry entry, bool braces_separators)
{
    // Time
    std::string t = Timing::TimePointToString_Pretty(entry.time);
    std::string result = "";
    result.append(t);
    // Message Level
    if (braces_separators) result.append("  [");
    else result.append("   ");
    if      (entry.level == Log_Level::ALL)    	result.append("  ALL  ");
    else if (entry.level == Log_Level::SEVERE) 	result.append("SEVERE ");
    else if (entry.level == Log_Level::ERROR)	result.append(" ERROR ");
    else if (entry.level == Log_Level::WARNING)	result.append("WARNING");
    else if (entry.level == Log_Level::PARAMETER) result.append(" PARAM ");
    else if (entry.level == Log_Level::INFO)    	result.append(" INFO  ");
    else if (entry.level == Log_Level::DEBUG)   	result.append(" DEBUG ");
    // Sender
    if (braces_separators) result.append("] [");
    else result.append("  ");
    if     (entry.sender == Log_Sender::ALL)  result.append("ALL ");
    else if(entry.sender == Log_Sender::IO)   result.append("IO  ");
    else if(entry.sender == Log_Sender::API)  result.append("API ");
    else if(entry.sender == Log_Sender::GNEB) result.append("GNEB");
    else if(entry.sender == Log_Sender::LLG)  result.append("LLG ");
    else if(entry.sender == Log_Sender::MMF)  result.append("MMF ");
    else if(entry.sender == Log_Sender::UI)   result.append("UI  ");
    // Chain Index
    if (braces_separators) result.append("] [");
    else result.append("  ");
    if (entry.idx_chain >= 0) result.append(std::to_string(entry.idx_chain));
    else result.append("--");
    // Image Index
    if (braces_separators) result.append("] [");
    else result.append("  ");
    if (entry.idx_image >= 0) result.append(std::to_string(entry.idx_image));
    else result.append("--");
    if (braces_separators) result.append("]  ");
    else result.append("   ");
    // Message string
    result.append(entry.message);
    // Return
    return result;
}