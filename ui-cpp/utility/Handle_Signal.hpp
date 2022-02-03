#pragma once
#ifndef UTILITY_SIGNAL_H
#define UTILITY_SIGNAL_H

#include <csignal>

namespace Utility
{
namespace Handle_Signal
{

void Handle_SigInt( int sig );

} // namespace Handle_Signal
} // namespace Utility

#endif