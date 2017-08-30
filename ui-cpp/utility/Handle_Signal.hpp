#pragma once
#ifndef UTILITY_SIGNAL_H
#define UTILITY_SIGNAL_H

#include <signal.h>

namespace Utility
{
    namespace Handle_Signal
    {
        void Handle_SigInt(int sig);
    }
}

#endif