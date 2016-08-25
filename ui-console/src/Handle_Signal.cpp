#include "Handle_Signal.h"

#include "Interface_Simulation.h"
#include "Interface_Log.h"

// TODO: Replace this
#include "Timing.h"

#include <memory>

struct State;
std::shared_ptr<State> extern state;

namespace Utility
{
    static system_clock::time_point t_last_sigint = system_clock::now();

    // Handle interrupt by signal SIGINT
    void Handle_Signal::Handle_SigInt(int sig)
    {
        // Have SIG_IGN (ignore) handle further SIGINTs from now
        signal(sig, SIG_IGN);

        // Check if the chain is initialized
        if (state != nullptr)
        {
            system_clock::time_point t_now = system_clock::now();

            if ( Timing::SecondsPassed(t_last_sigint, t_now) < 2.0 )
            {
                Log_Send(state.get(), Log_Level::All, Log_Sender::All, "SIGINT received! Received second time in less than 2s.");
                Log_Send(state.get(), Log_Level::All, Log_Sender::All, "                 Terminating Program.");
                Log_Append(state.get());
                exit(0);
            }
            else
            {
                Log_Send(state.get(), Log_Level::All, Log_Sender::All, "SIGINT received! All iteration_allowed are being set to false.");
                Log_Send(state.get(), Log_Level::All, Log_Sender::All, "                 Press again in less than 2s to terminate the Program.");
                Simulation_Stop_All(state.get());
            }
            Log_Append(state.get());
        }
        // No iterations started, exit the program
        else
        {
            Log_Send(state.get(), Log_Level::All, Log_Sender::All, "SIGINT received! Calling exit(0).");
            Log_Append(state.get());
            exit(0);
        }

        // Have this function handle the signal again
        signal(sig, Handle_SigInt);

        // Update time of last interrupt
        t_last_sigint = system_clock::now();
    }
}