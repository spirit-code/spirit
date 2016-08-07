#include "Handle_Signal.h"
#include "Logging.h"
#include "Interface_State.h"
#include "Timing.h"

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
                Log.Send(Log_Level::ALL, Log_Sender::ALL, "SIGINT received! Received second time in less than 2s.");
                Log.Send(Log_Level::ALL, Log_Sender::ALL, "                 Terminating Program.");
                Log.Append_to_File();
                exit(0);
            }
            else
            {
                Log.Send(Log_Level::ALL, Log_Sender::ALL, "SIGINT received! All iteration_allowed are being set to false.");
                Log.Send(Log_Level::ALL, Log_Sender::ALL, "                 Press again in less than 2s to terminate the Program.");
                state->active_chain->iteration_allowed = false;
                for (int i = 0; i < state->noi; ++i)
                {
                    state->active_chain->images[i]->iteration_allowed = false;
                }
            }
            Log.Append_to_File();
        }
        // No iterations started, exit the program
        else
        {
            Log.Send(Log_Level::ALL, Log_Sender::ALL, "SIGINT received! Calling exit(0).");
            Log.Append_to_File();
            exit(0);
        }

        // Have this function handle the signal again
        signal(sig, Handle_SigInt);

        // Update time of last interrupt
        t_last_sigint = system_clock::now();
    }
}