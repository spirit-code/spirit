#include "Signal.h"
#include "Logging.h"
#include "Interface_State.h"

std::shared_ptr<State> extern state;

namespace Utility
{
    // Handle interrupt by signal SIGINT
    void Signal::Handle_SigInt(int sig)
    {
        //std::shared_ptr<Data::Spin_System_Chain> extern c;
        // using Log_Level;
        // using Log_Sender;

        // Have SIG_IGN (ignore) handle further SIGINTs from now
        signal(sig, SIG_IGN);

        // Check if the chain is initialized
        if (state != nullptr)
        {
            Log.Send(Log_Level::ALL, Log_Sender::ALL, "SIGINT received! All iteration_allowed are being set to false.");
            state->c->iteration_allowed = false;
            for (int i = 0; i < state->noi; ++i)
            {
                state->c->images[i]->iteration_allowed = false;
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
    }
}