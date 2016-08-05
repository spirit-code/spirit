#include "Interface_State.h"

#include "Version.h"
#include "Logging.h"
#include "IO.h"
#include "Configurations.h"
#include "Configuration_Chain.h"


using namespace Utility;

State * setupState(const char * config_file)
{
    // Create the State
    State *state = new State();
    
    // Log
    Log.Send(Log_Level::ALL, Log_Sender::ALL,  "=====================================================");
    Log.Send(Log_Level::ALL, Log_Sender::ALL,  "============ MonoSpin State Initialising ============");
    Log.Send(Log_Level::ALL, Log_Sender::ALL,  "============     Version:  " + std::string(VERSION));
    Log.Send(Log_Level::INFO, Log_Sender::ALL, "============     Revision: " + std::string(VERSION_REVISION));
    Log.Send(Log_Level::ALL, Log_Sender::ALL,  "=====================================================");
    
    try
    {
        //---------------------- Read Log Levels ----------------------------------------
        IO::Log_Levels_from_Config(config_file);
        //-------------------------------------------------------------------------------

        //---------------------- initialize spin_systems --------------------------------
        // Create a system according to Config
        state->active_image = IO::Spin_System_from_Config(config_file);
        //-------------------------------------------------------------------------------

        //---------------------- set images' configurations -----------------------------
        // Parameters
        double dir[3] = { 0,0,1 };
        std::vector<double> pos = { 14.5, 14.5, 0 };
        // First image is homogeneous with a Skyrmion at pos
        Configurations::Random(*state->active_image);
        //Configurations::Skyrmion(*s1, pos, 6.0, 1.0, -90.0, false, false, false, false);
        //-------------------------------------------------------------------------------


        //----------------------- initialize spin system chain --------------------------
        // Get parameters
        auto params_gneb = std::shared_ptr<Data::Parameters_GNEB>(IO::GNEB_Parameters_from_Config(config_file));
        // Create the chain
        auto sv = std::vector<std::shared_ptr<Data::Spin_System>>();
        sv.push_back(state->active_image);
        state->active_chain = std::shared_ptr<Data::Spin_System_Chain>(new Data::Spin_System_Chain(sv, params_gneb, false));
        //-------------------------------------------------------------------------------
    }
	catch (Exception ex)
    {
		if (ex == Exception::System_not_Initialized)
			Log.Send(Utility::Log_Level::SEVERE, Utility::Log_Sender::IO, std::string("System not initialized - Terminating."));
		else if (ex == Exception::Simulated_domain_too_small)
			Log.Send(Utility::Log_Level::SEVERE, Utility::Log_Sender::ALL, std::string("CreateNeighbours:: Simulated domain is too small"));
		else if (ex == Exception::Not_Implemented)
			Log.Send(Utility::Log_Level::SEVERE, Utility::Log_Sender::ALL, std::string("Tried to use function which has not been implemented"));
		else
			Log.Send(Utility::Log_Level::SEVERE, Utility::Log_Sender::ALL, std::string("Unknown exception!"));
	}

    // active images
    state->idx_active_chain = 0;
    state->idx_active_image = 0;

    // Info
    state->noc = 1;
    state->noi = 1;
    state->nos = state->active_image->nos;

    // Methods
    state->methods_llg = std::vector<std::vector<std::shared_ptr<Engine::Method_LLG>>>(state->noc, std::vector<std::shared_ptr<Engine::Method_LLG>>(state->noi));
    state->methods_gneb = std::vector<std::shared_ptr<Engine::Method_GNEB>>(state->noi);
    state->methods_mmf = std::vector<std::shared_ptr<Engine::Method_MMF>>(state->noi);

    // Log
    Log.Send(Log_Level::ALL, Log_Sender::ALL, "=====================================================");
    Log.Send(Log_Level::ALL, Log_Sender::ALL, "============ MonoSpin State Initialised =============");
    Log.Send(Log_Level::ALL, Log_Sender::ALL, "============     NOS="+std::to_string(state->nos)+" NOI="+std::to_string(state->noi)+" NOC="+std::to_string(state->noc));
    Log.Send(Log_Level::ALL, Log_Sender::ALL, "=====================================================");
    
    // Return
    return state;
}

void from_indices(State * state, int & idx_image, int & idx_chain, std::shared_ptr<Data::Spin_System> & image, std::shared_ptr<Data::Spin_System_Chain> & chain)
{
    // Chain
    if (idx_chain < 0 || idx_chain == state->idx_active_chain)
    {
        chain = state->active_chain;
        idx_chain = state->idx_active_chain;
    }
    else
    {
        chain = state->active_chain;
        idx_chain = state->idx_active_chain;
    }
    
    // Image
    if ( idx_chain == state->idx_active_chain && (idx_image < 0 || idx_image == state->idx_active_image) )
    {
        image = state->active_image;
        idx_image = state->idx_active_image;
    }
    else
    {
        image = chain->images[idx_image];
        idx_image = idx_image;
    }
}