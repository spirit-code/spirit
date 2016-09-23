#include "Interface_State.h"

#include "Version.h"
#include "Logging.h"
#include "IO.h"
#include "Configurations.h"
#include "Configuration_Chain.h"


using namespace Utility;

State * State_Setup(const char * config_file)
{
    // Create the State
    State *state = new State();
    
    // Log
    Log(Log_Level::All, Log_Sender::All,  "=====================================================");
    Log(Log_Level::All, Log_Sender::All,  "============ Spirit State Initialising ==============");
    Log(Log_Level::All, Log_Sender::All,  "============     Version:  " + std::string(VERSION));
    Log(Log_Level::Info, Log_Sender::All, "============     Revision: " + std::string(VERSION_REVISION));
    Log(Log_Level::All, Log_Sender::All,  "=====================================================");
    
    try
    {
        //---------------------- Read Log Levels ----------------------------------------
        IO::Log_Levels_from_Config(config_file);
        //-------------------------------------------------------------------------------

        //---------------------- initialize spin_system ---------------------------------
        // Create a system according to Config
        state->active_image = IO::Spin_System_from_Config(config_file);
        //-------------------------------------------------------------------------------

        //---------------------- set images configuration -------------------------------
        // Parameters
        double dir[3] = { 0,0,1 };
        std::vector<double> pos = { 14.5, 14.5, 0 };
        // First image is homogeneous with a Skyrmion at pos
        Configurations::Random(*state->active_image);
        //Configurations::Skyrmion(*s1, pos, 6.0, 1.0, -90.0, false, false, false, false);
        //-------------------------------------------------------------------------------

        //----------------------- initialize spin system chain --------------------------
        // Get parameters
        auto params_gneb = std::shared_ptr<Data::Parameters_Method_GNEB>(IO::Parameters_Method_GNEB_from_Config(config_file));
        // Create the chain
        auto sv = std::vector<std::shared_ptr<Data::Spin_System>>();
        sv.push_back(state->active_image);
        state->active_chain = std::shared_ptr<Data::Spin_System_Chain>(new Data::Spin_System_Chain(sv, params_gneb, false));
        //-------------------------------------------------------------------------------

        //----------------------- initialize spin system chain collection ---------------
        // Get parameters
        auto params_mmf = std::shared_ptr<Data::Parameters_Method_MMF>(IO::Parameters_Method_MMF_from_Config(config_file));
        // Create the collection
        auto cv = std::vector<std::shared_ptr<Data::Spin_System_Chain>>();
        cv.push_back(state->active_chain);
        state->collection = std::shared_ptr<Data::Spin_System_Chain_Collection>(new Data::Spin_System_Chain_Collection(cv, params_mmf, false));
        //-------------------------------------------------------------------------------
    }
	catch (Exception ex)
    {
		if (ex == Exception::System_not_Initialized)
			Log(Utility::Log_Level::Severe, Utility::Log_Sender::IO, std::string("System not initialized - Terminating."));
		else if (ex == Exception::Simulated_domain_too_small)
			Log(Utility::Log_Level::Severe, Utility::Log_Sender::All, std::string("CreateNeighbours:: Simulated domain is too small"));
		else if (ex == Exception::Not_Implemented)
			Log(Utility::Log_Level::Severe, Utility::Log_Sender::All, std::string("Tried to use function which has not been implemented"));
		else
			Log(Utility::Log_Level::Severe, Utility::Log_Sender::All, std::string("Unknown exception!"));
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
    state->methods_gneb = std::vector<std::shared_ptr<Engine::Method_GNEB>>(state->noc);
    state->method_mmf = std::shared_ptr<Engine::Method_MMF>();

    // Log
    Log(Log_Level::All, Log_Sender::All, "=====================================================");
    Log(Log_Level::All, Log_Sender::All, "============ Spirit State Initialised ===============");
    Log(Log_Level::All, Log_Sender::All, "============     NOS="+std::to_string(state->nos)+" NOI="+std::to_string(state->noi)+" NOC="+std::to_string(state->noc));
    Log(Log_Level::All, Log_Sender::All, "=====================================================");
    
    // Return
    return state;
}

void State_Update(State * state)
{
    // Correct for removed chains - active_chain can maximally be noc-1
    if ( state->collection->idx_active_chain >= state->collection->noc )
        state->collection->idx_active_chain = state->collection->noc-1;
        
    // Update Chain
    state->idx_active_chain = state->collection->idx_active_chain;
    state->active_chain     = state->collection->chains[state->idx_active_chain];

    // Correct for removed images - active_image can maximally be noi-1
    if ( state->active_chain->idx_active_image >= state->active_chain->noi )
        state->active_chain->idx_active_image = state->active_chain->noi-1;

    // Update Image
    state->idx_active_image = state->active_chain->idx_active_image; 
    state->active_image     = state->active_chain->images[state->idx_active_image];

    // Update NOS, NOI, NOC
    state->noc = state->collection->noc;
    state->noi = state->active_chain->noi;
    state->nos = state->active_image->nos;
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