#include <Spirit/State.h>
#include <Spirit/Version.h>
#include "Spirit_Defines.h"
#include <data/State.hpp>
#include <io/IO.hpp>
#include <utility/Configurations.hpp>
#include <utility/Configuration_Chain.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

using namespace Utility;

State * State_Setup(const char * config_file, bool quiet)
{
    try
    {
        // Create the State
        State *state = new State();
        state->datetime_creation = system_clock::now();
        state->datetime_creation_string = Utility::Timing::TimePointToString(state->datetime_creation);
        state->config_file = config_file;
        state->quiet = quiet;

        //---------------------- Initial Block of Log messages -------------------------------------
        // Log version info
        Log(Log_Level::All,  Log_Sender::All, "=====================================================");
        Log(Log_Level::All,  Log_Sender::All, "========== Spirit State: Initialising... ============");
        Log(Log_Level::All,  Log_Sender::All, "==========     Version:  " + std::string(VERSION));
        
        // Log revision hash
        Log( Log_Level::All,  Log_Sender::All, "==========     Revision: " + 
             std::string(VERSION_REVISION));
        
        // Log if quiet mode
        if (state->quiet)
            Log( Log_Level::All, Log_Sender::All, std::string( "Going to run in QUIET mode" ) + 
                 std::string( " (only Error messages, no output files)" ) );
        
        // Log Config file info
        Log(Log_Level::All,  Log_Sender::All, "Config file: " + state->config_file);
        
        // Read Log Levels
        IO::Log_from_Config(state->config_file, state->quiet);
        Log(Log_Level::Info, Log_Sender::All, "=====================================================");
        Log(Log_Level::Info, Log_Sender::All, "========== Optimization Info");
        // Log OpenMP info
    	#ifdef _OPENMP
    		int nt = omp_get_max_threads();
    		Log(Log_Level::Info, Log_Sender::All, ("Using OpenMP (max. " + std::to_string(nt) + 
                " threads)").c_str() );
    	#else
    		Log(Log_Level::Info, Log_Sender::All, "Not using OpenMP");
        #endif
        // Log CUDA info
        #ifdef SPIRIT_USE_CUDA
    		Log(Log_Level::Info, Log_Sender::All, "Using CUDA");
        #else
    		Log(Log_Level::Info, Log_Sender::All, "Not using CUDA");
        #endif
        // Log Precision info
        #ifdef SPIRIT_SCALAR_TYPE_DOUBLE
    		Log(Log_Level::Info, Log_Sender::All, "Using double as scalar type");
        #endif
        #ifdef SPIRIT_SCALAR_TYPE_FLOAT
    		Log(Log_Level::Info, Log_Sender::All, "Using float as scalar type");
        #endif
        Log(Log_Level::All,  Log_Sender::All, "=====================================================");
        //------------------------------------------------------------------------------------------
        
        
        //---------------------- initialize spin_system ---------------------------------
        // Create a system according to Config
        state->active_image = IO::Spin_System_from_Config(state->config_file);
        //-------------------------------------------------------------------------------

        //---------------------- set image configuration --------------------------------
        Configurations::Random(*state->active_image);
        //-------------------------------------------------------------------------------

        //----------------------- initialize spin system chain --------------------------
        // Get parameters
        auto params_gneb = 
            std::shared_ptr<Data::Parameters_Method_GNEB>(
                IO::Parameters_Method_GNEB_from_Config( state->config_file, 
                                                        state->active_image->llg_parameters->pinning));
        
        // Create the chain
        auto sv = std::vector<std::shared_ptr<Data::Spin_System>>();
        sv.push_back(state->active_image);
        state->active_chain = std::shared_ptr<Data::Spin_System_Chain>(
            new Data::Spin_System_Chain(sv, params_gneb, false));
        //-------------------------------------------------------------------------------

        //----------------------- initialize spin system chain collection ---------------
        // Get parameters
        auto params_mmf = 
            std::shared_ptr<Data::Parameters_Method_MMF>(
                IO::Parameters_Method_MMF_from_Config( state->config_file, 
                                                       state->active_image->llg_parameters->pinning));
        
        // Create the collection
        auto cv = std::vector<std::shared_ptr<Data::Spin_System_Chain>>();
        cv.push_back(state->active_chain);
        state->collection = 
            std::shared_ptr<Data::Spin_System_Chain_Collection>(
                new Data::Spin_System_Chain_Collection( cv, params_mmf, false ) );
        //-------------------------------------------------------------------------------
        
        // active images
        state->idx_active_chain = 0;
        state->idx_active_image = 0;

        // Info
        state->noc = 1;
        state->noi = 1;
        state->nos = state->active_image->nos;

        // Methods
        state->simulation_information_image = 
            std::vector<std::vector<std::shared_ptr<Simulation_Information>>>( state->noc, 
                std::vector<std::shared_ptr<Simulation_Information>>(state->noi));
        state->simulation_information_chain = 
            std::vector<std::shared_ptr<Simulation_Information>>(state->noc);
        state->simulation_information_collection = std::shared_ptr<Simulation_Information>();

        // Save the config
        if (Log.save_input_initial)
        {
            std::string file = "input";
            if (Log.tag_time)
                file += "_" + state->datetime_creation_string;
            file += "_initial.cfg";
            State_To_Config(state, file.c_str(), state->config_file.c_str());
        }

        // Log
        Log(Log_Level::All, Log_Sender::All, "=====================================================");
        Log(Log_Level::All, Log_Sender::All, "============ Spirit State: Initialised ==============");
        Log(Log_Level::All, Log_Sender::All, "============     NOS="+std::to_string(state->nos)+
            " NOI="+std::to_string(state->noi) + " NOC="+std::to_string(state->noc) );
    	auto now = system_clock::now();
    	auto diff = Timing::DateTimePassed(now - state->datetime_creation);
    	Log(Log_Level::All, Log_Sender::All, "    Initialisation took " + diff);
    	Log(Log_Level::All, Log_Sender::All, "    Number of  Errors:  " + 
            std::to_string(Log_Get_N_Errors(state)));
    	Log(Log_Level::All, Log_Sender::All, "    Number of Warnings: " + 
            std::to_string(Log_Get_N_Warnings(state)));
        Log(Log_Level::All, Log_Sender::All, "=====================================================");
        Log.Append_to_File();
        
        // Return
        return state;
    }
    catch( ... )
    {
        Utility::Handle_Exception( );
        return nullptr;
    }
}

void State_Delete(State * state)
{
    try
    {
        Log(Log_Level::All, Log_Sender::All,  "=====================================================");
        Log(Log_Level::All, Log_Sender::All,  "============ Spirit State: Deleting... ==============");
        
        // Save the config
        if (Log.save_input_final)
        {
            std::string file = "input";
            if (Log.tag_time)
                file += "_" + state->datetime_creation_string;
            file += "_final.cfg";
            State_To_Config(state, file.c_str(), state->config_file.c_str());
        }
        
        // Timing
        auto now = system_clock::now();
        auto diff = Timing::DateTimePassed(now - state->datetime_creation);
        Log( Log_Level::All, Log_Sender::All,  "    State existed for " + diff );
        Log( Log_Level::All, Log_Sender::All,  "    Number of  Errors:  " + 
             std::to_string(Log_Get_N_Errors(state)) );
        Log( Log_Level::All, Log_Sender::All,  "    Number of Warnings: " + 
             std::to_string(Log_Get_N_Warnings(state)) );
    	
        // Delete
        delete(state);
        
        Log(Log_Level::All, Log_Sender::All,  "============== Spirit State: Deleted ================");
        Log(Log_Level::All, Log_Sender::All,  "=====================================================");
        Log.Append_to_File();
    }
    catch( ... )
    {
        Utility::Handle_Exception();
    }
}

void State_Update(State * state)
{
    try
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
    catch( ... )
    {
        Utility::Handle_Exception();
    }
}

void State_To_Config(State * state, const char * config_file, const char * original_config_file)
{
    try
    {
        std::string cfg = Log.output_folder + "/" + std::string(config_file);
        
        // Header
        std::string header = "###\n### Original configuration file was called\n###   " + 
                             std::string(original_config_file) + "\n###\n\n";
        IO::Append_String_to_File(header, cfg);
        // Folders
        IO::Folders_to_Config( cfg, state->active_image->llg_parameters, state->active_image->mc_parameters, 
                               state->active_chain->gneb_parameters, state->collection->parameters );
        // Log Parameters
        IO::Append_String_to_File("\n\n\n", cfg);
        IO::Log_Levels_to_Config(cfg);
        // Geometry
        IO::Append_String_to_File("\n\n\n", cfg);
        IO::Geometry_to_Config(cfg, state->active_image->geometry);
        // LLG
        IO::Append_String_to_File("\n\n\n", cfg);
        IO::Parameters_Method_LLG_to_Config(cfg, state->active_image->llg_parameters);
        // GNEB
        IO::Append_String_to_File("\n\n\n", cfg);
        IO::Parameters_Method_GNEB_to_Config(cfg, state->active_chain->gneb_parameters);
        // MMF
        IO::Append_String_to_File("\n\n\n", cfg);
        IO::Parameters_Method_MMF_to_Config(cfg, state->collection->parameters);
        // Hamiltonian
        IO::Append_String_to_File("\n\n\n", cfg);
        IO::Hamiltonian_to_Config(cfg, state->active_image->hamiltonian, state->active_image->geometry);
    }
    catch( ... )
    {
        Utility::Handle_Exception();
    }
}

const char * State_DateTime(State * state)
{
    try
    {
        return state->datetime_creation_string.c_str();
    }
    catch( ... )
    {
        Utility::Handle_Exception();
        return "00:00:00";
    }
}

void from_indices( const State * state, int & idx_image, int & idx_chain, 
                   std::shared_ptr<Data::Spin_System> & image, 
                   std::shared_ptr<Data::Spin_System_Chain> & chain )
{
    // In case of positive non-existing image_idx throw exception
    if ( idx_chain >= state->collection->noc )
        throw Exception::Non_existing_Chain;
    
    // Chain
    if ( idx_chain < 0 || idx_chain == state->idx_active_chain )
    {
        chain = state->active_chain;
        idx_chain = state->idx_active_chain;
    }
    else
    {
        chain = state->active_chain;
        idx_chain = state->idx_active_chain;
    }

    // In case of positive non-existing chain_idx throw exception    
    if (  idx_image >= state->active_chain->noi )
        throw Exception::Non_existing_Image;
    
    // Image
    if ( idx_chain == state->idx_active_chain && 
        (idx_image < 0 || idx_image == state->idx_active_image ) )
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