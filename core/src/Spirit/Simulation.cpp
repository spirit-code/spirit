#include <Spirit/State.h>
#include <Spirit/Simulation.h>
#include <Spirit/Chain.h>

#include <data/State.hpp>
#include <engine/Optimizer_Heun.hpp>
#include <engine/Optimizer_SIB.hpp>
#include <engine/Optimizer_SIB2.hpp>
#include <engine/Optimizer_NCG.hpp>
#include <engine/Optimizer_VP.hpp>
#include <engine/Method_LLG.hpp>
#include <engine/Method_MC.hpp>
#include <engine/Method_GNEB.hpp>
#include <engine/Method_MMF.hpp>
#include <utility/Logging.hpp>


void Simulation_SingleShot(State *state, const char * c_method_type, const char * c_optimizer_type, 
	int n_iterations, int n_iterations_log, int idx_image, int idx_chain)
{
    // Translate to string
    std::string method_type(c_method_type);
    std::string optimizer_type(c_optimizer_type);

    // Fetch correct indices and pointers for image and chain
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

     // Determine the method and chain(s) or image(s) involved
    std::shared_ptr<Engine::Method> method;
    std::shared_ptr<Engine::Optimizer> optim;
    if (method_type == "LLG")
    {
        image->iteration_allowed = true;
        if (n_iterations > 0) image->llg_parameters->n_iterations = n_iterations;
        if (n_iterations_log > 0) image->llg_parameters->n_iterations_log = n_iterations_log;
        method = std::shared_ptr<Engine::Method_LLG>(new Engine::Method_LLG(image, idx_image, idx_chain));
    }
    else if (method_type == "MC")
    {
        Log(Utility::Log_Level::Error, Utility::Log_Sender::API, "Monte Carlo is not yet implemented!");
        return;
    }
    else if (method_type == "GNEB")
    {
        if (Simulation_Running_LLG_Chain(state, idx_chain))
        {
            Log(Utility::Log_Level::Error, Utility::Log_Sender::API, "There are still LLG simulations running on the specified chain! Please stop them before starting a GNEB calculation.");
			return;
        }
        else
        {
            chain->iteration_allowed = true;
            if (n_iterations > 0) chain->gneb_parameters->n_iterations = n_iterations;
            if (n_iterations_log > 0) chain->gneb_parameters->n_iterations_log = n_iterations_log;
            method = std::shared_ptr<Engine::Method_GNEB>(new Engine::Method_GNEB(chain, idx_chain));
        }
    }
    else if (method_type == "MMF")
    {
        Log(Utility::Log_Level::Error, Utility::Log_Sender::API, "MMF is not yet implemented!");
        return;
        if (Simulation_Running_LLG_Anywhere(state) || Simulation_Running_GNEB_Anywhere(state))
        {
            Log(Utility::Log_Level::Error, Utility::Log_Sender::API, "There are still LLG or GNEB simulations running on the collection! Please stop them before starting a MMF calculation.");
			return;
        }
        else
        {
            state->collection->iteration_allowed = true;
            if (n_iterations > 0) state->collection->parameters->n_iterations = n_iterations;
            if (n_iterations_log > 0) state->collection->parameters->n_iterations_log = n_iterations_log;
            method = std::shared_ptr<Engine::Method_MMF>(new Engine::Method_MMF(state->collection, idx_chain));
        }
    }
	else
	{
		Log(Utility::Log_Level::Error, Utility::Log_Sender::API, "Invalid Method selected: " + method_type);
		return;
	}

    // Determine the Optimizer
    if (optimizer_type == "SIB")
    {
        optim = std::shared_ptr<Engine::Optimizer>(new Engine::Optimizer_SIB(method));
    }
    else if (optimizer_type == "SIB2")
    {
        optim = std::shared_ptr<Engine::Optimizer>(new Engine::Optimizer_SIB2(method));
    }
    else if (optimizer_type == "Heun")
    {
        optim = std::shared_ptr<Engine::Optimizer>(new Engine::Optimizer_Heun(method));
    }
    else if (optimizer_type == "NCG")
    {
        Log(Utility::Log_Level::Error, Utility::Log_Sender::API, "NCG is not yet implemented!");
        return;
        optim = std::shared_ptr<Engine::Optimizer>(new Engine::Optimizer_NCG(method));
    }
    else if (optimizer_type == "VP")
    {
        optim = std::shared_ptr<Engine::Optimizer>(new Engine::Optimizer_VP(method));
    }
	else
	{
		Log(Utility::Log_Level::Error, Utility::Log_Sender::API, "Invalid Optimizer selected: " + optimizer_type);
		return;
	}

    // One Iteration
    optim->Iteration();
}

void Simulation_PlayPause(State *state, const char * c_method_type, const char * c_optimizer_type,
    int n_iterations, int n_iterations_log, int idx_image, int idx_chain)
{
    // Translate to string
    std::string method_type(c_method_type);
    std::string optimizer_type(c_optimizer_type);

    // Fetch correct indices and pointers for image and chain
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);


    

    // Determine wether to stop or start a simulation
    if (image->iteration_allowed)
    {
        // Currently iterating image, so we stop
        image->Lock();
    	image->iteration_allowed = false;
        image->Unlock();
    }
    else if (chain->iteration_allowed)
    {
        // Currently iterating chain, so we stop
        chain->Lock();
    	chain->iteration_allowed = false;
        chain->Unlock();
    }
    else if (state->collection->iteration_allowed)
    {
        // Currently iterating collection, so we stop
        // collection->Lock();
        state->collection->iteration_allowed = false;
        // collection->Unlock();
    }
    else
    {
        // ------ Nothing is iterating, so we start a simulation ------

        // Lock the chain in order to prevent unexpected things
        chain->Lock();

        // Determine the method and chain(s) or image(s) involved
        std::shared_ptr<Engine::Method> method;
        std::shared_ptr<Engine::Optimizer> optim;
        if (method_type == "LLG")
        {
            image->iteration_allowed = true;
			if (n_iterations > 0) image->llg_parameters->n_iterations = n_iterations;
			if (n_iterations_log > 0) image->llg_parameters->n_iterations_log = n_iterations_log;
            method = std::shared_ptr<Engine::Method_LLG>(new Engine::Method_LLG(image, idx_image, idx_chain));
        }
        else if (method_type == "MC")
        {
            Log(Utility::Log_Level::Error, Utility::Log_Sender::API, "Monte Carlo is not implemented!");
            return;
        }
        else if (method_type == "GNEB")
        {
            if (Simulation_Running_LLG_Chain(state, idx_chain))
            {
                Log(Utility::Log_Level::Error, Utility::Log_Sender::API, "There are still LLG simulations running on the specified chain! Please stop them before starting a GNEB calculation.");
				return;
            }
            else if (Chain_Get_NOI(state, idx_chain) < 3)
            {
                Log(Utility::Log_Level::Error, Utility::Log_Sender::API, "There are less than 3 images in the specified chain! Please insert more before starting a GNEB calculation.");
				return;
            }
            else
            {
                chain->iteration_allowed = true;
                if (n_iterations > 0) chain->gneb_parameters->n_iterations = n_iterations;
                if (n_iterations_log > 0) chain->gneb_parameters->n_iterations_log = n_iterations_log;
                method = std::shared_ptr<Engine::Method_GNEB>(new Engine::Method_GNEB(chain, idx_chain));
            }
        }
        else if (method_type == "MMF")
        {
            Log(Utility::Log_Level::Error, Utility::Log_Sender::API, "MMF is not implemented!");
            return;
            if (Simulation_Running_LLG_Anywhere(state) || Simulation_Running_GNEB_Anywhere(state))
            {
                Log(Utility::Log_Level::Error, Utility::Log_Sender::API, "There are still LLG or GNEB simulations running on the collection! Please stop them before starting a MMF calculation.");
				return;
            }
            else
            {
                state->collection->iteration_allowed = true;
                if (n_iterations > 0) state->collection->parameters->n_iterations = n_iterations;
                if (n_iterations_log > 0) state->collection->parameters->n_iterations_log = n_iterations_log;
                method = std::shared_ptr<Engine::Method_MMF>(new Engine::Method_MMF(state->collection, idx_chain));
            }
        }
		else
		{
			Log(Utility::Log_Level::Error, Utility::Log_Sender::API, "Invalid Method selected: " + method_type);
			return;
		}

        // Determine the Optimizer
        if (optimizer_type == "SIB")
        {
            optim = std::shared_ptr<Engine::Optimizer>(new Engine::Optimizer_SIB(method));
        }
        else if (optimizer_type == "SIB2")
        {
            optim = std::shared_ptr<Engine::Optimizer>(new Engine::Optimizer_SIB2(method));
        }
        else if (optimizer_type == "Heun")
        {
            optim = std::shared_ptr<Engine::Optimizer>(new Engine::Optimizer_Heun(method));
        }
        else if (optimizer_type == "NCG")
        {
            Log(Utility::Log_Level::Error, Utility::Log_Sender::API, "NCG is not implemented!");
            return;
            optim = std::shared_ptr<Engine::Optimizer>(new Engine::Optimizer_NCG(method));
        }
        else if (optimizer_type == "VP")
        {
            optim = std::shared_ptr<Engine::Optimizer>(new Engine::Optimizer_VP(method));
        }
		else
		{
			Log(Utility::Log_Level::Error, Utility::Log_Sender::API, "Invalid Optimizer selected: "+optimizer_type);
			return;
		}

		// Create Simulation Information
		auto info = std::shared_ptr<Simulation_Information>(new Simulation_Information{ optim, method } );

        // Add to correct list
		if (method_type == "LLG")
		{
			state->simulation_information_llg[idx_chain][idx_image] = info;
		}
		else if (method_type == "MC")
		{
			// state->simulation_information_mc[idx_chain][idx_image] = info;
		}
		else if (method_type == "GNEB")
		{
			state->simulation_information_gneb[idx_chain] = info;
		}
		else if (method_type == "MMF")
		{
			state->simulation_information_mmf = info;
		}

        // Unlock chain in order to be able to iterate
        chain->Unlock();

        // Start the simulation
        optim->Iterate();
    }
}

void Simulation_Stop_All(State *state)
{
    // MMF
    // collection->Lock();
    state->collection->iteration_allowed = false;
    // collection->Unlock();

    // GNEB
    state->active_chain->Lock();
    state->active_chain->iteration_allowed = false;
    state->active_chain->Unlock();
    for (int i=0; i<state->noc; ++i)
    {
        state->collection->chains[i]->Lock();
        state->collection->chains[i]->iteration_allowed = false;
        state->collection->chains[i]->Unlock();
    }

    // LLG
    state->active_image->Lock();
    state->active_image->iteration_allowed = false;
    state->active_image->Unlock();
    for (int ichain=0; ichain<state->noc; ++ichain)
    {
        for (int img = 0; img < state->collection->chains[ichain]->noi; ++img)
        {
            state->collection->chains[ichain]->images[img]->Lock();
            state->collection->chains[ichain]->images[img]->iteration_allowed = false;
            state->collection->chains[ichain]->images[img]->Unlock();
        }
    }
}


float Simulation_Get_MaxTorqueComponent(State * state, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    if (Simulation_Running_LLG(state, idx_image, idx_chain))
	{
		if (state->simulation_information_llg[idx_chain][idx_image])
			return (float)state->simulation_information_llg[idx_chain][idx_image]->method->force_maxAbsComponent;
	}
	else if (Simulation_Running_GNEB(state, idx_chain))
    {
		if (state->simulation_information_gneb[idx_chain])
			return (float)state->simulation_information_gneb[idx_chain]->method->force_maxAbsComponent;
    }
	else if (Simulation_Running_MMF(state))
    {
		if (state->simulation_information_mmf)
			return (float)state->simulation_information_mmf->method->force_maxAbsComponent;
    }

	return 0;
}


float Simulation_Get_IterationsPerSecond(State *state, int idx_image, int idx_chain)
{
	// Fetch correct indices and pointers for image and chain
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

    if (Simulation_Running_LLG(state, idx_image, idx_chain))
	{
		if (state->simulation_information_llg[idx_chain][idx_image])
			return (float)state->simulation_information_llg[idx_chain][idx_image]->optimizer->getIterationsPerSecond();
	}
	else if (Simulation_Running_GNEB(state, idx_chain))
    {
		if (state->simulation_information_gneb[idx_chain])
			return (float)state->simulation_information_gneb[idx_chain]->optimizer->getIterationsPerSecond();
    }
	else if (Simulation_Running_MMF(state))
    {
		if (state->simulation_information_mmf)
			return (float)state->simulation_information_mmf->optimizer->getIterationsPerSecond();
    }

	return 0;
}


const char * Simulation_Get_Optimizer_Name(State *state, int idx_image, int idx_chain)
{
    // Fetch correct indices and pointers for image and chain
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

    if (Simulation_Running_LLG(state, idx_image, idx_chain))
	{
		if (state->simulation_information_llg[idx_chain][idx_image])
			return state->simulation_information_llg[idx_chain][idx_image]->optimizer->Name().c_str();
	}
	else if (Simulation_Running_GNEB(state, idx_chain))
    {
		if (state->simulation_information_gneb[idx_chain])
			return state->simulation_information_gneb[idx_chain]->optimizer->Name().c_str();
    }
	else if (Simulation_Running_MMF(state))
    {
		if (state->simulation_information_mmf)
			return state->simulation_information_mmf->optimizer->Name().c_str();
    }

	return "";
}

const char * Simulation_Get_Method_Name(State *state, int idx_image, int idx_chain)
{
    // Fetch correct indices and pointers for image and chain
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

    if (Simulation_Running_LLG(state, idx_image, idx_chain))
	{
		if (state->simulation_information_llg[idx_chain][idx_image])
			return state->simulation_information_llg[idx_chain][idx_image]->method->Name().c_str();
	}
	else if (Simulation_Running_GNEB(state, idx_chain))
    {
		if (state->simulation_information_gneb[idx_chain])
			return state->simulation_information_gneb[idx_chain]->method->Name().c_str();
    }
	else if (Simulation_Running_MMF(state))
    {
		if (state->simulation_information_mmf)
			return state->simulation_information_mmf->method->Name().c_str();
    }

	return "";
}

bool Simulation_Running_Any_Anywhere(State *state)
{
    if (Simulation_Running_LLG_Anywhere(state) ||
        Simulation_Running_GNEB_Anywhere(state) ||
        Simulation_Running_MMF(state)) return true;
    else return false;
}
bool Simulation_Running_LLG_Anywhere(State *state)
{
    bool running = false;
    for (int ichain=0; ichain<state->collection->noc; ++ichain)
    {
        if (Simulation_Running_LLG_Chain(state, ichain)) running = true;
    }
    return running;
}
bool Simulation_Running_GNEB_Anywhere(State *state)
{
    bool running = false;
    for (int i=0; i<state->collection->noc; ++i)
    {
        if (Simulation_Running_GNEB(state, i)) running = true;
    }
    return running;
}

bool Simulation_Running_LLG_Chain(State *state, int idx_chain)
{
    bool running = false;
    for (int img=0; img<state->collection->chains[idx_chain]->noi; ++img)
    {
        if (Simulation_Running_LLG(state, img, idx_chain)) running = true;
    }
    return running;
}

bool Simulation_Running_Any(State *state, int idx_image, int idx_chain)
{
    if (Simulation_Running_LLG(state, idx_image, idx_chain) ||
        Simulation_Running_GNEB(state, idx_chain) ||
        Simulation_Running_MMF(state))
        return true;
    else return false;
}
bool Simulation_Running_LLG(State *state, int idx_image, int idx_chain)
{
    // Fetch correct indices and pointers for image and chain
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    if (image->iteration_allowed) return true;
    else return false;
}
bool Simulation_Running_GNEB(State *state, int idx_chain)
{
    int idx_image = -1;
    // Fetch correct indices and pointers for image and chain
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    if (state->collection->chains[idx_chain]->iteration_allowed) return true;
    else return false;
}
bool Simulation_Running_MMF(State *state)
{
    if (state->collection->iteration_allowed) return true;
    else return false;
}