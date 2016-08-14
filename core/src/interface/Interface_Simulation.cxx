#include "Interface_Simulation.h"
#include "Interface_State.h"
#include "Logging.h"

#include "Optimizer.h"
#include "Optimizer_Heun.h"
#include "Optimizer_SIB.h"
#include "Optimizer_SIB2.h"
#include "Optimizer_CG.h"
#include "Optimizer_QM.h"
#include "Method.h"


extern "C" void Simulation_SingleShot(State *state, const char * c_method_type, const char * c_optimizer_type, 
	int n_iterations, int log_steps, int idx_image, int idx_chain)
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
        if (log_steps > 0) image->llg_parameters->log_steps = log_steps;
        method = std::shared_ptr<Engine::Method_LLG>(new Engine::Method_LLG(image, idx_image, idx_chain));
    }
    else if (method_type == "GNEB")
    {
        if (Simulation_Running_LLG_Chain(state, idx_chain))
        {
            Log(Utility::Log_Level::ERROR, Utility::Log_Sender::API, "There are still LLG simulations running on the specified chain! Please stop them before starting a GNEB calculation.");
        }
        else
        {
            chain->iteration_allowed = true;
            if (n_iterations > 0) chain->gneb_parameters->n_iterations = n_iterations;
            if (log_steps > 0) chain->gneb_parameters->log_steps = log_steps;
            method = std::shared_ptr<Engine::Method_GNEB>(new Engine::Method_GNEB(chain, idx_image, idx_chain));
        }
    }
    else if (method_type == "MMF")
    {
        if (Simulation_Running_LLG_Anywhere(state) || Simulation_Running_GNEB_Anywhere(state))
        {
            Log(Utility::Log_Level::ERROR, Utility::Log_Sender::API, "There are still LLG or GNEB simulations running on the collection! Please stop them before starting a MMF calculation.");
        }
        else
        {
            state->collection->iteration_allowed = true;
            if (n_iterations > 0) state->collection->parameters->n_iterations = n_iterations;
            if (log_steps > 0) state->collection->parameters->log_steps = log_steps;
            method = std::shared_ptr<Engine::Method_MMF>(new Engine::Method_MMF(state->collection, idx_image, idx_chain));
            Log(Utility::Log_Level::WARNING, Utility::Log_Sender::API, std::string("MMF Method selected, but not yet fully implemented!"));
        }
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
    else if (optimizer_type == "CG")
    {
        optim = std::shared_ptr<Engine::Optimizer>(new Engine::Optimizer_CG(method));
    }
    else if (optimizer_type == "QM")
    {
        optim = std::shared_ptr<Engine::Optimizer>(new Engine::Optimizer_QM(method));
    }

    // One Iteration
    optim->Iteration();
}

void Simulation_PlayPause(State *state, const char * c_method_type, const char * c_optimizer_type,
    int n_iterations, int log_steps, int idx_image, int idx_chain)
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
    	image->iteration_allowed = false;
    }
    else if (chain->iteration_allowed)
    {
        // Currently iterating chain, so we stop
    	chain->iteration_allowed = false;
    }
    else if (state->collection->iteration_allowed)
    {
        // Currently iterating collection, so we stop
        state->collection->iteration_allowed = false;
    }
    else
    {
        // ------ Nothing is iterating, so we start a simulation ------

        // Determine the method and chain(s) or image(s) involved
        std::shared_ptr<Engine::Method> method;
        std::shared_ptr<Engine::Optimizer> optim;
        if (method_type == "LLG")
        {
            image->iteration_allowed = true;
			if (n_iterations > 0) image->llg_parameters->n_iterations = n_iterations;
			if (log_steps > 0) image->llg_parameters->log_steps = log_steps;
            method = std::shared_ptr<Engine::Method_LLG>(new Engine::Method_LLG(image, idx_image, idx_chain));
        }
        else if (method_type == "GNEB")
        {
            if (Simulation_Running_LLG_Chain(state, idx_chain))
            {
                Log(Utility::Log_Level::ERROR, Utility::Log_Sender::API, "There are still LLG simulations running on the specified chain! Please stop them before starting a GNEB calculation.");
            }
            else
            {
                chain->iteration_allowed = true;
                if (n_iterations > 0) chain->gneb_parameters->n_iterations = n_iterations;
                if (log_steps > 0) chain->gneb_parameters->log_steps = log_steps;
                method = std::shared_ptr<Engine::Method_GNEB>(new Engine::Method_GNEB(chain, idx_image, idx_chain));
            }
        }
        else if (method_type == "MMF")
        {
            if (Simulation_Running_LLG_Anywhere(state) || Simulation_Running_GNEB_Anywhere(state))
            {
                Log(Utility::Log_Level::ERROR, Utility::Log_Sender::API, "There are still LLG or GNEB simulations running on the collection! Please stop them before starting a MMF calculation.");
            }
            else
            {
                state->collection->iteration_allowed = true;
                if (n_iterations > 0) state->collection->parameters->n_iterations = n_iterations;
                if (log_steps > 0) state->collection->parameters->log_steps = log_steps;
                method = std::shared_ptr<Engine::Method_MMF>(new Engine::Method_MMF(state->collection, idx_image, idx_chain));
                Log(Utility::Log_Level::WARNING, Utility::Log_Sender::API, std::string("MMF Method selected, but not yet fully implemented!"));
            }
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
        else if (optimizer_type == "CG")
        {
            optim = std::shared_ptr<Engine::Optimizer>(new Engine::Optimizer_CG(method));
        }
        else if (optimizer_type == "QM")
        {
            optim = std::shared_ptr<Engine::Optimizer>(new Engine::Optimizer_QM(method));
        }

        // TODO: how to add to list of optimizers?? how to remove when stopping??
        // state->optimizers.push_back(optim);

        // Start the simulation
        optim->Iterate();
    }
}

extern "C" void Simulation_Stop_All(State *state)
{
    // MMF
    state->collection->iteration_allowed = false;

    // GNEB
    state->active_chain->iteration_allowed = false;
    for (int i=0; i<state->noc; ++i)
    {
        state->collection->chains[i]->iteration_allowed = false;
    }

    // LLG
    state->active_image->iteration_allowed = false;
    for (int ichain=0; ichain<state->noc; ++ichain)
    {
        for (int img = 0; img < state->collection->chains[ichain]->noi; ++img)
        {
            state->collection->chains[ichain]->images[img]->iteration_allowed = false;
        }
    }
}

// TODO: how to do this correctly??
std::vector<double> Simulation_Get_IterationsPerSecond(State *state)
{
	std::vector<double> ret;

    // TODO: loop over all chains
    if (Simulation_Running_LLG(state))
	{
        // ret.push_back(state->optimizers[state->active_chain]->getIterationsPerSecond());
	}
	else if (Simulation_Running_GNEB(state))
    {
        
    }
	else if (Simulation_Running_GNEB(state))
    {

    }
	// {
	// 	for (unsigned int i = 0; i < this->llg_methods.size(); ++i)
	// 	{
	// 		if (state->active_chain->images[i]->iteration_allowed)
	// 		{
	// 			// TODO:
	// 			// ret.push_back(this->llg_methods[state->active_chain->images[i]]->getIterationsPerSecond());
	// 		}
	// 	}
	// }

	return ret;
}

extern "C" bool Simulation_Running_Any_Anywhere(State *state)
{
    if (Simulation_Running_LLG_Anywhere(state) ||
        Simulation_Running_GNEB_Anywhere(state) ||
        Simulation_Running_MMF(state)) return true;
    else return false;
}
extern "C" bool Simulation_Running_LLG_Anywhere(State *state)
{
    bool running = false;
    for (int ichain=0; ichain<state->collection->noc; ++ichain)
    {
        if (Simulation_Running_LLG_Chain(state, ichain)) running = true;
    }
    return running;
}
extern "C" bool Simulation_Running_GNEB_Anywhere(State *state)
{
    bool running = false;
    for (int i=0; i<state->collection->noc; ++i)
    {
        if (Simulation_Running_GNEB(state, i)) running = true;
    }
    return running;
}

extern "C" bool Simulation_Running_LLG_Chain(State *state, int idx_chain)
{
    bool running = false;
    for (int img=0; img<state->collection->chains[idx_chain]->noi; ++img)
    {
        if (Simulation_Running_LLG(state, img, idx_chain)) running = true;
    }
    return running;
}

extern "C" bool Simulation_Running_Any(State *state, int idx_image, int idx_chain)
{
    if (Simulation_Running_LLG(state, idx_image, idx_chain) ||
        Simulation_Running_GNEB(state, idx_chain) ||
        Simulation_Running_MMF(state))
        return true;
    else return false;
}
extern "C" bool Simulation_Running_LLG(State *state, int idx_image, int idx_chain)
{
    // Fetch correct indices and pointers for image and chain
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    if (image->iteration_allowed) return true;
    else return false;
}
extern "C" bool Simulation_Running_GNEB(State *state, int idx_chain)
{
    int idx_image = -1;
    // Fetch correct indices and pointers for image and chain
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    if (state->collection->chains[idx_chain]->iteration_allowed) return true;
    else return false;
}
extern "C" bool Simulation_Running_MMF(State *state)
{
    if (state->collection->iteration_allowed) return true;
    else return false;
}