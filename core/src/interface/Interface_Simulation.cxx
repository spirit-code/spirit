#include "Interface_Simulation.h"
#include "Interface_State.h"
#include "Optimizer.h"
#include "Optimizer_Heun.h"
#include "Optimizer_SIB.h"
#include "Optimizer_SIB2.h"
#include "Optimizer_CG.h"
#include "Optimizer_QM.h"
#include "Method.h"

void Simulation_PlayPause(State *state, const char * c_method_type, const char * c_optimizer_type, int n_iterations, int log_steps, int idx_image, int idx_chain)
{
    // Translate to string
    std::string method_type(c_method_type);
    std::string optimizer_type(c_optimizer_type);

    // Fetch correct indices and pointers for image and chain
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);


    

    // Determine wether to stop or start a simulation
    if (state->active_image->iteration_allowed)
    {
        // Currently iterating image, so we stop
    	state->active_image->iteration_allowed = false;
    	// state->active_chain->iteration_allowed = false; // should not be necessary, as only one of the two can be active at the same time
        // this->gneb_methods.erase(state->active_chain);
        // this->llg_methods.erase(state->active_image);
    }
    else if (state->active_chain->iteration_allowed)
    {
        // Currently iterating chain, so we stop
    	state->active_chain->iteration_allowed = false;
        // 	this->pushButton_PlayPause->setText("Play");
        // 	for (int i = 0; i < state->noi; ++i)
        // 	{
        // 		state->active_chain->images[i]->iteration_allowed = false;
        // 		this->llg_methods.erase(state->active_chain->images[i]);
        // 	}
        // 	this->gneb_methods.erase(state->active_chain);
    }
    else if (state->collection->iteration_allowed)
    {
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
            state->active_image->iteration_allowed = true;
			if (n_iterations > 0) state->active_image->llg_parameters->n_iterations = n_iterations;
			if (log_steps > 0) state->active_image->llg_parameters->log_steps = log_steps;
            method = std::shared_ptr<Engine::Method_LLG>(new Engine::Method_LLG(state->active_image, state->idx_active_image, state->idx_active_chain));
        }
        else if (method_type == "GNEB")
        {
    	    state->active_chain->iteration_allowed = true;
			if (n_iterations > 0) state->active_chain->gneb_parameters->n_iterations = n_iterations;
			if (log_steps > 0) state->active_chain->gneb_parameters->log_steps = log_steps;
            method = std::shared_ptr<Engine::Method_GNEB>(new Engine::Method_GNEB(state->active_chain, state->idx_active_image, state->idx_active_chain));
        }
        else if (method_type == "MMF")
        {
            state->collection->iteration_allowed = true;
			if (n_iterations > 0) state->collection->parameters->n_iterations = n_iterations;
			if (log_steps > 0) state->collection->parameters->log_steps = log_steps;
            method = std::shared_ptr<Engine::Method_MMF>(new Engine::Method_MMF(state->collection, state->idx_active_image, state->idx_active_chain));
            Utility::Log.Send(Utility::Log_Level::WARNING, Utility::Log_Sender::API, std::string("MMF Method selected, but not yet fully implemented!"));
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


extern "C" bool Simulation_Running_LLG(State *state)
{
    if (state->active_image->iteration_allowed) return true;
    else return false;
}
extern "C" bool Simulation_Running_GNEB(State *state)
{
    if (state->active_chain->iteration_allowed) return true;
    else return false;
}
extern "C" bool Simulation_Running_MMF(State *state)
{
    return false;
}