#include "Method_MMF.h"

namespace Engine
{
    Method_MMF::Method_MMF(std::shared_ptr<Data::Spin_System_Chain_Collection> collection, int idx_img, int idx_chain) :
        Method(collection->parameters, idx_img, idx_chain), collection(collection)
    {
		int noc = collection->noc;
		int nos = collection->chains[0]->images[0]->nos;

		this->SenderName = Utility::Log_Sender::MMF;

        // The systems we use are the last image of each respective chain
        for (int ichain = 0; ichain < noc; ++ichain)
		{
			this->systems.push_back(this->collection->chains[ichain]->images.back());
		}

		// Forces
		this->F_total    = std::vector<std::vector<double>>(noc, std::vector<double>(3 * nos));	// [noi][3nos]
		this->F_gradient = std::vector<std::vector<double>>(noc, std::vector<double>(3 * nos));	// [noi][3nos]
		this->F_mode     = std::vector<std::vector<double>>(noc, std::vector<double>(3 * nos));	// [noi][3nos]
    }

    void Method_MMF::Calculate_Force(std::vector<std::shared_ptr<std::vector<double>>> configurations, std::vector<std::vector<double>> & forces)
    {
        // Get Effective Fields of configurations
		for (int ichain = 0; ichain < collection->noc; ++ichain)
		{
			auto& image = *configurations[ichain];

			// The gradient force (unprojected) is simply the effective field
			this->systems[ichain]->hamiltonian->Effective_Field(image, F_gradient[ichain]);
		}

        // Get the Minimum Mode

        // Invert the gradient force along the minimum mode

        // Full force! Power up the horse!

    }
		
    // Check if the Forces are converged
    bool Method_MMF::Force_Converged()
    {
        return false;
    }

    void Method_MMF::Hook_Pre_Iteration()
    {

	}

    void Method_MMF::Hook_Post_Iteration()
    {
        // Update the chains' last images
    }

    void Method_MMF::Save_Current(std::string starttime, int iteration, bool final)
	{
        // Prepend copies of the current systems to their corresponding chains
        // - this way we will be able to look at the history of the optimizations

        // Recalculate the chains' Rx, E and interpolated values for their last two images
        // Maybe just reallocate and recalculate everything... this is maybe not done too often?

        // In the final save, we save all chains to file
        if (final)
        {

        }
    }

    void Method_MMF::Finalize()
    {
        this->collection->iteration_allowed=false;
    }

    // Optimizer name as string
    std::string Method_MMF::Name() { return "MMF"; }
}