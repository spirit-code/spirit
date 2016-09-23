#include "Method_MMF.h"
#include "Manifoldmath.h"

namespace Engine
{
    Method_MMF::Method_MMF(std::shared_ptr<Data::Spin_System_Chain_Collection> collection, int idx_chain) :
        Method(collection->parameters, -1, idx_chain), collection(collection)
    {
		int noc = collection->noc;
		int nos = collection->chains[0]->images[0]->nos;

		this->SenderName = Utility::Log_Sender::MMF;
        
        // The systems we use are the last image of each respective chain
        for (int ichain = 0; ichain < noc; ++ichain)
		{
			this->systems.push_back(this->collection->chains[ichain]->images.back());
		}

		// We assume that the systems are not converged before the first iteration
		this->force_maxAbsComponent = this->collection->parameters->force_convergence + 1.0;

		// Forces
		this->F_gradient   = std::vector<std::vector<double>>(noc, std::vector<double>(3 * nos));	// [noi][3nos]
		this->minimum_mode = std::vector<std::vector<double>>(noc, std::vector<double>(3 * nos));	// [noi][3nos]
    }

    void Method_MMF::Calculate_Force(std::vector<std::shared_ptr<std::vector<double>>> configurations, std::vector<std::vector<double>> & forces)
    {
        // Get Effective Fields of configurations
		for (int ichain = 0; ichain < collection->noc; ++ichain)
		{
			auto& image = *configurations[ichain];

			// The gradient force (unprojected) is simply the effective field
			this->systems[ichain]->hamiltonian->Effective_Field(image, F_gradient[ichain]);
		
            // Get the Minimum Mode
            // this->minimum_mode = ...

            // Invert the gradient force along the minimum mode
            Utility::Manifoldmath::Project_Reverse(F_gradient[ichain], minimum_mode[ichain]);

            // Copy out the forces
            forces[ichain] = F_gradient[ichain];
        }
    }
		
    // Check if the Forces are converged
    bool Method_MMF::Force_Converged()
    {
		if (this->force_maxAbsComponent < this->collection->parameters->force_convergence) return true;
		return false;
    }

    void Method_MMF::Hook_Pre_Iteration()
    {

	}

    void Method_MMF::Hook_Post_Iteration()
    {
        // --- Convergence Parameter Update
		this->force_maxAbsComponent = 0;
		for (int ichain = 0; ichain < collection->noc; ++ichain)
		{
			double fmax = this->Force_on_Image_MaxAbsComponent(*(systems[ichain]->spins), F_gradient[ichain]);
			if (fmax > this->force_maxAbsComponent) this->force_maxAbsComponent = fmax;
		}

        // --- Update the chains' last images
    }

    void Method_MMF::Save_Current(std::string starttime, int iteration, bool initial, bool final)
	{
        if (initial) return;
        // Insert copies of the current systems into their corresponding chains
        // - this way we will be able to look at the history of the optimizations
        for (int ichain=0; ichain<collection->noc; ++ichain)
        {
            // Copy the image
            auto copy = std::shared_ptr<Data::Spin_System>(new Data::Spin_System(*this->systems[ichain]));
            
            // Insert into chain
            auto chain = collection->chains[ichain];
            chain->noi++;
            chain->images.insert(chain->images.end(), copy);
            chain->climbing_image.insert(chain->climbing_image.end(), false);
            chain->falling_image.insert(chain->falling_image.end(), false);
        }

        // Reallocate and recalculate the chains' Rx, E and interpolated values for their last two images

        // Append Each chain's new image to it's corresponding archive

        // In the final save, we save all chains to file?
        if (final)
        {

        }
    }

    void Method_MMF::Finalize()
    {
        this->collection->iteration_allowed=false;
    }

    bool Method_MMF::Iterations_Allowed()
	{
		return this->collection->iteration_allowed;
	}

    // Optimizer name as string
    std::string Method_MMF::Name() { return "MMF"; }
}