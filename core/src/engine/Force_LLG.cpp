#include "Force_LLG.h"
#include "Manifoldmath.h"


namespace Engine
{
	Force_LLG::Force_LLG(std::shared_ptr<Data::Spin_System_Chain> c) : Force(c)
	{
	}

	void Force_LLG::Calculate(std::vector<std::vector<double>> & configurations, std::vector<std::vector<double>> & forces)
	{
		int noi = configurations.size();
		int nos = configurations[0].size() / 3;
		this->isConverged = std::vector<bool>(configurations.size(), false);
		this->maxAbsComponent = 0;

		// Loop over images to calculate the total Effective Field on each Image
		for (int img = 0; img < noi; ++img)
		{
			// The gradient force (unprojected) is simply the effective field
			this->c->images[img]->hamiltonian->Effective_Field(configurations[img], forces[img]);
			// Check for convergence
			auto fmax = this->Force_on_Image_MaxAbsComponent(configurations[img], forces[img]);
			if (fmax < this->c->gneb_parameters->force_convergence) this->isConverged[img] = true;
			if (fmax > this->maxAbsComponent) this->maxAbsComponent = fmax;
		}
	}

	bool Force_LLG::IsConverged()
	{
		return false;
		/*return std::all_of(this->isConverged.begin(),
							this->isConverged.end(),
							[](bool b) { return b; });*/
	}
}// end namespace Engine