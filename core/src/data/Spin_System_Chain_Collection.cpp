#include "Spin_System_Chain_Collection.h"

namespace Data
{
	Spin_System_Chain_Collection::Spin_System_Chain_Collection(std::vector<std::shared_ptr<Spin_System_Chain>> chains, std::shared_ptr<Data::Parameters_Method_MMF> parameters, bool iteration_allowed) :
		chains(chains), parameters(parameters), iteration_allowed(iteration_allowed)
	{
		this->noc = chains.size();

		this->idx_active_chain = 0;
    }
}