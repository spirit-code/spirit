#include "Spin_System_Chain_Collection.h"

namespace Data
{
	Spin_System_Chain_Collection::Spin_System_Chain_Collection(std::vector<std::shared_ptr<Spin_System_Chain>> chains, std::shared_ptr<Data::Parameters_MMF> parameters, bool iteration_allowed) :
		parameters(parameters)
	{
    }
}