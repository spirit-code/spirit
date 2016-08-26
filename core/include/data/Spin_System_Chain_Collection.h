#pragma once
#ifndef DATA_SPIN_SYSTEM_CHAIN_COLLECTION_H
#define DATA_SPIN_SYSTEM_CHAIN_COLLECTION_H

#include "Spin_System_Chain.h"
#include "Parameters_Method_MMF.h"

namespace Data
{
	class Spin_System_Chain_Collection
	{
	public:
		// Constructor
		Spin_System_Chain_Collection(std::vector<std::shared_ptr<Spin_System_Chain>> chains, std::shared_ptr<Data::Parameters_Method_MMF> parameters, bool iteration_allowed = false);
        
        int noc;	// Number of Chains
		std::vector<std::shared_ptr<Spin_System_Chain>> chains;
		int idx_active_chain;

        // Parameters for MMF Iterations
		std::shared_ptr<Data::Parameters_Method_MMF> parameters;

		// Are we allowed to iterate on this collection?
		bool iteration_allowed;
    };

}

#endif