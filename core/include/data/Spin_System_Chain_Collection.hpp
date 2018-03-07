#pragma once
#ifndef DATA_SPIN_SYSTEM_CHAIN_COLLECTION_H
#define DATA_SPIN_SYSTEM_CHAIN_COLLECTION_H

#include <data/Spin_System_Chain.hpp>
#include <data/Parameters_Method_MMF.hpp>

namespace Data
{
    class Spin_System_Chain_Collection
    {
    public:
        // Constructor
        Spin_System_Chain_Collection(std::vector<std::shared_ptr<Spin_System_Chain>> chains);
        
        int noc;	// Number of Chains
        std::vector<std::shared_ptr<Spin_System_Chain>> chains;
        int idx_active_chain;
    };
}

#endif