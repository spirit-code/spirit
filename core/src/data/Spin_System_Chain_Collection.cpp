#include <data/Spin_System_Chain_Collection.hpp>

namespace Data
{
    Spin_System_Chain_Collection::Spin_System_Chain_Collection(std::vector<std::shared_ptr<Spin_System_Chain>> chains) :
        chains(chains), noc(chains.size()), idx_active_chain(0)
    {
    }
}