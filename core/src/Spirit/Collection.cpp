#include <Spirit/Collection.h>
#include <utility/Exception.hpp>
#include <data/State.hpp>

int Collection_Get_NOC(State * state) noexcept
{
    try
    {
        return state->noc;
    }
    catch( ... )
    {
        Utility::Handle_Exception();
        return false;
    }
}

void Collection_next_Chain(State * state) noexcept
{

}

void Collection_prev_Chain(State * state) noexcept
{

}