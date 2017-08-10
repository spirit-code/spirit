#include <Spirit/System.h>
#include <Spirit/State.h>
#include <data/State.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

int System_Get_Index(State * state)
{
    return state->idx_active_image;
}

int System_Get_NOS(State * state, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    
    // Fetch correct indices and pointers
    try
    {
        from_indices( state, idx_image, idx_chain, image, chain );
    }
    catch( const Utility::Exception & ex )
    {
        Utility::Handle_exception( ex, idx_image, idx_chain );
        return 0;
    }
    
    return image->nos;
}

scalar * System_Get_Spin_Directions(State * state, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    
    // Fetch correct indices and pointers
    try
    {
        from_indices( state, idx_image, idx_chain, image, chain );
    }
    catch( const Utility::Exception & ex )
    {
        Utility::Handle_exception( ex, idx_image, idx_chain );
        return NULL;
    }
    
    return (scalar *)(*image->spins)[0].data();
}

scalar * System_Get_Effective_Field(State * state, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
    
    // Fetch correct indices and pointers
    try
    {
        from_indices( state, idx_image, idx_chain, image, chain );
    }
    catch( const Utility::Exception & ex )
    {
        Utility::Handle_exception( ex, idx_image, idx_chain );
        return NULL;
    }
    
	return image->effective_field[0].data();
}

float System_Get_Rx(State * state, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
    
    // Fetch correct indices and pointers
    try
    {
        from_indices( state, idx_image, idx_chain, image, chain );
    }
    catch( const Utility::Exception & ex )
    {
        Utility::Handle_exception( ex, idx_image, idx_chain );
        return 0;
    }
    
	return (float)chain->Rx[idx_image];
}

float System_Get_Energy(State * state, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    
    // Fetch correct indices and pointers
    try
    {
        from_indices( state, idx_image, idx_chain, image, chain );
    }
    catch( const Utility::Exception & ex )
    {
        Utility::Handle_exception( ex, idx_image, idx_chain );
        return 0;
    }
    
    return (float)image->E;
}

void System_Get_Energy_Array(State * state, float * energies, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    
    // Fetch correct indices and pointers
    try
    {
        from_indices( state, idx_image, idx_chain, image, chain );
    }
    catch( const Utility::Exception & ex )
    {
        Utility::Handle_exception( ex, idx_image, idx_chain );
        return ;
    }
    
    for (unsigned int i=0; i<image->E_array.size(); ++i)
    {
        energies[i] = (float)image->E_array[i].second;
    }
}

void System_Print_Energy_Array(State * state, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    
    // Fetch correct indices and pointers
    try
    {
        from_indices( state, idx_image, idx_chain, image, chain );
    }
    catch( const Utility::Exception & ex )
    {
        Utility::Handle_exception( ex, idx_image, idx_chain );
        return ;
    }
    
    scalar nd = 1/(scalar)image->nos;

    std::cerr << "E_tot = " << image->E*nd << "  ||  ";

    for (unsigned int i=0; i<image->E_array.size(); ++i)
    {
        std::cerr << image->E_array[i].first << " = " << image->E_array[i].second*nd;
        if (i < image->E_array.size()-1) std::cerr << "  |  ";
    }
    std::cerr << std::endl;
}

void System_Update_Data(State * state, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    
    // Fetch correct indices and pointers
    try
    {
        from_indices( state, idx_image, idx_chain, image, chain );
    }
    catch( const Utility::Exception & ex )
    {
        Utility::Handle_exception( ex, idx_image, idx_chain );
        return ;
    }
    
	image->Lock();
    image->UpdateEnergy();
	image->Unlock();
}