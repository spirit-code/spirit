#include <Spirit/Quantities.h>
#include <Spirit/Geometry.h>
#include <data/State.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

void Quantity_Get_Magnetization(State * state,  float m[3], int idx_image, int idx_chain)
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
        Utility::Handle_Exception( idx_image, idx_chain );
        return ;
    }
    
	image->Lock();

    auto mag = Engine::Vectormath::Magnetization(*image->spins);
	image->M = Vector3{ mag[0], mag[1], mag[2] };
	image->Unlock();

    for (int i=0; i<3; ++i) m[i] = (float)mag[i];
}

float Quantity_Get_Topological_Charge(State * state, int idx_image, int idx_chain)
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
        Utility::Handle_Exception( idx_image, idx_chain );
        return 0;
    }

    scalar charge = 0;
    int dimensionality = Geometry_Get_Dimensionality(state, idx_image, idx_chain);
    image->Lock();
    if (dimensionality == 2)
        charge = Engine::Vectormath::TopologicalCharge(*image->spins, image->geometry->spin_pos, image->geometry->triangulation());
	image->Unlock();

    return (float)charge;
}