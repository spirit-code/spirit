#include <Spirit/Transitions.h>
#include <Spirit/State.h>
#include <data/State.hpp>
#include <data/Spin_System_Chain.hpp>
#include <utility/Configuration_Chain.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

#include <memory>

void Transition_Homogeneous(State *state, int idx_1, int idx_2, int idx_chain)
{
	int idx_image = -1;
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
    
    // Use this when State implements chain collection: else c = state->collection[idx_chain];
	chain->Lock();
    Utility::Configuration_Chain::Homogeneous_Rotation(chain, idx_1, idx_2);
	for (int img = 0; img < chain->noi; ++img)
	{
		chain->gneb_parameters->pinning->Apply(*chain->images[img]->spins);
	}
	chain->Unlock();

	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
		"Set homogeneous transition between images " + std::to_string(idx_1+1) + " and " + std::to_string(idx_2+1) + ".", -1, idx_chain);
}

void Transition_Add_Noise_Temperature(State *state, float temperature, int idx_1, int idx_2, int idx_chain)
{
	int idx_image = -1;
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
    
    // Use this when State implements chain collection: else c = state->collection[idx_chain];
	chain->Lock();
    Utility::Configuration_Chain::Add_Noise_Temperature(chain, idx_1, idx_2, temperature);
	for (int img = 0; img < chain->noi; ++img)
	{
		chain->gneb_parameters->pinning->Apply(*chain->images[img]->spins);
	}
	chain->Unlock();

	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
		"Added noise with temperature T=" + std::to_string(temperature) + " to images " + std::to_string(idx_1+1) + " to " + std::to_string(idx_2+1) + ".", -1, idx_chain);
}