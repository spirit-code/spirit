#include <data/Spin_System_Chain_Collection.hpp>
#include <engine/Method.hpp>
#include <utility/Timing.hpp>

/*
	State
      The State struct is passed around in an application to make the
      simulation's state available.
	  The State contains all necessary Spin Systems (via chain and collection)
	  and provides a few utilities (pointers) to commonly used contents.
*/
struct State
{
	// Main data container: a collection of chains
	std::shared_ptr<Data::Spin_System_Chain_Collection> collection;
	// Currently active chain
	std::shared_ptr<Data::Spin_System_Chain> active_chain;
	// Currently active image
	std::shared_ptr<Data::Spin_System> active_image;
	// Spin System instance in clipboard
	std::shared_ptr<Data::Spin_System> clipboard_image;

	// Spin configuration in clipboard
	std::shared_ptr<vectorfield> clipboard_spins;

	// Info
	int nos /*Number of Spins*/, noi /*Number of Images*/, noc /*Number of Chains*/;
	int idx_active_image, idx_active_chain;

	// The Methods
	//    max. noi*noc LLG methods [noc][noi]
	std::vector<std::vector<std::shared_ptr<Engine::Method>>> method_image;
	//    max. noc GNEB methods [noc]
	std::vector<std::shared_ptr<Engine::Method>> method_chain;
	//    max. 1 MMF method
	std::shared_ptr<Engine::Method> method_collection;

	// Timepoint of creation
	system_clock::time_point datetime_creation;
	std::string datetime_creation_string;

	// Config file at creation
	std::string config_file;

	// Option to run quietly
	bool quiet;
};

// TODO: move this away somewhere?
// Behaviour for illegal (non-existing) idx_image and idx_chain:
// - In case of negative values the indices must be promoted to the ones of the idx_active_image 
//  and idx_active_chain. 
// - In case of negative (non-existing) indices the function should throw an exception before doing 
// any change to the corresponding variable (eg. )
void from_indices( const State * state, int & idx_image, int & idx_chain, 
                   std::shared_ptr<Data::Spin_System> & image, 
                   std::shared_ptr<Data::Spin_System_Chain> & chain );
