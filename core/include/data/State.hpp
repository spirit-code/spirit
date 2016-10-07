#include "Spin_System_Chain_Collection.hpp"
#include "Optimizer.hpp"
#include "Method_LLG.hpp"
#include "Method_GNEB.hpp"
#include "Method_MMF.hpp"

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

	// Info
	int nos /*Number of Spins*/, noi /*Number of Images*/, noc /*Number of Chains*/;
	int idx_active_image, idx_active_chain;

	// The Methods
	//    max. noi*noc LLG methods
	std::vector<std::vector<std::shared_ptr<Engine::Method_LLG>>> methods_llg; // [noc][noi]
																			   //    max. noc GNEB methods
	std::vector<std::shared_ptr<Engine::Method_GNEB>> methods_gneb; // [noc]
																	//    max. 1 MMF method
	std::shared_ptr<Engine::Method_MMF> method_mmf;
};


// TODO: move this away somewhere?
void from_indices(State * state, int & idx_image, int & idx_chain, std::shared_ptr<Data::Spin_System> & image, std::shared_ptr<Data::Spin_System_Chain> & chain);