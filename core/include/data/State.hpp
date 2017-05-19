#include <data/Spin_System_Chain_Collection.hpp>
#include <engine/Optimizer.hpp>
#include <engine/Method.hpp>
#include <utility/Timing.hpp>

/*
	Simulation_Information
		This struct contains the necessary instances to extract information
		during a running simulation.
		The Play_Pause function will insert this information in to the state
		appropriately.
*/
struct Simulation_Information
{
	std::shared_ptr<Engine::Optimizer> optimizer;
	std::shared_ptr<Engine::Method> method;
};

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
	std::vector<std::vector<std::shared_ptr<Simulation_Information>>> simulation_information_image;
	//    max. noc GNEB methods [noc]
	std::vector<std::shared_ptr<Simulation_Information>> simulation_information_chain;
	//    max. 1 MMF method
	std::shared_ptr<Simulation_Information> simulation_information_collection;

	// Timepoint of creation
	system_clock::time_point datetime_creation;
	std::string datetime_creation_string;
};


// TODO: move this away somewhere?
void from_indices(State * state, int & idx_image, int & idx_chain, std::shared_ptr<Data::Spin_System> & image, std::shared_ptr<Data::Spin_System_Chain> & chain);
