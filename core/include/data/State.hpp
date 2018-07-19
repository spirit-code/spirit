#include <engine/Method.hpp>
#include <utility/Timing.hpp>

/*
    State
        The State struct is passed around in an application to make the
        simulation's state available.
        The State contains all necessary Spin Systems (via chain)
        and provides a few utilities (pointers) to commonly used contents.
*/
struct State
{
    // Currently active chain
    std::shared_ptr<Data::Spin_System_Chain> chain;
    // Currently active image
    std::shared_ptr<Data::Spin_System> active_image;
    // Spin System instance in clipboard
    std::shared_ptr<Data::Spin_System> clipboard_image;

    // Spin configuration in clipboard
    std::shared_ptr<vectorfield> clipboard_spins;

    // Info
    int nos /*Number of Spins*/, noi /*Number of Images*/;
    int idx_active_image;

    // The Methods
    //    max. noi*noc methods on images [noc][noi]
    std::vector<std::shared_ptr<Engine::Method>> method_image;
    //    max. noc methods on the entire chain [noc]
    std::shared_ptr<Engine::Method> method_chain;

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
