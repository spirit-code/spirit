#pragma once
#ifndef SPIRIT_CORE_DATA_STATE_HPP
#define SPIRIT_CORE_DATA_STATE_HPP

#include <engine/Method.hpp>
#include <utility/Exception.hpp>
#include <utility/Timing.hpp>

/*
 * The State struct is passed around in an application to make the
 * simulation's state available.
 * The State contains all necessary Spin Systems (via chain)
 * and provides a few utilities (pointers) to commonly used contents.
 */
struct State
{
    // Currently "active" chain
    std::shared_ptr<Data::Spin_System_Chain> chain;
    // Currently "active" image
    std::shared_ptr<Data::Spin_System> active_image;
    // Spin System instance in clipboard
    std::shared_ptr<Data::Spin_System> clipboard_image;

    // Spin configuration in clipboard
    std::shared_ptr<vectorfield> clipboard_spins;

    // Number of Spins
    int nos{ 0 };
    // Number of Images
    int noi{ 0 };
    // Index of the urrently "active" image
    int idx_active_image{ 0 };

    // The Methods
    //    max. noi*noc methods on images [noc][noi]
    std::vector<std::shared_ptr<Engine::Method>> method_image{};
    //    max. noc methods on the entire chain [noc]
    std::shared_ptr<Engine::Method> method_chain{};

    // Timepoint of creation
    system_clock::time_point datetime_creation{ system_clock::now() };
    std::string datetime_creation_string{ Utility::Timing::TimePointToString( datetime_creation ) };

    // Config file at creation
    std::string config_file{ "" };

    // Option to run quietly
    bool quiet{ false };
};

// Check if the state pointer seems to point to a correctly initialized state
inline void check_state( const State * state )
{
    if( state == nullptr )
    {
        spirit_throw(
            Utility::Exception_Classifier::System_not_Initialized, Utility::Log_Level::Error,
            "The State pointer is invalid" );
    }
    if( state->chain == nullptr )
    {
        spirit_throw(
            Utility::Exception_Classifier::System_not_Initialized, Utility::Log_Level::Error,
            "The State seems to not be initialised correctly" );
    }
}

/*
 * Passed indices for a chain and an image in the corresponding chain, this function converts
 * negative indices into the corresponding index of the currently "active" chain and image.
 * The shared pointers are assigned to point to the corresponding instances.
 *
 * Behaviour for illegal (non-existing) idx_image and idx_chain:
 *  - In case of negative values the indices must be promoted to the ones of the idx_active_image
 *    and idx_active_chain.
 *  - In case of negative (non-existing) indices the function should throw an exception before doing
 *    any change to the corresponding variable (eg. )
 */
inline void from_indices(
    const State * state, int & idx_image, int & idx_chain, std::shared_ptr<Data::Spin_System> & image,
    std::shared_ptr<Data::Spin_System_Chain> & chain )
{
    check_state( state );

    // Chain
    idx_chain = 0;
    chain     = state->chain;

    // In case of positive non-existing chain_idx throw exception
    if( idx_image >= state->chain->noi )
    {
        spirit_throw(
            Utility::Exception_Classifier::Non_existing_Image, Utility::Log_Level::Warning,
            fmt::format(
                "Index {} points to non-existent image (NOI={}). No action taken.", idx_image, state->chain->noi ) );
    }

    // Image
    if( idx_image < 0 )
    {
        image     = state->active_image;
        idx_image = state->idx_active_image;
    }
    else
    {
        image = chain->images[idx_image];
    }
}

#endif