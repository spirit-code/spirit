#pragma once
#ifndef SPIRIT_CORE_DATA_STATE_HPP
#define SPIRIT_CORE_DATA_STATE_HPP

#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <engine/spin/Hamiltonian.hpp>
#include <engine/spin/Method.hpp>
#include <utility/Exception.hpp>

#include <fmt/chrono.h>
#include <fmt/format.h>

#include <chrono>
#include <memory>
#include <string>
#include <vector>

/*
 * The State struct is passed around in an application to make the
 * simulation's state available.
 * The State contains all necessary Spin Systems (via chain)
 * and provides a few utilities (pointers) to commonly used contents.
 */
struct State
{
    using hamiltonian_t = Engine::Spin::hamiltonian_t;
    using state_t       = typename hamiltonian_t::state_t;
    using chain_t       = Engine::Spin::chain_t;
    using system_t      = Engine::Spin::system_t;

    // Currently "active" chain
    std::shared_ptr<chain_t> chain;
    // Currently "active" image
    std::shared_ptr<system_t> active_image;
    // Spin System instance in clipboard
    std::shared_ptr<system_t> clipboard_image;

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
    std::chrono::system_clock::time_point datetime_creation = std::chrono::system_clock::now();
    std::string datetime_creation_string                    = fmt::format( "{:%Y-%m-%d_%H-%M-%S}", datetime_creation );

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

// Check if the given pointer is a null pointer and, if so, throw with a suitable message
inline void throw_if_nullptr( const void * ptr, const std::string_view name )
{
    if( ptr == nullptr )
    {
        spirit_throw(
            Utility::Exception_Classifier::API_GOT_NULLPTR, Utility::Log_Level::Error,
            fmt::format( "Got passed a null pointer for '{}'", name ) );
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
template<typename State>
[[nodiscard]] auto from_indices( const State * state, int & idx_image, int & idx_chain )
    -> std::pair<std::shared_ptr<typename State::system_t>, std::shared_ptr<typename State::chain_t>>
{
    check_state( state );

    std::shared_ptr<typename State::system_t> image;
    std::shared_ptr<typename State::chain_t> chain;

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

    return { image, chain };
}

#endif
