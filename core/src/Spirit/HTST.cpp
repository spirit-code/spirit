#include <Spirit/HTST.h>

#include <data/State.hpp>
#include <engine/HTST.hpp>
#include <engine/Sparse_HTST.hpp>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

using Spirit::Data::Spin_System;
using Spirit::Data::Spin_System_Chain;
using Spirit::Utility::Log_Level;
using Spirit::Utility::Log_Sender;

scalar HTST_Calculate(
    State * state, int idx_image_minimum, int idx_image_sp, int n_eigenmodes_keep, bool sparse, int idx_chain )
try
{
    std::shared_ptr<Spin_System> image_minimum, image_sp;
    std::shared_ptr<Spin_System_Chain> chain;
    from_indices( state, idx_image_minimum, idx_chain, image_minimum, chain );
    from_indices( state, idx_image_sp, idx_chain, image_sp, chain );

    auto & info       = chain->htst_info;
    info.minimum      = image_minimum;
    info.saddle_point = image_sp;

#ifndef SPIRIT_SKIP_HTST
    if( !sparse )
        Spirit::Engine::HTST::Calculate( chain->htst_info, n_eigenmodes_keep );
    else
        Spirit::Engine::Sparse_HTST::Calculate( chain->htst_info );
#endif

    return info.prefactor;
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
    return 0;
}

void HTST_Get_Info(
    State * state, scalar * temperature_exponent, scalar * me, scalar * Omega_0, scalar * s, scalar * volume_min,
    scalar * volume_sp, scalar * prefactor_dynamical, scalar * prefactor, int * n_eigenmodes_keep,
    int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Spin_System> image;
    std::shared_ptr<Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    if( temperature_exponent != nullptr )
        *temperature_exponent = chain->htst_info.temperature_exponent;

    if( me != nullptr )
        *me = chain->htst_info.me;

    if( Omega_0 != nullptr )
        *Omega_0 = chain->htst_info.Omega_0;

    if( s != nullptr )
        *s = chain->htst_info.s;

    if( volume_min != nullptr )
        *volume_min = chain->htst_info.volume_min;

    if( volume_sp != nullptr )
        *volume_sp = chain->htst_info.volume_sp;

    if( prefactor_dynamical != nullptr )
        *prefactor_dynamical = chain->htst_info.prefactor_dynamical;

    if( prefactor != nullptr )
        *prefactor = chain->htst_info.prefactor;

    if( n_eigenmodes_keep != nullptr )
        *n_eigenmodes_keep = chain->htst_info.n_eigenmodes_keep;
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

void HTST_Get_Eigenvalues_Min( State * state, scalar * eigenvalues_min, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Spin_System> image;
    std::shared_ptr<Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    if( chain->htst_info.sparse )
    {
        Log( Log_Level::Error, Log_Sender::API,
             "HTST_Get_Eigenvalues_Min: You tried to call this function after perfroming a sparse calculation. This is "
             "not allowed." );
        return;
    }

    throw_if_nullptr( eigenvalues_min, "eigenvalues_min" );
    for( int i = 0; i < chain->htst_info.eigenvalues_min.size(); ++i )
        eigenvalues_min[i] = chain->htst_info.eigenvalues_min[i];
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

void HTST_Get_Eigenvectors_Min( State * state, scalar * eigenvectors_min, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Spin_System> image;
    std::shared_ptr<Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    if( chain->htst_info.sparse )
    {
        Log( Log_Level::Error, Log_Sender::API,
             "HTST_Get_Eigenvectors_Min: You tried to call this function after perfroming a sparse calculation. This "
             "is not allowed." );
        return;
    }

    throw_if_nullptr( eigenvectors_min, "eigenvectors_min" );
    for( int i = 0; i < chain->htst_info.eigenvectors_min.size(); ++i )
        eigenvectors_min[i] = chain->htst_info.eigenvectors_min( i );
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

void HTST_Get_Eigenvalues_SP( State * state, scalar * eigenvalues_sp, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Spin_System> image;
    std::shared_ptr<Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    if( chain->htst_info.sparse )
    {
        Log( Log_Level::Error, Log_Sender::API,
             "HTST_Get_Eigenvalues_SP: You tried to call this function after perfroming a sparse calculation. This is "
             "not allowed." );
        return;
    }

    throw_if_nullptr( eigenvalues_sp, "eigenvalues_sp" );
    int nos = image->nos;
    for( int i = 0; i < 2 * nos && i < chain->htst_info.eigenvalues_sp.size(); ++i )
        eigenvalues_sp[i] = chain->htst_info.eigenvalues_sp[i];
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

void HTST_Get_Eigenvectors_SP( State * state, scalar * eigenvectors_sp, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Spin_System> image;
    std::shared_ptr<Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    if( chain->htst_info.sparse )
    {
        Log( Log_Level::Error, Log_Sender::API,
             "HTST_Get_Eigenvectors_SP: You tried to call this function after perfroming a sparse calculation. This is "
             "not allowed." );
        return;
    }

    throw_if_nullptr( eigenvectors_sp, "eigenvectors_sp" );
    for( int i = 0; i < chain->htst_info.eigenvectors_sp.size(); ++i )
        eigenvectors_sp[i] = chain->htst_info.eigenvectors_sp( i );
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

void HTST_Get_Velocities( State * state, scalar * velocities, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Spin_System> image;
    std::shared_ptr<Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    if( chain->htst_info.sparse )
    {
        Log( Log_Level::Error, Log_Sender::API,
             "HTST_Get_Velocities: You tried to call this function after perfroming a sparse calculation. This is not "
             "allowed." );
        return;
    }

    throw_if_nullptr( velocities, "velocities" );
    int nos = image->nos;
    for( int i = 0; i < 2 * nos * nos && i < chain->htst_info.perpendicular_velocity.size(); ++i )
        velocities[i] = chain->htst_info.perpendicular_velocity[i];
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}