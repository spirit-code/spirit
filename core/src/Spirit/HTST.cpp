#include <Spirit/HTST.h>

#include <data/State.hpp>
#include <engine/HTST.hpp>
#include <engine/Sparse_HTST.hpp>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

float HTST_Calculate(
    State * state, int idx_image_minimum, int idx_image_sp, int n_eigenmodes_keep, bool sparse, int idx_chain )
try
{
    std::shared_ptr<Data::Spin_System> image_minimum, image_sp;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices( state, idx_image_minimum, idx_chain, image_minimum, chain );
    from_indices( state, idx_image_sp, idx_chain, image_sp, chain );

    auto & info       = chain->htst_info;
    info.minimum      = image_minimum;
    info.saddle_point = image_sp;

#ifndef SPIRIT_SKIP_HTST
    if( !sparse )
        Engine::HTST::Calculate( chain->htst_info, n_eigenmodes_keep );
    else
        Engine::Sparse_HTST::Calculate( chain->htst_info );
#endif

    return (float)info.prefactor;
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
    return 0;
}

void HTST_Get_Info(
    State * state, float * temperature_exponent, float * me, float * Omega_0, float * s, float * volume_min,
    float * volume_sp, float * prefactor_dynamical, float * prefactor, int * n_eigenmodes_keep, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

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

void HTST_Get_Eigenvalues_Min( State * state, float * eigenvalues_min, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    if( chain->htst_info.sparse )
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::API,
             "HTST_Get_Eigenvalues_Min: You tried to call this function after perfroming a sparse calculation. This is "
             "not allowed." );
        return;
    }

    if( eigenvalues_min != nullptr )
    {
        for( int i = 0; i < chain->htst_info.eigenvalues_min.size(); ++i )
            eigenvalues_min[i] = chain->htst_info.eigenvalues_min[i];
    }
    else
        Log( Utility::Log_Level::Error, Utility::Log_Sender::API,
             "HTST_Get_Eigenvalues_Min: you passed a null pointer" );
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

void HTST_Get_Eigenvectors_Min( State * state, float * eigenvectors_min, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    if( chain->htst_info.sparse )
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::API,
             "HTST_Get_Eigenvectors_Min: You tried to call this function after perfroming a sparse calculation. This "
             "is not allowed." );
        return;
    }

    if( eigenvectors_min != nullptr )
    {
        for( int i = 0; i < chain->htst_info.eigenvectors_min.size(); ++i )
            eigenvectors_min[i] = chain->htst_info.eigenvectors_min( i );
    }
    else
        Log( Utility::Log_Level::Error, Utility::Log_Sender::API,
             "HTST_Get_Eigenvectors_Min: you passed a null pointer" );
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

void HTST_Get_Eigenvalues_SP( State * state, float * eigenvalues_sp, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    if( chain->htst_info.sparse )
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::API,
             "HTST_Get_Eigenvalues_SP: You tried to call this function after perfroming a sparse calculation. This is "
             "not allowed." );
        return;
    }

    if( eigenvalues_sp != nullptr )
    {
        int nos = image->nos;
        for( int i = 0; i < 2 * nos && i < chain->htst_info.eigenvalues_sp.size(); ++i )
            eigenvalues_sp[i] = chain->htst_info.eigenvalues_sp[i];
    }
    else
        Log( Utility::Log_Level::Error, Utility::Log_Sender::API,
             "HTST_Get_Eigenvalues_SP: you passed a null pointer" );
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

void HTST_Get_Eigenvectors_SP( State * state, float * eigenvectors_sp, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    if( chain->htst_info.sparse )
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::API,
             "HTST_Get_Eigenvectors_SP: You tried to call this function after perfroming a sparse calculation. This is "
             "not allowed." );
        return;
    }

    if( eigenvectors_sp != nullptr )
    {
        for( int i = 0; i < chain->htst_info.eigenvectors_sp.size(); ++i )
            eigenvectors_sp[i] = chain->htst_info.eigenvectors_sp( i );
    }
    else
        Log( Utility::Log_Level::Error, Utility::Log_Sender::API,
             "HTST_Get_Eigenvectors_SP: you passed a null pointer" );
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

void HTST_Get_Velocities( State * state, float * velocities, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    if( chain->htst_info.sparse )
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::API,
             "HTST_Get_Velocities: You tried to call this function after perfroming a sparse calculation. This is not "
             "allowed." );
        return;
    }

    if( velocities != nullptr )
    {
        int nos = image->nos;
        for( int i = 0; i < 2 * nos * nos && i < chain->htst_info.perpendicular_velocity.size(); ++i )
            velocities[i] = chain->htst_info.perpendicular_velocity[i];
    }
    else
        Log( Utility::Log_Level::Error, Utility::Log_Sender::API, "HTST_Get_Velocities: you passed a null pointer" );
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}