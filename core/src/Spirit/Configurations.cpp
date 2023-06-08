#include <Spirit/Configurations.h>
#include <Spirit/Spirit_Defines.h>

#include <data/State.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Configurations.hpp>
#include <utility/Constants.hpp>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

#include <fmt/format.h>
#include <Eigen/Dense>

namespace Configurations = Spirit::Utility::Configurations;
using Spirit::Data::Spin_System;
using Spirit::Data::Spin_System_Chain;
using Spirit::Utility::Log_Level;
using Spirit::Utility::Log_Sender;

std::function<bool( const Vector3 &, const Vector3 & )> get_filter(
    const Vector3 & position, const scalar r_cut_rectangular[3], scalar r_cut_cylindrical, scalar r_cut_spherical,
    bool inverted )
{
    bool no_cut_rectangular_x = r_cut_rectangular[0] < 0;
    bool no_cut_rectangular_y = r_cut_rectangular[1] < 0;
    bool no_cut_rectangular_z = r_cut_rectangular[2] < 0;
    bool no_cut_cylindrical   = r_cut_cylindrical < 0;
    bool no_cut_spherical     = r_cut_spherical < 0;

    std::function<bool( const Vector3 &, const Vector3 & )> filter;
    if( !inverted )
    {
        filter = [position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, no_cut_rectangular_x,
                  no_cut_rectangular_y, no_cut_rectangular_z, no_cut_cylindrical,
                  no_cut_spherical]( const Vector3 &, const Vector3 & positions )
        {
            Vector3 r_rectangular = positions - position;
            scalar r_cylindrical
                = std::sqrt( std::pow( positions[0] - position[0], 2 ) + std::pow( positions[1] - position[1], 2 ) );
            scalar r_spherical = ( positions - position ).norm();
            return ( no_cut_rectangular_x || std::abs( r_rectangular[0] ) < r_cut_rectangular[0] )
                   && ( no_cut_rectangular_y || std::abs( r_rectangular[1] ) < r_cut_rectangular[1] )
                   && ( no_cut_rectangular_z || std::abs( r_rectangular[2] ) < r_cut_rectangular[2] )
                   && ( no_cut_cylindrical || r_cylindrical < r_cut_cylindrical )
                   && ( no_cut_spherical || r_spherical < r_cut_spherical );
        };
    }
    else
    {
        filter = [position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, no_cut_rectangular_x,
                  no_cut_rectangular_y, no_cut_rectangular_z, no_cut_cylindrical,
                  no_cut_spherical]( const Vector3 & spin, const Vector3 & positions )
        {
            Vector3 r_rectangular = positions - position;
            scalar r_cylindrical
                = std::sqrt( std::pow( positions[0] - position[0], 2 ) + std::pow( positions[1] - position[1], 2 ) );
            scalar r_spherical = ( positions - position ).norm();
            return !(
                ( no_cut_rectangular_x || std::abs( r_rectangular[0] ) < r_cut_rectangular[0] )
                && ( no_cut_rectangular_y || std::abs( r_rectangular[1] ) < r_cut_rectangular[1] )
                && ( no_cut_rectangular_z || std::abs( r_rectangular[2] ) < r_cut_rectangular[2] )
                && ( no_cut_cylindrical || r_cylindrical < r_cut_cylindrical )
                && ( no_cut_spherical || r_spherical < r_cut_spherical ) );
        };
    }

    return filter;
}

std::string filter_to_string(
    const scalar position[3], const scalar r_cut_rectangular[3], scalar r_cut_cylindrical, scalar r_cut_spherical,
    bool inverted )
{
    std::string ret = "";

    if( position[0] != 0 || position[1] != 0 || position[2] != 0 )
        ret += fmt::format( "Position: ({}, {}, {}).", position[0], position[1], position[2] );

    if( r_cut_rectangular[0] <= 0 && r_cut_rectangular[1] <= 0 && r_cut_rectangular[2] <= 0 && r_cut_cylindrical <= 0
        && r_cut_spherical <= 0 && !inverted )
    {
        if( !ret.empty() )
            ret += " ";
        ret += "Entire space.";
    }
    else
    {
        if( r_cut_rectangular[0] > 0 || r_cut_rectangular[1] > 0 || r_cut_rectangular[2] > 0 )
        {
            if( !ret.empty() )
                ret += " ";
            ret += fmt::format(
                "Rectangular region: ({}, {}, {}).", r_cut_rectangular[0], r_cut_rectangular[1], r_cut_rectangular[2] );
        }
        if( r_cut_cylindrical > 0 )
        {
            if( !ret.empty() )
                ret += " ";
            ret += fmt::format( "Cylindrical region, r={}.", r_cut_cylindrical );
        }
        if( r_cut_spherical > 0 )
        {
            if( !ret.empty() )
                ret += " ";
            ret += fmt::format( "Spherical region, r={}.", r_cut_spherical );
        }
        if( inverted )
        {
            if( !ret.empty() )
                ret += " ";
            ret += "Inverted.";
        }
    }
    return ret;
}

void Configuration_To_Clipboard( State * state, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Spin_System> image;
    std::shared_ptr<Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    state->clipboard_spins = std::shared_ptr<vectorfield>( new vectorfield( *image->spins ) );
    Log( Log_Level::Info, Log_Sender::API, "Copied spin configuration to clipboard.", idx_image, idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Configuration_From_Clipboard(
    State * state, const scalar position[3], const scalar r_cut_rectangular[3], scalar r_cut_cylindrical,
    scalar r_cut_spherical, bool inverted, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Spin_System> image;
    std::shared_ptr<Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( position, "position" );
    throw_if_nullptr( r_cut_rectangular, "r_cut_rectangular" );

    // Get relative position
    Vector3 _pos{ position[0], position[1], position[2] };
    Vector3 vpos = image->geometry->center + _pos;

    // Create position filter
    auto filter = get_filter( vpos, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );

    // Apply configuration
    image->Lock();
    Configurations::Insert( *image, *state->clipboard_spins, 0, filter );
    image->geometry->Apply_Pinning( *image->spins );
    image->Unlock();

    auto filterstring = filter_to_string( position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );
    Log( Log_Level::Info, Log_Sender::API, "Set spin configuration from clipboard. " + filterstring, idx_image,
         idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

bool Configuration_From_Clipboard_Shift(
    State * state, const scalar shift[3], const scalar position[3], const scalar r_cut_rectangular[3],
    scalar r_cut_cylindrical, scalar r_cut_spherical, bool inverted, int idx_image, int idx_chain ) noexcept
try
{
    // Apply configuration
    if( state->clipboard_spins )
    {
        std::shared_ptr<Spin_System> image;
        std::shared_ptr<Spin_System_Chain> chain;

        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        throw_if_nullptr( position, "position" );
        throw_if_nullptr( r_cut_rectangular, "r_cut_rectangular" );

        // Get relative position
        Vector3 vpos{ position[0], position[1], position[2] };
        Vector3 vshift{ shift[0], shift[1], shift[2] };

        Vector3 decomposed = Spirit::Engine::Vectormath::decompose( vshift, image->geometry->bravais_vectors );

        int da = (int)std::round( decomposed[0] );
        int db = (int)std::round( decomposed[1] );
        int dc = (int)std::round( decomposed[2] );

        if( da == 0 && db == 0 && dc == 0 )
            return false;

        auto & geometry = *image->geometry;
        int delta       = geometry.n_cell_atoms * da + geometry.n_cell_atoms * geometry.n_cells[0] * db
                    + geometry.n_cell_atoms * geometry.n_cells[0] * geometry.n_cells[1] * dc;

        // Create position filter
        auto filter = get_filter( vpos, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );

        image->Lock();
        Spirit::Utility::Configurations::Insert( *image, *state->clipboard_spins, delta, filter );
        image->geometry->Apply_Pinning( *image->spins );
        image->Unlock();

        auto filterstring
            = filter_to_string( position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );
        Log( Log_Level::Info, Log_Sender::API, "Set shifted spin configuration from clipboard. " + filterstring,
             idx_image, idx_chain );
        return true;
    }
    else
    {
        Log( Log_Level::Info, Log_Sender::API, "Tried to insert configuration, but clipboard was empty.", idx_image,
             idx_chain );
        return false;
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return false;
}

void Configuration_Domain(
    State * state, const scalar direction[3], const scalar position[3], const scalar r_cut_rectangular[3],
    scalar r_cut_cylindrical, scalar r_cut_spherical, bool inverted, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Spin_System> image;
    std::shared_ptr<Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( direction, "direction" );
    throw_if_nullptr( position, "position" );
    throw_if_nullptr( r_cut_rectangular, "r_cut_rectangular" );

    // Get relative position
    Vector3 _pos{ position[0], position[1], position[2] };
    Vector3 vpos = image->geometry->center + _pos;

    // Create position filter
    auto filter = get_filter( vpos, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );

    // Apply configuration
    Vector3 vdir{ direction[0], direction[1], direction[2] };
    image->Lock();
    Spirit::Utility::Configurations::Domain( *image, vdir, filter );
    image->geometry->Apply_Pinning( *image->spins );
    image->Unlock();

    auto filterstring = filter_to_string( position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );
    Log( Log_Level::Info, Log_Sender::API,
         fmt::format(
             "Set domain configuration ({}, {}, {}). {}", direction[0], direction[1], direction[2], filterstring ),
         idx_image, idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

// void Configuration_DomainWall( State *state, const scalar pos[3], scalar v[3], bool greater,
//                                int idx_image, int idx_chain) noexcept
// {
//     std::shared_ptr<Spin_System> image;
//     std::shared_ptr<Spin_System_Chain> chain;
//     from_indices(state, idx_image, idx_chain, image, chain);

//     // Create position filter
//     Vector3 vpos{pos[0], pos[1], pos[2]};
//     std::function< bool( const Vector3&, const Vector3&) > filter = [vpos](const Vector3& spin,
//                         const Vector3& position)
//     {
//         scalar r = std::sqrt(std::pow(position[0] - vpos[0], 2) + std::pow(position[1] - vpos[1], 2));
//         if ( r < 3) return true;
//         return false;
//     };
//     // Apply configuration
//     Spirit::Utility::Configurations::Domain(*image, Vector3{ v[0],v[1],v[2] }, filter);
// }

void Configuration_PlusZ(
    State * state, const scalar position[3], const scalar r_cut_rectangular[3], scalar r_cut_cylindrical,
    scalar r_cut_spherical, bool inverted, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Spin_System> image;
    std::shared_ptr<Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( position, "position" );
    throw_if_nullptr( r_cut_rectangular, "r_cut_rectangular" );

    // Get relative position
    Vector3 _pos{ position[0], position[1], position[2] };
    Vector3 vpos = image->geometry->center + _pos;

    // Create position filter
    auto filter = get_filter( vpos, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );

    // Apply configuration
    Vector3 vdir{ 0, 0, 1 };
    image->Lock();
    Spirit::Utility::Configurations::Domain( *image, vdir, filter );
    image->geometry->Apply_Pinning( *image->spins );
    image->Unlock();

    auto filterstring = filter_to_string( position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );
    Log( Log_Level::Info, Log_Sender::API, "Set PlusZ configuration. " + filterstring, idx_image, idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Configuration_MinusZ(
    State * state, const scalar position[3], const scalar r_cut_rectangular[3], scalar r_cut_cylindrical,
    scalar r_cut_spherical, bool inverted, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Spin_System> image;
    std::shared_ptr<Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( position, "position" );
    throw_if_nullptr( r_cut_rectangular, "r_cut_rectangular" );

    // Get relative position
    Vector3 _pos{ position[0], position[1], position[2] };
    Vector3 vpos = image->geometry->center + _pos;

    // Create position filter
    auto filter = get_filter( vpos, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );

    // Apply configuration
    Vector3 vdir{ 0, 0, -1 };
    image->Lock();
    Spirit::Utility::Configurations::Domain( *image, vdir, filter );
    image->geometry->Apply_Pinning( *image->spins );
    image->Unlock();

    auto filterstring = filter_to_string( position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );
    Log( Log_Level::Info, Log_Sender::API, "Set MinusZ configuration. " + filterstring, idx_image, idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Configuration_Random(
    State * state, const scalar position[3], const scalar r_cut_rectangular[3], scalar r_cut_cylindrical,
    scalar r_cut_spherical, bool inverted, bool external, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Spin_System> image;
    std::shared_ptr<Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( position, "position" );
    throw_if_nullptr( r_cut_rectangular, "r_cut_rectangular" );

    // Get relative position
    Vector3 _pos{ position[0], position[1], position[2] };
    Vector3 vpos = image->geometry->center + _pos;

    // Create position filter
    auto filter = get_filter( vpos, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );

    // Apply configuration
    image->Lock();
    Spirit::Utility::Configurations::Random( *image, filter, external );
    image->geometry->Apply_Pinning( *image->spins );
    image->Unlock();

    auto filterstring = filter_to_string( position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );
    Log( Log_Level::Info, Log_Sender::API, "Set random configuration. " + filterstring, idx_image, idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Configuration_Add_Noise_Temperature(
    State * state, scalar temperature, const scalar position[3], const scalar r_cut_rectangular[3],
    scalar r_cut_cylindrical, scalar r_cut_spherical, bool inverted, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Spin_System> image;
    std::shared_ptr<Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( position, "position" );
    throw_if_nullptr( r_cut_rectangular, "r_cut_rectangular" );

    // Get relative position
    Vector3 _pos{ position[0], position[1], position[2] };
    Vector3 vpos = image->geometry->center + _pos;

    // Create position filter
    auto filter = get_filter( vpos, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );

    // Apply configuration
    image->Lock();
    Configurations::Add_Noise_Temperature( *image, temperature, 0, filter );
    image->geometry->Apply_Pinning( *image->spins );
    image->Unlock();

    auto filterstring = filter_to_string( position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );
    Log( Log_Level::Info, Log_Sender::API,
         fmt::format( "Added noise with temperature T={}. {}", temperature, filterstring ), idx_image, idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Configuration_Displace_Eigenmode( State * state, int idx_mode, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers for image and chain
    std::shared_ptr<Spin_System> image;
    std::shared_ptr<Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    if( idx_mode >= image->modes.size() )
        Log( Log_Level::Warning, Log_Sender::EMA,
             fmt::format(
                 "You tried to apply eigenmode number {}, "
                 "but you only calculated {} modes",
                 idx_mode, image->ema_parameters->n_modes ),
             idx_image, idx_chain );

    // The eigenmode was potentially not calculated, yet
    if( image->modes[idx_mode] == nullptr )
        Log( Log_Level::Warning, Log_Sender::EMA,
             fmt::format(
                 "Eigenmode number {} has not "
                 "yet been calculated.",
                 idx_mode ),
             idx_image, idx_chain );
    else
    {
        image->Lock();

        auto & spins = *image->spins;
        auto & mode  = *image->modes[idx_mode];
        int nos      = spins.size();

        scalarfield angles( nos );
        vectorfield axes( nos );

        // Find the angles and axes of rotation
        for( int idx = 0; idx < image->nos; idx++ )
        {
            angles[idx] = mode[idx].norm();
            axes[idx]   = spins[idx].cross( mode[idx] ).normalized();
        }

        // Scale the angles
        Spirit::Engine::Vectormath::scale( angles, image->ema_parameters->amplitude );

        // Rotate around axes by certain angles
        Spirit::Engine::Vectormath::rotate( spins, axes, angles, spins );

        image->Unlock();
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Configuration_Hopfion(
    State * state, scalar r, int order, const scalar position[3], const scalar r_cut_rectangular[3],
    scalar r_cut_cylindrical, scalar r_cut_spherical, bool inverted, const scalar normal[3], int idx_image,
    int idx_chain ) noexcept
try
{
    std::shared_ptr<Spin_System> image;
    std::shared_ptr<Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( position, "position" );
    throw_if_nullptr( r_cut_rectangular, "r_cut_rectangular" );
    throw_if_nullptr( normal, "normal" );

    // Get relative position
    Vector3 _pos{ position[0], position[1], position[2] };
    Vector3 vpos = image->geometry->center + _pos;

    // Set cutoff radius
    if( r_cut_spherical < 0 )
        r_cut_spherical = r * Spirit::Utility::Constants::Pi;

    // Create position filter
    auto filter = get_filter( vpos, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );

    // Apply configuration
    image->Lock();
    Configurations::Hopfion( *image, vpos, r, order, { normal[0], normal[1], normal[2] }, filter );
    image->geometry->Apply_Pinning( *image->spins );
    image->Unlock();

    auto filterstring = filter_to_string( position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );
    std::string parameterstring = fmt::format( "r={}", r );
    if( order != 1 )
        parameterstring += fmt::format( ", order={}", order );
    Log( Log_Level::Info, Log_Sender::API, "Set hopfion configuration, " + parameterstring + ". " + filterstring,
         idx_image, idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Configuration_Skyrmion(
    State * state, scalar r, scalar order, scalar phase, bool upDown, bool achiral, bool rl, const scalar position[3],
    const scalar r_cut_rectangular[3], scalar r_cut_cylindrical, scalar r_cut_spherical, bool inverted, int idx_image,
    int idx_chain ) noexcept
try
{
    std::shared_ptr<Spin_System> image;
    std::shared_ptr<Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( position, "position" );
    throw_if_nullptr( r_cut_rectangular, "r_cut_rectangular" );

    // Get relative position
    Vector3 _pos{ position[0], position[1], position[2] };
    Vector3 vpos = image->geometry->center + _pos;

    // Set cutoff radius
    if( r_cut_cylindrical < 0 )
        r_cut_cylindrical = r;

    // Create position filter
    auto filter = get_filter( vpos, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );

    // Apply configuration
    image->Lock();
    Configurations::Skyrmion( *image, vpos, r, order, phase, upDown, achiral, rl, false, filter );
    image->geometry->Apply_Pinning( *image->spins );
    image->Unlock();

    auto filterstring = filter_to_string( position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );
    std::string parameterstring = fmt::format( "r={}", r );
    if( order != 1 )
        parameterstring += fmt::format( ", order={}", order );
    if( phase != 0 )
        parameterstring += fmt::format( ", phase={}", phase );
    if( upDown )
        parameterstring += fmt::format( ", upDown={}", upDown );
    if( achiral )
        parameterstring += ", achiral";
    if( rl )
        parameterstring += fmt::format( ", rl={}", rl );
    Log( Log_Level::Info, Log_Sender::API, "Set skyrmion configuration, " + parameterstring + ". " + filterstring,
         idx_image, idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Configuration_DW_Skyrmion(
    State * state, scalar dw_radius, scalar dw_width, scalar order, scalar phase, bool upDown, bool achiral, bool rl,
    const scalar position[3], const scalar r_cut_rectangular[3], scalar r_cut_cylindrical, scalar r_cut_spherical,
    bool inverted, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Spin_System> image;
    std::shared_ptr<Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( position, "position" );
    throw_if_nullptr( r_cut_rectangular, "r_cut_rectangular" );

    // Get relative position
    Vector3 _pos{ position[0], position[1], position[2] };
    Vector3 vpos = image->geometry->center + _pos;

    // Set cutoff radius
    if( r_cut_cylindrical < 0 )
        r_cut_cylindrical = std::max( 3 * dw_radius, 3 * dw_width );

    // Create position filter
    auto filter = get_filter( vpos, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );

    // Apply configuration
    image->Lock();
    Configurations::DW_Skyrmion( *image, vpos, dw_radius, dw_width, order, phase, upDown, achiral, rl, filter );
    image->geometry->Apply_Pinning( *image->spins );
    image->Unlock();

    auto filterstring = filter_to_string( position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );
    std::string parameterstring = fmt::format( "dw_radius={}", dw_radius );
    parameterstring += fmt::format( ", dw_width={}", dw_width );
    if( order != 1 )
        parameterstring += fmt::format( ", order={}", order );
    if( phase != 0 )
        parameterstring += fmt::format( ", phase={}", phase );
    if( upDown )
        parameterstring += fmt::format( ", upDown={}", upDown );
    if( achiral )
        parameterstring += ", achiral";
    if( rl )
        parameterstring += fmt::format( ", rl={}", rl );
    Log( Log_Level::Info, Log_Sender::API,
         "Set 360 deg domain wall skyrmion configuration, " + parameterstring + ". " + filterstring, idx_image,
         idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Configuration_SpinSpiral(
    State * state, const char * direction_type, scalar q[3], scalar axis[3], scalar theta, const scalar position[3],
    const scalar r_cut_rectangular[3], scalar r_cut_cylindrical, scalar r_cut_spherical, bool inverted, int idx_image,
    int idx_chain ) noexcept
try
{
    std::shared_ptr<Spin_System> image;
    std::shared_ptr<Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( q, "q" );
    throw_if_nullptr( axis, "axis" );
    throw_if_nullptr( position, "position" );
    throw_if_nullptr( r_cut_rectangular, "r_cut_rectangular" );

    // Get relative position
    Vector3 _pos{ position[0], position[1], position[2] };
    Vector3 vpos = image->geometry->center + _pos;

    // Create position filter
    auto filter = get_filter( vpos, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );

    // Apply configuration
    std::string dir_type( direction_type );
    Vector3 vq{ q[0], q[1], q[2] };
    Vector3 vaxis{ axis[0], axis[1], axis[2] };
    image->Lock();
    Configurations::SpinSpiral( *image, dir_type, vq, vaxis, theta, filter );
    image->geometry->Apply_Pinning( *image->spins );
    image->Unlock();

    auto filterstring = filter_to_string( position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );
    std::string parameterstring = fmt::format(
        "W.r.t. {}, q=({}, {}, {}), axis=({}, {},{}), theta={}", direction_type, q[0], q[1], q[2], axis[0], axis[1],
        axis[2], theta );

    Log( Log_Level::Info, Log_Sender::API, "Set spin spiral configuration. " + parameterstring + ". " + filterstring,
         idx_image, idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Configuration_SpinSpiral_2q(
    State * state, const char * direction_type, scalar q1[3], scalar q2[3], scalar axis[3], scalar theta,
    const scalar position[3], const scalar r_cut_rectangular[3], scalar r_cut_cylindrical, scalar r_cut_spherical,
    bool inverted, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Spin_System> image;
    std::shared_ptr<Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( q1, "q1" );
    throw_if_nullptr( q2, "q2" );
    throw_if_nullptr( axis, "axis" );
    throw_if_nullptr( position, "position" );
    throw_if_nullptr( r_cut_rectangular, "r_cut_rectangular" );

    // Get relative position
    Vector3 _pos{ position[0], position[1], position[2] };
    Vector3 vpos = image->geometry->center + _pos;

    // Create position filter
    auto filter = get_filter( vpos, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );

    // Apply configuration
    std::string dir_type( direction_type );
    Vector3 vq1{ q1[0], q1[1], q1[2] };
    Vector3 vq2{ q2[0], q2[1], q2[2] };
    Vector3 vaxis{ axis[0], axis[1], axis[2] };
    image->Lock();
    Configurations::SpinSpiral( *image, dir_type, vq1, vq2, vaxis, theta, filter );
    image->Unlock();

    auto filterstring = filter_to_string( position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );

    std::string parameterstring = fmt::format(
        "W.r.t. {}, q1=({}, {}, {}), q2=({}, {}, {}), axis=({}, {},{}), theta={}", direction_type, q1[0], q1[1], q1[2],
        q2[0], q2[1], q2[2], axis[0], axis[1], axis[2], theta );

    Log( Log_Level::Info, Log_Sender::API, "Set spin spiral 2q configuration. " + parameterstring + ". " + filterstring,
         idx_image, idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

// Pinning
void Configuration_Set_Pinned(
    State * state, bool pinned, const scalar position[3], const scalar r_cut_rectangular[3], scalar r_cut_cylindrical,
    scalar r_cut_spherical, bool inverted, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Spin_System> image;
    std::shared_ptr<Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( position, "position" );
    throw_if_nullptr( r_cut_rectangular, "r_cut_rectangular" );

    // Get relative position
    Vector3 _pos{ position[0], position[1], position[2] };
    Vector3 vpos = image->geometry->center + _pos;

    // Create position filter
    auto filter = get_filter( vpos, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );

    // Apply configuration
    image->Lock();
    Configurations::Set_Pinned( *image, pinned, filter );
    image->Unlock();

    auto filterstring = filter_to_string( position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );
    Log( Log_Level::Info, Log_Sender::API, fmt::format( "Set pinned spins. {}", filterstring ), idx_image, idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

// Defects
void Configuration_Set_Atom_Type(
    State * state, int atom_type, const scalar position[3], const scalar r_cut_rectangular[3], scalar r_cut_cylindrical,
    scalar r_cut_spherical, bool inverted, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Spin_System> image;
    std::shared_ptr<Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( position, "position" );
    throw_if_nullptr( r_cut_rectangular, "r_cut_rectangular" );

    // Get relative position
    Vector3 _pos{ position[0], position[1], position[2] };
    Vector3 vpos = image->geometry->center + _pos;

    // Create position filter
    auto filter = get_filter( vpos, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );

    // Apply configuration
    image->Lock();
    Configurations::Set_Atom_Types( *image, atom_type, filter );
    image->Unlock();

    auto filterstring = filter_to_string( position, r_cut_rectangular, r_cut_cylindrical, r_cut_spherical, inverted );
    Log( Log_Level::Info, Log_Sender::API, fmt::format( "Set atom types to {}. {}", atom_type, filterstring ),
         idx_image, idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}