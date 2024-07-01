#include <Spirit/Hamiltonian.h>

#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <data/State.hpp>
#include <engine/Neighbours.hpp>
#include <engine/Vectormath.hpp>
#include <engine/spin/Hamiltonian.hpp>
#include <utility/Constants.hpp>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

#include <fmt/format.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <fstream>
#include <optional>

using namespace Utility;

/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Set Parameters ---------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

void Hamiltonian_Set_Boundary_Conditions(
    State * state, const bool * periodical, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );
    throw_if_nullptr( periodical, "periodical" );

    image->lock();
    try
    {
        image->hamiltonian->set_boundary_conditions( { periodical[0], periodical[1], periodical[2] } );
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image, idx_chain );
    }
    image->unlock();

    Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
         fmt::format( "Set boundary conditions to {} {} {}", periodical[0], periodical[1], periodical[2] ), idx_image,
         idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Hamiltonian_Set_Field(
    State * state, scalar magnitude, const scalar * normal, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );
    throw_if_nullptr( normal, "normal" );

    // Lock mutex because simulations may be running
    image->lock();
    try
    {
        // Normals
        Vector3 new_normal{ normal[0], normal[1], normal[2] };
        new_normal.normalize();

        // Into the Hamiltonian
        using Engine::Spin::Interaction::Zeeman;
        const auto error = image->hamiltonian->set_data<Zeeman>( magnitude * Constants::mu_B, new_normal );
        if( !error.has_value() )
            Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
                 fmt::format(
                     "Set external field to {}, direction ({}, {}, {})", magnitude, normal[0], normal[1], normal[2] ),
                 idx_image, idx_chain );
        else
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::API, *error, idx_image, idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image, idx_chain );
    }

    // Unlock mutex
    image->unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Hamiltonian_Set_Anisotropy(
    State * state, scalar magnitude, const scalar * normal, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );
    throw_if_nullptr( normal, "normal" );

    image->lock();
    try
    {
        int nos          = image->nos;
        int n_cell_atoms = image->hamiltonian->get_geometry().n_cell_atoms;

        // Indices and Magnitudes
        intfield new_indices( n_cell_atoms );
        scalarfield new_magnitudes( n_cell_atoms );
        for( int i = 0; i < n_cell_atoms; ++i )
        {
            new_indices[i]    = i;
            new_magnitudes[i] = magnitude;
        }
        // Normals
        Vector3 new_normal{ normal[0], normal[1], normal[2] };
        new_normal.normalize();
        vectorfield new_normals( n_cell_atoms, new_normal );
        // Update the Hamiltonian
        using Engine::Spin::Interaction::Anisotropy;
        const auto error = image->hamiltonian->set_data<Anisotropy>( new_indices, new_magnitudes, new_normals );

        if( !error.has_value() )
            Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
                 fmt::format(
                     "Set anisotropy to {}, direction ({}, {}, {})", magnitude, normal[0], normal[1], normal[2] ),
                 idx_image, idx_chain );
        else
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::API, *error, idx_image, idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image, idx_chain );
    }

    image->unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Hamiltonian_Set_Cubic_Anisotropy( State * state, scalar magnitude, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    image->lock();
    try
    {
        int nos          = image->nos;
        int n_cell_atoms = image->hamiltonian->get_geometry().n_cell_atoms;

        // Indices and Magnitudes
        intfield new_indices( n_cell_atoms );
        scalarfield new_magnitudes( n_cell_atoms );
        for( int i = 0; i < n_cell_atoms; ++i )
        {
            new_indices[i]    = i;
            new_magnitudes[i] = magnitude;
        }

        // Update the Hamiltonian
        using Engine::Spin::Interaction::Cubic_Anisotropy;
        const auto error = image->hamiltonian->set_data<Cubic_Anisotropy>( new_indices, new_magnitudes );

        if( !error.has_value() )
            Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
                 fmt::format( "Set cubic anisotropy to {}", magnitude ), idx_image, idx_chain );
        else
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::API, *error, idx_image, idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image, idx_chain );
    }

    image->unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Hamiltonian_Set_Biaxial_Anisotropy(
    State * state, const scalar * magnitude, const unsigned int exponents[][3], const scalar * primary,
    const scalar * secondary, int n_terms, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    image->lock();
    try
    {
        using Engine::Spin::Interaction::Biaxial_Anisotropy;
        std::optional<std::string> error   = std::nullopt;
        std::size_t new_on_site_terms_size = 0;
        if( image->hamiltonian->hasInteraction<Biaxial_Anisotropy>() )
        {
            int n_cell_atoms = image->hamiltonian->get_geometry().n_cell_atoms;

            const auto new_primary   = Vector3{ primary[0], primary[1], primary[2] }.normalized();
            const auto new_secondary = [&secondary, &new_primary]()
            {
                auto new_secondary = Vector3{ secondary[0], secondary[1], secondary[2] };
                new_secondary -= new_primary.dot( new_secondary ) * new_primary;
                new_secondary.normalize();
                return new_secondary;
            }();

            const Vector3 new_ternary = new_primary.cross( new_secondary ).normalized();

            field<PolynomialTerm> new_on_site_terms{};
            for( auto i = 0; i < n_terms; ++i )
            {
                if( magnitude[i] == 0 )
                    continue;

                new_on_site_terms.push_back(
                    PolynomialTerm{ magnitude[i], exponents[i][0], exponents[i][1], exponents[i][2] } );
            };

            // Indices and polynomial data
            intfield new_indices( n_cell_atoms );
            for( int i = 0; i < n_cell_atoms; ++i )
            {
                new_indices[i] = i;
            }
            field<PolynomialBasis> new_polynomial_bases( n_cell_atoms, { new_primary, new_secondary, new_ternary } );

            field<unsigned int> new_polynomial_site_p( n_cell_atoms == 0 ? 0 : n_cell_atoms + 1, 0u );
            std::generate(
                begin( new_polynomial_site_p ), end( new_polynomial_site_p ),
                [i = 0, n = new_on_site_terms.size()]() mutable { return ( i++ ) * n; } );

            field<PolynomialTerm> new_polynomial_terms{};
            new_polynomial_terms.reserve( n_cell_atoms * new_on_site_terms.size() );

            for( int i = 0; i < n_cell_atoms; ++i )
            {
                std::copy(
                    cbegin( new_on_site_terms ), cend( new_on_site_terms ),
                    std::back_inserter( new_polynomial_terms ) );
            }

            new_on_site_terms_size = new_on_site_terms.size();

            // Update the Hamiltonian
            error = image->hamiltonian->set_data<Biaxial_Anisotropy>(
                new_indices, new_polynomial_bases, new_polynomial_site_p, new_polynomial_terms );
        }
        if( !error.has_value() )
            Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
                 fmt::format( "Set {} terms for biaxial anisotropy", new_on_site_terms_size ), idx_image, idx_chain );
        else
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::API, *error, idx_image, idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image, idx_chain );
    }

    image->unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Hamiltonian_Set_Exchange( State * state, int n_shells, const scalar * jij, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );
    throw_if_nullptr( jij, "jij" );

    image->lock();
    try
    {
        using Engine::Spin::Interaction::Exchange;
        // Update the Hamiltonian
        const auto error = image->hamiltonian->set_data<Exchange>( scalarfield( jij, jij + n_shells ) );

        if( !error.has_value() )
        {
            std::string message = fmt::format( "Set exchange to {} shells", n_shells );
            if( n_shells > 0 )
                message += fmt::format( " Jij[0] = {}", jij[0] );
            Log( Utility::Log_Level::Info, Utility::Log_Sender::API, message, idx_image, idx_chain );
        }
        else
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::API, *error, idx_image, idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image, idx_chain );
    }

    image->unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Hamiltonian_Set_DMI(
    State * state, int n_shells, const scalar * dij, int chirality, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );
    throw_if_nullptr( dij, "dij" );

    if( chirality != SPIRIT_CHIRALITY_BLOCH && chirality != SPIRIT_CHIRALITY_NEEL
        && chirality != SPIRIT_CHIRALITY_BLOCH_INVERSE && chirality != SPIRIT_CHIRALITY_NEEL_INVERSE )
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::API,
             fmt::format( "Hamiltonian_Set_DMI: Invalid DM chirality {}", chirality ), idx_image, idx_chain );
        return;
    }

    image->lock();
    try
    {
        using Engine::Spin::Interaction::DMI;
        // Update the Hamiltonian
        const auto error = image->hamiltonian->set_data<DMI>( scalarfield( dij, dij + n_shells ), chirality );
        if( !error.has_value() )
        {
            std::string message = fmt::format( "Set dmi to {} shells", n_shells );
            if( n_shells > 0 )
                message += fmt::format( " Dij[0] = {}", dij[0] );
            Log( Utility::Log_Level::Info, Utility::Log_Sender::API, message, idx_image, idx_chain );
        }
        else
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::API, *error, idx_image, idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image, idx_chain );
    }

    image->unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Hamiltonian_Set_DDI(
    State * state, int ddi_method, int n_periodic_images[3], scalar cutoff_radius, bool pb_zero_padding, int idx_image,
    int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );
    throw_if_nullptr( n_periodic_images, "n_periodic_images" );

    image->lock();
    try
    {
        using Engine::Spin::Interaction::DDI;
        auto new_n_periodic_images = intfield( 3 );
        new_n_periodic_images[0]   = n_periodic_images[0];
        new_n_periodic_images[1]   = n_periodic_images[1];
        new_n_periodic_images[2]   = n_periodic_images[2];

        const auto error = image->hamiltonian->set_data<DDI>(
            Engine::Spin::DDI_Method( ddi_method ), cutoff_radius, pb_zero_padding, new_n_periodic_images );

        if( !error.has_value() )
            Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
                 fmt::format(
                     "Set ddi to method {}, periodic images {} {} {}, cutoff radius {} and pb_zero_padding {}",
                     ddi_method, n_periodic_images[0], n_periodic_images[1], n_periodic_images[2], cutoff_radius,
                     pb_zero_padding ),
                 idx_image, idx_chain );
        else
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::API, *error, idx_image, idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image, idx_chain );
    }

    image->unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Get Parameters ---------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

const char * Hamiltonian_Get_Name( State * state, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    return strdup( image->hamiltonian->Name().data() );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return nullptr;
}

void Hamiltonian_Get_Boundary_Conditions( State * state, bool * periodical, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );
    throw_if_nullptr( periodical, "periodical" );

    const auto & boundary_conditions = image->hamiltonian->get_boundary_conditions();
    periodical[0]                    = (bool)boundary_conditions[0];
    periodical[1]                    = (bool)boundary_conditions[1];
    periodical[2]                    = (bool)boundary_conditions[2];
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Hamiltonian_Get_Field( State * state, scalar * magnitude, scalar * normal, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );
    throw_if_nullptr( magnitude, "magnitude" );
    throw_if_nullptr( normal, "normal" );

    if( const auto * data = image->hamiltonian->data<Engine::Spin::Interaction::Zeeman>(); data != nullptr )
    {
        const scalar & field_magnitude = data->external_field_magnitude;
        const Vector3 & field_normal   = data->external_field_normal;

        if( field_magnitude > 0 )
        {
            *magnitude = field_magnitude / Constants::mu_B;
            normal[0]  = field_normal[0];
            normal[1]  = field_normal[1];
            normal[2]  = field_normal[2];
        }
        else
        {
            *magnitude = 0;
            normal[0]  = 0;
            normal[1]  = 0;
            normal[2]  = 1;
        }
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Hamiltonian_Get_Anisotropy(
    State * state, scalar * magnitude, scalar * normal, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );
    throw_if_nullptr( magnitude, "magnitude" );
    throw_if_nullptr( normal, "normal" );

    using Engine::Spin::Interaction::Anisotropy;
    if( const auto * data = image->hamiltonian->data<Anisotropy>(); data != nullptr )
    {
        const auto & anisotropy_indices    = data->indices;
        const auto & anisotropy_magnitudes = data->magnitudes;
        const auto & anisotropy_normals    = data->normals;
        if( !anisotropy_indices.empty() )
        {
            // Magnitude
            *magnitude = anisotropy_magnitudes[0];

            // Normal
            normal[0] = anisotropy_normals[0][0];
            normal[1] = anisotropy_normals[0][1];
            normal[2] = anisotropy_normals[0][2];
        }
        else
        {
            *magnitude = 0;
            normal[0]  = 0;
            normal[1]  = 0;
            normal[2]  = 1;
        }
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Hamiltonian_Get_Cubic_Anisotropy( State * state, scalar * magnitude, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );
    throw_if_nullptr( magnitude, "magnitude" );

    if( const auto * data = image->hamiltonian->data<Engine::Spin::Interaction::Cubic_Anisotropy>(); data != nullptr )
    {
        const auto & cubic_anisotropy_indices    = data->indices;
        const auto & cubic_anisotropy_magnitudes = data->magnitudes;

        if( !cubic_anisotropy_indices.empty() )
        {
            // Magnitude
            *magnitude = cubic_anisotropy_magnitudes[0];
        }
        else
        {
            *magnitude = 0;
        }
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

int Hamiltonian_Get_Biaxial_Anisotropy_N_Atoms( State * state, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    if( const auto * data = image->hamiltonian->data<Engine::Spin::Interaction::Biaxial_Anisotropy>(); data != nullptr )
    {
        return data->indices.size();
    }
    else
    {
        return 0;
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

int Hamiltonian_Get_Biaxial_Anisotropy_N_Terms( State * state, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    if( const auto * data = image->hamiltonian->data<Engine::Spin::Interaction::Biaxial_Anisotropy>(); data != nullptr )
    {
        return data->terms.size();
    }
    else
    {
        return 0;
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

void Hamiltonian_Get_Biaxial_Anisotropy(
    State * state, int * indices, scalar primary[][3], scalar secondary[][3], int * site_p, const int n_indices,
    scalar * magnitude, int exponents[][3], const int n_terms, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );
    throw_if_nullptr( indices, "indices" );
    throw_if_nullptr( primary, "primary" );
    throw_if_nullptr( secondary, "secondary" );
    throw_if_nullptr( site_p, "site_p" );
    throw_if_nullptr( magnitude, "magnitude" );
    throw_if_nullptr( exponents, "exponents" );

    if( const auto * data = image->hamiltonian->data<Engine::Spin::Interaction::Biaxial_Anisotropy>(); data != nullptr )
    {
        const auto & anisotropy_indices           = data->indices;
        const auto & anisotropy_polynomial_basis  = data->bases;
        const auto & anisotropy_polynomial_site_p = data->site_p;
        const auto & anisotropy_polynomial_terms  = data->terms;

        std::copy_n( cbegin( anisotropy_indices ), n_indices, indices );

        for( int j = 0; j < n_indices; ++j )
        {
            const auto & k1 = anisotropy_polynomial_basis[j].k1;
            std::copy( std::cbegin( k1 ), std::cend( k1 ), primary[j] );

            const auto & k2 = anisotropy_polynomial_basis[j].k2;
            std::copy( std::cbegin( k2 ), std::cend( k2 ), secondary[j] );
        }

        std::copy_n( cbegin( anisotropy_polynomial_site_p ), n_indices + 1, site_p );

        for( int i = 0; i < n_terms; ++i )
        {
            magnitude[i]    = anisotropy_polynomial_terms[i].coefficient;
            exponents[i][0] = anisotropy_polynomial_terms[i].n1;
            exponents[i][1] = anisotropy_polynomial_terms[i].n2;
            exponents[i][2] = anisotropy_polynomial_terms[i].n3;
        }
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Hamiltonian_Get_Exchange_Shells(
    State * state, int * n_shells, scalar * jij, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );
    throw_if_nullptr( n_shells, "n_shells" );
    throw_if_nullptr( jij, "jij" );

    if( const auto * data = image->hamiltonian->data<Engine::Spin::Interaction::Exchange>(); data != nullptr )
    {
        scalarfield exchange_shell_magnitudes = data->shell_magnitudes;
        *n_shells                             = exchange_shell_magnitudes.size();

        // Note the array needs to be correctly allocated beforehand!
        for( std::size_t i = 0; i < exchange_shell_magnitudes.size(); ++i )
        {
            jij[i] = exchange_shell_magnitudes[i];
        }
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

int Hamiltonian_Get_Exchange_N_Pairs( State * state, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    if( const auto * cache = image->hamiltonian->cache<Engine::Spin::Interaction::Exchange>(); cache != nullptr )
    {
        return cache->pairs.size();
    }

    return 0;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

void Hamiltonian_Get_Exchange_Pairs(
    State * state, int idx[][2], int translations[][3], scalar * Jij, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );
    throw_if_nullptr( Jij, "Jij" );

    if( const auto * cache = image->hamiltonian->cache<Engine::Spin::Interaction::Exchange>(); cache != nullptr )
    {
        const auto & exchange_pairs      = cache->pairs;
        const auto & exchange_magnitudes = cache->magnitudes;

        for( std::size_t i = 0; i < exchange_pairs.size() && i < exchange_magnitudes.size(); ++i )
        {
            const auto & pair  = exchange_pairs[i];
            idx[i][0]          = pair.i;
            idx[i][1]          = pair.j;
            translations[i][0] = pair.translations[0];
            translations[i][1] = pair.translations[1];
            translations[i][2] = pair.translations[2];
            Jij[i]             = exchange_magnitudes[i];
        }
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Hamiltonian_Get_DMI_Shells(
    State * state, int * n_shells, scalar * dij, int * chirality, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );
    throw_if_nullptr( n_shells, "n_shells" );
    throw_if_nullptr( dij, "dij" );
    throw_if_nullptr( chirality, "chirality" );

    if( const auto * data = image->hamiltonian->data<Engine::Spin::Interaction::DMI>(); data != nullptr )
    {
        const auto & dmi_shell_magnitudes = data->shell_magnitudes;
        const auto dmi_shell_chirality    = data->shell_chirality;

        *n_shells  = dmi_shell_magnitudes.size();
        *chirality = dmi_shell_chirality;

        for( int i = 0; i < *n_shells; ++i )
        {
            dij[i] = dmi_shell_magnitudes[i];
        }
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

int Hamiltonian_Get_DMI_N_Pairs( State * state, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    if( const auto * cache = image->hamiltonian->cache<Engine::Spin::Interaction::DMI>(); cache != nullptr )
    {
        return cache->pairs.size();
    }

    return 0;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

void Hamiltonian_Get_DDI(
    State * state, int * ddi_method, int n_periodic_images[3], scalar * cutoff_radius, bool * pb_zero_padding,
    int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );
    throw_if_nullptr( ddi_method, "ddi_method" );
    throw_if_nullptr( cutoff_radius, "cutoff_radius" );
    throw_if_nullptr( pb_zero_padding, "pb_zero_padding" );

    if( const auto * data = image->hamiltonian->data<Engine::Spin::Interaction::DDI>(); data != nullptr )
    {
        *ddi_method          = (int)data->method;
        n_periodic_images[0] = (int)data->n_periodic_images[0];
        n_periodic_images[1] = (int)data->n_periodic_images[1];
        n_periodic_images[2] = (int)data->n_periodic_images[2];
        *cutoff_radius       = data->cutoff_radius;
        *pb_zero_padding     = data->pb_zero_padding;
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void saveMatrix( const std::string & fname, const SpMatrixX & matrix )
{
    std::cout << "Saving matrix to file: " << fname << "\n";
    std::ofstream file( fname );
    if( file && file.is_open() )
    {
        file << matrix;
    }
    else
    {
        std::cerr << "Could not save matrix!";
    }
}

void saveTriplets( const std::string & fname, const SpMatrixX & matrix )
{

    std::cout << "Saving triplets to file: " << fname << "\n";
    std::ofstream file( fname );
    if( file && file.is_open() )
    {
        for( int k = 0; k < matrix.outerSize(); ++k )
        {
            for( SpMatrixX::InnerIterator it( matrix, k ); it; ++it )
            {
                file << it.row() << "\t"; // row index
                file << it.col() << "\t"; // col index (here it is equal to k)
                file << it.value() << "\n";
            }
        }
    }
    else
    {
        std::cerr << "Could not save matrix!";
    }
}

void Hamiltonian_Write_Hessian(
    State * state, const char * filename, bool triplet_format, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );
    throw_if_nullptr( filename, "filename" );

    // Compute hessian
    auto nos = image->hamiltonian->get_geometry().nos;
    SpMatrixX hessian( 3 * nos, 3 * nos );
    image->hamiltonian->Sparse_Hessian( *image->state, hessian );

    if( triplet_format )
        saveTriplets( std::string( filename ), hessian );
    else
        saveMatrix( std::string( filename ), hessian );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}
