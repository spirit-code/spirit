#include <Spirit/Hamiltonian.h>

#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <data/State.hpp>
#include <engine/Hamiltonian.hpp>
#include <engine/Neighbours.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Constants.hpp>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

#include <fmt/format.h>

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Utility;

/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Set Parameters ---------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

void Hamiltonian_Set_Boundary_Conditions(
    State * state, const bool * periodical, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( periodical, "periodical" );

    image->Lock();
    try
    {
        image->hamiltonian->boundary_conditions[0] = periodical[0];
        image->hamiltonian->boundary_conditions[1] = periodical[1];
        image->hamiltonian->boundary_conditions[2] = periodical[2];
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image, idx_chain );
    }
    image->Unlock();

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
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( normal, "normal" );

    // Lock mutex because simulations may be running
    image->Lock();
    try
    {
        // Set
        if( image->hamiltonian->Name() == "Heisenberg" )
        {
            // Normals
            Vector3 new_normal{ normal[0], normal[1], normal[2] };
            new_normal.normalize();

            // Into the Hamiltonian
            image->hamiltonian->getInteraction<Engine::Interaction::Zeeman>()->setParameters(
                magnitude * Constants::mu_B, new_normal );

            Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
                 fmt::format(
                     "Set external field to {}, direction ({}, {}, {})", magnitude, normal[0], normal[1], normal[2] ),
                 idx_image, idx_chain );
        }
        else
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
                 fmt::format( "External field cannot be set on {}", image->hamiltonian->Name() ), idx_image,
                 idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image, idx_chain );
    }

    // Unlock mutex
    image->Unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Hamiltonian_Set_Anisotropy(
    State * state, scalar magnitude, const scalar * normal, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( normal, "normal" );

    image->Lock();
    try
    {
        if( image->hamiltonian->Name() == "Heisenberg" )
        {
            int nos          = image->nos;
            int n_cell_atoms = image->geometry->n_cell_atoms;

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
            vectorfield new_normals( nos, new_normal );

            // Update the Hamiltonian
            image->hamiltonian->getInteraction<Engine::Interaction::Anisotropy>()->setParameters(
                new_indices, new_magnitudes, new_normals );

            Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
                 fmt::format(
                     "Set anisotropy to {}, direction ({}, {}, {})", magnitude, normal[0], normal[1], normal[2] ),
                 idx_image, idx_chain );
        }
        else
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
                 fmt::format( "Anisotropy cannot be set on {}", image->hamiltonian->Name() ), idx_image, idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image, idx_chain );
    }

    image->Unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Hamiltonian_Set_Cubic_Anisotropy( State * state, scalar magnitude, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    image->Lock();
    try
    {
        if( image->hamiltonian->Name() == "Heisenberg" )
        {
            int nos          = image->nos;
            int n_cell_atoms = image->geometry->n_cell_atoms;

            // Indices and Magnitudes
            intfield new_indices( n_cell_atoms );
            scalarfield new_magnitudes( n_cell_atoms );
            for( int i = 0; i < n_cell_atoms; ++i )
            {
                new_indices[i]    = i;
                new_magnitudes[i] = magnitude;
            }

            // Update the Hamiltonian
            image->hamiltonian->getInteraction<Engine::Interaction::Cubic_Anisotropy>()->setParameters(
                new_indices, new_magnitudes );

            Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
                 fmt::format( "Set cubic anisotropy to {}", magnitude ), idx_image, idx_chain );
        }
        else
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
                 fmt::format( "Cubic anisotropy cannot be set on {}", image->hamiltonian->Name() ), idx_image,
                 idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image, idx_chain );
    }

    image->Unlock();
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
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    image->Lock();
    try
    {
        if( auto * interaction = image->hamiltonian->getInteraction<Engine::Interaction::Biaxial_Anisotropy>();
            interaction != nullptr )
        {
            int n_cell_atoms = image->geometry->n_cell_atoms;

            const auto new_primary   = Vector3{ primary[0], primary[1], primary[2] }.normalized();
            const auto new_secondary = [&secondary, &new_primary]()
            {
                auto new_secondary = Vector3{ secondary[0], secondary[1], secondary[2] };
                new_secondary -= new_primary.dot( new_secondary ) * new_primary;
                new_secondary.normalize();
                return new_secondary;
            }();

            const Vector3 new_ternary = new_primary.cross( new_secondary ).normalized();

            field<PolynomialTerm> new_terms{};
            for( auto i = 0; i < n_terms; ++i )
            {
                if( magnitude[i] == 0 )
                    continue;

                new_terms.emplace_back(
                    PolynomialTerm{ magnitude[i], exponents[i][0], exponents[i][1], exponents[i][2] } );
            };

            AnisotropyPolynomial new_polynomial = { new_primary, new_secondary, new_ternary, new_terms };

            // Indices and Polynomials
            field<AnisotropyPolynomial> new_polynomials( n_cell_atoms, new_polynomial );
            intfield new_indices( n_cell_atoms );
            for( int i = 0; i < n_cell_atoms; ++i )
            {
                new_indices[i] = i;
            }

            // Update the Hamiltonian
            interaction->setParameters( new_indices, new_polynomials );

            Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
                 fmt::format( "Set {} terms for biaxial anisotropy", new_terms.size() ), idx_image, idx_chain );
        }
        else
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
                 fmt::format( "Biaxial anisotropy cannot be set on {}", image->hamiltonian->Name() ), idx_image,
                 idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image, idx_chain );
    }

    image->Unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Hamiltonian_Set_Exchange( State * state, int n_shells, const scalar * jij, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( jij, "jij" );

    image->Lock();
    try
    {
        if( image->hamiltonian->Name() == "Heisenberg" )
        {
            // Update the Hamiltonian
            image->hamiltonian->getInteraction<Engine::Interaction::Exchange>()->setParameters(
                scalarfield( jij, jij + n_shells ) );
            std::string message = fmt::format( "Set exchange to {} shells", n_shells );
            if( n_shells > 0 )
                message += fmt::format( " Jij[0] = {}", jij[0] );
            Log( Utility::Log_Level::Info, Utility::Log_Sender::API, message, idx_image, idx_chain );
        }
        else
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
                 fmt::format( "Exchange cannot be set on {}", image->hamiltonian->Name() ), idx_image, idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image, idx_chain );
    }

    image->Unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Hamiltonian_Set_DMI(
    State * state, int n_shells, const scalar * dij, int chirality, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( dij, "dij" );

    if( chirality != SPIRIT_CHIRALITY_BLOCH && chirality != SPIRIT_CHIRALITY_NEEL
        && chirality != SPIRIT_CHIRALITY_BLOCH_INVERSE && chirality != SPIRIT_CHIRALITY_NEEL_INVERSE )
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::API,
             fmt::format( "Hamiltonian_Set_DMI: Invalid DM chirality {}", chirality ), idx_image, idx_chain );
        return;
    }

    image->Lock();
    try
    {
        if( image->hamiltonian->Name() == "Heisenberg" )
        {
            // Update the Hamiltonian
            image->hamiltonian->getInteraction<Engine::Interaction::DMI>()->setParameters(
                scalarfield( dij, dij + n_shells ), chirality );

            std::string message = fmt::format( "Set dmi to {} shells", n_shells );
            if( n_shells > 0 )
                message += fmt::format( " Dij[0] = {}", dij[0] );
            Log( Utility::Log_Level::Info, Utility::Log_Sender::API, message, idx_image, idx_chain );
        }
        else
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
                 fmt::format( "DMI cannot be set on {}", image->hamiltonian->Name() ), idx_image, idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image, idx_chain );
    }

    image->Unlock();
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
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( n_periodic_images, "n_periodic_images" );

    image->Lock();
    try
    {
        if( image->hamiltonian->Name() == "Heisenberg" )
        {
            auto new_n_periodic_images = intfield( 3 );
            new_n_periodic_images[0]   = n_periodic_images[0];
            new_n_periodic_images[1]   = n_periodic_images[1];
            new_n_periodic_images[2]   = n_periodic_images[2];

            image->hamiltonian->getInteraction<Engine::Interaction::DDI>()->setParameters(
                Engine::DDI_Method( ddi_method ), new_n_periodic_images, pb_zero_padding, cutoff_radius );

            Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
                 fmt::format(
                     "Set ddi to method {}, periodic images {} {} {}, cutoff radius {} and pb_zero_padding {}",
                     ddi_method, n_periodic_images[0], n_periodic_images[1], n_periodic_images[2], cutoff_radius,
                     pb_zero_padding ),
                 idx_image, idx_chain );
        }
        else
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
                 fmt::format( "DDI cannot be set on {}", image->hamiltonian->Name() ), idx_image, idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image, idx_chain );
    }

    image->Unlock();
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
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    return strdup( std::string( image->hamiltonian->Name() ).c_str() );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return nullptr;
}

void Hamiltonian_Get_Boundary_Conditions( State * state, bool * periodical, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( periodical, "periodical" );

    periodical[0] = image->hamiltonian->boundary_conditions[0];
    periodical[1] = image->hamiltonian->boundary_conditions[1];
    periodical[2] = image->hamiltonian->boundary_conditions[2];
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Hamiltonian_Get_Field( State * state, scalar * magnitude, scalar * normal, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( magnitude, "magnitude" );
    throw_if_nullptr( normal, "normal" );

    if( image->hamiltonian->Name() == "Heisenberg" )
    {
        scalar field_magnitude = 0;
        Vector3 field_normal   = Vector3::Zero();
        image->hamiltonian->getInteraction<Engine::Interaction::Zeeman>()->getParameters(
            field_magnitude, field_normal );

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
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( magnitude, "magnitude" );
    throw_if_nullptr( normal, "normal" );

    if( image->hamiltonian->Name() == "Heisenberg" )
    {
        intfield anisotropy_indices;
        scalarfield anisotropy_magnitudes;
        vectorfield anisotropy_normals;
        image->hamiltonian->getInteraction<Engine::Interaction::Anisotropy>()->getParameters(
            anisotropy_indices, anisotropy_magnitudes, anisotropy_normals );
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
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( magnitude, "magnitude" );

    if( image->hamiltonian->Name() == "Heisenberg" )
    {
        intfield cubic_anisotropy_indices;
        scalarfield cubic_anisotropy_magnitudes;
        image->hamiltonian->getInteraction<Engine::Interaction::Cubic_Anisotropy>()->getParameters(
            cubic_anisotropy_indices, cubic_anisotropy_magnitudes );

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
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    try
    {
        if( auto * interaction = image->hamiltonian->getInteraction<Engine::Interaction::Biaxial_Anisotropy>();
            interaction != nullptr )
        {
            return interaction->getN_Atoms();
        }
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image, idx_chain );
    }

    return 0;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

void Hamiltonian_Get_Biaxial_Anisotropy_N_Terms(
    State * state, const int n_atoms, int n_terms[], int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    try
    {
        if( n_atoms == 0 )
            return;

        if( auto * interaction = image->hamiltonian->getInteraction<Engine::Interaction::Biaxial_Anisotropy>();
            interaction != nullptr )
        {
            n_terms[0] = 0;

            for( int i = 0; i < n_atoms; ++i )
                n_terms[i + 1] = n_terms[i] + interaction->getN_Terms( i );
        }
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image, idx_chain );
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Hamiltonian_Get_Biaxial_Anisotropy(
    State * state, const int * n_terms, int * indices, scalar primary[][3], scalar secondary[][3], scalar * magnitude,
    int exponents[][3], const int n_indices, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    try
    {
        if( auto * interaction = image->hamiltonian->getInteraction<Engine::Interaction::Biaxial_Anisotropy>();
            interaction != nullptr )
        {
            intfield anisotropy_indices;
            field<AnisotropyPolynomial> anisotropy_polynomials;

            interaction->getParameters( anisotropy_indices, anisotropy_polynomials );

            const auto n_read = std::min( n_indices, static_cast<decltype( n_indices )>( anisotropy_indices.size() ) );
            for( int j = 0; j < n_read; ++j )
            {
                indices[j] = anisotropy_indices[j];

                const auto & k1 = anisotropy_polynomials[j].k1;
                const auto & k2 = anisotropy_polynomials[j].k2;

#pragma unroll
                for( int i = 0; i < 3; ++i )
                {
                    primary[j][i]   = k1[i];
                    secondary[j][i] = k2[i];
                }

                // Magnitude
                const auto & terms = anisotropy_polynomials[j].terms;
                for( int i = 0, idx = n_terms[j]; idx < n_terms[j + 1]; ++i, ++idx )
                {
                    magnitude[idx]    = terms[i].coefficient;
                    exponents[idx][0] = terms[i].n1;
                    exponents[idx][1] = terms[i].n2;
                    exponents[idx][2] = terms[i].n3;
                }
            }
        }
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image, idx_chain );
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
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( n_shells, "n_shells" );
    throw_if_nullptr( jij, "jij" );

    if( image->hamiltonian->Name() == "Heisenberg" )
    {
        scalarfield exchange_shell_magnitudes;
        image->hamiltonian->getInteraction<Engine::Interaction::Exchange>()->getInitParameters(
            exchange_shell_magnitudes );

        *n_shells = exchange_shell_magnitudes.size();

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
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    if( image->hamiltonian->Name() == "Heisenberg" )
    {
        return image->hamiltonian->getInteraction<Engine::Interaction::Exchange>()->getN_Pairs();
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
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( Jij, "Jij" );

    if( image->hamiltonian->Name() == "Heisenberg" )
    {
        pairfield exchange_pairs;
        scalarfield exchange_magnitudes;
        image->hamiltonian->getInteraction<Engine::Interaction::Exchange>()->getParameters(
            exchange_pairs, exchange_magnitudes );

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
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( n_shells, "n_shells" );
    throw_if_nullptr( dij, "dij" );
    throw_if_nullptr( chirality, "chirality" );

    if( image->hamiltonian->Name() == "Heisenberg" )
    {
        scalarfield dmi_shell_magnitudes;
        int dmi_shell_chirality = 0;

        image->hamiltonian->getInteraction<Engine::Interaction::DMI>()->getInitParameters(
            dmi_shell_magnitudes, dmi_shell_chirality );

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
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    if( image->hamiltonian->Name() == "Heisenberg" )
    {
        return image->hamiltonian->getInteraction<Engine::Interaction::DMI>()->getN_Pairs();
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
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( ddi_method, "ddi_method" );
    throw_if_nullptr( cutoff_radius, "cutoff_radius" );
    throw_if_nullptr( pb_zero_padding, "pb_zero_padding" );

    if( image->hamiltonian->Name() == "Heisenberg" )
    {
        Engine::DDI_Method method{};
        intfield ddi_n_periodic_images;
        scalar ddi_cutoff_radius = 0;
        bool ddi_pb_zero_padding = false;

        image->hamiltonian->getInteraction<Engine::Interaction::DDI>()->getParameters(
            method, ddi_n_periodic_images, ddi_pb_zero_padding, ddi_cutoff_radius );

        *ddi_method          = (int)method;
        n_periodic_images[0] = (int)ddi_n_periodic_images[0];
        n_periodic_images[1] = (int)ddi_n_periodic_images[1];
        n_periodic_images[2] = (int)ddi_n_periodic_images[2];
        *cutoff_radius       = ddi_cutoff_radius;
        *pb_zero_padding     = ddi_pb_zero_padding;
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
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( filename, "filename" );

    // Compute hessian
    auto nos = image->geometry->nos;
    SpMatrixX hessian( 3 * nos, 3 * nos );
    image->hamiltonian->Sparse_Hessian( *image->spins, hessian );

    if( triplet_format )
        saveTriplets( std::string( filename ), hessian );
    else
        saveMatrix( std::string( filename ), hessian );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}
