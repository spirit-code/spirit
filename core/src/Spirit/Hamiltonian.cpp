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
            auto & ham =  image->hamiltonian;

            // Normals
            Vector3 new_normal{ normal[0], normal[1], normal[2] };
            new_normal.normalize();

            // Into the Hamiltonian
            ham->external_field_magnitude = magnitude * Constants::mu_B;
            ham->external_field_normal    = new_normal;

            // Update Energies
            ham->Update_Energy_Contributions();

            Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
                 fmt::format(
                     "Set external field to {}, direction ({}, {}, {})", magnitude, normal[0], normal[1], normal[2] ),
                 idx_image, idx_chain );
        }
        else
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
                 fmt::format("External field cannot be set on {}", image->hamiltonian->Name() ), idx_image, idx_chain );
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
                 fmt::format("Anisotropy cannot be set on {}", image->hamiltonian->Name() ), idx_image, idx_chain );
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
            auto & ham       = image->hamiltonian;
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
            //
            // Into the Hamiltonian
            ham->cubic_anisotropy_indices    = new_indices;
            ham->cubic_anisotropy_magnitudes = new_magnitudes;

            // Update Energies
            ham->Update_Energy_Contributions();

            Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
                 fmt::format( "Set cubic anisotropy to {}", magnitude ), idx_image, idx_chain );
        }
        else
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
                 fmt::format( "Cubic anisotropy cannot be set on {}", image->hamiltonian->Name() ), idx_image, idx_chain );
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
            auto & ham                     = image->hamiltonian;
            ham->exchange_shell_magnitudes = scalarfield( jij, jij + n_shells );
            ham->exchange_pairs_in         = pairfield( 0 );
            ham->exchange_magnitudes_in    = scalarfield( 0 );
            ham->updateInteractions();

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
            auto & ham                = image->hamiltonian;
            ham->dmi_shell_magnitudes = scalarfield( dij, dij + n_shells );
            ham->dmi_shell_chirality  = chirality;
            ham->dmi_pairs_in         = pairfield( 0 );
            ham->dmi_magnitudes_in    = scalarfield( 0 );
            ham->dmi_normals_in       = vectorfield( 0 );
            ham->updateInteractions();

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
            auto & ham = image->hamiltonian;

            ham->ddi_method               = Engine::DDI_Method( ddi_method );
            ham->ddi_n_periodic_images[0] = n_periodic_images[0];
            ham->ddi_n_periodic_images[1] = n_periodic_images[1];
            ham->ddi_n_periodic_images[2] = n_periodic_images[2];
            ham->ddi_cutoff_radius        = cutoff_radius;
            ham->ddi_pb_zero_padding      = pb_zero_padding;
            ham->updateInteractions();

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
        auto & ham =  image->hamiltonian;

        if( ham->external_field_magnitude > 0 )
        {
            // Magnitude
            *magnitude = ham->external_field_magnitude / Constants::mu_B;

            // Normal
            normal[0] = ham->external_field_normal[0];
            normal[1] = ham->external_field_normal[1];
            normal[2] = ham->external_field_normal[2];
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
        auto & ham =  image->hamiltonian;

        if( !ham->cubic_anisotropy_indices.empty() )
        {
            // Magnitude
            *magnitude = ham->cubic_anisotropy_magnitudes[0];
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
        auto & ham =  image->hamiltonian;

        *n_shells = ham->exchange_shell_magnitudes.size();

        // Note the array needs to be correctly allocated beforehand!
        for( std::size_t i = 0; i < ham->exchange_shell_magnitudes.size(); ++i )
        {
            jij[i] = ham->exchange_shell_magnitudes[i];
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
        auto & ham =  image->hamiltonian;
        return ham->exchange_pairs.size();
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
        auto & ham =  image->hamiltonian;

        for( std::size_t i = 0; i < ham->exchange_pairs.size() && i < ham->exchange_magnitudes.size(); ++i )
        {
            auto & pair        = ham->exchange_pairs[i];
            idx[i][0]          = pair.i;
            idx[i][1]          = pair.j;
            translations[i][0] = pair.translations[0];
            translations[i][1] = pair.translations[1];
            translations[i][2] = pair.translations[2];
            Jij[i]             = ham->exchange_magnitudes[i];
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
        auto & ham =  image->hamiltonian;

        *n_shells  = ham->dmi_shell_magnitudes.size();
        *chirality = ham->dmi_shell_chirality;

        for( int i = 0; i < *n_shells; ++i )
        {
            dij[i] = ham->dmi_shell_magnitudes[i];
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
        return image->hamiltonian->dmi_pairs.size();
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
        auto & ham =  image->hamiltonian;

        *ddi_method          = (int)ham->ddi_method;
        n_periodic_images[0] = (int)ham->ddi_n_periodic_images[0];
        n_periodic_images[1] = (int)ham->ddi_n_periodic_images[1];
        n_periodic_images[2] = (int)ham->ddi_n_periodic_images[2];
        *cutoff_radius       = ham->ddi_cutoff_radius;
        *pb_zero_padding     = ham->ddi_pb_zero_padding;
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
