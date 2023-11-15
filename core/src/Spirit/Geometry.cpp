#include <Spirit/Geometry.h>
#include <Spirit/Simulation.h>

#include <data/State.hpp>
#include <engine/Hamiltonian.hpp>
#include <engine/Indexing.hpp>
#include <utility/Exception.hpp>
#include <utility/Formatters_Eigen.hpp>
#include <utility/Logging.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>

void Helper_System_Set_Geometry( Data::Spin_System & system, const Data::Geometry & new_geometry )
{
    auto old_geometry = *system.geometry;

    int nos    = new_geometry.nos;
    system.nos = nos;

    // Move the vector-fields to the new geometry
    *system.spins = Engine::Indexing::change_dimensions(
        *system.spins, old_geometry.n_cell_atoms, old_geometry.n_cells, new_geometry.n_cell_atoms, new_geometry.n_cells,
        { 0, 0, 1 } );
    system.effective_field = Engine::Indexing::change_dimensions(
        system.effective_field, old_geometry.n_cell_atoms, old_geometry.n_cells, new_geometry.n_cell_atoms,
        new_geometry.n_cells, { 0, 0, 0 } );

    // Update the system geometry
    *system.geometry = new_geometry;

    // Update the Heisenberg Hamiltonian
    if( system.hamiltonian->Name() == "Heisenberg" )
        std::static_pointer_cast<Engine::Hamiltonian_Heisenberg>( system.hamiltonian )->Update_Interactions();
}

void Helper_State_Set_Geometry(
    State & state, const Data::Geometry & old_geometry, const Data::Geometry & new_geometry )
{
    // This requires simulations to be stopped, as Methods' temporary arrays may have the wrong size afterwards
    Simulation_Stop_All( &state );

    // Lock to avoid memory errors
    state.chain->Lock();
    try
    {
        // Modify all systems in the chain
        for( auto & system : state.chain->images )
        {
            Helper_System_Set_Geometry( *system, new_geometry );
        }
    }
    catch( ... )
    {
        spirit_handle_exception_api( -1, -1 );
    }
    // Unlock again
    state.chain->Unlock();

    // Retrieve total number of spins
    int nos = state.active_image->nos;

    // Update convenience integerin State
    state.nos = nos;

    // Deal with clipboard image of State
    if( state.clipboard_image )
    {
        auto & system = *state.clipboard_image;
        // Lock to avoid memory errors
        system.Lock();
        try
        {
            // Modify
            Helper_System_Set_Geometry( system, new_geometry );
        }
        catch( ... )
        {
            spirit_handle_exception_api( -1, -1 );
        }
        // Unlock
        system.Unlock();
    }

    // Deal with clipboard configuration of State
    if( state.clipboard_spins )
        *state.clipboard_spins = Engine::Indexing::change_dimensions(
            *state.clipboard_spins, old_geometry.n_cell_atoms, old_geometry.n_cells, new_geometry.n_cell_atoms,
            new_geometry.n_cells, { 0, 0, 1 } );

    // TODO: Deal with Methods
    // for (auto& chain_method_image : state.method_image)
    // {
    //     for (auto& method_image : chain_method_image)
    //     {
    //         method_image->Update_Geometry(new_geometry.n_cell_atoms, new_geometry.n_cells, new_geometry.n_cells);
    //     }
    // }
    // for (auto& method_chain : state.method_chain)
    // {
    //     method_chain->Update_Geometry(new_geometry.n_cell_atoms, new_geometry.n_cells, new_geometry.n_cells);
    // }
}

void Geometry_Set_Bravais_Lattice_Type( State * state, Bravais_Lattice_Type lattice_type ) noexcept
try
{
    check_state( state );

    std::string lattice_name;
    std::vector<Vector3> bravais_vectors;
    if( lattice_type == Bravais_Lattice_Irregular )
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::API,
             "Geometry_Set_Bravais_Lattice_Type: cannot set lattice type to Irregular", -1, -1 );
        return;
    }
    else if( lattice_type == Bravais_Lattice_Rectilinear )
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::API,
             "Geometry_Set_Bravais_Lattice_Type: cannot set lattice type to Rectilinear", -1, -1 );
        return;
    }
    else if( lattice_type == Bravais_Lattice_HCP )
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::API,
             "Geometry_Set_Bravais_Lattice_Type: cannot set lattice type to HCP", -1, -1 );
        return;
    }
    else if( lattice_type == Bravais_Lattice_SC )
    {
        lattice_name    = "Simple Cubic";
        bravais_vectors = Data::Geometry::BravaisVectorsSC();
    }
    else if( lattice_type == Bravais_Lattice_Hex2D )
    {
        lattice_name    = "2D Hexagonal";
        bravais_vectors = Data::Geometry::BravaisVectorsHex2D60();
    }
    else if( lattice_type == Bravais_Lattice_Hex2D_60 )
    {
        lattice_name    = "2D Hexagonal (60deg)";
        bravais_vectors = Data::Geometry::BravaisVectorsHex2D60();
    }
    else if( lattice_type == Bravais_Lattice_Hex2D_120 )
    {
        lattice_name    = "2D Hexagonal (120deg)";
        bravais_vectors = Data::Geometry::BravaisVectorsHex2D120();
    }
    else if( lattice_type == Bravais_Lattice_BCC )
    {
        lattice_name    = "Body-Centered Cubic";
        bravais_vectors = Data::Geometry::BravaisVectorsBCC();
    }
    else if( lattice_type == Bravais_Lattice_FCC )
    {
        lattice_name    = "Face-Centered Cubic";
        bravais_vectors = Data::Geometry::BravaisVectorsFCC();
    }
    else
    {
        spirit_throw(
            Utility::Exception_Classifier::System_not_Initialized, Utility::Log_Level::Error,
            fmt::format( "Unknown lattice type index '{}'", int( lattice_type ) ) );
    }

    // The new geometry
    const auto & old_geometry = *state->active_image->geometry;
    auto new_geometry         = Data::Geometry(
        bravais_vectors, old_geometry.n_cells, old_geometry.cell_atoms, old_geometry.cell_composition,
        old_geometry.lattice_constant, old_geometry.pinning, old_geometry.defects );

    // Update the State
    Helper_State_Set_Geometry( *state, old_geometry, new_geometry );

    Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
         fmt::format( "Set Bravais lattice type to {} for all Systems", lattice_name ), -1, -1 );
}
catch( ... )
{
    spirit_handle_exception_api( 0, 0 );
}

void Geometry_Set_N_Cells( State * state, int n_cells_i[3] ) noexcept
try
{
    check_state( state );
    throw_if_nullptr( n_cells_i, "n_cells_i" );

    // The new number of basis cells
    auto n_cells = intfield{ n_cells_i[0], n_cells_i[1], n_cells_i[2] };

    // The new geometry
    auto & old_geometry = *state->active_image->geometry;
    auto new_geometry   = Data::Geometry(
        old_geometry.bravais_vectors, n_cells, old_geometry.cell_atoms, old_geometry.cell_composition,
        old_geometry.lattice_constant, old_geometry.pinning, old_geometry.defects );

    // Update the State
    Helper_State_Set_Geometry( *state, old_geometry, new_geometry );

    Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
         fmt::format( "Set number of cells for all Systems: ({}, {}, {})", n_cells[0], n_cells[1], n_cells[2] ), -1,
         -1 );
}
catch( ... )
{
    spirit_handle_exception_api( 0, 0 );
}

void Geometry_Set_Cell_Atoms( State * state, int n_atoms, scalar ** atoms ) noexcept
try
{
    check_state( state );

    if( n_atoms < 1 )
    {
        spirit_throw(
            Utility::Exception_Classifier::System_not_Initialized, Utility::Log_Level::Error,
            fmt::format( "Cannot set number of atoms to less than one (you passed {})", n_atoms ) );
    }

    throw_if_nullptr( atoms, "atoms" );

    auto & old_geometry = *state->active_image->geometry;

    // The new arrays
    std::vector<Vector3> cell_atoms( 0 );
    std::vector<int> iatom( 0 );
    std::vector<int> atom_type( 0 );
    std::vector<scalar> mu_s( 0 );
    std::vector<scalar> concentration( 0 );

    // Basis cell atoms
    for( int i = 0; i < n_atoms; ++i )
    {
        if( atoms[i] == nullptr )
        {
            spirit_throw(
                Utility::Exception_Classifier::System_not_Initialized, Utility::Log_Level::Error,
                fmt::format( "Got passed a null pointer for atom {} of {}", i, n_atoms ) );
        }
        cell_atoms.emplace_back( atoms[i][0], atoms[i][1], atoms[i][2] );
    }

    // In the regular case, we re-generate information to make sure every atom
    // has its set of information
    if( !old_geometry.cell_composition.disordered )
    {
        for( int i = 0; i < n_atoms; ++i )
        {
            iatom.push_back( i );
            if( i < old_geometry.n_cell_atoms )
            {
                atom_type.push_back( old_geometry.cell_composition.atom_type[i] );
                mu_s.push_back( old_geometry.cell_composition.mu_s[i] );
            }
            else
            {
                atom_type.push_back( old_geometry.cell_composition.atom_type[0] );
                mu_s.push_back( old_geometry.cell_composition.mu_s[0] );
            }
        }
    }
    // In the disordered case, we take all previous information, which is
    // still valid. This may lead to new atoms not having any mu_s information.
    else
    {
        for( std::size_t i = 0; i < old_geometry.cell_composition.iatom.size(); ++i )
        {
            // If the atom index is within range, we keep the information
            if( old_geometry.cell_composition.iatom[i] < n_atoms )
            {
                atom_type.push_back( old_geometry.cell_composition.atom_type[i] );
                mu_s.push_back( old_geometry.cell_composition.mu_s[i] );
                concentration.push_back( old_geometry.cell_composition.concentration[i] );
            }
        }
    }

    Data::Basis_Cell_Composition new_composition{ old_geometry.cell_composition.disordered, iatom, atom_type, mu_s,
                                                  concentration };

    // The new geometry
    auto new_geometry = Data::Geometry(
        old_geometry.bravais_vectors, old_geometry.n_cells, cell_atoms, new_composition, old_geometry.lattice_constant,
        old_geometry.pinning, old_geometry.defects );

    // Update the State
    Helper_State_Set_Geometry( *state, old_geometry, new_geometry );

    Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
         fmt::format( "Set {} cell atoms for all Systems. cell_atom[0]={}", n_atoms, cell_atoms[0] ), -1, -1 );
    if( new_geometry.n_cell_atoms > old_geometry.n_cell_atoms )
        Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
             fmt::format(
                 "The basis cell size increased. Set {} additional values of mu_s to {}",
                 new_geometry.n_cell_atoms - old_geometry.n_cell_atoms, mu_s[0] ),
             -1, -1 );
}
catch( ... )
{
    spirit_handle_exception_api( 0, 0 );
}

void Geometry_Set_mu_s( State * state, scalar mu_s, int idx_image, int idx_chain ) noexcept
try
{
    check_state( state );

    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    try
    {
        auto & old_geometry = *state->active_image->geometry;

        auto new_composition = old_geometry.cell_composition;
        for( auto & m : new_composition.mu_s )
            m = mu_s;

        // The new geometry
        auto new_geometry = Data::Geometry(
            old_geometry.bravais_vectors, old_geometry.n_cells, old_geometry.cell_atoms, new_composition,
            old_geometry.lattice_constant, old_geometry.pinning, old_geometry.defects );

        // Update the State
        Helper_State_Set_Geometry( *state, old_geometry, new_geometry );

        Log( Utility::Log_Level::Info, Utility::Log_Sender::API, fmt::format( "Set mu_s to {}", mu_s ), idx_image,
             idx_chain );
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

void Geometry_Set_Cell_Atom_Types( State * state, int n_atoms, int * atom_types ) noexcept
try
{
    check_state( state );
    throw_if_nullptr( atom_types, "atom_types" );

    if( n_atoms < 1 )
    {
        spirit_throw(
            Utility::Exception_Classifier::System_not_Initialized, Utility::Log_Level::Error,
            fmt::format( "Cannot set atom types for less than one site (you passed {})", n_atoms ) );
    }

    auto & old_geometry = *state->active_image->geometry;

    auto new_composition = old_geometry.cell_composition;
    for( std::size_t i = 0; i < static_cast<std::size_t>( n_atoms ); ++i )
    {
        if( i < new_composition.iatom.size() )
            new_composition.atom_type[i] = atom_types[i];
    }

    // The new geometry
    auto new_geometry = Data::Geometry(
        old_geometry.bravais_vectors, old_geometry.n_cells, old_geometry.cell_atoms, new_composition,
        old_geometry.lattice_constant, old_geometry.pinning, old_geometry.defects );

    // Update the State
    Helper_State_Set_Geometry( *state, old_geometry, new_geometry );

    Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
         fmt::format( "Set {} types of basis cell atoms for all Systems. type[0]={}", n_atoms, atom_types[0] ), -1,
         -1 );
}
catch( ... )
{
    spirit_handle_exception_api( 0, 0 );
}

void Geometry_Set_Bravais_Vectors( State * state, scalar ta[3], scalar tb[3], scalar tc[3] ) noexcept
try
{
    check_state( state );
    throw_if_nullptr( ta, "ta" );
    throw_if_nullptr( tb, "tb" );
    throw_if_nullptr( tc, "tc" );

    // The new Bravais vectors
    std::vector<Vector3> bravais_vectors{
        Vector3{ ta[0], ta[1], ta[2] },
        Vector3{ tb[0], tb[1], tb[2] },
        Vector3{ tc[0], tc[1], tc[2] },
    };

    // The new geometry
    auto & old_geometry = *state->active_image->geometry;
    auto new_geometry   = Data::Geometry(
        bravais_vectors, old_geometry.n_cells, old_geometry.cell_atoms, old_geometry.cell_composition,
        old_geometry.lattice_constant, old_geometry.pinning, old_geometry.defects );

    // Update the State
    Helper_State_Set_Geometry( *state, old_geometry, new_geometry );

    Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
         fmt::format(
             "Set Bravais vectors for all Systems: ({}), ({}), ({})", bravais_vectors[0], bravais_vectors[1],
             bravais_vectors[2] ),
         -1, -1 );
}
catch( ... )
{
    spirit_handle_exception_api( 0, 0 );
}

void Geometry_Set_Lattice_Constant( State * state, scalar lattice_constant ) noexcept
try
{
    check_state( state );

    // The new geometry
    auto & old_geometry = *state->active_image->geometry;
    auto new_geometry   = Data::Geometry(
        old_geometry.bravais_vectors, old_geometry.n_cells, old_geometry.cell_atoms, old_geometry.cell_composition,
        lattice_constant, old_geometry.pinning, old_geometry.defects );

    // Update the State
    Helper_State_Set_Geometry( *state, old_geometry, new_geometry );

    Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
         fmt::format( "Set lattice constant for all Systems to {}", lattice_constant ), -1, -1 );
}
catch( ... )
{
    spirit_handle_exception_api( 0, 0 );
}

int Geometry_Get_NOS( State * state ) noexcept
{
    check_state( state );
    return state->nos;
}

scalar * Geometry_Get_Positions( State * state, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    // TODO: we should also check if idx_image < 0 and log the promotion to idx_active_image

    return (scalar *)image->geometry->positions[0].data();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return nullptr;
}

int * Geometry_Get_Atom_Types( State * state, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    // TODO: we should also check if idx_image < 0 and log the promotion to idx_active_image

    return (int *)image->geometry->atom_types.data();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return nullptr;
}

void Geometry_Get_Bounds( State * state, scalar min[3], scalar max[3], int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( min, "min" );
    throw_if_nullptr( max, "max" );

    // TODO: we should also check if idx_image < 0 and log the promotion to idx_active_image

    auto g = image->geometry;
    for( std::uint8_t dim = 0; dim < 3; ++dim )
    {
        min[dim] = g->bounds_min[dim];
        max[dim] = g->bounds_max[dim];
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

// Get Center as array (x,y,z)
void Geometry_Get_Center( State * state, scalar center[3], int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( center, "center" );

    // TODO: we should also check if idx_image < 0 and log the promotion to idx_active_image

    auto g = image->geometry;
    for( std::uint8_t dim = 0; dim < 3; ++dim )
    {
        center[dim] = g->center[dim];
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Geometry_Get_Cell_Bounds( State * state, scalar min[3], scalar max[3], int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( min, "min" );
    throw_if_nullptr( max, "max" );

    // TODO: we should also check if idx_image < 0 and log the promotion to idx_active_image

    auto g = image->geometry;
    for( std::uint8_t dim = 0; dim < 3; ++dim )
    {
        min[dim] = g->cell_bounds_min[dim];
        max[dim] = g->cell_bounds_max[dim];
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

// Get bravais lattice type
Bravais_Lattice_Type Geometry_Get_Bravais_Lattice_Type( State * state, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    return Bravais_Lattice_Type( image->geometry->classifier );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return Bravais_Lattice_Irregular;
}

// Get bravais vectors ta, tb, tc
void Geometry_Get_Bravais_Vectors(
    State * state, scalar a[3], scalar b[3], scalar c[3], int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( a, "a" );
    throw_if_nullptr( b, "b" );
    throw_if_nullptr( c, "c" );

    // TODO: we should also check if idx_image < 0 and log the promotion to idx_active_image

    auto g = image->geometry;
    for( std::uint8_t dim = 0; dim < 3; ++dim )
    {
        a[dim] = g->bravais_vectors[dim][0];
        b[dim] = g->bravais_vectors[dim][1];
        c[dim] = g->bravais_vectors[dim][2];
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

// Get number of atoms in a basis cell
int Geometry_Get_N_Cell_Atoms( State * state, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    // TODO: we should also check if idx_image < 0 and log the promotion to idx_active_image

    return image->geometry->n_cell_atoms;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

// Get basis cell atoms
int Geometry_Get_Cell_Atoms( State * state, scalar ** atoms, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    auto g = image->geometry;
    if( atoms != nullptr )
        *atoms = reinterpret_cast<scalar *>( g->cell_atoms[0].data() );

    return g->cell_atoms.size();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

void Geometry_Get_mu_s( State * state, scalar * mu_s, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( mu_s, "mu_s" );

    for( int i = 0; i < image->geometry->n_cell_atoms; ++i )
        mu_s[i] = image->geometry->mu_s[i];
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

// Get number of basis cells in the three translation directions
void Geometry_Get_N_Cells( State * state, int n_cells[3], int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( n_cells, "n_cells" );

    // TODO: we should also check if idx_image < 0 and log the promotion to idx_active_image

    auto g     = image->geometry;
    n_cells[0] = g->n_cells[0];
    n_cells[1] = g->n_cells[1];
    n_cells[2] = g->n_cells[2];
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

int Geometry_Get_Dimensionality( State * state, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    // TODO: we should also check if idx_image < 0 and log the promotion to idx_active_image

    auto g = image->geometry;
    return g->dimensionality;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

int Geometry_Get_Triangulation(
    State * state, const int ** indices_ptr, int n_cell_step, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    // TODO: we should also check if idx_image < 0 and log the promotion to idx_active_image

    auto g                 = image->geometry;
    const auto & triangles = g->triangulation( n_cell_step );
    if( indices_ptr != nullptr )
    {
        *indices_ptr = reinterpret_cast<const int *>( triangles.data() );
    }
    return triangles.size();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

int Geometry_Get_Triangulation_Ranged(
    State * state, const int ** indices_ptr, int n_cell_step, int ranges[6], int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( ranges, "ranges" );

    // TODO: we should also check if idx_image < 0 and log the promotion to idx_active_image
    std::array<int, 6> range = { ranges[0], ranges[1], ranges[2], ranges[3], ranges[4], ranges[5] };
    auto g                   = image->geometry;
    const auto & triangles   = g->triangulation( n_cell_step, range );
    if( indices_ptr != nullptr )
    {
        *indices_ptr = reinterpret_cast<const int *>( triangles.data() );
    }
    return triangles.size();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

int Geometry_Get_Tetrahedra(
    State * state, const int ** indices_ptr, int n_cell_step, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices( state, idx_image, idx_chain, image, chain );

    auto g                  = image->geometry;
    const auto & tetrahedra = g->tetrahedra( n_cell_step );
    if( indices_ptr != nullptr )
    {
        *indices_ptr = reinterpret_cast<const int *>( tetrahedra.data() );
    }
    return tetrahedra.size();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

int Geometry_Get_Tetrahedra_Ranged(
    State * state, const int ** indices_ptr, int n_cell_step, int ranges[6], int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices( state, idx_image, idx_chain, image, chain );
    throw_if_nullptr( ranges, "ranges" );

    auto g                   = image->geometry;
    std::array<int, 6> range = { ranges[0], ranges[1], ranges[2], ranges[3], ranges[4], ranges[5] };

    const auto & tetrahedra = g->tetrahedra( n_cell_step, range );
    if( indices_ptr != nullptr )
    {
        *indices_ptr = reinterpret_cast<const int *>( tetrahedra.data() );
    }
    return tetrahedra.size();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}
