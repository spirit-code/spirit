#include <Spirit/Geometry.h>
#include <data/State.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Hamiltonian_Heisenberg.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>


void Helper_System_Set_Geometry(std::shared_ptr<Data::Spin_System> system, const Data::Geometry & new_geometry)
{
    // *system->geometry = new_geometry;
    auto old_geometry = *system->geometry;

    // Spins
    int nos_old = system->nos;
    int nos = new_geometry.nos;
    system->nos = nos;
    
    // Move the vector-fields to the new geometry
    *system->spins = Engine::Vectormath::change_dimensions(*system->spins, old_geometry, new_geometry, {0,0,1});
    system->effective_field = Engine::Vectormath::change_dimensions(system->effective_field, old_geometry, new_geometry, {0,0,0});

    // Update the system geometry
    *system->geometry = new_geometry;

    // Parameters
    system->llg_parameters->pinning->mask_unpinned =
        Engine::Vectormath::change_dimensions(system->llg_parameters->pinning->mask_unpinned, old_geometry, new_geometry, 1);

    // Heisenberg Hamiltonian
    if (system->hamiltonian->Name() == "Heisenberg")
        std::static_pointer_cast<Engine::Hamiltonian_Heisenberg>(system->hamiltonian)->Update_Interactions();
}

void Helper_State_Set_Geometry(State * state, const Data::Geometry & old_geometry, const Data::Geometry & new_geometry)
{
    // Deal with all systems in all chains
    for (auto& chain : state->collection->chains)
    {
        // Lock to avoid memory errors
        chain->Lock();
        try
        {
            // Modify all systems in the chain
            for (auto& system : chain->images)
            {
                Helper_System_Set_Geometry(system, new_geometry);
            }
        }
        catch( ... )
        {
            spirit_handle_exception_api(-1, -1);
        }
        // Unlock again
        chain->Unlock();
    }

    // Retrieve total number of spins
    int nos = state->active_image->nos;

    // Update convenience integerin State
    state->nos = nos;

    // Deal with clipboard image of State
    auto& system = state->clipboard_image;
    if (system)
    {
        // Lock to avoid memory errors
        system->Lock();
        try
        {
            // Modify
            Helper_System_Set_Geometry(system, new_geometry);
        }
        catch( ... )
        {
            spirit_handle_exception_api(-1, -1);
        }
        // Unlock
        system->Unlock();
    }

    // Deal with clipboard configuration of State
    if (state->clipboard_spins)
        *state->clipboard_spins = Engine::Vectormath::change_dimensions(*state->clipboard_spins, old_geometry, new_geometry, {0,0,1});

    // TODO: Deal with Methods
    // for (auto& chain_method_image : state->method_image)
    // {
    //     for (auto& method_image : chain_method_image)
    //     {
    //         method_image->Update_Geometry(new_geometry.n_cell_atoms, new_geometry.n_cells, new_geometry.n_cells);
    //     }
    // }
    // for (auto& method_chain : state->method_chain)
    // {
    //     method_chain->Update_Geometry(new_geometry.n_cell_atoms, new_geometry.n_cells, new_geometry.n_cells);
    // }
}

void Geometry_Set_Bravais_Lattice(State *state, const char * c_bravais_lattice) noexcept
{
    try
    {
        std::string bravais_lattice = c_bravais_lattice;
        std::vector<Vector3> bravais_vectors;
        if (bravais_lattice == "sc")
            bravais_vectors = Data::Geometry::BravaisVectorsSC();
        else if (bravais_lattice == "fcc")
            bravais_vectors = Data::Geometry::BravaisVectorsFCC();
        else if (bravais_lattice == "bcc")
            bravais_vectors = Data::Geometry::BravaisVectorsBCC();
        else if (bravais_lattice == "hex2d60")
            bravais_vectors = Data::Geometry::BravaisVectorsHex2D60();
        else if (bravais_lattice == "hex2d120")
            bravais_vectors = Data::Geometry::BravaisVectorsHex2D120();
        else
        {
            Log(Utility::Log_Level::Error, Utility::Log_Sender::API, fmt::format("Invalid input to Geometry_Set_Bravais_Lattice: '{}'", bravais_lattice), -1, -1);
            return;
        }
        
        // The new geometry
        auto& old_geometry = *state->active_image->geometry;
        auto  new_geometry = Data::Geometry(bravais_vectors,
            old_geometry.n_cells, old_geometry.cell_atoms, old_geometry.cell_atom_types, old_geometry.lattice_constant);

        // Update the State
        Helper_State_Set_Geometry(state, old_geometry, new_geometry);

        Log(Utility::Log_Level::Warning, Utility::Log_Sender::API,
            fmt::format("Set Bravais lattice type to {} for all Systems", bravais_lattice), -1, -1);
    }
    catch( ... )
    {
        spirit_handle_exception_api(0, 0);
    }
}

void Geometry_Set_N_Cells(State * state, int n_cells_i[3]) noexcept
{
    try
    {
        // The new number of basis cells
        auto n_cells = intfield{n_cells_i[0], n_cells_i[1], n_cells_i[2]};

        // The new geometry
        auto& old_geometry = *state->active_image->geometry;
        auto  new_geometry = Data::Geometry(old_geometry.bravais_vectors,
            n_cells, old_geometry.cell_atoms, old_geometry.cell_atom_types, old_geometry.lattice_constant);

        // Update the State
        Helper_State_Set_Geometry(state, old_geometry, new_geometry);

        Log(Utility::Log_Level::Warning, Utility::Log_Sender::API, fmt::format("Set number of cells for all Systems: ({}, {}, {})", n_cells[0], n_cells[1], n_cells[2]), -1, -1);
    }
    catch( ... )
    {
        spirit_handle_exception_api(0, 0);
    }
}

void Geometry_Set_Cell_Atoms(State *state, int n_atoms, float ** atoms) noexcept
{
    try
    {
        // The new cell atoms
        std::vector<Vector3> cell_atoms(0);
        for (int i=0; i<n_atoms; ++i)
        {
            cell_atoms.push_back({atoms[i][0], atoms[i][1], atoms[i][2]});
        }

        // The new geometry
        auto& old_geometry = *state->active_image->geometry;
        auto  new_geometry = Data::Geometry(old_geometry.bravais_vectors,
            old_geometry.n_cells, cell_atoms, old_geometry.cell_atom_types, old_geometry.lattice_constant);

        // Update the State
        Helper_State_Set_Geometry(state, old_geometry, new_geometry);

        Log(Utility::Log_Level::Warning, Utility::Log_Sender::API, fmt::format("Set {} cell atoms for all Systems. cell_atom[0]={}", n_atoms, cell_atoms[0]), -1, -1);
    }
    catch( ... )
    {
        spirit_handle_exception_api(0, 0);
    }
}

void Geometry_Set_Cell_Atom_Types(State *state, int n_atoms, int * atom_types) noexcept
{
    try
    {
        // The new atom types
        intfield cell_atom_types(0);
        for (int i=0; i<n_atoms; ++i)
        {
            cell_atom_types.push_back(atom_types[i]);
        }

        // The new geometry
        auto& old_geometry = *state->active_image->geometry;
        auto  new_geometry = Data::Geometry(old_geometry.bravais_vectors,
            old_geometry.n_cells, old_geometry.cell_atoms, cell_atom_types, old_geometry.lattice_constant);

        // Update the State
        Helper_State_Set_Geometry(state, old_geometry, new_geometry);

        Log(Utility::Log_Level::Warning, Utility::Log_Sender::API, fmt::format("Set {} types of basis cell atoms for all Systems. type[0]={}", n_atoms, cell_atom_types[0]), -1, -1);
    }
    catch( ... )
    {
        spirit_handle_exception_api(0, 0);
    }
}

void Geometry_Set_Bravais_Vectors(State *state, float ta[3], float tb[3], float tc[3]) noexcept
{
    try
    {
        // The new Bravais vectors
        std::vector<Vector3> bravais_vectors{
            Vector3{ta[0], ta[1], ta[2]},
            Vector3{tb[0], tb[1], tb[2]},
            Vector3{tc[0], tc[1], tc[2]}};

        // The new geometry
        auto& old_geometry = *state->active_image->geometry;
        auto  new_geometry = Data::Geometry(bravais_vectors,
            old_geometry.n_cells, old_geometry.cell_atoms, old_geometry.cell_atom_types, old_geometry.lattice_constant);

        // Update the State
        Helper_State_Set_Geometry(state, old_geometry, new_geometry);

        Log(Utility::Log_Level::Warning, Utility::Log_Sender::API,
            fmt::format("Set Bravais vectors for all Systems: ({}), ({}), ({})", bravais_vectors[0], bravais_vectors[1], bravais_vectors[2]), -1, -1);
    }
    catch( ... )
    {
        spirit_handle_exception_api(0, 0);
    }
}

void Geometry_Set_Lattice_Constant(State *state, float lattice_constant) noexcept
{
    try
    {
        // The new geometry
        auto& old_geometry = *state->active_image->geometry;
        auto  new_geometry = Data::Geometry(old_geometry.bravais_vectors,
            old_geometry.n_cells, old_geometry.cell_atoms, old_geometry.cell_atom_types, lattice_constant);

        // Update the State
        Helper_State_Set_Geometry(state, old_geometry, new_geometry);

        Log(Utility::Log_Level::Warning, Utility::Log_Sender::API, fmt::format("Set lattice constant for all Systems to {}", lattice_constant), -1, -1);
    }
    catch( ... )
    {
        spirit_handle_exception_api(0, 0);
    }
}



int Geometry_Get_NOS(State * state) noexcept
{
    return state->nos;
}

scalar * Geometry_Get_Positions( State * state, int idx_image, int idx_chain ) noexcept
{
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
        spirit_handle_exception_api(idx_image, idx_chain);
        return nullptr;
    }
}

int * Geometry_Get_Atom_Types( State * state, int idx_image, int idx_chain ) noexcept
{
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
        spirit_handle_exception_api(idx_image, idx_chain);
        return nullptr;
    }
}

void Geometry_Get_Bounds( State *state, float min[3], float max[3], int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;

        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        // TODO: we should also check if idx_image < 0 and log the promotion to idx_active_image

        auto g = image->geometry;
        for (int dim=0; dim<3; ++dim)
        {
            min[dim] = (float)g->bounds_min[dim];
            max[dim] = (float)g->bounds_max[dim];
        }
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

// Get Center as array (x,y,z)
void Geometry_Get_Center(State *state, float center[3], int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        // TODO: we should also check if idx_image < 0 and log the promotion to idx_active_image
            
        auto g = image->geometry;
        for (int dim=0; dim<3; ++dim)
        {
            center[dim] = (float)g->center[dim];
        }
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

void Geometry_Get_Cell_Bounds( State *state, float min[3], float max[3], int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
            
        // TODO: we should also check if idx_image < 0 and log the promotion to idx_active_image
        
        auto g = image->geometry;
        for (int dim=0; dim<3; ++dim)
        {
            min[dim] = (float)g->cell_bounds_min[dim];
            max[dim] = (float)g->cell_bounds_max[dim];
        }
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

// Get bravais lattice type
Bravais_Lattice_Type Geometry_Get_Bravais_Type(State *state, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        return Bravais_Lattice_Type(image->geometry->classifier);
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
        return Bravais_Lattice_Irregular;
    }
}

// Get bravais vectors ta, tb, tc
void Geometry_Get_Bravais_Vectors( State *state, float a[3], float b[3], float c[3], 
                                 int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        // TODO: we should also check if idx_image < 0 and log the promotion to idx_active_image
        
        auto g = image->geometry;
        for (int dim=0; dim<3; ++dim)
        {
            a[dim] = (float)g->bravais_vectors[dim][0];
            b[dim] = (float)g->bravais_vectors[dim][1];
            c[dim] = (float)g->bravais_vectors[dim][2];
        }
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

// Get number of atoms in a basis cell
int Geometry_Get_N_Cell_Atoms(State *state, int idx_image, int idx_chain) noexcept
{
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
        spirit_handle_exception_api(idx_image, idx_chain);
        return false;
    }
}

// Get basis cell atoms
int Geometry_Get_Cell_Atoms(State *state, scalar ** atoms, int idx_image, int idx_chain)
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;

        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );

        auto g = image->geometry;
        if (atoms != nullptr)
            *atoms = reinterpret_cast<scalar *>(g->cell_atoms[0].data());

        return g->cell_atoms.size();
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
        return 0;
    }
}

// Get number of basis cells in the three translation directions
void Geometry_Get_N_Cells(State *state, int n_cells[3], int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        // TODO: we should also check if idx_image < 0 and log the promotion to idx_active_image
        
        auto g = image->geometry;
        n_cells[0] = g->n_cells[0];
        n_cells[1] = g->n_cells[1];
        n_cells[2] = g->n_cells[2];
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

// Get translation vectors ta, tb, tc
void Geometry_Get_Translation_Vectors( State *state, float ta[3], float tb[3], float tc[3], 
                                       int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        // TODO: we should also check if idx_image < 0 and log the promotion to idx_active_image
        
        auto g = image->geometry;
        for (int dim=0; dim<3; ++dim)
        {
            ta[dim] = (float)g->bravais_vectors[0][dim];
            tb[dim] = (float)g->bravais_vectors[1][dim];
            tc[dim] = (float)g->bravais_vectors[2][dim];
        }
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

int Geometry_Get_Dimensionality(State * state, int idx_image, int idx_chain) noexcept
{
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
        spirit_handle_exception_api(idx_image, idx_chain);
        return 0;
    }
}


int Geometry_Get_Triangulation( State * state, const int ** indices_ptr, int n_cell_step, 
                                int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
      
        // TODO: we should also check if idx_image < 0 and log the promotion to idx_active_image
        
        auto g = image->geometry;
        auto& triangles = g->triangulation(n_cell_step);
        if (indices_ptr != nullptr) {
            *indices_ptr = reinterpret_cast<const int *>(triangles.data());
        }
        return triangles.size();
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
        return 0;
    }
}

int Geometry_Get_Tetrahedra( State * state, const int ** indices_ptr, int n_cell_step, 
                             int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        from_indices(state, idx_image, idx_chain, image, chain);

        auto g = image->geometry;
        auto& tetrahedra = g->tetrahedra(n_cell_step);
        if (indices_ptr != nullptr) {
            *indices_ptr = reinterpret_cast<const int *>(tetrahedra.data());
        }
        return tetrahedra.size();
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
        return 0;
    }
}