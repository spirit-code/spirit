#include <Spirit/Geometry.h>
#include <data/State.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>


void Geometry_Set_N_Cells(State * state, int n_cells_i[3]) noexcept
{
    // The new number of spins
    auto n_cells = intfield{n_cells_i[0], n_cells_i[1], n_cells_i[2]};
    int nos = n_cells[0]*n_cells[1]*n_cells[2]*Geometry_Get_N_Cell_Atoms(state);

    // Deal with all systems in all chains
    for (auto& chain : state->collection->chains)
    {
		// Lock to avoid memory errors
		chain->Lock();
		// Modify all systems in the chain
        for (auto& system : chain->images)
        {
            int nos_old = system->nos;
            system->nos = nos;

            // Geometry
            auto ge = system->geometry;
            *system->geometry = Data::Geometry(ge->bravais_vectors,
                n_cells, ge->cell_atoms, ge->cell_atom_types, ge->lattice_constant);
            
            // Spins
            // TODO: ordering of spins should be considered and date potentially extrapolated -> write a function for this
			system->spins->resize(nos);
			system->effective_field.resize(nos);
			for (int i = nos_old; i<nos; ++i) (*system->spins)[i] = Vector3{ 0, 0, 1 };
			for (int i = nos_old; i<nos; ++i) system->effective_field[i] = Vector3{ 0, 0, 1 };

            // Parameters
            // TODO: properly re-generate pinning
            system->llg_parameters->pinning->mask_unpinned = intfield(nos, 1);

            // Hamiltonian
            // TODO: can we do this nicer than resizing everything?
			// TODO: how to resize with correct ordering of data?
			system->hamiltonian->Update_From_Geometry();
        }
		// Unlock again
		chain->Unlock();
    }

    // Update convenience integers across everywhere
    state->nos = nos;
    // Deal with clipboard image of State
    if (state->clipboard_image)
    {
        auto& system = state->clipboard_image;
        int nos_old = system->nos;
        system->nos = nos;

		// Lock to avoid memory errors
		system->Lock();
        
        // Geometry
        auto ge = system->geometry;
        *system->geometry = Data::Geometry(ge->bravais_vectors,
            n_cells, ge->cell_atoms, ge->cell_atom_types, ge->lattice_constant);
        
        // Spins
        // TODO: ordering of spins should be considered -> write a Configurations function for this
        system->spins->resize(nos);
		system->effective_field.resize(nos);
		for (int i = nos_old; i<nos; ++i) (*system->spins)[i] = Vector3{ 0, 0, 1 };
		for (int i = nos_old; i<nos; ++i) system->effective_field[i] = Vector3{ 0, 0, 1 };
        
        // Parameters
        // TODO: properly re-generate pinning
        system->llg_parameters->pinning->mask_unpinned = intfield(nos, 1);

		// Unlock
		system->Unlock();
	}
    // Deal with clipboard configuration of State
	if (state->clipboard_spins)
	{
		// TODO: the previous configuration should be extended, not overwritten
		state->clipboard_spins = std::shared_ptr<vectorfield>(new vectorfield(nos, { 0, 0, 1 }));
	}

    // TODO: the Hamiltonians may contain arrays that depend on system size

	Log(Utility::Log_Level::Warning, Utility::Log_Sender::API, "Set number of cells for all Systems: (" + std::to_string(n_cells[0]) + ", " + std::to_string(n_cells[1]) + ", " + std::to_string(n_cells[2]) + ")", -1, -1);
}

void Geometry_Set_Cell_Atoms(State *state, int n_atoms, float ** atoms) noexcept
{
    Log(Utility::Log_Level::Warning, Utility::Log_Sender::API, "Geometry_Set_Cell_Atoms is not yet implemented", -1, -1);
}

void Geometry_Set_Translation_Vectors(State *state, float ta[3], float tb[3], float tc[3]) noexcept
{
    Log(Utility::Log_Level::Warning, Utility::Log_Sender::API, "Geometry_Get_Translation_Vectors is not yet implemented", -1, -1);
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

// Get basis vectors ta, tb, tc
void Geometry_Get_Basis_Vectors( State *state, float a[3], float b[3], float c[3], 
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

// TODO: Get basis atoms
// void Geometry_Get_Cell_Atoms(State *state, float * n_atoms, float ** atoms)
// {
//     auto g = state->active_image->geometry;
//     *n_atoms = g->n_cell_atoms;
// }

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