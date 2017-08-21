#include <Spirit/Geometry.h>
#include <data/State.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

scalar * Geometry_Get_Spin_Positions( State * state, int idx_image, int idx_chain )
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        // TODO: we should also check if idx_image < 0 and log the promotion to idx_active_image
        
        return (scalar *)image->geometry->spin_pos[0].data();
    }
    catch( ... )
    {
        Utility::Handle_Exception( idx_image, idx_chain );
        return nullptr;
    }
}

int * Geometry_Get_Atom_Types( State * state, int idx_image, int idx_chain )
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
        Utility::Handle_Exception( idx_image, idx_chain );
        return nullptr;
    }
}

void Geometry_Get_Bounds( State *state, float min[3], float max[3], int idx_image, int idx_chain )
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
        Utility::Handle_Exception( idx_image, idx_chain );
    }
}

// Get Center as array (x,y,z)
void Geometry_Get_Center(State *state, float center[3], int idx_image, int idx_chain)
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
        Utility::Handle_Exception( idx_image, idx_chain );
    }
}

void Geometry_Get_Cell_Bounds( State *state, float min[3], float max[3], int idx_image, int idx_chain )
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
        Utility::Handle_Exception( idx_image, idx_chain );
    }   
}

// Get basis vectors ta, tb, tc
void Geometry_Get_Basis_Vectors( State *state, float a[3], float b[3], float c[3], 
                                 int idx_image, int idx_chain )
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
            a[dim] = (float)g->basis[dim][0];
            b[dim] = (float)g->basis[dim][1];
            c[dim] = (float)g->basis[dim][2];
        }
    }
    catch( ... )
    {
        Utility::Handle_Exception( idx_image, idx_chain );
    }
}

// TODO: Get basis atoms
// void Geometry_Get_Basis_Atoms(State *state, float * n_atoms, float ** atoms)
// {
//     auto g = state->active_image->geometry;
//     *n_atoms = g->n_spins_basic_domain;
// }

// Get number of atoms in a basis cell
int Geometry_Get_N_Basis_Atoms(State *state, int idx_image, int idx_chain)
{
    try
    {
    	std::shared_ptr<Data::Spin_System> image;
    	std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        // TODO: we should also check if idx_image < 0 and log the promotion to idx_active_image
        
        return image->geometry->n_spins_basic_domain;
    }
    catch( ... )
    {
        Utility::Handle_Exception( idx_image, idx_chain );
        return false;
    }
}

// Get number of basis cells in the three translation directions
void Geometry_Get_N_Cells(State *state, int n_cells[3], int idx_image, int idx_chain)
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
        Utility::Handle_Exception( idx_image, idx_chain );
    }
}

// Get translation vectors ta, tb, tc
void Geometry_Get_Translation_Vectors( State *state, float ta[3], float tb[3], float tc[3], 
                                       int idx_image, int idx_chain )
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
            ta[dim] = (float)g->translation_vectors[0][dim];
            tb[dim] = (float)g->translation_vectors[1][dim];
            tc[dim] = (float)g->translation_vectors[2][dim];
        }
    }
    catch( ... )
    {
        Utility::Handle_Exception( idx_image, idx_chain );
    }
}

int Geometry_Get_Dimensionality(State * state, int idx_image, int idx_chain)
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
        Utility::Handle_Exception( idx_image, idx_chain );
        return 0;
    }
}


int Geometry_Get_Triangulation( State * state, const int ** indices_ptr, int n_cell_step, 
                                int idx_image, int idx_chain )
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
        Utility::Handle_Exception( idx_image, idx_chain );
        return 0;
    }
}

int Geometry_Get_Tetrahedra( State * state, const int ** indices_ptr, int n_cell_step, 
                             int idx_image, int idx_chain )
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
        Utility::Handle_Exception( idx_image, idx_chain );
        return 0;        
    }
}