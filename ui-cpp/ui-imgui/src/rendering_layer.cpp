#include <rendering_layer.hpp>

#include <Spirit/Chain.h>
#include <Spirit/Configurations.h>
#include <Spirit/Geometry.h>
#include <Spirit/Log.h>
#include <Spirit/Simulation.h>
#include <Spirit/System.h>

#include <fmt/format.h>

RenderingLayer::RenderingLayer( std::shared_ptr<State> state ) : state( state ) {}

void RenderingLayer::draw( int display_w, int display_h )
{
    if( !gl_initialized_ )
        initialize_gl();

    if( needs_data_ )
    {
        update_vf_directions();
        needs_redraw_ = true;
    }
    if( needs_redraw_ )
    {
        view.setFramebufferSize( float( display_w ), float( display_h ) );
        view.draw();
    }
}

void RenderingLayer::needs_redraw()
{
    needs_redraw_ = true;
}

void RenderingLayer::needs_data()
{
    needs_data_ = true;
}

void RenderingLayer::initialize_gl()
{
    view.setOption<VFRendering::ArrowRenderer::Option::CONE_RADIUS>( 0.125f );
    view.setOption<VFRendering::ArrowRenderer::Option::CONE_HEIGHT>( 0.3f );
    view.setOption<VFRendering::ArrowRenderer::Option::CYLINDER_RADIUS>( 0.0625f );
    view.setOption<VFRendering::ArrowRenderer::Option::CYLINDER_HEIGHT>( 0.35f );
    view.setOption<VFRendering::View::Option::BACKGROUND_COLOR>(
        { background_colour.x, background_colour.y, background_colour.z } );

    this->update_vf_geometry();
    this->update_vf_directions();

    view.setOption<VFRendering::View::Option::COLORMAP_IMPLEMENTATION>(
        VFRendering::Utilities::getColormapImplementation( VFRendering::Utilities::Colormap::HSV ) );

    arrow_renderer_ptr = std::make_shared<VFRendering::ArrowRenderer>( view, vectorfield );
    system_renderers.push_back( arrow_renderer_ptr );

    view.renderers(
        { { std::make_shared<VFRendering::CombinedRenderer>( view, system_renderers ), { { 0, 0, 1, 1 } } } } );

    gl_initialized_ = true;
}

void RenderingLayer::update_vf_geometry()
{
    int nos = System_Get_NOS( state.get() );
    int n_cells[3];
    Geometry_Get_N_Cells( this->state.get(), n_cells );
    int n_cell_atoms = Geometry_Get_N_Cell_Atoms( this->state.get() );

    int n_cells_draw[3] = { std::max( 1, n_cells[0] / n_cell_step ), std::max( 1, n_cells[1] / n_cell_step ),
                            std::max( 1, n_cells[2] / n_cell_step ) };
    int nos_draw        = n_cell_atoms * n_cells_draw[0] * n_cells_draw[1] * n_cells_draw[2];

    // Positions of the vectorfield
    std::vector<glm::vec3> positions = std::vector<glm::vec3>( nos_draw );

    // ToDo: Update the pointer to our Data instead of copying Data?
    // Positions
    //        get pointer
    scalar * spin_pos;
    int * atom_types;
    spin_pos   = Geometry_Get_Positions( state.get() );
    atom_types = Geometry_Get_Atom_Types( state.get() );
    int icell  = 0;
    for( int cell_c = 0; cell_c < n_cells_draw[2]; cell_c++ )
    {
        for( int cell_b = 0; cell_b < n_cells_draw[1]; cell_b++ )
        {
            for( int cell_a = 0; cell_a < n_cells_draw[0]; cell_a++ )
            {
                for( int ibasis = 0; ibasis < n_cell_atoms; ++ibasis )
                {
                    int idx = ibasis
                              + n_cell_atoms * n_cell_step
                                    * ( +cell_a + n_cells[0] * cell_b + n_cells[0] * n_cells[1] * cell_c );
                    positions[icell] = glm::vec3( spin_pos[3 * idx], spin_pos[1 + 3 * idx], spin_pos[2 + 3 * idx] );
                    ++icell;
                }
            }
        }
    }

    // Generate the right geometry (triangles and tetrahedra)
    VFRendering::Geometry geometry;
    VFRendering::Geometry geometry_surf2D;
    //      get tetrahedra
    if( Geometry_Get_Dimensionality( state.get() ) == 3 )
    {
        if( n_cell_step > 1
            && ( n_cells[0] / n_cell_step < 2 || n_cells[1] / n_cell_step < 2 || n_cells[2] / n_cell_step < 2 ) )
        {
            geometry = VFRendering::Geometry( positions, {}, {}, true );
        }
        else
        {
            const std::array<VFRendering::Geometry::index_type, 4> * tetrahedra_indices_ptr = nullptr;
            int num_tetrahedra                                                              = Geometry_Get_Tetrahedra(
                state.get(), reinterpret_cast<const int **>( &tetrahedra_indices_ptr ), n_cell_step );
            std::vector<std::array<VFRendering::Geometry::index_type, 4>> tetrahedra_indices(
                tetrahedra_indices_ptr, tetrahedra_indices_ptr + num_tetrahedra );
            geometry = VFRendering::Geometry( positions, {}, tetrahedra_indices, false );
        }
    }
    else if( Geometry_Get_Dimensionality( state.get() ) == 2 )
    {
        // Determine two basis vectors
        std::array<glm::vec3, 2> basis;
        float eps = 1e-6;
        for( int i = 1, j = 0; i < nos && j < 2; ++i )
        {
            if( glm::length( positions[i] - positions[0] ) > eps )
            {
                if( j < 1 )
                {
                    basis[j] = glm::normalize( positions[i] - positions[0] );
                    ++j;
                }
                else
                {
                    if( 1 - std::abs( glm::dot( basis[0], glm::normalize( positions[i] - positions[0] ) ) ) > eps )
                    {
                        basis[j] = glm::normalize( positions[i] - positions[0] );
                        ++j;
                    }
                }
            }
        }
        glm::vec3 normal = glm::normalize( glm::cross( basis[0], basis[1] ) );
        // By default, +z is up, which is where we want the normal oriented towards
        if( glm::dot( normal, glm::vec3{ 0, 0, 1 } ) < 1e-6 )
            normal = -normal;

        // Rectilinear with one basis atom
        if( n_cell_atoms == 1 && std::abs( glm::dot( basis[0], basis[1] ) ) < 1e-6 )
        {
            std::vector<float> xs( n_cells_draw[0] ), ys( n_cells_draw[1] ), zs( n_cells_draw[2] );
            for( int i = 0; i < n_cells_draw[0]; ++i )
                xs[i] = positions[i].x;
            for( int i = 0; i < n_cells_draw[1]; ++i )
                ys[i] = positions[i * n_cells_draw[0]].y;
            for( int i = 0; i < n_cells_draw[2]; ++i )
                zs[i] = positions[i * n_cells_draw[0] * n_cells_draw[1]].z;
            geometry = VFRendering::Geometry::rectilinearGeometry( xs, ys, zs );
            for( int i = 0; i < n_cells_draw[0]; ++i )
                xs[i] = ( positions[i] - normal ).x;
            for( int i = 0; i < n_cells_draw[1]; ++i )
                ys[i] = ( positions[i * n_cells_draw[0]] - normal ).y;
            for( int i = 0; i < n_cells_draw[2]; ++i )
                zs[i] = ( positions[i * n_cells_draw[0] * n_cells_draw[1]] - normal ).z;
            geometry_surf2D = VFRendering::Geometry::rectilinearGeometry( xs, ys, zs );
        }
        // All others
        else
        {
            const std::array<VFRendering::Geometry::index_type, 3> * triangle_indices_ptr = nullptr;
            int num_triangles                                                             = Geometry_Get_Triangulation(
                state.get(), reinterpret_cast<const int **>( &triangle_indices_ptr ), n_cell_step );
            std::vector<std::array<VFRendering::Geometry::index_type, 3>> triangle_indices(
                triangle_indices_ptr, triangle_indices_ptr + num_triangles );
            geometry = VFRendering::Geometry( positions, triangle_indices, {}, true );
            for( int i = 0; i < nos_draw; ++i )
                positions[i] = positions[i] - normal;
            geometry_surf2D = VFRendering::Geometry( positions, triangle_indices, {}, true );
        }

        // Update the vectorfield geometry
        vectorfield_surf2D.updateGeometry( geometry_surf2D );
    }
    else
    {
        geometry = VFRendering::Geometry( positions, {}, {}, true );
    }

    // Update the vectorfield
    vectorfield.updateGeometry( geometry );
}

void RenderingLayer::update_vf_directions()
{
    int nos = System_Get_NOS( state.get() );
    int n_cells[3];
    Geometry_Get_N_Cells( this->state.get(), n_cells );
    int n_cell_atoms = Geometry_Get_N_Cell_Atoms( this->state.get() );

    int n_cells_draw[3] = { std::max( 1, n_cells[0] / n_cell_step ), std::max( 1, n_cells[1] / n_cell_step ),
                            std::max( 1, n_cells[2] / n_cell_step ) };
    int nos_draw        = n_cell_atoms * n_cells_draw[0] * n_cells_draw[1] * n_cells_draw[2];

    // Directions of the vectorfield
    std::vector<glm::vec3> directions = std::vector<glm::vec3>( nos_draw );

    // ToDo: Update the pointer to our Data instead of copying Data?
    // Directions
    //        get pointer
    scalar * spins;
    int * atom_types;
    atom_types = Geometry_Get_Atom_Types( state.get() );
    // if( this->m_source == 0 )
    spins = System_Get_Spin_Directions( state.get() );
    // else if( this->m_source == 1 )
    //     spins = System_Get_Effective_Field( state.get() );
    // else spins = System_Get_Spin_Directions( state.get() );

    //        copy
    /*positions.assign(spin_pos, spin_pos + 3*nos);
    directions.assign(spins, spins + 3*nos);*/
    int icell = 0;
    for( int cell_c = 0; cell_c < n_cells_draw[2]; cell_c++ )
    {
        for( int cell_b = 0; cell_b < n_cells_draw[1]; cell_b++ )
        {
            for( int cell_a = 0; cell_a < n_cells_draw[0]; cell_a++ )
            {
                for( int ibasis = 0; ibasis < n_cell_atoms; ++ibasis )
                {
                    int idx = ibasis + n_cell_atoms * cell_a * n_cell_step
                              + n_cell_atoms * n_cells[0] * cell_b * n_cell_step
                              + n_cell_atoms * n_cells[0] * n_cells[1] * cell_c * n_cell_step;
                    // std::cerr << idx << " " << icell << std::endl;
                    directions[icell] = glm::vec3( spins[3 * idx], spins[1 + 3 * idx], spins[2 + 3 * idx] );
                    if( atom_types[idx] < 0 )
                        directions[icell] *= 0;
                    ++icell;
                }
            }
        }
    }
    // //        rescale if effective field
    // if( this->m_source == 1 )
    // {
    //     float max_length = 0;
    //     for( auto direction : directions )
    //     {
    //         max_length = std::max( max_length, glm::length( direction ) );
    //     }
    //     if( max_length > 0 )
    //     {
    //         for( auto & direction : directions )
    //         {
    //             direction /= max_length;
    //         }
    //     }
    // }

    // Update the vectorfield
    vectorfield.updateVectors( directions );

    if( Geometry_Get_Dimensionality( state.get() ) == 2 )
        vectorfield_surf2D.updateVectors( directions );
}
