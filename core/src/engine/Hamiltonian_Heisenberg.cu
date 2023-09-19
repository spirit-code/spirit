#ifdef SPIRIT_USE_CUDA

#include <data/Spin_System.hpp>
#include <engine/Backend_par.hpp>
#include <engine/FFT.hpp>
#include <engine/Hamiltonian_Heisenberg.hpp>
#include <engine/Indexing.hpp>
#include <engine/Neighbours.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Constants.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <complex>

using namespace Data;
using namespace Utility;
namespace C = Utility::Constants;
using Engine::Indexing::check_atom_type;
using Engine::Indexing::cu_check_atom_type;
using Engine::Indexing::cu_idx_from_pair;
using Engine::Indexing::cu_tupel_from_idx;
using Engine::Indexing::idx_from_pair;
using Engine::Indexing::idx_from_translations;

namespace Engine
{
// Construct a Heisenberg Hamiltonian with pairs
Hamiltonian_Heisenberg::Hamiltonian_Heisenberg(
    std::shared_ptr<Data::Geometry> geometry, const intfield & boundary_conditions, const NormalVector & external_field,
    const VectorfieldData & anisotropy, const ScalarfieldData & cubic_anisotropy, const ScalarPairfieldData & exchange,
    const VectorPairfieldData & dmi, const QuadrupletfieldData & quadruplet, const DDI_Method ddi_method,
    const DDI_Data & ddi_data )
        : Hamiltonian( boundary_conditions ),
          external_field_magnitude( external_field.magnitude * C::mu_B ),
          external_field_normal( external_field.normal ),
          anisotropy_indices( anisotropy.indices ),
          anisotropy_magnitudes( anisotropy.magnitudes ),
          anisotropy_normals( anisotropy.normals ),
          cubic_anisotropy_indices( cubic_anisotropy.indices ),
          cubic_anisotropy_magnitudes( cubic_anisotropy.magnitudes ),
          exchange_shell_magnitudes( 0 ),
          exchange_pairs_in( exchange.pairs ),
          exchange_magnitudes_in( exchange.magnitudes ),
          dmi_shell_magnitudes( 0 ),
          dmi_shell_chirality( 0 ),
          dmi_pairs_in( dmi.pairs ),
          dmi_magnitudes_in( dmi.magnitudes ),
          dmi_normals_in( dmi.normals ),
          ddi_method( ddi_method ),
          ddi_n_periodic_images( ddi_data.n_periodic_images ),
          ddi_pb_zero_padding( ddi_data.pb_zero_padding ),
          ddi_cutoff_radius( ddi_data.radius ),
          quadruplets( quadruplet.quadruplets ),
          quadruplet_magnitudes( quadruplet.magnitudes ),
          geometry( std::move( geometry ) ),
          fft_plan_spins( FFT::FFT_Plan() ),
          fft_plan_reverse( FFT::FFT_Plan() )
{
    // Generate interaction pairs, constants etc.
    this->Update_Interactions();
}

// Construct a Heisenberg Hamiltonian from shells
Hamiltonian_Heisenberg::Hamiltonian_Heisenberg(
    std::shared_ptr<Data::Geometry> geometry, const intfield & boundary_conditions, const NormalVector & external_field,
    const VectorfieldData & anisotropy, const ScalarfieldData & cubic_anisotropy,
    const scalarfield & exchange_shell_magnitudes, const scalarfield & dmi_shell_magnitudes, int dm_chirality,
    const QuadrupletfieldData & quadruplet, const DDI_Method ddi_method, const DDI_Data & ddi_data )
        : Hamiltonian( boundary_conditions ),
          external_field_magnitude( external_field.magnitude * C::mu_B ),
          external_field_normal( external_field.normal ),
          anisotropy_indices( anisotropy.indices ),
          anisotropy_magnitudes( anisotropy.magnitudes ),
          anisotropy_normals( anisotropy.normals ),
          cubic_anisotropy_indices( cubic_anisotropy.indices ),
          cubic_anisotropy_magnitudes( cubic_anisotropy.magnitudes ),
          exchange_shell_magnitudes( exchange_shell_magnitudes ),
          exchange_pairs_in( 0 ),
          exchange_magnitudes_in( 0 ),
          dmi_shell_magnitudes( dmi_shell_magnitudes ),
          dmi_shell_chirality( dm_chirality ),
          dmi_pairs_in( 0 ),
          dmi_magnitudes_in( 0 ),
          dmi_normals_in( 0 ),
          ddi_method( ddi_method ),
          ddi_n_periodic_images( ddi_data.n_periodic_images ),
          ddi_pb_zero_padding( ddi_data.pb_zero_padding ),
          ddi_cutoff_radius( ddi_data.radius ),
          quadruplets( quadruplet.quadruplets ),
          quadruplet_magnitudes( quadruplet.magnitudes ),
          geometry( std::move( geometry ) ),
          fft_plan_spins( FFT::FFT_Plan() ),
          fft_plan_reverse( FFT::FFT_Plan() )
{
    // Generate interaction pairs, constants etc.
    this->Update_Interactions();
}

void Hamiltonian_Heisenberg::Update_Interactions()
{
    // When parallelising (cuda or openmp), we need all neighbours per spin
    const bool use_redundant_neighbours = true;

    // Exchange
    this->exchange_pairs      = pairfield( 0 );
    this->exchange_magnitudes = scalarfield( 0 );
    if( exchange_shell_magnitudes.size() > 0 )
    {
        // Generate Exchange neighbours
        intfield exchange_shells( 0 );
        Neighbours::Get_Neighbours_in_Shells(
            *geometry, exchange_shell_magnitudes.size(), exchange_pairs, exchange_shells, use_redundant_neighbours );
        for( unsigned int ipair = 0; ipair < exchange_pairs.size(); ++ipair )
        {
            this->exchange_magnitudes.push_back( exchange_shell_magnitudes[exchange_shells[ipair]] );
        }
    }
    else
    {
        // Use direct list of pairs
        this->exchange_pairs      = this->exchange_pairs_in;
        this->exchange_magnitudes = this->exchange_magnitudes_in;
        if( use_redundant_neighbours )
        {
            for( int i = 0; i < exchange_pairs_in.size(); ++i )
            {
                auto & p = exchange_pairs_in[i];
                auto & t = p.translations;
                this->exchange_pairs.push_back( Pair{ p.j, p.i, { -t[0], -t[1], -t[2] } } );
                this->exchange_magnitudes.push_back( exchange_magnitudes_in[i] );
            }
        }
    }

    // DMI
    this->dmi_pairs      = pairfield( 0 );
    this->dmi_magnitudes = scalarfield( 0 );
    this->dmi_normals    = vectorfield( 0 );
    if( dmi_shell_magnitudes.size() > 0 )
    {
        // Generate DMI neighbours and normals
        intfield dmi_shells( 0 );
        Neighbours::Get_Neighbours_in_Shells(
            *geometry, dmi_shell_magnitudes.size(), dmi_pairs, dmi_shells, use_redundant_neighbours );
        for( unsigned int ineigh = 0; ineigh < dmi_pairs.size(); ++ineigh )
        {
            this->dmi_normals.push_back(
                Neighbours::DMI_Normal_from_Pair( *geometry, dmi_pairs[ineigh], dmi_shell_chirality ) );
            this->dmi_magnitudes.push_back( dmi_shell_magnitudes[dmi_shells[ineigh]] );
        }
    }
    else
    {
        // Use direct list of pairs
        this->dmi_pairs      = this->dmi_pairs_in;
        this->dmi_magnitudes = this->dmi_magnitudes_in;
        this->dmi_normals    = this->dmi_normals_in;
        for( int i = 0; i < dmi_pairs_in.size(); ++i )
        {
            auto & p = dmi_pairs_in[i];
            auto & t = p.translations;
            this->dmi_pairs.push_back( Pair{ p.j, p.i, { -t[0], -t[1], -t[2] } } );
            this->dmi_magnitudes.push_back( dmi_magnitudes_in[i] );
            this->dmi_normals.push_back( -dmi_normals_in[i] );
        }
    }

    // Dipole-dipole (cutoff)
    if( this->ddi_method == DDI_Method::Cutoff )
        this->ddi_pairs = Engine::Neighbours::Get_Pairs_in_Radius( *this->geometry, this->ddi_cutoff_radius );
    else
        this->ddi_pairs = field<Pair>{};

    this->ddi_magnitudes = scalarfield( this->ddi_pairs.size() );
    this->ddi_normals    = vectorfield( this->ddi_pairs.size() );

    for( unsigned int i = 0; i < this->ddi_pairs.size(); ++i )
    {
        Engine::Neighbours::DDI_from_Pair(
            *this->geometry,
            { this->ddi_pairs[i].i,
              this->ddi_pairs[i].j,
              { this->ddi_pairs[i].translations[0], this->ddi_pairs[i].translations[1],
                this->ddi_pairs[i].translations[2] } },
            this->ddi_magnitudes[i], this->ddi_normals[i] );
    }
    // Dipole-dipole (FFT)
    this->Prepare_DDI();

    // Update, which terms still contribute
    this->Update_Energy_Contributions();
}

void Hamiltonian_Heisenberg::Update_Energy_Contributions()
{
    this->energy_contributions_per_spin = std::vector<std::pair<std::string, scalarfield>>( 0 );

    // External field
    if( std::abs( this->external_field_magnitude ) > 1e-60 )
    {
        this->energy_contributions_per_spin.push_back( { "Zeeman", scalarfield( 0 ) } );
        this->idx_zeeman = this->energy_contributions_per_spin.size() - 1;
    }
    else
        this->idx_zeeman = -1;
    // Anisotropy
    if( this->anisotropy_indices.size() > 0 )
    {
        this->energy_contributions_per_spin.push_back( { "Anisotropy", scalarfield( 0 ) } );
        this->idx_anisotropy = this->energy_contributions_per_spin.size() - 1;
    }
    else
        this->idx_anisotropy = -1;
    // Cubic anisotropy
    if( this->cubic_anisotropy_indices.size() > 0 )
    {
        this->energy_contributions_per_spin.push_back( { "Cubic anisotropy", scalarfield( 0 ) } );
        this->idx_cubic_anisotropy = this->energy_contributions_per_spin.size() - 1;
    }
    else
        this->idx_cubic_anisotropy = -1;
    // Exchange
    if( this->exchange_pairs.size() > 0 )
    {
        this->energy_contributions_per_spin.push_back( { "Exchange", scalarfield( 0 ) } );
        this->idx_exchange = this->energy_contributions_per_spin.size() - 1;
    }
    else
        this->idx_exchange = -1;
    // DMI
    if( this->dmi_pairs.size() > 0 )
    {
        this->energy_contributions_per_spin.push_back( { "DMI", scalarfield( 0 ) } );
        this->idx_dmi = this->energy_contributions_per_spin.size() - 1;
    }
    else
        this->idx_dmi = -1;
    // Dipole-Dipole
    if( this->ddi_method != DDI_Method::None )
    {
        this->energy_contributions_per_spin.push_back( { "DDI", scalarfield( 0 ) } );
        this->idx_ddi = this->energy_contributions_per_spin.size() - 1;
    }
    else
        this->idx_ddi = -1;
    // Quadruplets
    if( this->quadruplets.size() > 0 )
    {
        this->energy_contributions_per_spin.push_back( { "Quadruplets", scalarfield( 0 ) } );
        this->idx_quadruplet = this->energy_contributions_per_spin.size() - 1;
    }
    else
        this->idx_quadruplet = -1;
}

void Hamiltonian_Heisenberg::Energy_Contributions_per_Spin(
    const vectorfield & spins, std::vector<std::pair<std::string, scalarfield>> & contributions )
{
    if( contributions.size() != this->energy_contributions_per_spin.size() )
    {
        contributions = this->energy_contributions_per_spin;
    }
    int nos = spins.size();
    for( auto & pair : contributions )
    {
        // Allocate if not already allocated
        if( pair.second.size() != nos )
            pair.second = scalarfield( nos, 0 );
        // Otherwise set to zero
        else
            for( auto & pair : contributions )
                Vectormath::fill( pair.second, 0 );
    }

    // External field
    if( this->idx_zeeman >= 0 )
        E_Zeeman( spins, contributions[idx_zeeman].second );

    // Anisotropy
    if( this->idx_anisotropy >= 0 )
        E_Anisotropy( spins, contributions[idx_anisotropy].second );

    // Cubic anisotropy
    if( this->idx_cubic_anisotropy >= 0 )
        E_Cubic_Anisotropy( spins, contributions[idx_cubic_anisotropy].second );

    // Exchange
    if( this->idx_exchange >= 0 )
        E_Exchange( spins, contributions[idx_exchange].second );
    // DMI
    if( this->idx_dmi >= 0 )
        E_DMI( spins, contributions[idx_dmi].second );
    // DDI
    if( this->idx_ddi >= 0 )
        E_DDI( spins, contributions[idx_ddi].second );

    // Quadruplets
    if( this->idx_quadruplet >= 0 )
        E_Quadruplet( spins, contributions[idx_quadruplet].second );
}

__global__ void CU_E_Zeeman(
    const Vector3 * spins, const int * atom_types, const int n_cell_atoms, const scalar * mu_s,
    const scalar external_field_magnitude, const Vector3 external_field_normal, scalar * Energy, size_t n_cells_total )
{
    for( auto icell = blockIdx.x * blockDim.x + threadIdx.x; icell < n_cells_total; icell += blockDim.x * gridDim.x )
    {
        for( int ibasis = 0; ibasis < n_cell_atoms; ++ibasis )
        {
            int ispin = n_cell_atoms * icell + ibasis;
            if( cu_check_atom_type( atom_types[ispin] ) )
                Energy[ispin] -= mu_s[ispin] * external_field_magnitude * external_field_normal.dot( spins[ispin] );
        }
    }
}
void Hamiltonian_Heisenberg::E_Zeeman( const vectorfield & spins, scalarfield & Energy )
{
    int size = geometry->n_cells_total;
    CU_E_Zeeman<<<( size + 1023 ) / 1024, 1024>>>(
        spins.data(), this->geometry->atom_types.data(), geometry->n_cell_atoms, geometry->mu_s.data(),
        this->external_field_magnitude, this->external_field_normal, Energy.data(), size );
    CU_CHECK_AND_SYNC();
}

__global__ void CU_E_Anisotropy(
    const Vector3 * spins, const int * atom_types, const int n_cell_atoms, const int n_anisotropies,
    const int * anisotropy_indices, const scalar * anisotropy_magnitude, const Vector3 * anisotropy_normal,
    scalar * Energy, size_t n_cells_total )
{
    for( auto icell = blockIdx.x * blockDim.x + threadIdx.x; icell < n_cells_total; icell += blockDim.x * gridDim.x )
    {
        for( int iani = 0; iani < n_anisotropies; ++iani )
        {
            int ispin = icell * n_cell_atoms + anisotropy_indices[iani];
            if( cu_check_atom_type( atom_types[ispin] ) )
                Energy[ispin] -= anisotropy_magnitude[iani] * pow( anisotropy_normal[iani].dot( spins[ispin] ), 2 );
        }
    }
}
void Hamiltonian_Heisenberg::E_Anisotropy( const vectorfield & spins, scalarfield & Energy )
{
    int size = geometry->n_cells_total;
    CU_E_Anisotropy<<<( size + 1023 ) / 1024, 1024>>>(
        spins.data(), this->geometry->atom_types.data(), this->geometry->n_cell_atoms, this->anisotropy_indices.size(),
        this->anisotropy_indices.data(), this->anisotropy_magnitudes.data(), this->anisotropy_normals.data(),
        Energy.data(), size );
    CU_CHECK_AND_SYNC();
}

__global__ void CU_E_Cubic_Anisotropy(
    const Vector3 * spins, const int * atom_types, const int n_cell_atoms, const int n_anisotropies,
    const int * anisotropy_indices, const scalar * anisotropy_magnitude, scalar * Energy, size_t n_cells_total )
{
    for( auto icell = blockIdx.x * blockDim.x + threadIdx.x; icell < n_cells_total; icell += blockDim.x * gridDim.x )
    {
        for( int iani = 0; iani < n_anisotropies; ++iani )
        {
            int ispin = icell * n_cell_atoms + anisotropy_indices[iani];
            if( cu_check_atom_type( atom_types[ispin] ) )
                Energy[ispin]
                    -= anisotropy_magnitude[iani] / 2
                       * ( pow( spins[ispin][0], 4.0 ) + pow( spins[ispin][1], 4.0 ) + pow( spins[ispin][2], 4.0 ) );
        }
    }
}
void Hamiltonian_Heisenberg::E_Cubic_Anisotropy( const vectorfield & spins, scalarfield & Energy )
{
    int size = geometry->n_cells_total;
    CU_E_Cubic_Anisotropy<<<( size + 1023 ) / 1024, 1024>>>(
        spins.data(), this->geometry->atom_types.data(), this->geometry->n_cell_atoms,
        this->cubic_anisotropy_indices.size(), this->cubic_anisotropy_indices.data(),
        this->cubic_anisotropy_magnitudes.data(), Energy.data(), size );
    CU_CHECK_AND_SYNC();
}

__global__ void CU_E_Exchange(
    const Vector3 * spins, const int * atom_types, const int * boundary_conditions, const int * n_cells,
    int n_cell_atoms, int n_pairs, const Pair * pairs, const scalar * magnitudes, scalar * Energy, size_t size )
{
    int bc[3] = { boundary_conditions[0], boundary_conditions[1], boundary_conditions[2] };
    int nc[3] = { n_cells[0], n_cells[1], n_cells[2] };

    for( auto icell = blockIdx.x * blockDim.x + threadIdx.x; icell < size; icell += blockDim.x * gridDim.x )
    {
        for( auto ipair = 0; ipair < n_pairs; ++ipair )
        {
            int ispin = pairs[ipair].i + icell * n_cell_atoms;
            int jspin = cu_idx_from_pair( ispin, bc, nc, n_cell_atoms, atom_types, pairs[ipair] );
            if( jspin >= 0 )
            {
                Energy[ispin] -= 0.5 * magnitudes[ipair] * spins[ispin].dot( spins[jspin] );
            }
        }
    }
}
void Hamiltonian_Heisenberg::E_Exchange( const vectorfield & spins, scalarfield & Energy )
{
    int size = geometry->n_cells_total;
    CU_E_Exchange<<<( size + 1023 ) / 1024, 1024>>>(
        spins.data(), this->geometry->atom_types.data(), boundary_conditions.data(), geometry->n_cells.data(),
        geometry->n_cell_atoms, this->exchange_pairs.size(), this->exchange_pairs.data(),
        this->exchange_magnitudes.data(), Energy.data(), size );
    CU_CHECK_AND_SYNC();
}

__global__ void CU_E_DMI(
    const Vector3 * spins, const int * atom_types, const int * boundary_conditions, const int * n_cells,
    int n_cell_atoms, int n_pairs, const Pair * pairs, const scalar * magnitudes, const Vector3 * normals,
    scalar * Energy, size_t size )
{
    int bc[3] = { boundary_conditions[0], boundary_conditions[1], boundary_conditions[2] };
    int nc[3] = { n_cells[0], n_cells[1], n_cells[2] };

    for( auto icell = blockIdx.x * blockDim.x + threadIdx.x; icell < size; icell += blockDim.x * gridDim.x )
    {
        for( auto ipair = 0; ipair < n_pairs; ++ipair )
        {
            int ispin = pairs[ipair].i + icell * n_cell_atoms;
            int jspin = cu_idx_from_pair( ispin, bc, nc, n_cell_atoms, atom_types, pairs[ipair] );
            if( jspin >= 0 )
            {
                Energy[ispin] -= 0.5 * magnitudes[ipair] * normals[ipair].dot( spins[ispin].cross( spins[jspin] ) );
            }
        }
    }
}
void Hamiltonian_Heisenberg::E_DMI( const vectorfield & spins, scalarfield & Energy )
{
    int size = geometry->n_cells_total;
    CU_E_DMI<<<( size + 1023 ) / 1024, 1024>>>(
        spins.data(), this->geometry->atom_types.data(), boundary_conditions.data(), geometry->n_cells.data(),
        geometry->n_cell_atoms, this->dmi_pairs.size(), this->dmi_pairs.data(), this->dmi_magnitudes.data(),
        this->dmi_normals.data(), Energy.data(), size );
    CU_CHECK_AND_SYNC();
}

void Hamiltonian_Heisenberg::E_DDI( const vectorfield & spins, scalarfield & Energy )
{
    if( this->ddi_method == DDI_Method::FFT )
        this->E_DDI_FFT( spins, Energy );
    else if( this->ddi_method == DDI_Method::Cutoff )
    {
        // TODO: Merge these implementations in the future
        if( ddi_cutoff_radius >= 0 )
            this->E_DDI_Cutoff( spins, Energy );
        else
            this->E_DDI_Direct( spins, Energy );
    }
}

void Hamiltonian_Heisenberg::E_DDI_Direct( const vectorfield & spins, scalarfield & Energy )
{
    vectorfield gradients_temp;
    gradients_temp.resize( geometry->nos );
    Vectormath::fill( gradients_temp, { 0, 0, 0 } );
    this->Gradient_DDI_Direct( spins, gradients_temp );

#pragma omp parallel for
    for( int ispin = 0; ispin < geometry->nos; ispin++ )
    {
        Energy[ispin] += 0.5 * spins[ispin].dot( gradients_temp[ispin] );
    }
}

void Hamiltonian_Heisenberg::E_DDI_Cutoff( const vectorfield & spins, scalarfield & Energy )
{
    // //scalar mult = -mu_B*mu_B*1.0 / 4.0 / Pi; // multiply with mu_B^2
    // scalar mult = 0.5*0.0536814951168; // mu_0*mu_B**2/(4pi*10**-30) -- the translations are in angstr�m, so the
    // |r|[m] becomes |r|[m]*10^-10
    // // scalar result = 0.0;

    // for (unsigned int i_pair = 0; i_pair < ddi_pairs.size(); ++i_pair)
    // {
    //     if (ddi_magnitudes[i_pair] > 0.0)
    //     {
    //         for (int da = 0; da < geometry->n_cells[0]; ++da)
    //         {
    //             for (int db = 0; db < geometry->n_cells[1]; ++db)
    //             {
    //                 for (int dc = 0; dc < geometry->n_cells[2]; ++dc)
    //                 {
    //                     std::array<int, 3 > translations = { da, db, dc };
    //                     // int idx_i = ddi_pairs[i_pair].i;
    //                     // int idx_j = ddi_pairs[i_pair].j;
    //                     int idx_i = idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations);
    //                     int idx_j = idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations,
    //                     ddi_pairs[i_pair].translations); Energy[idx_i] -= mult / std::pow(ddi_magnitudes[i_pair], 3.0) *
    //                         (3 * spins[idx_j].dot(ddi_normals[i_pair]) * spins[idx_i].dot(ddi_normals[i_pair]) -
    //                         spins[idx_i].dot(spins[idx_j]));
    //                     Energy[idx_j] -= mult / std::pow(ddi_magnitudes[i_pair], 3.0) *
    //                         (3 * spins[idx_j].dot(ddi_normals[i_pair]) * spins[idx_i].dot(ddi_normals[i_pair]) -
    //                         spins[idx_i].dot(spins[idx_j]));
    //                 }
    //             }
    //         }
    //     }
    // }
} // end DipoleDipole

// TODO: add dot_scaled to Vectormath and use that
__global__ void CU_E_DDI_FFT(
    scalar * Energy, const Vector3 * spins, const Vector3 * gradients, const int nos, const int n_cell_atoms,
    const scalar * mu_s )
{
    for( int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nos; idx += blockDim.x * gridDim.x )
    {
        Energy[idx] += 0.5 * spins[idx].dot( gradients[idx] );
    }
}
void Hamiltonian_Heisenberg::E_DDI_FFT( const vectorfield & spins, scalarfield & Energy )
{
    // todo maybe the gradient should be cached somehow, it is quite inefficient to calculate it
    // again just for the energy
    vectorfield gradients_temp( geometry->nos );
    Vectormath::fill( gradients_temp, { 0, 0, 0 } );
    this->Gradient_DDI( spins, gradients_temp );
    CU_E_DDI_FFT<<<( geometry->nos + 1023 ) / 1024, 1024>>>(
        Energy.data(), spins.data(), gradients_temp.data(), geometry->nos, geometry->n_cell_atoms,
        geometry->mu_s.data() );

    // === DEBUG: begin gradient comparison ===
    // vectorfield gradients_temp_dir;
    // gradients_temp_dir.resize(this->geometry->nos);
    // Vectormath::fill(gradients_temp_dir, {0,0,0});
    // Gradient_DDI_Direct(spins, gradients_temp_dir);

    // //get deviation
    // std::array<scalar, 3> deviation = {0,0,0};
    // std::array<scalar, 3> avg = {0,0,0};
    // std::array<scalar, 3> avg_ft = {0,0,0};

    // for(int i = 0; i < this->geometry->nos; i++)
    // {
    //     for(int d = 0; d < 3; d++)
    //     {
    //         deviation[d] += std::pow(gradients_temp[i][d] - gradients_temp_dir[i][d], 2);
    //         avg[d] += gradients_temp_dir[i][d];
    //         avg_ft[d] += gradients_temp[i][d];
    //     }
    // }
    // std::cerr << "Avg. Gradient (Direct) = " << avg[0]/this->geometry->nos << " " << avg[1]/this->geometry->nos << "
    // " << avg[2]/this->geometry->nos << std::endl; std::cerr << "Avg. Gradient (FFT)    = " <<
    // avg_ft[0]/this->geometry->nos << " " << avg_ft[1]/this->geometry->nos << " " << avg_ft[2]/this->geometry->nos <<
    // std::endl; std::cerr << "Relative Error in %    = " << (avg_ft[0]/avg[0]-1)*100 << " " <<
    // (avg_ft[1]/avg[1]-1)*100 << " " << (avg_ft[2]/avg[2]-1)*100 << std::endl; std::cerr << "Avg. Deviation         =
    // " << std::pow(deviation[0]/this->geometry->nos, 0.5) << " " << std::pow(deviation[1]/this->geometry->nos, 0.5) <<
    // " " << std::pow(deviation[2]/this->geometry->nos, 0.5) << std::endl; std::cerr << " ---------------- " <<
    // std::endl;
    // ==== DEBUG: end gradient comparison ====

} // end DipoleDipole
void Hamiltonian_Heisenberg::Gradient_DDI_Direct( const vectorfield & spins, vectorfield & gradient )
{
    scalar mult = C::mu_0 * C::mu_B * C::mu_B / ( 4 * C::Pi * 1e-30 );
    scalar d, d3, d5;
    Vector3 diff;
    Vector3 diff_img;

    int img_a = boundary_conditions[0] == 0 ? 0 : ddi_n_periodic_images[0];
    int img_b = boundary_conditions[1] == 0 ? 0 : ddi_n_periodic_images[1];
    int img_c = boundary_conditions[2] == 0 ? 0 : ddi_n_periodic_images[2];

    for( int idx1 = 0; idx1 < geometry->nos; idx1++ )
    {
        for( int idx2 = 0; idx2 < geometry->nos; idx2++ )
        {
            auto & m2 = spins[idx2];

            diff       = this->geometry->positions[idx2] - this->geometry->positions[idx1];
            scalar Dxx = 0, Dxy = 0, Dxz = 0, Dyy = 0, Dyz = 0, Dzz = 0;

            for( int a_pb = -img_a; a_pb <= img_a; a_pb++ )
            {
                for( int b_pb = -img_b; b_pb <= img_b; b_pb++ )
                {
                    for( int c_pb = -img_c; c_pb <= img_c; c_pb++ )
                    {
                        diff_img
                            = diff
                              + a_pb * geometry->n_cells[0] * geometry->bravais_vectors[0] * geometry->lattice_constant
                              + b_pb * geometry->n_cells[1] * geometry->bravais_vectors[1] * geometry->lattice_constant
                              + c_pb * geometry->n_cells[2] * geometry->bravais_vectors[2] * geometry->lattice_constant;
                        d = diff_img.norm();
                        if( d > 1e-10 )
                        {
                            d3 = d * d * d;
                            d5 = d * d * d * d * d;
                            Dxx += mult * ( 3 * diff_img[0] * diff_img[0] / d5 - 1 / d3 );
                            Dxy += mult * 3 * diff_img[0] * diff_img[1] / d5; // same as Dyx
                            Dxz += mult * 3 * diff_img[0] * diff_img[2] / d5; // same as Dzx
                            Dyy += mult * ( 3 * diff_img[1] * diff_img[1] / d5 - 1 / d3 );
                            Dyz += mult * 3 * diff_img[1] * diff_img[2] / d5; // same as Dzy
                            Dzz += mult * ( 3 * diff_img[2] * diff_img[2] / d5 - 1 / d3 );
                        }
                    }
                }
            }

            auto & mu = geometry->mu_s[idx2];

            gradient[idx1][0]
                -= ( Dxx * m2[0] + Dxy * m2[1] + Dxz * m2[2] ) * geometry->mu_s[idx1] * geometry->mu_s[idx2];
            gradient[idx1][1]
                -= ( Dxy * m2[0] + Dyy * m2[1] + Dyz * m2[2] ) * geometry->mu_s[idx1] * geometry->mu_s[idx2];
            gradient[idx1][2]
                -= ( Dxz * m2[0] + Dyz * m2[1] + Dzz * m2[2] ) * geometry->mu_s[idx1] * geometry->mu_s[idx2];
        }
    }
}

void Hamiltonian_Heisenberg::E_Quadruplet( const vectorfield & spins, scalarfield & Energy )
{
    for( unsigned int iquad = 0; iquad < quadruplets.size(); ++iquad )
    {
        const auto & quad = quadruplets[iquad];

        int i = quad.i;
        int j = quad.j;
        int k = quad.k;
        int l = quad.l;

        const auto & d_j = quad.d_j;
        const auto & d_k = quad.d_k;
        const auto & d_l = quad.d_l;

        for( int da = 0; da < geometry->n_cells[0]; ++da )
        {
            for( int db = 0; db < geometry->n_cells[1]; ++db )
            {
                for( int dc = 0; dc < geometry->n_cells[2]; ++dc )
                {
                    int ispin = i + idx_from_translations( geometry->n_cells, geometry->n_cell_atoms, { da, db, dc } );
                    int jspin = idx_from_pair(
                        ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                        { i, j, { d_j[0], d_j[1], d_j[2] } } );
                    int kspin = idx_from_pair(
                        ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                        { i, k, { d_k[0], d_k[1], d_k[2] } } );
                    int lspin = idx_from_pair(
                        ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                        { i, l, { d_l[0], d_l[1], d_l[2] } } );

                    if( ispin >= 0 && jspin >= 0 && kspin >= 0 && lspin >= 0 )
                    {
                        Energy[ispin] -= 0.25 * quadruplet_magnitudes[iquad] * ( spins[ispin].dot( spins[jspin] ) )
                                         * ( spins[kspin].dot( spins[lspin] ) );
                        Energy[jspin] -= 0.25 * quadruplet_magnitudes[iquad] * ( spins[ispin].dot( spins[jspin] ) )
                                         * ( spins[kspin].dot( spins[lspin] ) );
                        Energy[kspin] -= 0.25 * quadruplet_magnitudes[iquad] * ( spins[ispin].dot( spins[jspin] ) )
                                         * ( spins[kspin].dot( spins[lspin] ) );
                        Energy[lspin] -= 0.25 * quadruplet_magnitudes[iquad] * ( spins[ispin].dot( spins[jspin] ) )
                                         * ( spins[kspin].dot( spins[lspin] ) );
                    }
                }
            }
        }
    }
}

scalar Hamiltonian_Heisenberg::Energy_Single_Spin( int ispin, const vectorfield & spins )
{
    scalar Energy = 0;
    if( check_atom_type( this->geometry->atom_types[ispin] ) )
    {
        int icell   = ispin / this->geometry->n_cell_atoms;
        int ibasis  = ispin - icell * this->geometry->n_cell_atoms;
        auto & mu_s = this->geometry->mu_s;
        Pair pair_inv;

        // External field
        if( this->idx_zeeman >= 0 )
            Energy -= mu_s[ispin] * this->external_field_magnitude * this->external_field_normal.dot( spins[ispin] );

        // Anisotropy
        if( this->idx_anisotropy >= 0 )
        {
            for( int iani = 0; iani < anisotropy_indices.size(); ++iani )
            {
                if( anisotropy_indices[iani] == ibasis )
                {
                    if( check_atom_type( this->geometry->atom_types[ispin] ) )
                        Energy -= this->anisotropy_magnitudes[iani]
                                  * std::pow( anisotropy_normals[iani].dot( spins[ispin] ), 2.0 );
                }
            }
        }

        // Cubic Anisotropy
        if( this->idx_cubic_anisotropy >= 0 )
        {
            for( int iani = 0; iani < cubic_anisotropy_indices.size(); ++iani )
            {
                if( cubic_anisotropy_indices[iani] == ibasis )
                {
                    if( check_atom_type( this->geometry->atom_types[ispin] ) )
                        Energy -= this->cubic_anisotropy_magnitudes[iani] / 2
                                  * ( std::pow( spins[ispin][0], 4.0 ) + std::pow( spins[ispin][1], 4.0 )
                                      + std::pow( spins[ispin][2], 4.0 ) );
                }
            }
        }

        // Exchange
        if( this->idx_exchange >= 0 )
        {
            for( unsigned int ipair = 0; ipair < exchange_pairs.size(); ++ipair )
            {
                const auto & pair = exchange_pairs[ipair];
                if( pair.i == ibasis )
                {
                    int jspin = idx_from_pair(
                        ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                        pair );
                    if( jspin >= 0 )
                        Energy -= this->exchange_magnitudes[ipair] * spins[ispin].dot( spins[jspin] );
                }
            }
        }

        // DMI
        if( this->idx_dmi >= 0 )
        {
            for( unsigned int ipair = 0; ipair < dmi_pairs.size(); ++ipair )
            {
                const auto & pair = dmi_pairs[ipair];
                if( pair.i == ibasis )
                {
                    int jspin = idx_from_pair(
                        ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                        pair );
                    if( jspin >= 0 )
                        Energy -= this->dmi_magnitudes[ipair]
                                  * this->dmi_normals[ipair].dot( spins[ispin].cross( spins[jspin] ) );
                }
            }
        }

        // TODO: Quadruplets
        if( this->idx_quadruplet >= 0 )
        {
        }
    }
    return Energy;
}

void Hamiltonian_Heisenberg::Gradient( const vectorfield & spins, vectorfield & gradient )
{
    // Set to zero
    Vectormath::fill( gradient, { 0, 0, 0 } );

    // External field
    if( idx_zeeman >= 0 )
        this->Gradient_Zeeman( gradient );

    // Anisotropy
    if( idx_anisotropy >= 0 )
        this->Gradient_Anisotropy( spins, gradient );

    // Cubic_anisotropy
    if( idx_cubic_anisotropy >= 0 )
        this->Gradient_Cubic_Anisotropy( spins, gradient );

    //    Exchange
    if( idx_exchange >= 0 )
        this->Gradient_Exchange( spins, gradient );

    //    DMI
    if( idx_dmi >= 0 )
        this->Gradient_DMI( spins, gradient );

    //    DDI
    if( idx_ddi >= 0 )
        this->Gradient_DDI( spins, gradient );

    //    Quadruplet
    if( idx_quadruplet )
        this->Gradient_Quadruplet( spins, gradient );
}

void Hamiltonian_Heisenberg::Gradient_and_Energy( const vectorfield & spins, vectorfield & gradient, scalar & energy )
{
    // Set to zero
    Vectormath::fill( gradient, { 0, 0, 0 } );
    energy = 0;

    auto N    = spins.size();
    auto s    = spins.data();
    auto mu_s = geometry->mu_s.data();
    auto g    = gradient.data();

    // Anisotropy
    if( idx_anisotropy >= 0 )
        this->Gradient_Anisotropy( spins, gradient );

    // Exchange
    if( idx_exchange >= 0 )
        this->Gradient_Exchange( spins, gradient );

    // DMI
    if( idx_dmi >= 0 )
        this->Gradient_DMI( spins, gradient );

    // DDI
    if( idx_ddi >= 0 )
        this->Gradient_DDI( spins, gradient );

    energy += Backend::par::reduce( N, [s, g] SPIRIT_LAMBDA( int idx ) { return 0.5 * g[idx].dot( s[idx] ); } );

    // Cubic Anisotropy
    if( idx_cubic_anisotropy >= 0 )
    {
        this->Gradient_Cubic_Anisotropy( spins, gradient );

        scalarfield energy_cubic( N );
        this->E_Cubic_Anisotropy( spins, energy_cubic );
        energy += Vectormath::sum( energy_cubic );
    }

    // External field
    if( idx_zeeman >= 0 )
    {
        Vector3 ext_field = external_field_normal * external_field_magnitude;
        this->Gradient_Zeeman( gradient );
        energy += Backend::par::reduce(
            N, [s, ext_field, mu_s] SPIRIT_LAMBDA( int idx ) { return -mu_s[idx] * ext_field.dot( s[idx] ); } );
    }

    // Quadruplets
    if( idx_quadruplet > 0 )
    {
        // Kind of a bandaid fix
        this->Gradient_Quadruplet( spins, gradient );
        if( energy_contributions_per_spin[idx_quadruplet].second.size() != spins.size() )
        {
            energy_contributions_per_spin[idx_quadruplet].second.resize( spins.size() );
        };
        Vectormath::fill( energy_contributions_per_spin[idx_quadruplet].second, 0 );
        E_Quadruplet( spins, energy_contributions_per_spin[idx_quadruplet].second );
        energy += Vectormath::sum( energy_contributions_per_spin[idx_quadruplet].second );
    }
}

__global__ void CU_Gradient_Zeeman(
    const int * atom_types, const int n_cell_atoms, const scalar * mu_s, const scalar external_field_magnitude,
    const Vector3 external_field_normal, Vector3 * gradient, size_t n_cells_total )
{
    for( auto icell = blockIdx.x * blockDim.x + threadIdx.x; icell < n_cells_total; icell += blockDim.x * gridDim.x )
    {
        for( int ibasis = 0; ibasis < n_cell_atoms; ++ibasis )
        {
            int ispin = n_cell_atoms * icell + ibasis;
            if( cu_check_atom_type( atom_types[ispin] ) )
                gradient[ispin] -= mu_s[ispin] * external_field_magnitude * external_field_normal;
        }
    }
}
void Hamiltonian_Heisenberg::Gradient_Zeeman( vectorfield & gradient )
{
    int size = geometry->n_cells_total;
    CU_Gradient_Zeeman<<<( size + 1023 ) / 1024, 1024>>>(
        this->geometry->atom_types.data(), geometry->n_cell_atoms, geometry->mu_s.data(),
        this->external_field_magnitude, this->external_field_normal, gradient.data(), size );
    CU_CHECK_AND_SYNC();
}

__global__ void CU_Gradient_Anisotropy(
    const Vector3 * spins, const int * atom_types, const int n_cell_atoms, const int n_anisotropies,
    const int * anisotropy_indices, const scalar * anisotropy_magnitude, const Vector3 * anisotropy_normal,
    Vector3 * gradient, size_t n_cells_total )
{
    for( auto icell = blockIdx.x * blockDim.x + threadIdx.x; icell < n_cells_total; icell += blockDim.x * gridDim.x )
    {
        for( int iani = 0; iani < n_anisotropies; ++iani )
        {
            int ispin = icell * n_cell_atoms + anisotropy_indices[iani];
            if( cu_check_atom_type( atom_types[ispin] ) )
            {
                scalar sc = -2 * anisotropy_magnitude[iani] * anisotropy_normal[iani].dot( spins[ispin] );
                gradient[ispin] += sc * anisotropy_normal[iani];
            }
        }
    }
}
void Hamiltonian_Heisenberg::Gradient_Anisotropy( const vectorfield & spins, vectorfield & gradient )
{
    int size = geometry->n_cells_total;
    CU_Gradient_Anisotropy<<<( size + 1023 ) / 1024, 1024>>>(
        spins.data(), this->geometry->atom_types.data(), this->geometry->n_cell_atoms, this->anisotropy_indices.size(),
        this->anisotropy_indices.data(), this->anisotropy_magnitudes.data(), this->anisotropy_normals.data(),
        gradient.data(), size );
    CU_CHECK_AND_SYNC();
}

__global__ void CU_Gradient_Cubic_Anisotropy(
    const Vector3 * spins, const int * atom_types, const int n_cell_atoms, const int n_anisotropies,
    const int * anisotropy_indices, const scalar * anisotropy_magnitude, Vector3 * gradient, size_t n_cells_total )
{
    for( auto icell = blockIdx.x * blockDim.x + threadIdx.x; icell < n_cells_total; icell += blockDim.x * gridDim.x )
    {
        for( int iani = 0; iani < n_anisotropies; ++iani )
        {
            int ispin = icell * n_cell_atoms + anisotropy_indices[iani];
            if( cu_check_atom_type( atom_types[ispin] ) )
            {
                for( int icomp = 0; icomp < 3; ++icomp )
                {
                    gradient[ispin][icomp] -= 2.0 * anisotropy_magnitude[iani] * pow( spins[ispin][icomp], 3.0 );
                }
            }
        }
    }
}
void Hamiltonian_Heisenberg::Gradient_Cubic_Anisotropy( const vectorfield & spins, vectorfield & gradient )
{
    int size = geometry->n_cells_total;
    CU_Gradient_Cubic_Anisotropy<<<( size + 1023 ) / 1024, 1024>>>(
        spins.data(), this->geometry->atom_types.data(), this->geometry->n_cell_atoms,
        this->cubic_anisotropy_indices.size(), this->cubic_anisotropy_indices.data(),
        this->cubic_anisotropy_magnitudes.data(), gradient.data(), size );
    CU_CHECK_AND_SYNC();
}

__global__ void CU_Gradient_Exchange(
    const Vector3 * spins, const int * atom_types, const int * boundary_conditions, const int * n_cells,
    int n_cell_atoms, int n_pairs, const Pair * pairs, const scalar * magnitudes, Vector3 * gradient, size_t size )
{
    int bc[3] = { boundary_conditions[0], boundary_conditions[1], boundary_conditions[2] };
    int nc[3] = { n_cells[0], n_cells[1], n_cells[2] };

    for( auto icell = blockIdx.x * blockDim.x + threadIdx.x; icell < size; icell += blockDim.x * gridDim.x )
    {
        for( auto ipair = 0; ipair < n_pairs; ++ipair )
        {
            int ispin = pairs[ipair].i + icell * n_cell_atoms;
            int jspin = cu_idx_from_pair( ispin, bc, nc, n_cell_atoms, atom_types, pairs[ipair] );
            if( jspin >= 0 )
            {
                gradient[ispin] -= magnitudes[ipair] * spins[jspin];
            }
        }
    }
}
void Hamiltonian_Heisenberg::Gradient_Exchange( const vectorfield & spins, vectorfield & gradient )
{
    int size = geometry->n_cells_total;
    CU_Gradient_Exchange<<<( size + 1023 ) / 1024, 1024>>>(
        spins.data(), this->geometry->atom_types.data(), boundary_conditions.data(), geometry->n_cells.data(),
        geometry->n_cell_atoms, this->exchange_pairs.size(), this->exchange_pairs.data(),
        this->exchange_magnitudes.data(), gradient.data(), size );
    CU_CHECK_AND_SYNC();
}

__global__ void CU_Gradient_DMI(
    const Vector3 * spins, const int * atom_types, const int * boundary_conditions, const int * n_cells,
    int n_cell_atoms, int n_pairs, const Pair * pairs, const scalar * magnitudes, const Vector3 * normals,
    Vector3 * gradient, size_t size )
{
    int bc[3] = { boundary_conditions[0], boundary_conditions[1], boundary_conditions[2] };
    int nc[3] = { n_cells[0], n_cells[1], n_cells[2] };

    for( auto icell = blockIdx.x * blockDim.x + threadIdx.x; icell < size; icell += blockDim.x * gridDim.x )
    {
        for( auto ipair = 0; ipair < n_pairs; ++ipair )
        {
            int ispin = pairs[ipair].i + icell * n_cell_atoms;
            int jspin = cu_idx_from_pair( ispin, bc, nc, n_cell_atoms, atom_types, pairs[ipair] );
            if( jspin >= 0 )
            {
                gradient[ispin] -= magnitudes[ipair] * spins[jspin].cross( normals[ipair] );
            }
        }
    }
}
void Hamiltonian_Heisenberg::Gradient_DMI( const vectorfield & spins, vectorfield & gradient )
{
    int size = geometry->n_cells_total;
    CU_Gradient_DMI<<<( size + 1023 ) / 1024, 1024>>>(
        spins.data(), this->geometry->atom_types.data(), boundary_conditions.data(), geometry->n_cells.data(),
        geometry->n_cell_atoms, this->dmi_pairs.size(), this->dmi_pairs.data(), this->dmi_magnitudes.data(),
        this->dmi_normals.data(), gradient.data(), size );
    CU_CHECK_AND_SYNC();
}

void Hamiltonian_Heisenberg::Gradient_DDI( const vectorfield & spins, vectorfield & gradient )
{
    if( this->ddi_method == DDI_Method::FFT )
        this->Gradient_DDI_FFT( spins, gradient );
    else if( this->ddi_method == DDI_Method::Cutoff )
    {
        // TODO: Merge these implementations in the future
        if( this->ddi_cutoff_radius >= 0 )
            this->Gradient_DDI_Cutoff( spins, gradient );
        else
            this->Gradient_DDI_Direct( spins, gradient );
    }
}

void Hamiltonian_Heisenberg::Gradient_DDI_Cutoff( const vectorfield & spins, vectorfield & gradient )
{
    // TODO
}

__global__ void CU_FFT_Pointwise_Mult(
    FFT::FFT_cpx_type * ft_D_matrices, FFT::FFT_cpx_type * ft_spins, FFT::FFT_cpx_type * res_mult,
    int * iteration_bounds, int * inter_sublattice_lookup, FFT::StrideContainer dipole_stride,
    FFT::StrideContainer spin_stride )
{
    int nos = iteration_bounds[0] * iteration_bounds[1] * iteration_bounds[2] * iteration_bounds[3];
    int tupel[4];

    for( int ispin = blockIdx.x * blockDim.x + threadIdx.x; ispin < nos; ispin += blockDim.x * gridDim.x )
    {
        cu_tupel_from_idx( ispin, tupel, iteration_bounds, 4 ); // tupel now is {i_b1, a, b, c}
        int i_b1 = tupel[0], a = tupel[1], b = tupel[2], c = tupel[3];

        // Index to the first component of the spin (Remember: not the same as ispin, because we also have the spin
        // component stride!)
        int idx_b1 = i_b1 * spin_stride.basis + a * spin_stride.a + b * spin_stride.b + c * spin_stride.c;

        // Collect the intersublattice contributions
        FFT::FFT_cpx_type res_temp_x{ 0, 0 }, res_temp_y{ 0, 0 }, res_temp_z{ 0, 0 };
        for( int i_b2 = 0; i_b2 < iteration_bounds[0]; i_b2++ )
        {
            // Index to the first component of the second spin
            int idx_b2 = i_b2 * spin_stride.basis + a * spin_stride.a + b * spin_stride.b + c * spin_stride.c;

            int & b_inter = inter_sublattice_lookup[i_b1 + i_b2 * iteration_bounds[0]];
            // Index of the dipole matrix "connecting" the two spins
            int idx_d = b_inter * dipole_stride.basis + a * dipole_stride.a + b * dipole_stride.b + c * dipole_stride.c;

            // Fourier transformed components of the second spin
            auto & fs_x = ft_spins[idx_b2];
            auto & fs_y = ft_spins[idx_b2 + 1 * spin_stride.comp];
            auto & fs_z = ft_spins[idx_b2 + 2 * spin_stride.comp];

            // Fourier transformed components of the dipole matrix
            auto & fD_xx = ft_D_matrices[idx_d];
            auto & fD_xy = ft_D_matrices[idx_d + 1 * dipole_stride.comp];
            auto & fD_xz = ft_D_matrices[idx_d + 2 * dipole_stride.comp];
            auto & fD_yy = ft_D_matrices[idx_d + 3 * dipole_stride.comp];
            auto & fD_yz = ft_D_matrices[idx_d + 4 * dipole_stride.comp];
            auto & fD_zz = ft_D_matrices[idx_d + 5 * dipole_stride.comp];

            FFT::addTo( res_temp_x, FFT::mult3D( fD_xx, fD_xy, fD_xz, fs_x, fs_y, fs_z ) );
            FFT::addTo( res_temp_y, FFT::mult3D( fD_xy, fD_yy, fD_yz, fs_x, fs_y, fs_z ) );
            FFT::addTo( res_temp_z, FFT::mult3D( fD_xz, fD_yz, fD_zz, fs_x, fs_y, fs_z ) );
        }
        // Add the temporary result
        FFT::addTo( res_mult[idx_b1 + 0 * spin_stride.comp], res_temp_x, true );
        FFT::addTo( res_mult[idx_b1 + 1 * spin_stride.comp], res_temp_y, true );
        FFT::addTo( res_mult[idx_b1 + 2 * spin_stride.comp], res_temp_z, true );
    }
}

__global__ void CU_Write_FFT_Gradients(
    FFT::FFT_real_type * resiFFT, Vector3 * gradient, FFT::StrideContainer spin_stride, int * iteration_bounds,
    int n_cell_atoms, scalar * mu_s, int sublattice_size )
{
    int nos = iteration_bounds[0] * iteration_bounds[1] * iteration_bounds[2] * iteration_bounds[3];
    int tupel[4];
    int idx_pad;

    for( int idx_orig = blockIdx.x * blockDim.x + threadIdx.x; idx_orig < nos; idx_orig += blockDim.x * gridDim.x )
    {
        cu_tupel_from_idx( idx_orig, tupel, iteration_bounds, 4 ); // tupel now is {ib, a, b, c}
        idx_pad = tupel[0] * spin_stride.basis + tupel[1] * spin_stride.a + tupel[2] * spin_stride.b
                  + tupel[3] * spin_stride.c;
        gradient[idx_orig][0] -= mu_s[idx_orig] * resiFFT[idx_pad] / sublattice_size;
        gradient[idx_orig][1] -= mu_s[idx_orig] * resiFFT[idx_pad + 1 * spin_stride.comp] / sublattice_size;
        gradient[idx_orig][2] -= mu_s[idx_orig] * resiFFT[idx_pad + 2 * spin_stride.comp] / sublattice_size;
    }
}

void Hamiltonian_Heisenberg::Gradient_DDI_FFT( const vectorfield & spins, vectorfield & gradient )
{
    auto & ft_D_matrices = transformed_dipole_matrices;

    auto & ft_spins = fft_plan_spins.cpx_ptr;

    auto & res_iFFT = fft_plan_reverse.real_ptr;
    auto & res_mult = fft_plan_reverse.cpx_ptr;

    int number_of_mults = it_bounds_pointwise_mult[0] * it_bounds_pointwise_mult[1] * it_bounds_pointwise_mult[2]
                          * it_bounds_pointwise_mult[3];

    FFT_Spins( spins );

    // TODO: also parallelize over i_b1
    // Loop over basis atoms (i.e sublattices) and add contribution of each sublattice
    CU_FFT_Pointwise_Mult<<<( spins.size() + 1023 ) / 1024, 1024>>>(
        ft_D_matrices.data(), ft_spins.data(), res_mult.data(), it_bounds_pointwise_mult.data(),
        inter_sublattice_lookup.data(), dipole_stride, spin_stride );
    // cudaDeviceSynchronize();
    // std::cerr << "\n\n>>>>>>>>>>>  Pointwise_Mult       <<<<<<<<<\n";
    // for( int i = 0; i < 10; i++ )
    //     std::cout << ( res_mult[i].x ) << " " << ( res_mult[i].y ) << " ";
    // std::cerr << "\n>=====================================<\n\n";

    FFT::batch_iFour_3D( fft_plan_reverse );

    CU_Write_FFT_Gradients<<<( geometry->nos + 1023 ) / 1024, 1024>>>(
        res_iFFT.data(), gradient.data(), spin_stride, it_bounds_write_gradients.data(), geometry->n_cell_atoms,
        geometry->mu_s.data(), sublattice_size );
} // end Field_DipoleDipole

void Hamiltonian_Heisenberg::Gradient_Quadruplet( const vectorfield & spins, vectorfield & gradient )
{
    for( unsigned int iquad = 0; iquad < quadruplets.size(); ++iquad )
    {
        const auto & quad = quadruplets[iquad];

        int i = quad.i;
        int j = quad.j;
        int k = quad.k;
        int l = quad.l;

        const auto & d_j = quad.d_j;
        const auto & d_k = quad.d_k;
        const auto & d_l = quad.d_l;

        for( int da = 0; da < geometry->n_cells[0]; ++da )
        {
            for( int db = 0; db < geometry->n_cells[1]; ++db )
            {
                for( int dc = 0; dc < geometry->n_cells[2]; ++dc )
                {
                    int ispin = i + idx_from_translations( geometry->n_cells, geometry->n_cell_atoms, { da, db, dc } );
                    int jspin = idx_from_pair(
                        ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                        { i, j, { d_j[0], d_j[1], d_j[2] } } );
                    int kspin = idx_from_pair(
                        ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                        { i, k, { d_k[0], d_k[1], d_k[2] } } );
                    int lspin = idx_from_pair(
                        ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                        { i, l, { d_l[0], d_l[1], d_l[2] } } );

                    if( ispin >= 0 && jspin >= 0 && kspin >= 0 && lspin >= 0 )
                    {
                        gradient[ispin]
                            -= quadruplet_magnitudes[iquad] * spins[jspin] * ( spins[kspin].dot( spins[lspin] ) );
                        gradient[jspin]
                            -= quadruplet_magnitudes[iquad] * spins[ispin] * ( spins[kspin].dot( spins[lspin] ) );
                        gradient[kspin]
                            -= quadruplet_magnitudes[iquad] * ( spins[ispin].dot( spins[jspin] ) ) * spins[lspin];
                        gradient[lspin]
                            -= quadruplet_magnitudes[iquad] * ( spins[ispin].dot( spins[jspin] ) ) * spins[kspin];
                    }
                }
            }
        }
    }
}

void Hamiltonian_Heisenberg::Hessian( const vectorfield & spins, MatrixX & hessian )
{
    int nos     = spins.size();
    const int N = geometry->n_cell_atoms;

    // --- Set to zero
    hessian.setZero();

    // --- Single Spin elements
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( int iani = 0; iani < anisotropy_indices.size(); ++iani )
        {
            int ispin = icell * N + anisotropy_indices[iani];
            if( check_atom_type( this->geometry->atom_types[ispin] ) )
            {
                for( int alpha = 0; alpha < 3; ++alpha )
                {
                    for( int beta = 0; beta < 3; ++beta )
                    {
                        int i = 3 * ispin + alpha;
                        int j = 3 * ispin + alpha;
                        hessian( i, j ) += -2.0 * this->anisotropy_magnitudes[iani]
                                           * this->anisotropy_normals[iani][alpha]
                                           * this->anisotropy_normals[iani][beta];
                    }
                }
            }
        }
    }

    // --- Spin Pair elements
    // Exchange
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( unsigned int i_pair = 0; i_pair < exchange_pairs.size(); ++i_pair )
        {
            int ispin = exchange_pairs[i_pair].i + icell * geometry->n_cell_atoms;
            int jspin = idx_from_pair(
                ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                exchange_pairs[i_pair] );
            if( jspin >= 0 )
            {
                for( int alpha = 0; alpha < 3; ++alpha )
                {
                    int i = 3 * ispin + alpha;
                    int j = 3 * jspin + alpha;

                    hessian( i, j ) += -exchange_magnitudes[i_pair];
                }
            }
        }
    }

    // DMI
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( unsigned int i_pair = 0; i_pair < dmi_pairs.size(); ++i_pair )
        {
            int ispin = dmi_pairs[i_pair].i + icell * geometry->n_cell_atoms;
            int jspin = idx_from_pair(
                ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                dmi_pairs[i_pair] );
            if( jspin >= 0 )
            {
                int i = 3 * ispin;
                int j = 3 * jspin;

                hessian( i + 2, j + 1 ) += dmi_magnitudes[i_pair] * dmi_normals[i_pair][0];
                hessian( i + 1, j + 2 ) += -dmi_magnitudes[i_pair] * dmi_normals[i_pair][0];
                hessian( i + 0, j + 2 ) += dmi_magnitudes[i_pair] * dmi_normals[i_pair][1];
                hessian( i + 2, j + 0 ) += -dmi_magnitudes[i_pair] * dmi_normals[i_pair][1];
                hessian( i + 1, j + 0 ) += dmi_magnitudes[i_pair] * dmi_normals[i_pair][2];
                hessian( i + 0, j + 1 ) += -dmi_magnitudes[i_pair] * dmi_normals[i_pair][2];
            }
        }
    }

    // Tentative Dipole-Dipole (only works for open boundary conditions)
    if( ddi_method != DDI_Method::None )
    {
        scalar mult = C::mu_0 * C::mu_B * C::mu_B / ( 4 * C::Pi * 1e-30 );
        for( int idx1 = 0; idx1 < geometry->nos; idx1++ )
        {
            for( int idx2 = 0; idx2 < geometry->nos; idx2++ )
            {
                auto diff  = this->geometry->positions[idx2] - this->geometry->positions[idx1];
                scalar d   = diff.norm(), d3, d5;
                scalar Dxx = 0, Dxy = 0, Dxz = 0, Dyy = 0, Dyz = 0, Dzz = 0;
                if( d > 1e-10 )
                {
                    d3 = d * d * d;
                    d5 = d * d * d * d * d;
                    Dxx += mult * ( 3 * diff[0] * diff[0] / d5 - 1 / d3 );
                    Dxy += mult * 3 * diff[0] * diff[1] / d5; // same as Dyx
                    Dxz += mult * 3 * diff[0] * diff[2] / d5; // same as Dzx
                    Dyy += mult * ( 3 * diff[1] * diff[1] / d5 - 1 / d3 );
                    Dyz += mult * 3 * diff[1] * diff[2] / d5; // same as Dzy
                    Dzz += mult * ( 3 * diff[2] * diff[2] / d5 - 1 / d3 );
                }

                int i = 3 * idx1;
                int j = 3 * idx2;

                hessian( i + 0, j + 0 ) += -geometry->mu_s[idx1] * geometry->mu_s[idx2] * ( Dxx );
                hessian( i + 1, j + 0 ) += -geometry->mu_s[idx1] * geometry->mu_s[idx2] * ( Dxy );
                hessian( i + 2, j + 0 ) += -geometry->mu_s[idx1] * geometry->mu_s[idx2] * ( Dxz );
                hessian( i + 0, j + 1 ) += -geometry->mu_s[idx1] * geometry->mu_s[idx2] * ( Dxy );
                hessian( i + 1, j + 1 ) += -geometry->mu_s[idx1] * geometry->mu_s[idx2] * ( Dyy );
                hessian( i + 2, j + 1 ) += -geometry->mu_s[idx1] * geometry->mu_s[idx2] * ( Dyz );
                hessian( i + 0, j + 2 ) += -geometry->mu_s[idx1] * geometry->mu_s[idx2] * ( Dxz );
                hessian( i + 1, j + 2 ) += -geometry->mu_s[idx1] * geometry->mu_s[idx2] * ( Dyz );
                hessian( i + 2, j + 2 ) += -geometry->mu_s[idx1] * geometry->mu_s[idx2] * ( Dzz );
            }
        }
    }

    // // TODO: Dipole-Dipole
    // for (unsigned int i_pair = 0; i_pair < this->DD_indices.size(); ++i_pair)
    // {
    //     // indices
    //     int idx_1 = DD_indices[i_pair][0];
    //     int idx_2 = DD_indices[i_pair][1];
    //     // prefactor
    //     scalar prefactor = 0.0536814951168
    //         * mu_s[idx_1] * mu_s[idx_2]
    //         / std::pow(DD_magnitude[i_pair], 3);
    //     // components
    //     for (int alpha = 0; alpha < 3; ++alpha)
    //     {
    //         for (int beta = 0; beta < 3; ++beta)
    //         {
    //             int idx_h = idx_1 + alpha*nos + 3 * nos*(idx_2 + beta*nos);
    //             if (alpha == beta)
    //                 hessian[idx_h] += prefactor;
    //             hessian[idx_h] += -3.0*prefactor*DD_normal[i_pair][alpha] * DD_normal[i_pair][beta];
    //         }
    //     }
    // }

    // TODO: Quadruplets
}

void Hamiltonian_Heisenberg::Sparse_Hessian( const vectorfield & spins, SpMatrixX & hessian )
{
    int nos     = spins.size();
    const int N = geometry->n_cell_atoms;

    typedef Eigen::Triplet<scalar> T;
    std::vector<T> tripletList;
    tripletList.reserve(
        geometry->n_cells_total
        * ( anisotropy_indices.size() * 9 + exchange_pairs.size() * 2 + dmi_pairs.size() * 3 ) );

    // --- Single Spin elements
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( int iani = 0; iani < anisotropy_indices.size(); ++iani )
        {
            int ispin = icell * N + anisotropy_indices[iani];
            if( check_atom_type( this->geometry->atom_types[ispin] ) )
            {
                for( int alpha = 0; alpha < 3; ++alpha )
                {
                    for( int beta = 0; beta < 3; ++beta )
                    {
                        int i      = 3 * ispin + alpha;
                        int j      = 3 * ispin + alpha;
                        scalar res = -2.0 * this->anisotropy_magnitudes[iani] * this->anisotropy_normals[iani][alpha]
                                     * this->anisotropy_normals[iani][beta];
                        tripletList.push_back( T( i, j, res ) );
                    }
                }
            }
        }
    }

    // --- Spin Pair elements
    // Exchange
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( unsigned int i_pair = 0; i_pair < exchange_pairs.size(); ++i_pair )
        {
            int ispin = exchange_pairs[i_pair].i + icell * geometry->n_cell_atoms;
            int jspin = idx_from_pair(
                ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                exchange_pairs[i_pair] );
            if( jspin >= 0 )
            {
                for( int alpha = 0; alpha < 3; ++alpha )
                {
                    int i = 3 * ispin + alpha;
                    int j = 3 * jspin + alpha;

                    tripletList.push_back( T( i, j, -exchange_magnitudes[i_pair] ) );
#ifndef SPIRIT_USE_OPENMP
                    tripletList.push_back( T( j, i, -exchange_magnitudes[i_pair] ) );
#endif
                }
            }
        }
    }

    // DMI
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( unsigned int i_pair = 0; i_pair < dmi_pairs.size(); ++i_pair )
        {
            int ispin = dmi_pairs[i_pair].i + icell * geometry->n_cell_atoms;
            int jspin = idx_from_pair(
                ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                dmi_pairs[i_pair] );
            if( jspin >= 0 )
            {
                int i = 3 * ispin;
                int j = 3 * jspin;

                tripletList.push_back( T( i + 2, j + 1, dmi_magnitudes[i_pair] * dmi_normals[i_pair][0] ) );
                tripletList.push_back( T( i + 1, j + 2, -dmi_magnitudes[i_pair] * dmi_normals[i_pair][0] ) );
                tripletList.push_back( T( i + 0, j + 2, dmi_magnitudes[i_pair] * dmi_normals[i_pair][1] ) );
                tripletList.push_back( T( i + 2, j + 0, -dmi_magnitudes[i_pair] * dmi_normals[i_pair][1] ) );
                tripletList.push_back( T( i + 1, j + 0, dmi_magnitudes[i_pair] * dmi_normals[i_pair][2] ) );
                tripletList.push_back( T( i + 0, j + 1, -dmi_magnitudes[i_pair] * dmi_normals[i_pair][2] ) );

#ifndef SPIRIT_USE_OPENMP
                tripletList.push_back( T( j + 1, i + 2, dmi_magnitudes[i_pair] * dmi_normals[i_pair][0] ) );
                tripletList.push_back( T( j + 2, i + 1, -dmi_magnitudes[i_pair] * dmi_normals[i_pair][0] ) );
                tripletList.push_back( T( j + 2, i + 0, dmi_magnitudes[i_pair] * dmi_normals[i_pair][1] ) );
                tripletList.push_back( T( j + 0, i + 2, -dmi_magnitudes[i_pair] * dmi_normals[i_pair][1] ) );
                tripletList.push_back( T( j + 0, i + 1, dmi_magnitudes[i_pair] * dmi_normals[i_pair][2] ) );
                tripletList.push_back( T( j + 1, i + 0, -dmi_magnitudes[i_pair] * dmi_normals[i_pair][2] ) );
#endif
            }
        }
    }

    hessian.setFromTriplets( tripletList.begin(), tripletList.end() );
}

__global__ void CU_Write_FFT_Spin_Input(
    FFT::FFT_real_type * fft_spin_inputs, const Vector3 * spins, int * iteration_bounds,
    FFT::StrideContainer spin_stride, scalar * mu_s )
{
    int nos = iteration_bounds[0] * iteration_bounds[1] * iteration_bounds[2] * iteration_bounds[3];
    int tupel[4];
    int idx_pad;
    for( int idx_orig = blockIdx.x * blockDim.x + threadIdx.x; idx_orig < nos; idx_orig += blockDim.x * gridDim.x )
    {
        cu_tupel_from_idx( idx_orig, tupel, iteration_bounds, 4 ); // tupel now is {ib, a, b, c}
        idx_pad = tupel[0] * spin_stride.basis + tupel[1] * spin_stride.a + tupel[2] * spin_stride.b
                  + tupel[3] * spin_stride.c;
        fft_spin_inputs[idx_pad]                        = spins[idx_orig][0] * mu_s[idx_orig];
        fft_spin_inputs[idx_pad + 1 * spin_stride.comp] = spins[idx_orig][1] * mu_s[idx_orig];
        fft_spin_inputs[idx_pad + 2 * spin_stride.comp] = spins[idx_orig][2] * mu_s[idx_orig];
    }
}

void Hamiltonian_Heisenberg::FFT_Spins( const vectorfield & spins )
{
    CU_Write_FFT_Spin_Input<<<( geometry->nos + 1023 ) / 1024, 1024>>>(
        fft_plan_spins.real_ptr.data(), spins.data(), it_bounds_write_spins.data(), spin_stride,
        geometry->mu_s.data() );
    FFT::batch_Four_3D( fft_plan_spins );
}

__global__ void CU_Write_FFT_Dipole_Input(
    FFT::FFT_real_type * fft_dipole_inputs, int * iteration_bounds, const Vector3 * translation_vectors,
    int n_cell_atoms, Vector3 * cell_atom_translations, int * n_cells, int * inter_sublattice_lookup, int * img,
    FFT::StrideContainer dipole_stride )
{
    int tupel[3];
    int sublattice_size = iteration_bounds[0] * iteration_bounds[1] * iteration_bounds[2];
    // Prefactor of ddi interaction
    scalar mult = C::mu_0 * C::mu_B * C::mu_B / ( 4 * C::Pi * 1e-30 );
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < sublattice_size; i += blockDim.x * gridDim.x )
    {
        cu_tupel_from_idx( i, tupel, iteration_bounds, 3 ); // tupel now is {a, b, c}
        auto & a    = tupel[0];
        auto & b    = tupel[1];
        auto & c    = tupel[2];
        int b_inter = -1;
        for( int i_b1 = 0; i_b1 < n_cell_atoms; ++i_b1 )
        {
            for( int i_b2 = 0; i_b2 < n_cell_atoms; ++i_b2 )
            {
                if( i_b1 != i_b2 || i_b1 == 0 )
                {
                    b_inter++;
                    inter_sublattice_lookup[i_b1 + i_b2 * n_cell_atoms] = b_inter;

                    int a_idx  = a < n_cells[0] ? a : a - iteration_bounds[0];
                    int b_idx  = b < n_cells[1] ? b : b - iteration_bounds[1];
                    int c_idx  = c < n_cells[2] ? c : c - iteration_bounds[2];
                    scalar Dxx = 0, Dxy = 0, Dxz = 0, Dyy = 0, Dyz = 0, Dzz = 0;

                    Vector3 diff;

                    // Iterate over periodic images
                    for( int a_pb = -img[0]; a_pb <= img[0]; a_pb++ )
                    {
                        for( int b_pb = -img[1]; b_pb <= img[1]; b_pb++ )
                        {
                            for( int c_pb = -img[2]; c_pb <= img[2]; c_pb++ )
                            {
                                diff = ( a_idx + a_pb * n_cells[0] ) * translation_vectors[0]
                                       + ( b_idx + b_pb * n_cells[1] ) * translation_vectors[1]
                                       + ( c_idx + c_pb * n_cells[2] ) * translation_vectors[2]
                                       + cell_atom_translations[i_b1] - cell_atom_translations[i_b2];

                                if( diff.norm() > 1e-10 )
                                {
                                    auto d  = diff.norm();
                                    auto d3 = d * d * d;
                                    auto d5 = d * d * d * d * d;
                                    Dxx += mult * ( 3 * diff[0] * diff[0] / d5 - 1 / d3 );
                                    Dxy += mult * 3 * diff[0] * diff[1] / d5; // same as Dyx
                                    Dxz += mult * 3 * diff[0] * diff[2] / d5; // same as Dzx
                                    Dyy += mult * ( 3 * diff[1] * diff[1] / d5 - 1 / d3 );
                                    Dyz += mult * 3 * diff[1] * diff[2] / d5; // same as Dzy
                                    Dzz += mult * ( 3 * diff[2] * diff[2] / d5 - 1 / d3 );
                                }
                            }
                        }
                    }

                    int idx = b_inter * dipole_stride.basis + a * dipole_stride.a + b * dipole_stride.b
                              + c * dipole_stride.c;
                    fft_dipole_inputs[idx]                          = Dxx;
                    fft_dipole_inputs[idx + 1 * dipole_stride.comp] = Dxy;
                    fft_dipole_inputs[idx + 2 * dipole_stride.comp] = Dxz;
                    fft_dipole_inputs[idx + 3 * dipole_stride.comp] = Dyy;
                    fft_dipole_inputs[idx + 4 * dipole_stride.comp] = Dyz;
                    fft_dipole_inputs[idx + 5 * dipole_stride.comp] = Dzz;
                }
                else
                {
                    inter_sublattice_lookup[i_b1 + i_b2 * n_cell_atoms] = 0;
                }
            }
        }
    }
}

void Hamiltonian_Heisenberg::FFT_Dipole_Matrices( FFT::FFT_Plan & fft_plan_dipole, int img_a, int img_b, int img_c )
{
    auto & fft_dipole_inputs = fft_plan_dipole.real_ptr;

    field<int> img = { img_a, img_b, img_c };

    // Work around to make bravais vectors and cell_atoms available to GPU as they are currently saves as std::vectors
    // and not fields ...
    auto translation_vectors    = field<Vector3>();
    auto cell_atom_translations = field<Vector3>();

    for( int i = 0; i < 3; i++ )
        translation_vectors.push_back( geometry->lattice_constant * geometry->bravais_vectors[i] );

    for( int i = 0; i < geometry->n_cell_atoms; i++ )
        cell_atom_translations.push_back( geometry->positions[i] );

    CU_Write_FFT_Dipole_Input<<<( sublattice_size + 1023 ) / 1024, 1024>>>(
        fft_dipole_inputs.data(), it_bounds_write_dipole.data(), translation_vectors.data(), geometry->n_cell_atoms,
        cell_atom_translations.data(), geometry->n_cells.data(), inter_sublattice_lookup.data(), img.data(),
        dipole_stride );
    FFT::batch_Four_3D( fft_plan_dipole );
}

void Hamiltonian_Heisenberg::Prepare_DDI()
{
    Clean_DDI();

    if( ddi_method != DDI_Method::FFT )
        return;

    // We perform zero-padding in a lattice direction if the dimension of the system is greater than 1 *and*
    //  - the boundary conditions are open, or
    //  - the boundary conditions are periodic and zero-padding is explicitly requested
    n_cells_padded.resize( 3 );
    for( int i = 0; i < 3; i++ )
    {
        n_cells_padded[i]         = geometry->n_cells[i];
        bool perform_zero_padding = geometry->n_cells[i] > 1 && ( boundary_conditions[i] == 0 || ddi_pb_zero_padding );
        if( perform_zero_padding )
            n_cells_padded[i] *= 2;
    }
    sublattice_size = n_cells_padded[0] * n_cells_padded[1] * n_cells_padded[2];

    inter_sublattice_lookup.resize( geometry->n_cell_atoms * geometry->n_cell_atoms );

    // We dont need to transform over length 1 dims
    std::vector<int> fft_dims;
    for( int i = 2; i >= 0; i-- ) // Notice that reverse order is important!
    {
        if( n_cells_padded[i] > 1 )
            fft_dims.push_back( n_cells_padded[i] );
    }

    // Count how many distinct inter-lattice contributions we need to store
    n_inter_sublattice = 0;
    for( int i = 0; i < geometry->n_cell_atoms; i++ )
    {
        for( int j = 0; j < geometry->n_cell_atoms; j++ )
        {
            if( i != 0 && i == j )
                continue;
            n_inter_sublattice++;
        }
    }

    // Set the iteration bounds for the nested for loops that are flattened in the kernels
    it_bounds_write_spins
        = { geometry->n_cell_atoms, geometry->n_cells[0], geometry->n_cells[1], geometry->n_cells[2] };

    it_bounds_write_dipole = { n_cells_padded[0], n_cells_padded[1], n_cells_padded[2] };

    it_bounds_pointwise_mult = { geometry->n_cell_atoms,
                                 ( n_cells_padded[0] / 2 + 1 ), // due to redundancy in real fft
                                 n_cells_padded[1], n_cells_padded[2] };

    it_bounds_write_gradients
        = { geometry->n_cell_atoms, geometry->n_cells[0], geometry->n_cells[1], geometry->n_cells[2] };

    FFT::FFT_Plan fft_plan_dipole = FFT::FFT_Plan( fft_dims, false, 6 * n_inter_sublattice, sublattice_size );
    fft_plan_spins                = FFT::FFT_Plan( fft_dims, false, 3 * geometry->n_cell_atoms, sublattice_size );
    fft_plan_reverse              = FFT::FFT_Plan( fft_dims, true, 3 * geometry->n_cell_atoms, sublattice_size );

    field<int *> temp_s = { &spin_stride.comp, &spin_stride.basis, &spin_stride.a, &spin_stride.b, &spin_stride.c };
    field<int *> temp_d
        = { &dipole_stride.comp, &dipole_stride.basis, &dipole_stride.a, &dipole_stride.b, &dipole_stride.c };
    ;
    FFT::get_strides(
        temp_s, { 3, this->geometry->n_cell_atoms, n_cells_padded[0], n_cells_padded[1], n_cells_padded[2] } );
    FFT::get_strides( temp_d, { 6, n_inter_sublattice, n_cells_padded[0], n_cells_padded[1], n_cells_padded[2] } );

    // Perform FFT of dipole matrices
    int img_a = boundary_conditions[0] == 0 ? 0 : ddi_n_periodic_images[0];
    int img_b = boundary_conditions[1] == 0 ? 0 : ddi_n_periodic_images[1];
    int img_c = boundary_conditions[2] == 0 ? 0 : ddi_n_periodic_images[2];

    FFT_Dipole_Matrices( fft_plan_dipole, img_a, img_b, img_c );
    transformed_dipole_matrices = std::move( fft_plan_dipole.cpx_ptr );
    if( save_dipole_matrices )
    {
        dipole_matrices = std::move( fft_plan_dipole.real_ptr );
    }
} // End prepare

void Hamiltonian_Heisenberg::Clean_DDI()
{
    fft_plan_spins   = FFT::FFT_Plan();
    fft_plan_reverse = FFT::FFT_Plan();
}

// Hamiltonian name as string
static const std::string name = "Heisenberg";
const std::string & Hamiltonian_Heisenberg::Name() const
{
    return name;
}
} // namespace Engine

#endif
