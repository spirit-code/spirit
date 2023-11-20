#include <data/Spin_System.hpp>
#include <engine/Backend_par.hpp>
#include <engine/Hamiltonian.hpp>
#include <engine/Indexing.hpp>
#include <engine/Neighbours.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Constants.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#ifndef SPIRIT_USE_CUDA
#include <algorithm>
#else
#include <complex> // TODO: check if I need complex for the CUDA implementation
#endif

using namespace Data;
using namespace Utility;
namespace C = Utility::Constants;
using Engine::Indexing::check_atom_type;
using Engine::Indexing::idx_from_pair;
using Engine::Indexing::idx_from_translations;

namespace Engine
{

Hamiltonian::Hamiltonian( std::shared_ptr<Geometry> geometry, intfield boundary_conditions )
        : geometry( std::move( geometry ) ),
          boundary_conditions( std::move( boundary_conditions ) ),
          name_update_paused( false ),
          hamiltonian_class( HAMILTONIAN_CLASS::GENERIC ),
          spin_stride(),
          dipole_stride()
{
    // legacy block
    external_field_magnitude    = 0;
    external_field_normal       = { 0, 0, 1 };
    anisotropy_indices          = intfield( 0 );
    anisotropy_magnitudes       = scalarfield( 0 );
    anisotropy_normals          = vectorfield( 0 );
    cubic_anisotropy_indices    = intfield( 0 );
    cubic_anisotropy_magnitudes = scalarfield( 0 );
    exchange_shell_magnitudes   = scalarfield( 0 );
    exchange_pairs_in           = pairfield( 0 );
    exchange_magnitudes_in      = scalarfield( 0 );
    dmi_shell_magnitudes        = scalarfield( 0 );
    dmi_shell_chirality         = 0;
    dmi_pairs_in                = pairfield( 0 );
    dmi_magnitudes_in           = scalarfield( 0 );
    dmi_normals_in              = vectorfield( 0 );
    quadruplets                 = quadrupletfield( 0 );
    quadruplet_magnitudes       = scalarfield( 0 );
    fft_plan_spins              = FFT::FFT_Plan();
    fft_plan_reverse            = FFT::FFT_Plan();

    ddi_method                  = DDI_Method::None;
    ddi_n_periodic_images       = intfield( 0 );
    ddi_pb_zero_padding         = false;
    ddi_cutoff_radius           = 0;
    n_inter_sublattice          = 0;
    sublattice_size             = 0;
    n_cells_padded              = field<int>( 0 );
    transformed_dipole_matrices = field<FFT::FFT_cpx_type>( 0 );
    dipole_matrices             = field<FFT::FFT_real_type>( 0 );
    save_dipole_matrices        = false;
    it_bounds_pointwise_mult    = field<int>( 0 );
    it_bounds_write_gradients   = field<int>( 0 );
    it_bounds_write_spins       = field<int>( 0 );
    it_bounds_write_dipole      = field<int>( 0 );

    // init to 0, because initializing to -1 can fail silently
    energy_contributions_per_spin = Data::vectorlabeled<scalarfield>( 0 );
    idx_zeeman                    = 0;
    idx_exchange                  = 0;
    idx_dmi                       = 0;
    idx_anisotropy                = 0;
    idx_cubic_anisotropy          = 0;
    idx_ddi                       = 0;
    idx_quadruplet                = 0;

    prng             = std::mt19937( 94199188 );
    distribution_int = std::uniform_int_distribution<int>( 0, 1 );
    this->updateInteractions(); // should be a nop, but has to be here semantically
    this->updateName();
}

// Construct a Heisenberg Hamiltonian with pairs
Hamiltonian::Hamiltonian(
    std::shared_ptr<Data::Geometry> geometry, const intfield & boundary_conditions, const NormalVector & external_field,
    const VectorfieldData & anisotropy, const ScalarfieldData & cubic_anisotropy, const ScalarPairfieldData & exchange,
    const VectorPairfieldData & dmi, const QuadrupletfieldData & quadruplet, const DDI_Method ddi_method,
    const DDI_Data & ddi_data )
        : geometry( std::move( geometry ) ),
          boundary_conditions( std::move( boundary_conditions ) ),
          interactions( 0 ),
          active_interactions_size( 0 ),
          common_interactions_size( 0 ),
          name_update_paused( false ),
          hamiltonian_class( HAMILTONIAN_CLASS::GENERIC ),
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
          fft_plan_spins( FFT::FFT_Plan() ),
          fft_plan_reverse( FFT::FFT_Plan() ),
          spin_stride(),
          dipole_stride()
{
    n_inter_sublattice          = 0;
    sublattice_size             = 0;
    n_cells_padded              = field<int>( 0 );
    transformed_dipole_matrices = field<FFT::FFT_cpx_type>( 0 );
    dipole_matrices             = field<FFT::FFT_real_type>( 0 );
    save_dipole_matrices        = false;

    // init to 0, because initializing to -1 can fail silently
    energy_contributions_per_spin = decltype( energy_contributions_per_spin )( 0 );
    idx_zeeman                    = 0;
    idx_exchange                  = 0;
    idx_dmi                       = 0;
    idx_anisotropy                = 0;
    idx_cubic_anisotropy          = 0;
    idx_ddi                       = 0;
    idx_quadruplet                = 0;

    it_bounds_pointwise_mult  = field<int>( 0 );
    it_bounds_write_gradients = field<int>( 0 );
    it_bounds_write_spins     = field<int>( 0 );
    it_bounds_write_dipole    = field<int>( 0 );

    // Generate interaction pairs, constants etc.
    this->updateInteractions();
    this->updateName();
}

// Construct a Heisenberg Hamiltonian from shells
Hamiltonian::Hamiltonian(
    std::shared_ptr<Data::Geometry> geometry, const intfield & boundary_conditions, const NormalVector & external_field,
    const VectorfieldData & anisotropy, const ScalarfieldData & cubic_anisotropy,
    const scalarfield & exchange_shell_magnitudes, const scalarfield & dmi_shell_magnitudes, int dm_chirality,
    const QuadrupletfieldData & quadruplet, const DDI_Method ddi_method, const DDI_Data & ddi_data )
        : geometry( std::move( geometry ) ),
          boundary_conditions( std::move( boundary_conditions ) ),
          interactions( 0 ),
          active_interactions_size( 0 ),
          common_interactions_size( 0 ),
          name_update_paused( false ),
          hamiltonian_class( HAMILTONIAN_CLASS::GENERIC ),
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
          fft_plan_spins( FFT::FFT_Plan() ),
          fft_plan_reverse( FFT::FFT_Plan() ),
          spin_stride(),
          dipole_stride()
{
    n_inter_sublattice          = 0;
    sublattice_size             = 0;
    n_cells_padded              = field<int>( 0 );
    transformed_dipole_matrices = field<FFT::FFT_cpx_type>( 0 );
    dipole_matrices             = field<FFT::FFT_real_type>( 0 );
    save_dipole_matrices        = false;

    // init to 0, because initializing to -1 can fail silently
    energy_contributions_per_spin = decltype( energy_contributions_per_spin )( 0 );
    idx_zeeman                    = 0;
    idx_exchange                  = 0;
    idx_dmi                       = 0;
    idx_anisotropy                = 0;
    idx_cubic_anisotropy          = 0;
    idx_ddi                       = 0;
    idx_quadruplet                = 0;

    it_bounds_pointwise_mult  = field<int>( 0 );
    it_bounds_write_gradients = field<int>( 0 );
    it_bounds_write_spins     = field<int>( 0 );
    it_bounds_write_dipole    = field<int>( 0 );

    // Generate interaction pairs, constants etc.
    this->updateInteractions();
    this->updateName();
}

Hamiltonian::Hamiltonian( const Hamiltonian & other )
        : geometry( other.geometry ),
          boundary_conditions( other.boundary_conditions ),
          interactions( 0 ),
          active_interactions_size( other.active_interactions_size ),
          common_interactions_size( other.common_interactions_size ),
          prng( other.prng ),
          distribution_int( other.distribution_int ),
          name_update_paused( other.name_update_paused ),
          hamiltonian_class( other.hamiltonian_class ),
          class_name( other.class_name )
{
    external_field_magnitude    = other.external_field_magnitude;
    external_field_normal       = other.external_field_normal;
    anisotropy_indices          = other.anisotropy_indices;
    anisotropy_magnitudes       = other.anisotropy_magnitudes;
    anisotropy_normals          = other.anisotropy_normals;
    cubic_anisotropy_indices    = other.cubic_anisotropy_indices;
    cubic_anisotropy_magnitudes = other.cubic_anisotropy_magnitudes;
    exchange_shell_magnitudes   = other.exchange_shell_magnitudes;
    exchange_pairs_in           = other.exchange_pairs_in;
    exchange_magnitudes_in      = other.exchange_magnitudes_in;
    exchange_pairs              = other.exchange_pairs;
    exchange_magnitudes         = other.exchange_magnitudes;
    dmi_shell_magnitudes        = other.dmi_shell_magnitudes;
    dmi_shell_chirality         = other.dmi_shell_chirality;
    dmi_pairs_in                = other.dmi_pairs_in;
    dmi_magnitudes_in           = other.dmi_magnitudes_in;
    dmi_normals_in              = other.dmi_normals_in;
    dmi_pairs                   = other.dmi_pairs;
    dmi_magnitudes              = other.dmi_magnitudes;
    dmi_normals                 = other.dmi_normals;
    ddi_method                  = other.ddi_method;
    ddi_n_periodic_images       = other.ddi_n_periodic_images;
    ddi_pb_zero_padding         = other.ddi_pb_zero_padding;
    ddi_cutoff_radius           = other.ddi_cutoff_radius;
    ddi_pairs                   = other.ddi_pairs;
    ddi_magnitudes              = other.ddi_magnitudes;
    ddi_normals                 = other.ddi_normals;
    quadruplets                 = other.quadruplets;
    quadruplet_magnitudes       = other.quadruplet_magnitudes;

    idx_zeeman                    = other.idx_zeeman;
    idx_anisotropy                = other.idx_anisotropy;
    idx_cubic_anisotropy          = other.idx_cubic_anisotropy;
    idx_exchange                  = other.idx_exchange;
    idx_dmi                       = other.idx_dmi;
    idx_ddi                       = other.idx_ddi;
    idx_quadruplet                = other.idx_quadruplet;
    energy_contributions_per_spin = other.energy_contributions_per_spin;

    fft_plan_spins              = other.fft_plan_spins;
    fft_plan_reverse            = other.fft_plan_reverse;
    transformed_dipole_matrices = other.transformed_dipole_matrices;
    save_dipole_matrices        = other.save_dipole_matrices;
    dipole_matrices             = other.dipole_matrices;
    n_inter_sublattice          = other.n_inter_sublattice;
    inter_sublattice_lookup     = other.inter_sublattice_lookup;
    n_cells_padded              = other.n_cells_padded;
    sublattice_size             = other.sublattice_size;
    spin_stride                 = other.spin_stride;
    dipole_stride               = other.dipole_stride;

    it_bounds_pointwise_mult  = other.it_bounds_pointwise_mult;
    it_bounds_write_gradients = other.it_bounds_write_gradients;
    it_bounds_write_spins     = other.it_bounds_write_spins;
    it_bounds_write_dipole    = other.it_bounds_write_dipole;

    interactions.reserve( other.interactions.capacity() );
    for( const auto & interaction : other.interactions )
        interactions.emplace_back( interaction->clone( this ) );
}

void Hamiltonian::updateInteractions()
{
#if defined( SPIRIT_USE_OPENMP ) || defined( SPIRIT_USE_CUDA )
    // When parallelising (cuda or openmp), we need all neighbours per spin
    const bool use_redundant_neighbours = true;
#else
    // When running on a single thread, we can ignore redundant neighbours
    const bool use_redundant_neighbours = false;
#endif

    // Exchange
    this->exchange_pairs      = pairfield( 0 );
    this->exchange_magnitudes = scalarfield( 0 );
    if( !exchange_shell_magnitudes.empty() )
    {
        // Generate Exchange neighbours
        intfield exchange_shells( 0 );
        Neighbours::Get_Neighbours_in_Shells(
            *geometry, exchange_shell_magnitudes.size(), exchange_pairs, exchange_shells, use_redundant_neighbours );
        for( std::size_t ipair = 0; ipair < exchange_pairs.size(); ++ipair )
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
            for( std::size_t i = 0; i < exchange_pairs_in.size(); ++i )
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
    if( !dmi_shell_magnitudes.empty() )
    {
        // Generate DMI neighbours and normals
        intfield dmi_shells( 0 );
        Neighbours::Get_Neighbours_in_Shells(
            *geometry, dmi_shell_magnitudes.size(), dmi_pairs, dmi_shells, use_redundant_neighbours );
        for( std::size_t ineigh = 0; ineigh < dmi_pairs.size(); ++ineigh )
        {
            this->dmi_normals.push_back(
                Neighbours::DMI_Normal_from_Pair( *geometry, dmi_pairs[ineigh], this->dmi_shell_chirality ) );
            this->dmi_magnitudes.push_back( dmi_shell_magnitudes[dmi_shells[ineigh]] );
        }
    }
    else
    {
        // Use direct list of pairs
        this->dmi_pairs      = this->dmi_pairs_in;
        this->dmi_magnitudes = this->dmi_magnitudes_in;
        this->dmi_normals    = this->dmi_normals_in;
        if( use_redundant_neighbours )
        {
            for( std::size_t i = 0; i < dmi_pairs_in.size(); ++i )
            {
                auto & p = dmi_pairs_in[i];
                auto & t = p.translations;
                this->dmi_pairs.push_back( Pair{ p.j, p.i, { -t[0], -t[1], -t[2] } } );
                this->dmi_magnitudes.push_back( dmi_magnitudes_in[i] );
                this->dmi_normals.push_back( -dmi_normals_in[i] );
            }
        }
    }

    // Dipole-dipole (cutoff)
    if( this->ddi_method == DDI_Method::Cutoff )
        this->ddi_pairs = Engine::Neighbours::Get_Pairs_in_Radius( *this->geometry, this->ddi_cutoff_radius );
    else
        this->ddi_pairs = field<Pair>{};

    this->ddi_magnitudes = scalarfield( this->ddi_pairs.size() );
    this->ddi_normals    = vectorfield( this->ddi_pairs.size() );

    for( std::size_t i = 0; i < this->ddi_pairs.size(); ++i )
    {
        Engine::Neighbours::DDI_from_Pair(
            *this->geometry,
            {
                this->ddi_pairs[i].i,
                this->ddi_pairs[i].j,
#ifndef SPIRIT_USE_CUDA
                this->ddi_pairs[i].translations,
#else
                { this->ddi_pairs[i].translations[0], this->ddi_pairs[i].translations[1],
                  this->ddi_pairs[i].translations[2] }
#endif
            },
            this->ddi_magnitudes[i], this->ddi_normals[i] );
    };
    // Dipole-dipole
    this->Prepare_DDI();

    // Update, which terms still contribute
    this->Update_Energy_Contributions();
}

void Hamiltonian::Update_Energy_Contributions()
{
    this->energy_contributions_per_spin = vectorlabeled<scalarfield>( 0 );

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

void Hamiltonian::Energy_Contributions_per_Spin( const vectorfield & spins, vectorlabeled<scalarfield> & contributions )
{
    if( contributions.size() != this->energy_contributions_per_spin.size() )
    {
        contributions = this->energy_contributions_per_spin;
    }

    auto nos = spins.size();
    for( auto & contrib : contributions )
    {
        // Allocate if not already allocated
        if( contrib.second.size() != nos )
            contrib.second = scalarfield( nos, 0 );
        // Otherwise set to zero
        else
            Vectormath::fill( contrib.second, 0 );
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

void Hamiltonian::E_Zeeman( const vectorfield & spins, scalarfield & energy )
{
#ifdef SPIRIT_USE_CUDA
    int size = geometry->n_cells_total;
    CU_E_Zeeman<<<( size + 1023 ) / 1024, 1024>>>(
        spins.data(), this->geometry->atom_types.data(), geometry->n_cell_atoms, geometry->mu_s.data(),
        this->external_field_magnitude, this->external_field_normal, energy.data(), size );
    CU_CHECK_AND_SYNC();
#else
    const int N = geometry->n_cell_atoms;
    auto & mu_s = this->geometry->mu_s;

#pragma omp parallel for
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( int ibasis = 0; ibasis < N; ++ibasis )
        {
            int ispin = icell * N + ibasis;
            if( check_atom_type( this->geometry->atom_types[ispin] ) )
                energy[ispin]
                    -= mu_s[ispin] * this->external_field_magnitude * this->external_field_normal.dot( spins[ispin] );
        }
    }
#endif
}

void Hamiltonian::E_Anisotropy( const vectorfield & spins, scalarfield & energy )
{
#ifdef SPIRIT_USE_CUDA
    int size = geometry->n_cells_total;
    CU_E_Anisotropy<<<( size + 1023 ) / 1024, 1024>>>(
        spins.data(), this->geometry->atom_types.data(), this->geometry->n_cell_atoms, this->anisotropy_indices.size(),
        this->anisotropy_indices.data(), this->anisotropy_magnitudes.data(), this->anisotropy_normals.data(),
        energy.data(), size );
    CU_CHECK_AND_SYNC();
#else
    const int N = geometry->n_cell_atoms;

#pragma omp parallel for
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( int iani = 0; iani < anisotropy_indices.size(); ++iani )
        {
            int ispin = icell * N + anisotropy_indices[iani];
            if( check_atom_type( this->geometry->atom_types[ispin] ) )
                energy[ispin] -= this->anisotropy_magnitudes[iani]
                                 * std::pow( anisotropy_normals[iani].dot( spins[ispin] ), 2.0 );
        }
    }
#endif
}

void Hamiltonian::E_Cubic_Anisotropy( const vectorfield & spins, scalarfield & energy )
{
#ifdef SPIRIT_USE_CUDA
    int size = geometry->n_cells_total;
    CU_E_Cubic_Anisotropy<<<( size + 1023 ) / 1024, 1024>>>(
        spins.data(), this->geometry->atom_types.data(), this->geometry->n_cell_atoms,
        this->cubic_anisotropy_indices.size(), this->cubic_anisotropy_indices.data(),
        this->cubic_anisotropy_magnitudes.data(), energy.data(), size );
    CU_CHECK_AND_SYNC();
#else
    const int N = geometry->n_cell_atoms;

#pragma omp parallel for
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( int iani = 0; iani < cubic_anisotropy_indices.size(); ++iani )
        {
            int ispin = icell * N + cubic_anisotropy_indices[iani];
            if( check_atom_type( this->geometry->atom_types[ispin] ) )
                energy[ispin] -= this->cubic_anisotropy_magnitudes[iani] / 2
                                 * ( std::pow( spins[ispin][0], 4.0 ) + std::pow( spins[ispin][1], 4.0 )
                                     + std::pow( spins[ispin][2], 4.0 ) );
        }
    }
#endif
}

void Hamiltonian::E_Exchange( const vectorfield & spins, scalarfield & energy )
{
#ifdef SPIRIT_USE_CUDA
    int size = geometry->n_cells_total;
    CU_E_Exchange<<<( size + 1023 ) / 1024, 1024>>>(
        spins.data(), this->geometry->atom_types.data(), boundary_conditions.data(), geometry->n_cells.data(),
        geometry->n_cell_atoms, this->exchange_pairs.size(), this->exchange_pairs.data(),
        this->exchange_magnitudes.data(), energy.data(), size );
    CU_CHECK_AND_SYNC();
#else

#pragma omp parallel for
    for( unsigned int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( unsigned int i_pair = 0; i_pair < exchange_pairs.size(); ++i_pair )
        {
            int ispin = exchange_pairs[i_pair].i + icell * geometry->n_cell_atoms;
            int jspin = idx_from_pair(
                ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                exchange_pairs[i_pair] );
            if( jspin >= 0 )
            {
                energy[ispin] -= 0.5 * exchange_magnitudes[i_pair] * spins[ispin].dot( spins[jspin] );
#ifndef SPIRIT_USE_OPENMP
                energy[jspin] -= 0.5 * exchange_magnitudes[i_pair] * spins[ispin].dot( spins[jspin] );
#endif
            }
        }
    }
#endif
}

void Hamiltonian::E_DMI( const vectorfield & spins, scalarfield & energy )
{
#ifdef SPIRIT_USE_CUDA
    int size = geometry->n_cells_total;
    CU_E_DMI<<<( size + 1023 ) / 1024, 1024>>>(
        spins.data(), this->geometry->atom_types.data(), boundary_conditions.data(), geometry->n_cells.data(),
        geometry->n_cell_atoms, this->dmi_pairs.size(), this->dmi_pairs.data(), this->dmi_magnitudes.data(),
        this->dmi_normals.data(), energy.data(), size );
    CU_CHECK_AND_SYNC();
#else
#pragma omp parallel for
    for( unsigned int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( unsigned int i_pair = 0; i_pair < dmi_pairs.size(); ++i_pair )
        {
            int ispin = dmi_pairs[i_pair].i + icell * geometry->n_cell_atoms;
            int jspin = idx_from_pair(
                ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                dmi_pairs[i_pair] );
            if( jspin >= 0 )
            {
                energy[ispin]
                    -= 0.5 * dmi_magnitudes[i_pair] * dmi_normals[i_pair].dot( spins[ispin].cross( spins[jspin] ) );
#ifndef SPIRIT_USE_OPENMP
                energy[jspin]
                    -= 0.5 * dmi_magnitudes[i_pair] * dmi_normals[i_pair].dot( spins[ispin].cross( spins[jspin] ) );
#endif
            }
        }
    }
#endif
}

void Hamiltonian::E_DDI( const vectorfield & spins, scalarfield & energy )
{
    if( this->ddi_method == DDI_Method::FFT )
        this->E_DDI_FFT( spins, energy );
    else if( this->ddi_method == DDI_Method::Cutoff )
    {
        // TODO: Merge these implementations in the future
        if( ddi_cutoff_radius >= 0 )
            this->E_DDI_Cutoff( spins, energy );
        else
            this->E_DDI_Direct( spins, energy );
    }
}

void Hamiltonian::E_DDI_Direct( const vectorfield & spins, scalarfield & energy )
{
    vectorfield gradients_temp;
    gradients_temp.resize( geometry->nos );
    Vectormath::fill( gradients_temp, { 0, 0, 0 } );
    this->Gradient_DDI_Direct( spins, gradients_temp );

#pragma omp parallel for
    for( int ispin = 0; ispin < geometry->nos; ispin++ )
        energy[ispin] += 0.5 * spins[ispin].dot( gradients_temp[ispin] );
}

void Hamiltonian::E_DDI_Cutoff( const vectorfield & spins, scalarfield & energy )
{
#ifdef SPIRIT_USE_CUDA
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
    //                     ddi_pairs[i_pair].translations); energy[idx_i] -= mult / std::pow(ddi_magnitudes[i_pair], 3.0) *
    //                         (3 * spins[idx_j].dot(ddi_normals[i_pair]) * spins[idx_i].dot(ddi_normals[i_pair]) -
    //                         spins[idx_i].dot(spins[idx_j]));
    //                     energy[idx_j] -= mult / std::pow(ddi_magnitudes[i_pair], 3.0) *
    //                         (3 * spins[idx_j].dot(ddi_normals[i_pair]) * spins[idx_i].dot(ddi_normals[i_pair]) -
    //                         spins[idx_i].dot(spins[idx_j]));
    //                 }
    //             }
    //         }
    //     }
    // }
#else

    auto & mu_s = this->geometry->mu_s;
    // The translations are in angstr�m, so the |r|[m] becomes |r|[m]*10^-10
    const scalar mult = C::mu_0 * std::pow( C::mu_B, 2 ) / ( 4 * C::Pi * 1e-30 );

    scalar result = 0.0;

    for( unsigned int i_pair = 0; i_pair < ddi_pairs.size(); ++i_pair )
    {
        if( ddi_magnitudes[i_pair] > 0.0 )
        {
            for( int da = 0; da < geometry->n_cells[0]; ++da )
            {
                for( int db = 0; db < geometry->n_cells[1]; ++db )
                {
                    for( int dc = 0; dc < geometry->n_cells[2]; ++dc )
                    {
                        std::array<int, 3> translations = { da, db, dc };

                        int i = ddi_pairs[i_pair].i;
                        int j = ddi_pairs[i_pair].j;
                        int ispin
                            = i + idx_from_translations( geometry->n_cells, geometry->n_cell_atoms, translations );
                        int jspin = idx_from_pair(
                            ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                            ddi_pairs[i_pair] );
                        if( jspin >= 0 )
                        {
                            energy[ispin] -= 0.5 * mu_s[ispin] * mu_s[jspin] * mult
                                             / std::pow( ddi_magnitudes[i_pair], 3.0 )
                                             * ( 3 * spins[ispin].dot( ddi_normals[i_pair] )
                                                     * spins[jspin].dot( ddi_normals[i_pair] )
                                                 - spins[ispin].dot( spins[jspin] ) );
                        }
                    }
                }
            }
        }
    }
#endif
} // end DipoleDipole

void Hamiltonian::E_DDI_FFT( const vectorfield & spins, scalarfield & energy )
{
#ifdef SPIRIT_USE_CUDA
    // todo maybe the gradient should be cached somehow, it is quite inefficient to calculate it
    // again just for the energy
    vectorfield gradients_temp( geometry->nos );
    Vectormath::fill( gradients_temp, { 0, 0, 0 } );
    this->Gradient_DDI( spins, gradients_temp );
    CU_E_DDI_FFT<<<( geometry->nos + 1023 ) / 1024, 1024>>>(
        energy.data(), spins.data(), gradients_temp.data(), geometry->nos, geometry->n_cell_atoms,
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

#else
    scalar Energy_DDI = 0;
    vectorfield gradients_temp;
    gradients_temp.resize( geometry->nos );
    Vectormath::fill( gradients_temp, { 0, 0, 0 } );
    this->Gradient_DDI_FFT( spins, gradients_temp );

// TODO: add dot_scaled to Vectormath and use that
#pragma omp parallel for
    for( int ispin = 0; ispin < geometry->nos; ispin++ )
    {
        energy[ispin] += 0.5 * spins[ispin].dot( gradients_temp[ispin] );
        // Energy_DDI    += 0.5 * spins[ispin].dot(gradients_temp[ispin]);
    }
#endif
} // end DipoleDipole

void Hamiltonian::E_Quadruplet( const vectorfield & spins, scalarfield & energy )
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
#ifdef SPIRIT_USE_CUDA
                    int jspin = idx_from_pair(
                        ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                        { i, j, { d_j[0], d_j[1], d_j[2] } } );
                    int kspin = idx_from_pair(
                        ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                        { i, k, { d_k[0], d_k[1], d_k[2] } } );
                    int lspin = idx_from_pair(
                        ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                        { i, l, { d_l[0], d_l[1], d_l[2] } } );
#else
                    int jspin = idx_from_pair(
                        ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                        { i, j, d_j } );
                    int kspin = idx_from_pair(
                        ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                        { i, k, d_k } );
                    int lspin = idx_from_pair(
                        ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                        { i, l, d_l } );
#endif
                    if( ispin >= 0 && jspin >= 0 && kspin >= 0 && lspin >= 0 )
                    {
                        energy[ispin] -= 0.25 * quadruplet_magnitudes[iquad] * ( spins[ispin].dot( spins[jspin] ) )
                                         * ( spins[kspin].dot( spins[lspin] ) );
                        energy[jspin] -= 0.25 * quadruplet_magnitudes[iquad] * ( spins[ispin].dot( spins[jspin] ) )
                                         * ( spins[kspin].dot( spins[lspin] ) );
                        energy[kspin] -= 0.25 * quadruplet_magnitudes[iquad] * ( spins[ispin].dot( spins[jspin] ) )
                                         * ( spins[kspin].dot( spins[lspin] ) );
                        energy[lspin] -= 0.25 * quadruplet_magnitudes[iquad] * ( spins[ispin].dot( spins[jspin] ) )
                                         * ( spins[kspin].dot( spins[lspin] ) );
                    }
                }
            }
        }
    }
}

scalar Hamiltonian::Energy_Single_Spin( int ispin, const vectorfield & spins )
{
    scalar energy = 0;
    if( check_atom_type( this->geometry->atom_types[ispin] ) )
    {
        int icell   = ispin / this->geometry->n_cell_atoms;
        int ibasis  = ispin - icell * this->geometry->n_cell_atoms;
        auto & mu_s = this->geometry->mu_s;
        Pair pair_inv;

        // External field
        if( this->idx_zeeman >= 0 )
            energy -= mu_s[ispin] * this->external_field_magnitude * this->external_field_normal.dot( spins[ispin] );

        // Anisotropy
        if( this->idx_anisotropy >= 0 )
        {
            for( int iani = 0; iani < anisotropy_indices.size(); ++iani )
            {
                if( anisotropy_indices[iani] == ibasis )
                {
                    if( check_atom_type( this->geometry->atom_types[ispin] ) )
                        energy -= this->anisotropy_magnitudes[iani]
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
                        energy -= this->cubic_anisotropy_magnitudes[iani] / 2
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
                        energy -= this->exchange_magnitudes[ipair] * spins[ispin].dot( spins[jspin] );
                }
#if !( defined( SPIRIT_USE_OPENMP ) || defined( SPIRIT_USE_CUDA ) )
                if( pair.j == ibasis )
                {
                    const auto & t = pair.translations;
                    pair_inv       = Pair{ pair.j, pair.i, { -t[0], -t[1], -t[2] } };
                    int jspin      = idx_from_pair(
                        ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                        pair_inv );
                    if( jspin >= 0 )
                        energy -= this->exchange_magnitudes[ipair] * spins[ispin].dot( spins[jspin] );
                }
#endif
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
                        energy -= this->dmi_magnitudes[ipair]
                                  * this->dmi_normals[ipair].dot( spins[ispin].cross( spins[jspin] ) );
                }
#if !( defined( SPIRIT_USE_OPENMP ) || defined( SPIRIT_USE_CUDA ) )
                if( pair.j == ibasis )
                {
                    const auto & t = pair.translations;
                    pair_inv       = Pair{ pair.j, pair.i, { -t[0], -t[1], -t[2] } };
                    int jspin      = idx_from_pair(
                        ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                        pair_inv );
                    if( jspin >= 0 )
                        energy += this->dmi_magnitudes[ipair]
                                  * this->dmi_normals[ipair].dot( spins[ispin].cross( spins[jspin] ) );
                }
#endif
            }
        }

        // TODO: Quadruplets
        if( this->idx_quadruplet >= 0 )
        {
        }
    }
    return energy;
}

void Hamiltonian::Gradient( const vectorfield & spins, vectorfield & gradient )
{
    // Set to zero
    Vectormath::fill( gradient, { 0, 0, 0 } );

    // External field
    if( idx_zeeman >= 0 )
        this->Gradient_Zeeman( gradient );

    // Anisotropy
    if( idx_anisotropy >= 0 )
        this->Gradient_Anisotropy( spins, gradient );

    // Cubic Anisotropy
    if( idx_cubic_anisotropy >= 0 )
        this->Gradient_Cubic_Anisotropy( spins, gradient );

    // Exchange
    if( idx_exchange >= 0 )
        this->Gradient_Exchange( spins, gradient );

    // DMI
    if( idx_dmi >= 0 )
        this->Gradient_DMI( spins, gradient );

    // DDI
    if( idx_ddi >= 0 )
        this->Gradient_DDI( spins, gradient );

    // Quadruplets
    if( idx_quadruplet >= 0 )
        this->Gradient_Quadruplet( spins, gradient );
}

void Hamiltonian::Gradient_and_Energy( const vectorfield & spins, vectorfield & gradient, scalar & energy )
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

void Hamiltonian::Gradient_Zeeman( vectorfield & gradient )
{
#ifdef SPIRIT_USE_CUDA
    int size = geometry->n_cells_total;
    CU_Gradient_Zeeman<<<( size + 1023 ) / 1024, 1024>>>(
        this->geometry->atom_types.data(), geometry->n_cell_atoms, geometry->mu_s.data(),
        this->external_field_magnitude, this->external_field_normal, gradient.data(), size );
    CU_CHECK_AND_SYNC();
#else
    const int N = geometry->n_cell_atoms;
    auto & mu_s = this->geometry->mu_s;

#pragma omp parallel for
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( int ibasis = 0; ibasis < N; ++ibasis )
        {
            int ispin = icell * N + ibasis;
            if( check_atom_type( this->geometry->atom_types[ispin] ) )
                gradient[ispin] -= mu_s[ispin] * this->external_field_magnitude * this->external_field_normal;
        }
    }
#endif
}

void Hamiltonian::Gradient_Anisotropy( const vectorfield & spins, vectorfield & gradient )
{
#ifdef SPIRIT_USE_CUDA
    int size = geometry->n_cells_total;
    CU_Gradient_Anisotropy<<<( size + 1023 ) / 1024, 1024>>>(
        spins.data(), this->geometry->atom_types.data(), this->geometry->n_cell_atoms, this->anisotropy_indices.size(),
        this->anisotropy_indices.data(), this->anisotropy_magnitudes.data(), this->anisotropy_normals.data(),
        gradient.data(), size );
    CU_CHECK_AND_SYNC();
#else
    const int N = geometry->n_cell_atoms;

#pragma omp parallel for
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( int iani = 0; iani < anisotropy_indices.size(); ++iani )
        {
            int ispin = icell * N + anisotropy_indices[iani];
            if( check_atom_type( this->geometry->atom_types[ispin] ) )
                gradient[ispin] -= 2.0 * this->anisotropy_magnitudes[iani] * this->anisotropy_normals[iani]
                                   * anisotropy_normals[iani].dot( spins[ispin] );
        }
    }
#endif
}

void Hamiltonian::Gradient_Cubic_Anisotropy( const vectorfield & spins, vectorfield & gradient )
{
#ifdef SPIRIT_USE_CUDA
    int size = geometry->n_cells_total;
    CU_Gradient_Cubic_Anisotropy<<<( size + 1023 ) / 1024, 1024>>>(
        spins.data(), this->geometry->atom_types.data(), this->geometry->n_cell_atoms,
        this->cubic_anisotropy_indices.size(), this->cubic_anisotropy_indices.data(),
        this->cubic_anisotropy_magnitudes.data(), gradient.data(), size );
    CU_CHECK_AND_SYNC();
#else
    const int N = geometry->n_cell_atoms;

#pragma omp parallel for
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( int iani = 0; iani < cubic_anisotropy_indices.size(); ++iani )
        {
            int ispin = icell * N + cubic_anisotropy_indices[iani];
            if( check_atom_type( this->geometry->atom_types[ispin] ) )
                for( int icomp = 0; icomp < 3; ++icomp )
                {
                    gradient[ispin][icomp]
                        -= 2.0 * this->cubic_anisotropy_magnitudes[iani] * std::pow( spins[ispin][icomp], 3.0 );
                }
        }
    }
#endif
}

void Hamiltonian::Gradient_Exchange( const vectorfield & spins, vectorfield & gradient )
{
#ifdef SPIRIT_USE_CUDA
    int size = geometry->n_cells_total;
    CU_Gradient_Exchange<<<( size + 1023 ) / 1024, 1024>>>(
        spins.data(), this->geometry->atom_types.data(), boundary_conditions.data(), geometry->n_cells.data(),
        geometry->n_cell_atoms, this->exchange_pairs.size(), this->exchange_pairs.data(),
        this->exchange_magnitudes.data(), gradient.data(), size );
    CU_CHECK_AND_SYNC();
#else
#pragma omp parallel for
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
                gradient[ispin] -= exchange_magnitudes[i_pair] * spins[jspin];
#ifndef SPIRIT_USE_OPENMP
                gradient[jspin] -= exchange_magnitudes[i_pair] * spins[ispin];
#endif
            }
        }
    }
#endif
}

void Hamiltonian::Gradient_DMI( const vectorfield & spins, vectorfield & gradient )
{
#ifdef SPIRIT_USE_CUDA
    int size = geometry->n_cells_total;
    CU_Gradient_DMI<<<( size + 1023 ) / 1024, 1024>>>(
        spins.data(), this->geometry->atom_types.data(), boundary_conditions.data(), geometry->n_cells.data(),
        geometry->n_cell_atoms, this->dmi_pairs.size(), this->dmi_pairs.data(), this->dmi_magnitudes.data(),
        this->dmi_normals.data(), gradient.data(), size );
    CU_CHECK_AND_SYNC();
#else
#pragma omp parallel for
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
                gradient[ispin] -= dmi_magnitudes[i_pair] * spins[jspin].cross( dmi_normals[i_pair] );
#ifndef SPIRIT_USE_OPENMP
                gradient[jspin] += dmi_magnitudes[i_pair] * spins[ispin].cross( dmi_normals[i_pair] );
#endif
            }
        }
    }
#endif
}

void Hamiltonian::Gradient_DDI( const vectorfield & spins, vectorfield & gradient )
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

void Hamiltonian::Gradient_DDI_Cutoff( const vectorfield & spins, vectorfield & gradient )
{
#ifdef SPIRIT_USE_CUDA
// TODO
#else
    const auto & mu_s = this->geometry->mu_s;
    // The translations are in angstr�m, so the |r|[m] becomes |r|[m]*10^-10
    const scalar mult = C::mu_0 * C::mu_B * C::mu_B / ( 4 * C::Pi * 1e-30 );

    for( unsigned int i_pair = 0; i_pair < ddi_pairs.size(); ++i_pair )
    {
        if( ddi_magnitudes[i_pair] > 0.0 )
        {
            for( int da = 0; da < geometry->n_cells[0]; ++da )
            {
                for( int db = 0; db < geometry->n_cells[1]; ++db )
                {
                    for( int dc = 0; dc < geometry->n_cells[2]; ++dc )
                    {
                        scalar skalar_contrib           = mult / std::pow( ddi_magnitudes[i_pair], 3.0 );
                        std::array<int, 3> translations = { da, db, dc };

                        int i = ddi_pairs[i_pair].i;
                        int j = ddi_pairs[i_pair].j;
                        int ispin
                            = i + idx_from_translations( geometry->n_cells, geometry->n_cell_atoms, translations );
                        int jspin = idx_from_pair(
                            ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                            ddi_pairs[i_pair] );
                        if( jspin >= 0 )
                        {
                            gradient[ispin] -= mu_s[jspin] * mu_s[ispin] * skalar_contrib
                                               * ( 3 * ddi_normals[i_pair] * spins[jspin].dot( ddi_normals[i_pair] )
                                                   - spins[jspin] );
                        }
                    }
                }
            }
        }
    }
#endif
} // end Field_DipoleDipole

void Hamiltonian::Gradient_DDI_FFT( const vectorfield & spins, vectorfield & gradient )
{
#ifdef SPIRIT_USE_CUDA
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
#else
    // Size of original geometry
    int Na = geometry->n_cells[0];
    int Nb = geometry->n_cells[1];
    int Nc = geometry->n_cells[2];

    FFT_Spins( spins );

    auto & ft_D_matrices = transformed_dipole_matrices;
    auto & ft_spins      = fft_plan_spins.cpx_ptr;

    auto & res_iFFT = fft_plan_reverse.real_ptr;
    auto & res_mult = fft_plan_reverse.cpx_ptr;

    // Workaround for compability with intel compiler
    const int c_n_cell_atoms               = geometry->n_cell_atoms;
    const int * c_it_bounds_pointwise_mult = it_bounds_pointwise_mult.data();

// Loop over basis atoms (i.e sublattices)
#pragma omp parallel for collapse( 4 )
    for( int i_b1 = 0; i_b1 < c_n_cell_atoms; ++i_b1 )
    {
        for( int c = 0; c < c_it_bounds_pointwise_mult[2]; ++c )
        {
            for( int b = 0; b < c_it_bounds_pointwise_mult[1]; ++b )
            {
                for( int a = 0; a < c_it_bounds_pointwise_mult[0]; ++a )
                {
                    // Collect the intersublattice contributions
                    for( int i_b2 = 0; i_b2 < c_n_cell_atoms; ++i_b2 )
                    {
                        // Look up at which position the correct D-matrices are saved
                        int & b_inter = inter_sublattice_lookup[i_b1 + i_b2 * geometry->n_cell_atoms];

                        int idx_b2
                            = i_b2 * spin_stride.basis + a * spin_stride.a + b * spin_stride.b + c * spin_stride.c;
                        int idx_b1
                            = i_b1 * spin_stride.basis + a * spin_stride.a + b * spin_stride.b + c * spin_stride.c;
                        int idx_d = b_inter * dipole_stride.basis + a * dipole_stride.a + b * dipole_stride.b
                                    + c * dipole_stride.c;

                        auto & fs_x = ft_spins[idx_b2];
                        auto & fs_y = ft_spins[idx_b2 + 1 * spin_stride.comp];
                        auto & fs_z = ft_spins[idx_b2 + 2 * spin_stride.comp];

                        auto & fD_xx = ft_D_matrices[idx_d];
                        auto & fD_xy = ft_D_matrices[idx_d + 1 * dipole_stride.comp];
                        auto & fD_xz = ft_D_matrices[idx_d + 2 * dipole_stride.comp];
                        auto & fD_yy = ft_D_matrices[idx_d + 3 * dipole_stride.comp];
                        auto & fD_yz = ft_D_matrices[idx_d + 4 * dipole_stride.comp];
                        auto & fD_zz = ft_D_matrices[idx_d + 5 * dipole_stride.comp];

                        FFT::addTo(
                            res_mult[idx_b1 + 0 * spin_stride.comp],
                            FFT::mult3D( fD_xx, fD_xy, fD_xz, fs_x, fs_y, fs_z ), i_b2 == 0 );
                        FFT::addTo(
                            res_mult[idx_b1 + 1 * spin_stride.comp],
                            FFT::mult3D( fD_xy, fD_yy, fD_yz, fs_x, fs_y, fs_z ), i_b2 == 0 );
                        FFT::addTo(
                            res_mult[idx_b1 + 2 * spin_stride.comp],
                            FFT::mult3D( fD_xz, fD_yz, fD_zz, fs_x, fs_y, fs_z ), i_b2 == 0 );
                    }
                }
            } // end iteration over padded lattice cells
        }     // end iteration over second sublattice
    }

    // Inverse Fourier Transform
    FFT::batch_iFour_3D( fft_plan_reverse );

    // Workaround for compability with intel compiler
    const int * c_n_cells = geometry->n_cells.data();

    // Place the gradients at the correct positions and mult with correct mu
    for( int c = 0; c < c_n_cells[2]; ++c )
    {
        for( int b = 0; b < c_n_cells[1]; ++b )
        {
            for( int a = 0; a < c_n_cells[0]; ++a )
            {
                for( int i_b1 = 0; i_b1 < c_n_cell_atoms; ++i_b1 )
                {
                    int idx_orig = i_b1 + geometry->n_cell_atoms * ( a + Na * ( b + Nb * c ) );
                    int idx      = i_b1 * spin_stride.basis + a * spin_stride.a + b * spin_stride.b + c * spin_stride.c;
                    gradient[idx_orig][0] -= geometry->mu_s[idx_orig] * res_iFFT[idx] / sublattice_size;
                    gradient[idx_orig][1]
                        -= geometry->mu_s[idx_orig] * res_iFFT[idx + 1 * spin_stride.comp] / sublattice_size;
                    gradient[idx_orig][2]
                        -= geometry->mu_s[idx_orig] * res_iFFT[idx + 2 * spin_stride.comp] / sublattice_size;
                }
            }
        }
    } // end iteration sublattice 1
#endif
} // end Field_DipoleDipole

void Hamiltonian::Gradient_DDI_Direct( const vectorfield & spins, vectorfield & gradient )
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

            gradient[idx1][0]
                -= ( Dxx * m2[0] + Dxy * m2[1] + Dxz * m2[2] ) * geometry->mu_s[idx1] * geometry->mu_s[idx2];
            gradient[idx1][1]
                -= ( Dxy * m2[0] + Dyy * m2[1] + Dyz * m2[2] ) * geometry->mu_s[idx1] * geometry->mu_s[idx2];
            gradient[idx1][2]
                -= ( Dxz * m2[0] + Dyz * m2[1] + Dzz * m2[2] ) * geometry->mu_s[idx1] * geometry->mu_s[idx2];
        }
    }
}

void Hamiltonian::Gradient_Quadruplet( const vectorfield & spins, vectorfield & gradient )
{
    for( unsigned int iquad = 0; iquad < quadruplets.size(); ++iquad )
    {
        const auto & quad = quadruplets[iquad];

        const int i = quad.i;
        const int j = quad.j;
        const int k = quad.k;
        const int l = quad.l;

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
#ifdef SPIRIT_USE_CUDA
                    int jspin = idx_from_pair(
                        ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                        { i, j, { d_j[0], d_j[1], d_j[2] } } );
                    int kspin = idx_from_pair(
                        ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                        { i, k, { d_k[0], d_k[1], d_k[2] } } );
                    int lspin = idx_from_pair(
                        ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                        { i, l, { d_l[0], d_l[1], d_l[2] } } );
#else
                    int jspin = idx_from_pair(
                        ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                        { i, j, d_j } );
                    int kspin = idx_from_pair(
                        ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                        { i, k, d_k } );
                    int lspin = idx_from_pair(
                        ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types,
                        { i, l, d_l } );
#endif

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

void Hamiltonian::Hessian( const vectorfield & spins, MatrixX & hessian )
{
    int nos     = spins.size();
    const int N = geometry->n_cell_atoms;

    // --- Set to zero
    hessian.setZero();

    // --- Single Spin elements
#pragma omp parallel for
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
#pragma omp parallel for
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
#if !( defined( SPIRIT_USE_OPENMP ) || defined( SPIRIT_USE_CUDA ) )
                    hessian( j, i ) += -exchange_magnitudes[i_pair];
#endif
                }
            }
        }
    }

    // DMI
#pragma omp parallel for
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

#if !( defined( SPIRIT_USE_OPENMP ) || defined( SPIRIT_USE_CUDA ) )
                hessian( j + 1, i + 2 ) += dmi_magnitudes[i_pair] * dmi_normals[i_pair][0];
                hessian( j + 2, i + 1 ) += -dmi_magnitudes[i_pair] * dmi_normals[i_pair][0];
                hessian( j + 2, i + 0 ) += dmi_magnitudes[i_pair] * dmi_normals[i_pair][1];
                hessian( j + 0, i + 2 ) += -dmi_magnitudes[i_pair] * dmi_normals[i_pair][1];
                hessian( j + 0, i + 1 ) += dmi_magnitudes[i_pair] * dmi_normals[i_pair][2];
                hessian( j + 1, i + 0 ) += -dmi_magnitudes[i_pair] * dmi_normals[i_pair][2];
#endif
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

void Hamiltonian::Sparse_Hessian( const vectorfield & spins, SpMatrixX & hessian )
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
#if !( defined( SPIRIT_USE_OPENMP ) || defined( SPIRIT_USE_CUDA ) )
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

#if !( defined( SPIRIT_USE_OPENMP ) || defined( SPIRIT_USE_CUDA ) )
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

void Hamiltonian::FFT_Spins( const vectorfield & spins )
{
#ifdef SPIRIT_USE_CUDA
    CU_Write_FFT_Spin_Input<<<( geometry->nos + 1023 ) / 1024, 1024>>>(
        fft_plan_spins.real_ptr.data(), spins.data(), it_bounds_write_spins.data(), spin_stride,
        geometry->mu_s.data() );
#else
    // size of original geometry
    int Na           = geometry->n_cells[0];
    int Nb           = geometry->n_cells[1];
    int Nc           = geometry->n_cells[2];
    int n_cell_atoms = geometry->n_cell_atoms;

    auto & fft_spin_inputs = fft_plan_spins.real_ptr;

// iterate over the **original** system
#pragma omp parallel for collapse( 4 )
    for( int c = 0; c < Nc; ++c )
    {
        for( int b = 0; b < Nb; ++b )
        {
            for( int a = 0; a < Na; ++a )
            {
                for( int bi = 0; bi < n_cell_atoms; ++bi )
                {
                    int idx_orig = bi + n_cell_atoms * ( a + Na * ( b + Nb * c ) );
                    int idx      = bi * spin_stride.basis + a * spin_stride.a + b * spin_stride.b + c * spin_stride.c;

                    fft_spin_inputs[idx]                        = spins[idx_orig][0] * geometry->mu_s[idx_orig];
                    fft_spin_inputs[idx + 1 * spin_stride.comp] = spins[idx_orig][1] * geometry->mu_s[idx_orig];
                    fft_spin_inputs[idx + 2 * spin_stride.comp] = spins[idx_orig][2] * geometry->mu_s[idx_orig];
                }
            }
        }
    } // end iteration over basis
#endif
    FFT::batch_Four_3D( fft_plan_spins );
}

void Hamiltonian::FFT_Dipole_Matrices( FFT::FFT_Plan & fft_plan_dipole, int img_a, int img_b, int img_c )
{
#ifdef SPIRIT_USE_CUDA
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
#else
    // Prefactor of DDI
    scalar mult = C::mu_0 * C::mu_B * C::mu_B / ( 4 * C::Pi * 1e-30 );

    // Size of original geometry
    int Na = geometry->n_cells[0];
    int Nb = geometry->n_cells[1];
    int Nc = geometry->n_cells[2];

    auto & fft_dipole_inputs = fft_plan_dipole.real_ptr;

    int b_inter = -1;
    for( int i_b1 = 0; i_b1 < geometry->n_cell_atoms; ++i_b1 )
    {
        for( int i_b2 = 0; i_b2 < geometry->n_cell_atoms; ++i_b2 )
        {
            if( i_b1 == i_b2 && i_b1 != 0 )
            {
                inter_sublattice_lookup[i_b1 + i_b2 * geometry->n_cell_atoms] = 0;
                continue;
            }
            b_inter++;
            inter_sublattice_lookup[i_b1 + i_b2 * geometry->n_cell_atoms] = b_inter;

            // Iterate over the padded system
            const int * c_n_cells_padded = n_cells_padded.data();

            std::array<scalar, 3> cell_sizes = { geometry->lattice_constant * geometry->bravais_vectors[0].norm(),
                                                 geometry->lattice_constant * geometry->bravais_vectors[1].norm(),
                                                 geometry->lattice_constant * geometry->bravais_vectors[2].norm() };

#pragma omp parallel for collapse( 3 )
            for( int c = 0; c < c_n_cells_padded[2]; ++c )
            {
                for( int b = 0; b < c_n_cells_padded[1]; ++b )
                {
                    for( int a = 0; a < c_n_cells_padded[0]; ++a )
                    {
                        int a_idx  = a < Na ? a : a - n_cells_padded[0];
                        int b_idx  = b < Nb ? b : b - n_cells_padded[1];
                        int c_idx  = c < Nc ? c : c - n_cells_padded[2];
                        scalar Dxx = 0, Dxy = 0, Dxz = 0, Dyy = 0, Dyz = 0, Dzz = 0;
                        Vector3 diff;
                        // Iterate over periodic images
                        for( int a_pb = -img_a; a_pb <= img_a; a_pb++ )
                        {
                            for( int b_pb = -img_b; b_pb <= img_b; b_pb++ )
                            {
                                for( int c_pb = -img_c; c_pb <= img_c; c_pb++ )
                                {
                                    diff = geometry->lattice_constant
                                           * ( ( a_idx + a_pb * Na + geometry->cell_atoms[i_b1][0]
                                                 - geometry->cell_atoms[i_b2][0] )
                                                   * geometry->bravais_vectors[0]
                                               + ( b_idx + b_pb * Nb + geometry->cell_atoms[i_b1][1]
                                                   - geometry->cell_atoms[i_b2][1] )
                                                     * geometry->bravais_vectors[1]
                                               + ( c_idx + c_pb * Nc + geometry->cell_atoms[i_b1][2]
                                                   - geometry->cell_atoms[i_b2][2] )
                                                     * geometry->bravais_vectors[2] );

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
                }
            }
        }
    }
#endif
    FFT::batch_Four_3D( fft_plan_dipole );
}

void Hamiltonian::Prepare_DDI()
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

    FFT::FFT_Init();

// Workaround for bug in kissfft
// kissfft_ndr does not perform one-dimensional FFTs properly
#if !( defined( SPIRIT_USE_FFTW ) || defined( SPIRIT_USE_CUDA ) )
    int number_of_one_dims = 0;
    for( int i = 0; i < 3; i++ )
        if( n_cells_padded[i] == 1 && ++number_of_one_dims > 1 )
            n_cells_padded[i] = 2;
#endif

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

    // Create FFT plans
#ifdef SPIRIT_USE_CUDA
    // Set the iteration bounds for the nested for loops that are flattened in the kernels
    it_bounds_write_spins
        = { geometry->n_cell_atoms, geometry->n_cells[0], geometry->n_cells[1], geometry->n_cells[2] };

    it_bounds_write_dipole = { n_cells_padded[0], n_cells_padded[1], n_cells_padded[2] };

    it_bounds_pointwise_mult = { geometry->n_cell_atoms,
                                 ( n_cells_padded[0] / 2 + 1 ), // due to redundancy in real fft
                                 n_cells_padded[1], n_cells_padded[2] };

    it_bounds_write_gradients
        = { geometry->n_cell_atoms, geometry->n_cells[0], geometry->n_cells[1], geometry->n_cells[2] };
#endif
    FFT::FFT_Plan fft_plan_dipole = FFT::FFT_Plan( fft_dims, false, 6 * n_inter_sublattice, sublattice_size );
    fft_plan_spins                = FFT::FFT_Plan( fft_dims, false, 3 * geometry->n_cell_atoms, sublattice_size );
    fft_plan_reverse              = FFT::FFT_Plan( fft_dims, true, 3 * geometry->n_cell_atoms, sublattice_size );

#if defined( SPIRIT_USE_FFTW ) || defined( SPIRIT_USE_CUDA )
    field<int *> temp_s = { &spin_stride.comp, &spin_stride.basis, &spin_stride.a, &spin_stride.b, &spin_stride.c };
    field<int *> temp_d
        = { &dipole_stride.comp, &dipole_stride.basis, &dipole_stride.a, &dipole_stride.b, &dipole_stride.c };
    ;
    FFT::get_strides(
        temp_s, { 3, this->geometry->n_cell_atoms, n_cells_padded[0], n_cells_padded[1], n_cells_padded[2] } );
    FFT::get_strides( temp_d, { 6, n_inter_sublattice, n_cells_padded[0], n_cells_padded[1], n_cells_padded[2] } );
#ifndef SPIRIT_USE_CUDA
    it_bounds_pointwise_mult = { ( n_cells_padded[0] / 2 + 1 ), // due to redundancy in real fft
                                 n_cells_padded[1], n_cells_padded[2] };
#endif
#else
    field<int *> temp_s = { &spin_stride.a, &spin_stride.b, &spin_stride.c, &spin_stride.comp, &spin_stride.basis };
    field<int *> temp_d
        = { &dipole_stride.a, &dipole_stride.b, &dipole_stride.c, &dipole_stride.comp, &dipole_stride.basis };
    ;
    FFT::get_strides(
        temp_s, { n_cells_padded[0], n_cells_padded[1], n_cells_padded[2], 3, this->geometry->n_cell_atoms } );
    FFT::get_strides( temp_d, { n_cells_padded[0], n_cells_padded[1], n_cells_padded[2], 6, n_inter_sublattice } );
    it_bounds_pointwise_mult = { n_cells_padded[0], n_cells_padded[1], n_cells_padded[2] };
    ( it_bounds_pointwise_mult[fft_dims.size() - 1] /= 2 )++;
#endif

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

void Hamiltonian::Clean_DDI()
{
    fft_plan_spins   = FFT::FFT_Plan();
    fft_plan_reverse = FFT::FFT_Plan();
}

void Hamiltonian::Hessian_FD( const vectorfield & spins, MatrixX & hessian )
{
    // This is a regular finite difference implementation (probably not very efficient)
    // using the differences between gradient values (not function)
    // see https://v8doc.sas.com/sashtml/ormp/chap5/sect28.htm

    std::size_t nos = spins.size();

    vectorfield spins_pi( nos );
    vectorfield spins_mi( nos );
    vectorfield spins_pj( nos );
    vectorfield spins_mj( nos );

    spins_pi = spins;
    spins_mi = spins;
    spins_pj = spins;
    spins_mj = spins;

    vectorfield grad_pi( nos );
    vectorfield grad_mi( nos );
    vectorfield grad_pj( nos );
    vectorfield grad_mj( nos );

    for( std::size_t i = 0; i < nos; ++i )
    {
        for( std::size_t j = 0; j < nos; ++j )
        {
            for( std::uint8_t alpha = 0; alpha < 3; ++alpha )
            {
                for( std::uint8_t beta = 0; beta < 3; ++beta )
                {
                    // Displace
                    spins_pi[i][alpha] += delta;
                    spins_mi[i][alpha] -= delta;
                    spins_pj[j][beta] += delta;
                    spins_mj[j][beta] -= delta;

                    // Calculate Hessian component
                    this->Gradient( spins_pi, grad_pi );
                    this->Gradient( spins_mi, grad_mi );
                    this->Gradient( spins_pj, grad_pj );
                    this->Gradient( spins_mj, grad_mj );

                    hessian( 3 * i + alpha, 3 * j + beta )
                        = 0.25 / delta
                          * ( grad_pj[i][alpha] - grad_mj[i][alpha] + grad_pi[j][beta] - grad_mi[j][beta] );

                    // Un-Displace
                    spins_pi[i][alpha] -= delta;
                    spins_mi[i][alpha] += delta;
                    spins_pj[j][beta] -= delta;
                    spins_mj[j][beta] += delta;
                }
            }
        }
    }
}

void Hamiltonian::Gradient_FD( const vectorfield & spins, vectorfield & gradient )
{
    std::size_t nos = spins.size();

    // Calculate finite difference
    vectorfield spins_plus( nos );
    vectorfield spins_minus( nos );

    spins_plus  = spins;
    spins_minus = spins;

    for( std::size_t i = 0; i < nos; ++i )
    {
        for( std::uint8_t dim = 0; dim < 3; ++dim )
        {
            // Displace
            spins_plus[i][dim] += delta;
            spins_minus[i][dim] -= delta;

            // Calculate gradient component
            scalar E_plus    = this->Energy( spins_plus );
            scalar E_minus   = this->Energy( spins_minus );
            gradient[i][dim] = 0.5 * ( E_plus - E_minus ) / delta;

            // Un-Displace
            spins_plus[i][dim] -= delta;
            spins_minus[i][dim] += delta;
        }
    }
}

scalar Hamiltonian::Energy( const vectorfield & spins )
{
    scalar sum  = 0;
    auto energy = Energy_Contributions( spins );
    for( const auto & E : energy )
        sum += E.second;
    return sum;
}

Data::vectorlabeled<scalar> Hamiltonian::Energy_Contributions( const vectorfield & spins )
{
    Energy_Contributions_per_Spin( spins, this->energy_contributions_per_spin );
    vectorlabeled<scalar> energy( Number_of_Interactions() );
    for( std::size_t i = 0; i < energy.size(); ++i )
    {
        energy[i] = { this->energy_contributions_per_spin[i].first,
                      Vectormath::sum( this->energy_contributions_per_spin[i].second ) };
    }
    return energy;
}

std::size_t Hamiltonian::Number_of_Interactions()
{
    // TODO: integrate this with `Hamiltonian::getActiveInteractionsSize()`
    return energy_contributions_per_spin.size();
}

void Hamiltonian::updateName()
{
    if( name_update_paused )
        return;

    if( interactions.size() == 1 && hasInteraction<Interaction::Gaussian>() )
        hamiltonian_class = HAMILTONIAN_CLASS::GAUSSIAN;
    else if( !hasInteraction<Interaction::Gaussian>() )
        hamiltonian_class = HAMILTONIAN_CLASS::HEISENBERG;
    else
        hamiltonian_class = HAMILTONIAN_CLASS::GENERIC;

    class_name = hamiltonianClassName( hamiltonian_class );
}

// Hamiltonian name as string
std::string_view Hamiltonian::Name() const
{
    return class_name;
};

} // namespace Engine
