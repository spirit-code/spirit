#include <data/Spin_System.hpp>
#include <engine/Backend_par.hpp>
#include <engine/Indexing.hpp>
#include <engine/Vectormath.hpp>
#include <engine/spin/interaction/Biaxial_Anisotropy.hpp>
#include <utility/Constants.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Data;
using namespace Utility;
using Engine::Indexing::check_atom_type;

#ifdef SPIRIT_USE_CUDA
using Engine::Indexing::cu_check_atom_type;
#endif

namespace Engine
{

namespace Spin
{

namespace Interaction
{

Biaxial_Anisotropy::Biaxial_Anisotropy(
    Common::Interaction::Owner * hamiltonian, intfield indices, field<PolynomialBasis> bases, field<unsigned int> site_ptr,
    field<PolynomialTerm> terms ) noexcept
        : Interaction::Base<Biaxial_Anisotropy>( hamiltonian, scalarfield( 0 ) ),
          indices( std::move( indices ) ),
          bases( std::move( bases ) ),
          site_p( std::move( site_ptr ) ),
          terms( std::move( terms ) )
{
    this->updateGeometry();
}

void Biaxial_Anisotropy::updateFromGeometry( const Geometry & geometry ) {}

bool Biaxial_Anisotropy::is_contributing() const
{
    return !indices.empty();
}

#ifdef SPIRIT_USE_CUDA
__global__ void CU_E_Biaxial_Anisotropy(
    const Vector3 * spins, const int * atom_types, const int n_cell_atoms, const int n_anisotropies,
    const int * indices, const PolynomialBasis * bases, const unsigned int * site_p, const PolynomialTerm * terms,
    scalar * energy, size_t n_cells_total )
{
    for( auto icell = blockIdx.x * blockDim.x + threadIdx.x; icell < n_cells_total; icell += blockDim.x * gridDim.x )
    {
        for( int iani = 0; iani < n_anisotropies; ++iani )
        {
            int ispin = icell * n_cell_atoms + indices[iani];
            if( cu_check_atom_type( atom_types[ispin] ) )
            {
                const scalar s1 = bases[iani].k1.dot( spins[ispin] );
                const scalar s2 = bases[iani].k2.dot( spins[ispin] );
                const scalar s3 = bases[iani].k3.dot( spins[ispin] );

                const scalar sin_theta_2 = 1 - s1 * s1;

                scalar result = 0;
                for( int iterm = site_p[iani]; iterm < site_p[iani + 1]; ++iterm )
                {
                    const auto & [coeff, n1, n2, n3] = terms[iterm];
                    result += coeff * pow( sin_theta_2, n1 ) * pow( s2, n2 ) * pow( s3, n3 );
                }
                energy[ispin] += result;
            }
        }
    }
}
#endif

void Biaxial_Anisotropy::Energy_per_Spin( const vectorfield & spins, scalarfield & energy )
{
    const auto & geometry = getGeometry();
    const int N           = geometry.n_cell_atoms;

#ifdef SPIRIT_USE_CUDA
    const int size = geometry.n_cells_total;
    CU_E_Biaxial_Anisotropy<<<( size + 1023 ) / 1024, 1024>>>(
        spins.data(), geometry.atom_types.data(), geometry.n_cell_atoms, this->indices.size(), this->indices.data(),
        this->bases.data(), this->site_p.data(), this->terms.data(), energy.data(), size );
    CU_CHECK_AND_SYNC();
#else

#pragma omp parallel for
    for( int icell = 0; icell < geometry.n_cells_total; ++icell )
    {
        for( int iani = 0; iani < indices.size(); ++iani )
        {
            int ispin = icell * N + indices[iani];
            if( check_atom_type( geometry.atom_types[ispin] ) )
            {
                const scalar s1 = bases[iani].k1.dot( spins[ispin] );
                const scalar s2 = bases[iani].k2.dot( spins[ispin] );
                const scalar s3 = bases[iani].k3.dot( spins[ispin] );

                const scalar sin_theta_2 = 1 - s1 * s1;

                scalar result = 0;
                for( auto iterm = site_p[iani]; iterm < site_p[iani + 1]; ++iterm )
                {
                    const auto & [coeff, n1, n2, n3] = terms[iterm];
                    result += coeff * pow( sin_theta_2, n1 ) * pow( s2, n2 ) * pow( s3, n3 );
                }
                energy[ispin] += result;
            }
        }
    }
#endif
}

// Calculate the total energy for a single spin to be used in Monte Carlo.
//      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
scalar Biaxial_Anisotropy::Energy_Single_Spin( const int ispin, const vectorfield & spins )
{
    scalar energy         = 0;
    const auto & geometry = getGeometry();
    const int N           = geometry.n_cell_atoms;

    int icell  = ispin / N;
    int ibasis = ispin - icell * geometry.n_cell_atoms;

    for( int iani = 0; iani < indices.size(); ++iani )
    {
        if( indices[iani] == ibasis )
        {
            if( check_atom_type( geometry.atom_types[ispin] ) )
            {
                const scalar s1 = bases[iani].k1.dot( spins[ispin] );
                const scalar s2 = bases[iani].k2.dot( spins[ispin] );
                const scalar s3 = bases[iani].k3.dot( spins[ispin] );

                const scalar sin_theta_2 = 1 - s1 * s1;

                scalar result = 0;
                for( auto iterm = site_p[iani]; iterm < site_p[iani + 1]; ++iterm )
                {
                    const auto & [coeff, n1, n2, n3] = terms[iterm];
                    result += coeff * pow( sin_theta_2, n1 ) * pow( s2, n2 ) * pow( s3, n3 );
                }
                energy += result;
            }
        }
    }
    return energy;
};

template<typename F>
void Biaxial_Anisotropy::Hessian_Impl( const vectorfield & spins, F f )
{
    const auto & geometry = getGeometry();
    const int N           = geometry.n_cell_atoms;

    // --- Single Spin elements
#pragma omp parallel for
    for( int icell = 0; icell < geometry.n_cells_total; ++icell )
    {
        for( int iani = 0; iani < indices.size(); ++iani )
        {
            int ispin = icell * N + indices[iani];
            if( check_atom_type( geometry.atom_types[ispin] ) )
            {
                const auto & [k1, k2, k3] = bases[iani];

                const scalar s1 = k1.dot( spins[ispin] );
                const scalar s2 = k2.dot( spins[ispin] );
                const scalar s3 = k3.dot( spins[ispin] );

                const scalar st2 = 1 - s1 * s1;

                for( auto iterm = site_p[iani]; iterm < site_p[iani + 1]; ++iterm )
                {
                    const auto & [coeff, n1, n2, n3] = terms[iterm];

                    const scalar a = pow( s2, n2 );
                    const scalar b = pow( s3, n3 );
                    const scalar c = pow( st2, n1 );
                    // clang-format off
                    const scalar p_11 = n1 <= 1 ? 0
                        : 2 * n1 * ( 2 * n1 * s1 * s1 - 1 ) * ( coeff * a * b * pow( st2, n1 - 2 ) );
                    const scalar p_22 = n2 <= 1 ? 0
                        : n2 * ( n2 - 1 ) * ( coeff * b * c * pow( s2, n2 - 2 ) );
                    const scalar p_33 = n3 <= 1 ? 0
                        : n3 * ( n3 - 1 ) * ( coeff * a * c * pow( s3, n3 - 2 ) );
                    const scalar p_12 = n2 == 0 || n1 == 0 ? 0
                        : b * coeff * n2 * pow( s2, n2 - 1 ) * ( -2 * n1 * s1 ) * pow( s1, n1 - 1 );
                    const scalar p_13 = n3 == 0 || n1 == 0 ? 0
                        : a * coeff * n3 * pow( s3, n3 - 1 ) * ( -2 * n1 * s1 ) * pow( s1, n1 - 1 );
                    const scalar p_23 = n2 == 0 || n3 == 0 ? 0
                        : c * coeff * n2 * pow( s2, n2 - 1 ) * n3 * pow( s3, n3 - 1 );
                    // clang-format on

#pragma unroll
                    for( int alpha = 0; alpha < 3; ++alpha )
                    {
#pragma unroll
                        for( int beta = 0; beta < 3; ++beta )
                        {
                            f( 3 * ispin + alpha, 3 * ispin + beta,
                               k1[alpha] * ( p_11 * k1[beta] + p_12 * k2[beta] + p_13 * k3[beta] )
                                   + k2[alpha] * ( p_12 * k1[beta] + p_22 * k2[beta] + p_23 * k3[beta] )
                                   + k3[alpha] * ( p_13 * k1[beta] + p_23 * k2[beta] + p_33 * k3[beta] ) );
                        }
                    }
                }
            }
        }
    }
}

void Biaxial_Anisotropy::Hessian( const vectorfield & spins, MatrixX & hessian )
{
    Hessian_Impl( spins, [&hessian]( const auto i, const auto j, const scalar value ) { hessian( i, j ) += value; } );
};

void Biaxial_Anisotropy::Sparse_Hessian( const vectorfield & spins, std::vector<triplet> & hessian )
{
    Hessian_Impl(
        spins, [&hessian]( const auto i, const auto j, const scalar value ) { hessian.emplace_back( i, j, value ); } );
};

#ifdef SPIRIT_USE_CUDA
__global__ void CU_Gradient_Biaxial_Anisotropy(
    const Vector3 * spins, const int * atom_types, const int n_cell_atoms, const int n_anisotropies,
    const int * indices, const PolynomialBasis * bases, const unsigned int * site_p, const PolynomialTerm * terms,
    Vector3 * gradient, size_t n_cells_total )
{
    for( auto icell = blockIdx.x * blockDim.x + threadIdx.x; icell < n_cells_total; icell += blockDim.x * gridDim.x )
    {
        for( int iani = 0; iani < n_anisotropies; ++iani )
        {
            int ispin = icell * n_cell_atoms + indices[iani];
            if( cu_check_atom_type( atom_types[ispin] ) )
            {
                Vector3 result = Vector3::Zero();

                const auto & [k1, k2, k3] = bases[iani];

                const scalar s1 = k1.dot( spins[ispin] );
                const scalar s2 = k2.dot( spins[ispin] );
                const scalar s3 = k3.dot( spins[ispin] );

                const scalar sin_theta_2 = 1 - s1 * s1;

                for( auto iterm = site_p[iani]; iterm < site_p[iani + 1]; ++iterm )
                {
                    const auto & [coeff, n1, n2, n3] = terms[iterm];

                    const scalar a = pow( s2, n2 );
                    const scalar b = pow( s3, n3 );
                    const scalar c = pow( sin_theta_2, n1 );

                    if( n1 > 0 )
                        result += k1 * ( coeff * a * b * n1 * ( -2.0 * s1 * pow( sin_theta_2, n1 - 1 ) ) );
                    if( n2 > 0 )
                        result += k2 * ( coeff * b * c * n2 * pow( s2, n2 - 1 ) );
                    if( n3 > 0 )
                        result += k3 * ( coeff * a * c * n3 * pow( s3, n3 - 1 ) );
                }

                gradient[ispin] += result;
            }
        }
    }
}
#endif

void Biaxial_Anisotropy::Gradient( const vectorfield & spins, vectorfield & gradient )
{
    const auto & geometry = getGeometry();
    const int N           = geometry.n_cell_atoms;

#ifdef SPIRIT_USE_CUDA
    const int size                 = geometry.n_cells_total;
    static constexpr int blockSize = 768;
    CU_Gradient_Biaxial_Anisotropy<<<( size - 1 + blockSize ) / blockSize, blockSize>>>(
        spins.data(), geometry.atom_types.data(), geometry.n_cell_atoms, this->indices.size(), this->indices.data(),
        this->bases.data(), this->site_p.data(), this->terms.data(), gradient.data(), size );
    CU_CHECK_AND_SYNC();
#else

#pragma omp parallel for
    for( int icell = 0; icell < geometry.n_cells_total; ++icell )
    {
        for( int iani = 0; iani < indices.size(); ++iani )
        {
            int ispin = icell * N + indices[iani];
            if( check_atom_type( geometry.atom_types[ispin] ) )
            {
                const auto & [k1, k2, k3] = bases[iani];

                const scalar s1 = k1.dot( spins[ispin] );
                const scalar s2 = k2.dot( spins[ispin] );
                const scalar s3 = k3.dot( spins[ispin] );

                const scalar sin_theta_2 = 1 - s1 * s1;

                Vector3 result = Vector3::Zero();
                for( auto iterm = site_p[iani]; iterm < site_p[iani + 1]; ++iterm )
                {
                    const auto & [coeff, n1, n2, n3] = terms[iterm];

                    const scalar a = pow( s2, n2 );
                    const scalar b = pow( s3, n3 );
                    const scalar c = pow( sin_theta_2, n1 );

                    if( n1 > 0 )
                        result += k1 * ( coeff * a * b * n1 * ( -2.0 * s1 * pow( sin_theta_2, n1 - 1 ) ) );
                    if( n2 > 0 )
                        result += k2 * ( coeff * b * c * n2 * pow( s2, n2 - 1 ) );
                    if( n3 > 0 )
                        result += k3 * ( coeff * a * c * n3 * pow( s3, n3 - 1 ) );
                }

                gradient[ispin] += result;
            }
        }
    }
#endif
};

} // namespace Interaction

} // namespace Spin

} // namespace Engine
