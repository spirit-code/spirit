#include <data/Spin_System.hpp>
#include <engine/Backend_par.hpp>
#include <engine/Indexing.hpp>
#include <engine/Vectormath.hpp>
#include <engine/interaction/Biaxial_Anisotropy.hpp>
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

namespace Interaction
{

Biaxial_Anisotropy::Biaxial_Anisotropy(
    Hamiltonian * hamiltonian, intfield indices, field<AnisotropyPolynomial> polynomials ) noexcept
        : Interaction::Base<Biaxial_Anisotropy>( hamiltonian, scalarfield( 0 ) ),
          anisotropy_indices( std::move( indices ) ),
          anisotropy_polynomials( std::move( polynomials ) )
{
    this->updateGeometry();
}

void Biaxial_Anisotropy::updateFromGeometry( const Geometry * geometry ) {}

bool Biaxial_Anisotropy::is_contributing() const
{
    return !anisotropy_indices.empty();
}

#ifdef SPIRIT_USE_CUDA
__global__ void CU_E_Biaxial_Anisotropy(
    const Vector3 * spins, const int * atom_types, const int n_cell_atoms, const int n_anisotropies,
    const int * anisotropy_indices, const AnisotropyPolynomial * anisotropy_polynomials, scalar * energy,
    size_t n_cells_total )
{
    for( auto icell = blockIdx.x * blockDim.x + threadIdx.x; icell < n_cells_total; icell += blockDim.x * gridDim.x )
    {
        for( int iani = 0; iani < n_anisotropies; ++iani )
        {
            int ispin = icell * n_cell_atoms + anisotropy_indices[iani];
            if( cu_check_atom_type( atom_types[ispin] ) )
            {
                const auto & poly = anisotropy_polynomials[iani];
                const scalar s1   = poly.k1.dot( spins[ispin] );
                const scalar s2   = poly.k2.dot( spins[ispin] );
                const scalar s3   = poly.k3.dot( spins[ispin] );

                const scalar sin_theta_2 = 1 - s1 * s1;

                scalar result = 0;
                for( const auto & [coeff, n1, n2, n3] : poly.terms )
                {
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
    const auto * geometry = hamiltonian->geometry.get();
    const int N           = geometry->n_cell_atoms;

#ifdef SPIRIT_USE_CUDA
    int size = geometry->n_cells_total;
    CU_E_Biaxial_Anisotropy<<<( size + 1023 ) / 1024, 1024>>>(
        spins.data(), geometry->atom_types.data(), geometry->n_cell_atoms, this->anisotropy_indices.size(),
        this->anisotropy_indices.data(), this->anisotropy_polynomials.data(), energy.data(), size );
    CU_CHECK_AND_SYNC();
#else

#pragma omp parallel for
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( int iani = 0; iani < anisotropy_indices.size(); ++iani )
        {
            int ispin = icell * N + anisotropy_indices[iani];
            if( check_atom_type( geometry->atom_types[ispin] ) )
            {
                const auto & poly = anisotropy_polynomials[iani];
                const scalar s1   = poly.k1.dot( spins[ispin] );
                const scalar s2   = poly.k2.dot( spins[ispin] );
                const scalar s3   = poly.k3.dot( spins[ispin] );

                const scalar sin_theta_2 = 1 - s1 * s1;

                scalar result = 0;
                for( const auto & [coeff, n1, n2, n3] : poly.terms )
                {
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
    const auto * geometry = hamiltonian->geometry.get();
    const int N           = geometry->n_cell_atoms;

    int icell  = ispin / N;
    int ibasis = ispin - icell * geometry->n_cell_atoms;

    for( int iani = 0; iani < anisotropy_indices.size(); ++iani )
    {
        if( anisotropy_indices[iani] == ibasis )
        {
            if( check_atom_type( geometry->atom_types[ispin] ) )
            {
                const auto & poly = anisotropy_polynomials[iani];
                const scalar s1   = poly.k1.dot( spins[ispin] );
                const scalar s2   = poly.k2.dot( spins[ispin] );
                const scalar s3   = poly.k3.dot( spins[ispin] );

                const scalar sin_theta_2 = 1 - s1 * s1;

                scalar result = 0;
                for( const auto & [coeff, n1, n2, n3] : anisotropy_polynomials[iani].terms )
                {
                    result += coeff * pow( sin_theta_2, n1 ) * pow( s2, n2 ) * pow( s3, n3 );
                }
                energy += result;
            }
        }
    }
    return energy;
};

void Biaxial_Anisotropy::Hessian( const vectorfield & spins, MatrixX & hessian )
{
    const auto * geometry = hamiltonian->geometry.get();
    const int N           = geometry->n_cell_atoms;

    // --- Single Spin elements
#pragma omp parallel for
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( int iani = 0; iani < anisotropy_indices.size(); ++iani )
        {
            int ispin = icell * N + anisotropy_indices[iani];
            if( check_atom_type( geometry->atom_types[ispin] ) )
            {
                const auto & poly = anisotropy_polynomials[iani];
                const scalar s1   = poly.k1.dot( spins[ispin] );
                const scalar s2   = poly.k2.dot( spins[ispin] );
                const scalar s3   = poly.k3.dot( spins[ispin] );

                const scalar st2 = 1 - s1 * s1;

                for( const auto & [coeff, n1, n2, n3] : poly.terms )
                {

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

                            hessian( 3 * ispin + alpha, 3 * ispin + beta )
                                += poly.k1[alpha]
                                       * ( p_11 * poly.k1[beta] + p_12 * poly.k2[beta] + p_13 * poly.k3[beta] )
                                   + poly.k2[alpha]
                                         * ( p_12 * poly.k1[beta] + p_22 * poly.k2[beta] + p_23 * poly.k3[beta] )
                                   + poly.k3[alpha]
                                         * ( p_13 * poly.k1[beta] + p_23 * poly.k2[beta] + p_33 * poly.k3[beta] );
                        }
                    }
                }
            }
        }
    }
};

void Biaxial_Anisotropy::Sparse_Hessian( const vectorfield & spins, std::vector<triplet> & hessian )
{
    const auto * geometry = hamiltonian->geometry.get();
    const int N           = geometry->n_cell_atoms;

    // --- Single Spin elements
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( int iani = 0; iani < anisotropy_indices.size(); ++iani )
        {
            int ispin = icell * N + anisotropy_indices[iani];
            if( check_atom_type( geometry->atom_types[ispin] ) )
            {
                const auto & poly = anisotropy_polynomials[iani];
                const scalar s1   = poly.k1.dot( spins[ispin] );
                const scalar s2   = poly.k2.dot( spins[ispin] );
                const scalar s3   = poly.k3.dot( spins[ispin] );

                const scalar st2 = 1 - s1 * s1;

                for( const auto & [coeff, n1, n2, n3] : poly.terms )
                {

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

                            hessian.emplace_back(
                                3 * ispin + alpha, 3 * ispin + beta,
                                poly.k1[alpha] * ( p_11 * poly.k1[beta] + p_12 * poly.k2[beta] + p_13 * poly.k3[beta] )
                                    + poly.k2[alpha]
                                          * ( p_12 * poly.k1[beta] + p_22 * poly.k2[beta] + p_23 * poly.k3[beta] )
                                    + poly.k3[alpha]
                                          * ( p_13 * poly.k1[beta] + p_23 * poly.k2[beta] + p_33 * poly.k3[beta] ) );
                        }
                    }
                }
            }
        }
    }
};

#ifdef SPIRIT_USE_CUDA
__global__ void CU_Gradient_Biaxial_Anisotropy(
    const Vector3 * spins, const int * atom_types, const int n_cell_atoms, const int n_anisotropies,
    const int * anisotropy_indices, const AnisotropyPolynomial * anisotropy_polynomials, Vector3 * gradient,
    size_t n_cells_total )
{
    for( auto icell = blockIdx.x * blockDim.x + threadIdx.x; icell < n_cells_total; icell += blockDim.x * gridDim.x )
    {
        for( int iani = 0; iani < anisotropy_indices.size(); ++iani )
        {
            int ispin = icell * N + anisotropy_indices[iani];
            if( check_atom_type( geometry->atom_types[ispin] ) )
            {
                Vector3 result = Vector3::Zero();

                const auto & poly = anisotropy_polynomials[iani];
                const scalar s1   = poly.k1.dot( spins[ispin] );
                const scalar s2   = poly.k2.dot( spins[ispin] );
                const scalar s3   = poly.k3.dot( spins[ispin] );

                const scalar sin_theta_2 = 1 - s1 * s1;

                for( const auto & [coeff, n1, n2, n3] : poly.terms )
                {
                    using std::pow;
                    const scalar a = pow( s2, n2 );
                    const scalar b = pow( s3, n3 );
                    const scalar c = pow( sin_theta_2, n1 );

                    if( n1 > 0 )
                        result += poly.k1 * ( coeff * a * b * n1 * ( -2 * s1 * pow( sin_theta_2, n1 - 1 ) ) );
                    if( n2 > 0 )
                        result += poly.k2 * ( coeff * b * c * n2 * pow( s2, n2 - 1 ) );
                    if( n3 > 0 )
                        result += poly.k3 * ( coeff * a * c * n3 * pow( s3, n3 - 1 ) );
                }

                gradient[ispin] += result;
            }
        }
    }
}
#endif

void Biaxial_Anisotropy::Gradient( const vectorfield & spins, vectorfield & gradient )
{
    const auto * geometry = hamiltonian->geometry.get();
    const int N           = geometry->n_cell_atoms;

#ifdef SPIRIT_USE_CUDA
    int size = geometry->n_cells_total;
    CU_Gradient_Biaxial_Anisotropy<<<( size + 1023 ) / 1024, 1024>>>(
        spins.data(), geometry->atom_types.data(), geometry->n_cell_atoms, this->anisotropy_indices.size(),
        this->anisotropy_indices.data(), this->anisotropy_polynomials.data(), gradient.data(), size );
    CU_CHECK_AND_SYNC();
#else

#pragma omp parallel for
    for( int icell = 0; icell < geometry->n_cells_total; ++icell )
    {
        for( int iani = 0; iani < anisotropy_indices.size(); ++iani )
        {
            int ispin = icell * N + anisotropy_indices[iani];
            if( check_atom_type( geometry->atom_types[ispin] ) )
            {
                Vector3 result = Vector3::Zero();

                const auto & poly = anisotropy_polynomials[iani];
                const scalar s1   = poly.k1.dot( spins[ispin] );
                const scalar s2   = poly.k2.dot( spins[ispin] );
                const scalar s3   = poly.k3.dot( spins[ispin] );

                const scalar sin_theta_2 = 1 - s1 * s1;

                for( const auto & [coeff, n1, n2, n3] : poly.terms )
                {
                    const scalar a = pow( s2, n2 );
                    const scalar b = pow( s3, n3 );
                    const scalar c = pow( sin_theta_2, n1 );

                    if( n1 > 0 )
                        result += poly.k1 * ( coeff * a * b * n1 * ( -2 * s1 * pow( sin_theta_2, n1 - 1 ) ) );
                    if( n2 > 0 )
                        result += poly.k2 * ( coeff * b * c * n2 * pow( s2, n2 - 1 ) );
                    if( n3 > 0 )
                        result += poly.k3 * ( coeff * a * c * n3 * pow( s3, n3 - 1 ) );
                }

                gradient[ispin] += result;
            }
        }
    }
#endif
};

} // namespace Interaction

} // namespace Engine
