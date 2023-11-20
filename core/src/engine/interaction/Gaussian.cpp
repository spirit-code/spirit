#include <engine/Vectormath.hpp>
#include <engine/Indexing.hpp>
#include <engine/interaction/Gaussian.hpp>
#include <utility/Exception.hpp>

using namespace Data;

#ifdef SPIRIT_USE_CUDA
using Engine::Indexing::cu_check_atom_type;
using Engine::Indexing::cu_idx_from_pair;
using Engine::Indexing::cu_tupel_from_idx;
#endif

namespace Engine
{

namespace Interaction
{

Gaussian::Gaussian( Hamiltonian * hamiltonian, scalarfield amplitude, scalarfield width, vectorfield center ) noexcept
        : Interaction::Base<Gaussian>( hamiltonian, scalarfield( 0 ) ),
          n_gaussians( amplitude.size() ),
          amplitude( std::move( amplitude ) ),
          width( std::move( width ) ),
          center( std::move( center ) )
{
    this->updateGeometry();
}

void Gaussian::updateFromGeometry( const Geometry * geometry ) {}

bool Gaussian::is_contributing() const
{
    return n_gaussians > 0;
}

void Gaussian::Energy_per_Spin( const vectorfield & spins, scalarfield & energy )
{
    std::size_t nos = spins.size();

    for( int i = 0; i < this->n_gaussians; ++i )
    {
        for( std::size_t ispin = 0; ispin < nos; ++ispin )
        {
            // Energy contribution
            energy[ispin] += this->Energy_Single_Spin_Single_Gaussian( ispin, i, spins );
        }
    }
}

scalar Gaussian::Energy_Single_Spin( int ispin, const vectorfield & spins )
{
    scalar energy = 0;
    for( int i = 0; i < this->n_gaussians; ++i )
    {
        // Energy contribution
        energy += this->Energy_Single_Spin_Single_Gaussian( ispin, i, spins );
    }
    return energy;
}

scalar Gaussian::Energy_Single_Spin_Single_Gaussian( const int ispin, const int igauss, const vectorfield & spins )
{
    // Distance between spin and gaussian center
    scalar l = 1 - this->center[igauss].dot( spins[ispin] );
    return this->amplitude[igauss] * std::exp( -std::pow( l, 2 ) / ( 2.0 * std::pow( this->width[igauss], 2 ) ) );
}

void Gaussian::Hessian( const vectorfield & spins, MatrixX & hessian )
{
    std::size_t nos = spins.size();
    for( std::size_t ispin = 0; ispin < nos; ++ispin )
    {
        // Calculate Hessian
        for( int igauss = 0; igauss < this->n_gaussians; ++igauss )
        {
            // Distance between spin and gaussian center
            scalar l = 1 - this->center[igauss].dot( spins[ispin] );
            // Prefactor for all alpha, beta
            scalar prefactor
                = this->amplitude[igauss] * std::exp( -std::pow( l, 2 ) / ( 2.0 * std::pow( this->width[igauss], 2 ) ) )
                  / std::pow( this->width[igauss], 2 ) * ( std::pow( l, 2 ) / std::pow( this->width[igauss], 2 ) - 1 );
            // Effective Field contribution
            for( std::uint8_t alpha = 0; alpha < 3; ++alpha )
            {
                for( std::uint8_t beta = 0; beta < 3; ++beta )
                {
                    std::size_t i = 3 * ispin + alpha;
                    std::size_t j = 3 * ispin + beta;
                    hessian( i, j ) += prefactor * this->center[igauss][alpha] * this->center[igauss][beta];
                }
            }
        }
    }
}

void Gaussian::Sparse_Hessian( const vectorfield & spins, std::vector<triplet> & hessian )
{
    // Not implemented
    spirit_throw(
        Utility::Exception_Classifier::Not_Implemented, Utility::Log_Level::Error,
        "Tried to use Sparse_Hessian() of the of the Interaction::Gaussian!" );
};

void Gaussian::Gradient( const vectorfield & spins, vectorfield & gradient )
{
    std::size_t nos = spins.size();

    for( std::size_t ispin = 0; ispin < nos; ++ispin )
    {
        // Calculate gradient
        for( int i = 0; i < this->n_gaussians; ++i )
        {
            // Distance between spin and gaussian center
            scalar l
                = 1
                  - this->center[i].dot( spins[ispin] ); // Utility::Manifoldmath::Dist_Greatcircle(this->center[i], n);
            // Scalar product of spin and gaussian center
            // scalar nc = 0;
            // for (int dim = 0; dim < 3; ++dim) nc += spins[ispin + dim*nos] * this->center[i][dim];
            // Prefactor
            scalar prefactor = this->amplitude[i]
                               * std::exp( -std::pow( l, 2 ) / ( 2.0 * std::pow( this->width[i], 2 ) ) ) * l
                               / std::pow( this->width[i], 2 );
            // Gradient contribution
            gradient[ispin] += prefactor * this->center[i];
        }
    }
}

} // namespace Interaction

} // namespace Engine
