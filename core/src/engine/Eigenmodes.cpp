#include <engine/Eigenmodes.hpp>
#include <engine/Manifoldmath.hpp>
#include <engine/Vectormath.hpp>
// #include <engine/Backend_par.hpp>

#include <SymEigsSolver.h> // Also includes <MatOp/DenseSymMatProd.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <fmt/format.h>
#include <fmt/ostream.h>

namespace Engine
{
namespace Eigenmodes
{
using Utility::Log_Level;
using Utility::Log_Sender;

void Check_Eigenmode_Parameters( std::shared_ptr<Data::Spin_System> system )
{
    int nos        = system->nos;
    auto & n_modes = system->ema_parameters->n_modes;
    if( n_modes > 2 * nos - 2 )
    {
        n_modes = 2 * nos - 2;
        system->modes.resize( n_modes );
        system->eigenvalues.resize( n_modes );

        Log( Log_Level::Warning, Log_Sender::EMA,
             fmt::format(
                 "Number of eigenmodes declared in "
                 "EMA Parameters is too large. The number is set to {}",
                 n_modes ) );
    }
    if( n_modes != system->modes.size() )
        system->modes.resize( n_modes );

    // Initial check of selected_mode
    auto & n_mode_follow = system->ema_parameters->n_mode_follow;
    if( n_mode_follow > n_modes - 1 )
    {
        Log( Log_Level::Warning, Log_Sender::EMA,
             fmt::format(
                 "Eigenmode number {} is not "
                 "available. The largest eigenmode ({}) is used instead",
                 n_mode_follow, n_modes - 1 ) );
        n_mode_follow = n_modes - 1;
    }
}

void Calculate_Eigenmodes( std::shared_ptr<Data::Spin_System> system, int idx_img, int idx_chain )
{
    int nos = system->nos;

    Check_Eigenmode_Parameters( system );

    auto & n_modes = system->ema_parameters->n_modes;

    // vectorfield mode(nos, Vector3{1, 0, 0});
    vectorfield spins_initial = *system->spins;

    Log( Log_Level::Info, Log_Sender::EMA, fmt::format( "Started calculation of {} Eigenmodes ", n_modes ), idx_img,
         idx_chain );

    // Calculate the Eigenmodes
    vectorfield gradient( nos );
    MatrixX hessian( 3 * nos, 3 * nos );

    // The gradient (unprojected)
    system->hamiltonian->Gradient( spins_initial, gradient );
    auto mask = system->geometry->mask_unpinned.data();
    auto g    = gradient.data();
    // Backend::par::apply(gradient.size(), [g, mask] SPIRIT_LAMBDA (int idx) {
    //     g[idx] = mask[idx]*g[idx];
    // });
    Vectormath::set_c_a( 1, gradient, gradient, system->geometry->mask_unpinned );

    // The Hessian (unprojected)
    system->hamiltonian->Hessian( spins_initial, hessian );

    // Get the eigenspectrum
    MatrixX hessian_constrained = MatrixX::Zero( 2 * nos, 2 * nos );
    MatrixX tangent_basis       = MatrixX::Zero( 3 * nos, 2 * nos );
    VectorX eigenvalues;
    MatrixX eigenvectors;
    bool successful = Eigenmodes::Hessian_Partial_Spectrum(
        system->ema_parameters, spins_initial, gradient, hessian, n_modes, tangent_basis, hessian_constrained,
        eigenvalues, eigenvectors );

    if( successful )
    {
        // get every mode and save it to system->modes
        for( int i = 0; i < n_modes; i++ )
        {
            // Extract the minimum mode (transform evec_lowest_2N back to 3N)
            VectorX evec_3N = tangent_basis * eigenvectors.col( i );

            // dynamically allocate the system->modes
            system->modes[i] = std::shared_ptr<vectorfield>( new vectorfield( nos, Vector3{ 1, 0, 0 } ) );

            // Set the modes
            for( int j = 0; j < nos; j++ )
                ( *system->modes[i] )[j] = { evec_3N[3 * j], evec_3N[3 * j + 1], evec_3N[3 * j + 2] };

            // get the eigenvalues
            system->eigenvalues[i] = eigenvalues( i );
        }

        Log( Log_Level::Info, Log_Sender::All, fmt::format( "Finished calculation of {} Eigenmodes ", n_modes ),
             idx_img, idx_chain );

        int ev_print = std::min( n_modes, 100 );
        Log( Log_Level::Info, Log_Sender::EMA,
             fmt::format( "Eigenvalues: {}", eigenvalues.head( ev_print ).transpose() ), idx_img, idx_chain );
    }
    else
    {
        //// TODO: What to do then?
        Log( Log_Level::Warning, Log_Sender::All, "Something went wrong in eigenmode calculation...", idx_img,
             idx_chain );
    }
}

bool Hessian_Full_Spectrum(
    const std::shared_ptr<Data::Parameters_Method> parameters, const vectorfield & spins, const vectorfield & gradient,
    const MatrixX & hessian, MatrixX & tangent_basis, MatrixX & hessian_constrained, VectorX & eigenvalues,
    MatrixX & eigenvectors )
{
    int nos = spins.size();

    // Calculate the final Hessian to use for the minimum mode
    // TODO: add option to choose different Hessian calculation
    hessian_constrained = MatrixX::Zero( 2 * nos, 2 * nos );
    tangent_basis       = MatrixX::Zero( 3 * nos, 2 * nos );
    Manifoldmath::hessian_bordered( spins, gradient, hessian, tangent_basis, hessian_constrained );
    // Manifoldmath::hessian_projected(spins, gradient, hessian, tangent_basis, hessian_constrained);
    // Manifoldmath::hessian_weingarten(spins, gradient, hessian, tangent_basis, hessian_constrained);
    // Manifoldmath::hessian_spherical(spins, gradient, hessian, tangent_basis, hessian_constrained);
    // Manifoldmath::hessian_covariant(spins, gradient, hessian, tangent_basis, hessian_constrained);

    // Create and initialize a Eigen solver. Note: the hessian matrix should be symmetric!
    Eigen::SelfAdjointEigenSolver<MatrixX> hessian_spectrum( hessian_constrained );

    // Extract real eigenvalues
    eigenvalues = hessian_spectrum.eigenvalues().real();
    // Retrieve the real eigenvectors
    eigenvectors = hessian_spectrum.eigenvectors().real();

    // Return whether the calculation was successful
    return true;
}

bool Hessian_Partial_Spectrum(
    const std::shared_ptr<Data::Parameters_Method> parameters, const vectorfield & spins, const vectorfield & gradient,
    const MatrixX & hessian, int n_modes, MatrixX & tangent_basis, MatrixX & hessian_constrained, VectorX & eigenvalues,
    MatrixX & eigenvectors )
{
    int nos = spins.size();

    // Restrict number of calculated modes to [1,2N)
    n_modes = std::max( 1, std::min( 2 * nos - 2, n_modes ) );

    // If we have only one spin, we can only calculate the full spectrum
    if( n_modes == nos )
        return Hessian_Full_Spectrum(
            parameters, spins, gradient, hessian, tangent_basis, hessian_constrained, eigenvalues, eigenvectors );

    // Calculate the final Hessian to use for the minimum mode
    // TODO: add option to choose different Hessian calculation
    hessian_constrained = MatrixX::Zero( 2 * nos, 2 * nos );
    tangent_basis       = MatrixX::Zero( 3 * nos, 2 * nos );
    Manifoldmath::hessian_bordered( spins, gradient, hessian, tangent_basis, hessian_constrained );
// Manifoldmath::hessian_projected(spins, gradient, hessian, tangent_basis, hessian_constrained);
// Manifoldmath::hessian_weingarten(spins, gradient, hessian, tangent_basis, hessian_constrained);
// Manifoldmath::hessian_spherical(spins, gradient, hessian, tangent_basis, hessian_constrained);
// Manifoldmath::hessian_covariant(spins, gradient, hessian, tangent_basis, hessian_constrained);

// Remove degrees of freedom of pinned spins
#ifdef SPIRIT_ENABLE_PINNING
    for( int i = 0; i < nos; ++i )
    {
        // TODO: pinning is now in Data::Geometry
        // if (!parameters->pinning->mask_unpinned[i])
        // {
        //     // Remove interaction block
        //     for (int j=0; j<nos; ++j)
        //     {
        //         hessian_constrained.block<2,2>(2*i,2*j).setZero();
        //         hessian_constrained.block<2,2>(2*j,2*i).setZero();
        //     }
        //     // Set diagonal matrix entries of pinned spins to a large value
        //     hessian_constrained.block<2,2>(2*i,2*i).setZero();
        //     hessian_constrained.block<2,2>(2*i,2*i).diagonal().setConstant(nos*1e5);
        // }
    }
#endif // SPIRIT_ENABLE_PINNING

    // Create the Spectra Matrix product operation
    Spectra::DenseSymMatProd<scalar> op( hessian_constrained );
    // Create and initialize a Spectra solver
    Spectra::SymEigsSolver<scalar, Spectra::SMALLEST_ALGE, Spectra::DenseSymMatProd<scalar>> hessian_spectrum(
        &op, n_modes, 2 * nos );
    hessian_spectrum.init();

    // Compute the specified spectrum, sorted by smallest real eigenvalue
    int nconv = hessian_spectrum.compute( 1000, 1e-10, int( Spectra::SMALLEST_ALGE ) );

    // Extract real eigenvalues
    eigenvalues = hessian_spectrum.eigenvalues().real();

    // Retrieve the real eigenvectors
    eigenvectors = hessian_spectrum.eigenvectors().real();

    // Return whether the calculation was successful
    return ( hessian_spectrum.info() == Spectra::SUCCESSFUL ) && ( nconv > 0 );
}

} // namespace Eigenmodes
} // namespace Engine
