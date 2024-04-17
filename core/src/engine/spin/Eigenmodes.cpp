#include <engine/Backend.hpp>
#include <engine/Manifoldmath.hpp>
#include <engine/Vectormath.hpp>
#include <engine/spin/Eigenmodes.hpp>
#include <utility/Formatters_Eigen.hpp>

#include <Spectra/MatOp/SparseSymMatProd.h> // Also includes <Spectra/MatOp/DenseSymMatProd.h>
#include <Spectra/SymEigsSolver.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <memory>

using Utility::Log_Level;
using Utility::Log_Sender;

namespace Engine
{

namespace Spin
{

namespace Eigenmodes
{

void Check_Eigenmode_Parameters( State::system_t & system )
{
    int nos        = system.nos;
    auto & n_modes = system.ema_parameters->n_modes;
    if( n_modes > 2 * nos - 2 )
    {
        n_modes = 2 * nos - 2;
        system.modes.resize( n_modes );
        system.eigenvalues.resize( n_modes );

        Log( Log_Level::Warning, Log_Sender::EMA,
             fmt::format(
                 "Number of eigenmodes declared in "
                 "EMA Parameters is too large. The number is set to {}",
                 n_modes ) );
    }
    if( n_modes != system.modes.size() )
        system.modes.resize( n_modes );

    // Initial check of selected_mode
    auto & n_mode_follow = system.ema_parameters->n_mode_follow;
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

void Calculate_Eigenmodes( State::system_t & system, int idx_img, int idx_chain )
{
    int nos = system.nos;

    Check_Eigenmode_Parameters( system );

    auto & n_modes = system.ema_parameters->n_modes;

    // vectorfield mode(nos, Vector3{1, 0, 0});
    vectorfield spins_initial = *system.spins;

    Log( Log_Level::Info, Log_Sender::EMA, fmt::format( "Started calculation of {} Eigenmodes ", n_modes ), idx_img,
         idx_chain );

    // Calculate the Eigenmodes
    vectorfield gradient( nos );

    // The gradient (unprojected)
    system.hamiltonian->Gradient( spins_initial, gradient );
    // auto mask = system.geometry->mask_unpinned.data();
    // auto g    = gradient.data();
    // Backend::for_each_n( SPIRIT_PAR Backend::make_counting_iterator( 0 ), gradient.size(), [g, mask]
    // SPIRIT_LAMBDA (int idx) {
    //     g[idx] = mask[idx]*g[idx];
    // });
    Vectormath::set_c_a( 1, gradient, gradient, system.hamiltonian->get_geometry().mask_unpinned );

    VectorX eigenvalues;
    MatrixX eigenvectors;
    SpMatrixX tangent_basis = SpMatrixX( 3 * nos, 2 * nos );

    bool sparse     = system.ema_parameters->sparse;
    bool successful = false;
    if( sparse )
    {
        // The Hessian (unprojected)
        SpMatrixX hessian( 3 * nos, 3 * nos );
        system.hamiltonian->Sparse_Hessian( spins_initial, hessian );
        // Get the eigenspectrum
        SpMatrixX hessian_constrained = SpMatrixX( 2 * nos, 2 * nos );

        successful = Eigenmodes::Sparse_Hessian_Partial_Spectrum(
            system.ema_parameters, spins_initial, gradient, hessian, n_modes, tangent_basis, hessian_constrained,
            eigenvalues, eigenvectors );
    }
    else
    {
        // The Hessian (unprojected)
        MatrixX hessian( 3 * nos, 3 * nos );
        system.hamiltonian->Hessian( spins_initial, hessian );
        // Get the eigenspectrum
        MatrixX hessian_constrained = MatrixX::Zero( 2 * nos, 2 * nos );
        MatrixX _tangent_basis      = MatrixX( tangent_basis );

        successful = Eigenmodes::Hessian_Partial_Spectrum(
            system.ema_parameters, spins_initial, gradient, hessian, n_modes, _tangent_basis, hessian_constrained,
            eigenvalues, eigenvectors );

        tangent_basis = _tangent_basis.sparseView();
    }

    if( successful )
    {
        // get every mode and save it to system.modes
        for( int i = 0; i < n_modes; i++ )
        {
            // Extract the minimum mode (transform evec_lowest_2N back to 3N)
            VectorX evec_3N = tangent_basis * eigenvectors.col( i );

            // dynamically allocate the system.modes
            system.modes[i] = std::make_shared<vectorfield>( nos, Vector3{ 1, 0, 0 } );

            // Set the modes
            for( int j = 0; j < nos; j++ )
                ( *system.modes[i] )[j] = { evec_3N[3 * j], evec_3N[3 * j + 1], evec_3N[3 * j + 2] };

            // get the eigenvalues
            system.eigenvalues[i] = eigenvalues( i );
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
    std::size_t nos = spins.size();

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
    const MatrixX & hessian, std::size_t n_modes, MatrixX & tangent_basis, MatrixX & hessian_constrained,
    VectorX & eigenvalues, MatrixX & eigenvectors )
{
    std::size_t nos = spins.size();

    // Restrict number of calculated modes to [1,2N)
    n_modes = std::max( static_cast<std::size_t>( 1 ), std::min( 2 * nos - 2, n_modes ) );

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
    for( std::size_t i = 0; i < nos; ++i )
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
    Spectra::SymEigsSolver<Spectra::DenseSymMatProd<scalar>> hessian_spectrum( op, n_modes, 2 * nos );
    hessian_spectrum.init();

    // Compute the specified spectrum, sorted by smallest real eigenvalue
    int nconv
        = hessian_spectrum.compute( Spectra::SortRule::SmallestAlge, 1000, 1e-10, Spectra::SortRule::SmallestAlge );

    // Extract real eigenvalues
    eigenvalues = hessian_spectrum.eigenvalues().real();

    // Retrieve the real eigenvectors
    eigenvectors = hessian_spectrum.eigenvectors().real();

    // Return whether the calculation was successful
    return ( hessian_spectrum.info() == Spectra::CompInfo::Successful ) && ( nconv > 0 );
}

bool Sparse_Hessian_Partial_Spectrum(
    const std::shared_ptr<Data::Parameters_Method> parameters, const vectorfield & spins, const vectorfield & gradient,
    const SpMatrixX & hessian, int n_modes, SpMatrixX & tangent_basis, SpMatrixX & hessian_constrained,
    VectorX & eigenvalues, MatrixX & eigenvectors )
{
    int nos = spins.size();

    // Restrict number of calculated modes to [1,2N)
    n_modes = std::max( 1, std::min( 2 * nos - 2, n_modes ) );

    // Calculate the final Hessian to use for the minimum mode
    Manifoldmath::sparse_tangent_basis_spherical( spins, tangent_basis );

    SpMatrixX hessian_constrained_3N = SpMatrixX( 3 * nos, 3 * nos );
    Manifoldmath::sparse_hessian_bordered_3N( spins, gradient, hessian, hessian_constrained_3N );

    hessian_constrained = tangent_basis.transpose() * hessian_constrained_3N * tangent_basis;

    // TODO: Pinning (see non-sparse function for)

    hessian_constrained.makeCompressed();
    int ncv = std::min( 2 * nos, std::max( 2 * n_modes + 1, 20 ) ); // This is the default value used by scipy.sparse
    int max_iter = 20 * nos;

    // Create the Spectra Matrix product operation
    Spectra::SparseSymMatProd<scalar> op( hessian_constrained );
    // Create and initialize a Spectra solver
    Spectra::SymEigsSolver<Spectra::SparseSymMatProd<scalar>> hessian_spectrum( op, n_modes, ncv );
    hessian_spectrum.init();

    // Compute the specified spectrum, sorted by smallest real eigenvalue
    int nconv
        = hessian_spectrum.compute( Spectra::SortRule::SmallestAlge, max_iter, 1e-10, Spectra::SortRule::SmallestAlge );

    // Extract real eigenvalues
    eigenvalues = hessian_spectrum.eigenvalues().real();

    // Retrieve the real eigenvectors
    eigenvectors = hessian_spectrum.eigenvectors().real();

    // Return whether the calculation was successful
    return ( hessian_spectrum.info() == Spectra::CompInfo::Successful ) && ( nconv > 0 );
}

} // namespace Eigenmodes

} // namespace Spin

} // namespace Engine
