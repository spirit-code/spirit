#include <Spirit/Geometry.h>
#include <Spirit/Quantities.h>

#include <data/State.hpp>
#include <engine/Eigenmodes.hpp>
#include <engine/Manifoldmath.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Constants.hpp>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>

namespace C = Utility::Constants;

void Quantity_Get_Magnetization( State * state, float m[3], int idx_image, int idx_chain )
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    // image->Lock(); // Mutex locks in these functions may cause problems with the performance of UIs

    auto mag = Engine::Vectormath::Magnetization( *image->spins );
    image->M = Vector3{ mag[0], mag[1], mag[2] };

    // image->Unlock();

    for( int i = 0; i < 3; ++i )
        m[i] = (float)mag[i];
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

float Quantity_Get_Topological_Charge( State * state, int idx_image, int idx_chain )
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    // image->Lock(); // Mutex locks in these functions may cause problems with the performance of UIs

    scalar charge      = 0;
    int dimensionality = Geometry_Get_Dimensionality( state, idx_image, idx_chain );
    if( dimensionality == 2 )
        charge = Engine::Vectormath::TopologicalCharge(
            *image->spins, *image->geometry, image->hamiltonian->boundary_conditions );

    // image->Unlock();

    return (float)charge;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

void check_modes(
    const vectorfield & image, const vectorfield & grad, const MatrixX & tangent_basis, const VectorX & eigenvalues,
    const MatrixX & eigenvectors_2N, const vectorfield & minimum_mode )
{
    using namespace Engine;
    using namespace Utility;

    int nos = image.size();

    // ////////////////////////////////////////////////////////////////
    // // Check for complex numbers in the eigenvalues
    // if (std::abs(hessian_spectrum.eigenvalues().imag()[0]) > 1e-8)
    //     std::cerr << "     >>>>>>>> WARNING  nonzero complex EW    WARNING" << std::endl;
    // for (int ispin=0; ispin<nos; ++ispin)
    // {
    //     if (std::abs(hessian_spectrum.eigenvectors().col(0).imag()[0]) > 1e-8)
    //         std::cerr << "     >>>>>>>> WARNING  nonzero complex EV x  WARNING" << std::endl;
    //     if (std::abs(hessian_spectrum.eigenvectors().col(0).imag()[1]) > 1e-8)
    //         std::cerr << "     >>>>>>>> WARNING  nonzero complex EV y  WARNING" << std::endl;
    //     if (std::abs(hessian_spectrum.eigenvectors().col(0).imag()[2]) > 1e-8)
    //         std::cerr << "     >>>>>>>> WARNING  nonzero complex EV z  WARNING" << std::endl;
    // }
    // ////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////
    // For one of the tests
    auto grad_tangential = grad;
    Manifoldmath::project_tangential( grad_tangential, image );
    // Get the tangential gradient in 2N-representation
    Eigen::Ref<VectorX> grad_tangent_3N = Eigen::Map<VectorX>( grad_tangential[0].data(), 3 * nos );
    VectorX grad_tangent_2N             = tangent_basis.transpose() * grad_tangent_3N;
    // Eigenstuff
    scalar eval_lowest     = eigenvalues[0];
    VectorX evec_lowest_2N = eigenvectors_2N.col( 0 );
    VectorX evec_lowest_3N = tangent_basis * evec_lowest_2N;
    /////////
    // Norms
    scalar image_norm        = Manifoldmath::norm( image );
    scalar grad_norm         = Manifoldmath::norm( grad );
    scalar grad_tangent_norm = Manifoldmath::norm( grad_tangential );
    scalar mode_norm         = Manifoldmath::norm( minimum_mode );
    scalar mode_norm_2N      = evec_lowest_2N.norm();
    // Scalar products
    scalar mode_dot_image = std::abs(
        Vectormath::dot( minimum_mode, image ) / mode_norm ); // mode should be orthogonal to image in 3N-space
    scalar mode_grad_angle
        = std::abs( evec_lowest_3N.dot( grad_tangent_3N ) / evec_lowest_3N.norm() / grad_tangent_3N.norm() );
    scalar mode_grad_angle_2N
        = std::abs( evec_lowest_2N.dot( grad_tangent_2N ) / evec_lowest_2N.norm() / grad_tangent_2N.norm() );
    // Do some more checks to ensure the mode fulfills our requirements
    bool bad_image_norm = 1e-8 < std::abs( image_norm - std::sqrt( (scalar)nos ) ); // image norm should be sqrt(nos)
    bool bad_grad_norm  = 1e-8 > grad_norm;                // gradient should not be a zero vector
    bool bad_grad_tangent_norm = 1e-8 > grad_tangent_norm; // gradient should not be a zero vector in tangent space
    bool bad_mode_norm         = 1e-8 > mode_norm;         // mode should not be a zero vector
    /////////
    bool bad_mode_dot_image     = 1e-10 < mode_dot_image;    // mode should be orthogonal to image in 3N-space
    bool bad_mode_grad_angle    = 1e-8 > mode_grad_angle;    // mode should not be orthogonal to gradient in 3N-space
    bool bad_mode_grad_angle_2N = 1e-8 > mode_grad_angle_2N; // mode should not be orthogonal to gradient in 2N-space
    /////////
    bool eval_nonzero = 1e-8 < std::abs( eval_lowest );
    /////////
    if( bad_image_norm || bad_mode_norm || bad_grad_norm || bad_grad_tangent_norm || bad_mode_dot_image
        || ( eval_nonzero && ( bad_mode_grad_angle || bad_mode_grad_angle_2N ) ) )
    {
        // scalar theta, phi;
        // Manifoldmath::spherical_from_cartesian(image[1], theta, phi);
        std::cerr << "-------------------------" << std::endl;
        std::cerr << "BAD MODE! evalue =      " << eigenvalues[0] << std::endl;
        // std::cerr << "image (theta,phi):      " << theta << " " << phi << std::endl;
        std::cerr << "image norm:             " << image_norm << std::endl;
        std::cerr << "mode norm:              " << mode_norm << std::endl;
        std::cerr << "mode norm 2N:           " << mode_norm_2N << std::endl;
        std::cerr << "grad norm:              " << grad_norm << std::endl;
        std::cerr << "grad norm tangential:   " << grad_tangent_norm << std::endl;
        if( bad_image_norm )
            std::cerr << "   image norm is not equal to sqrt(nos): " << image_norm << std::endl;
        if( bad_mode_norm )
            std::cerr << "   mode norm is too small: " << mode_norm << std::endl;
        if( bad_grad_norm )
            std::cerr << "   gradient norm is too small: " << grad_norm << std::endl;
        if( bad_mode_dot_image )
        {
            std::cerr << "   mode NOT TANGENTIAL to SPINS: " << mode_dot_image << std::endl;
            std::cerr << "             >>> check the (3N x 2N) spherical basis matrix" << std::endl;
        }
        if( eval_nonzero && ( bad_mode_grad_angle || bad_mode_grad_angle_2N ) )
        {
            std::cerr << "   mode is ORTHOGONAL to GRADIENT: 3N = " << mode_grad_angle << std::endl;
            std::cerr << "                              >>>  2N = " << mode_grad_angle_2N << std::endl;
        }
        std::cerr << "-------------------------" << std::endl;
    }
    ////////////////////////////////////////////////////////////////
}

void Quantity_Get_Grad_Force_MinimumMode(
    State * state, float * f_grad, float * eval, float * mode, float * forces, int idx_image, int idx_chain )
{
    using namespace Engine;
    using namespace Utility;

    std::shared_ptr<Data::Spin_System> system;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices( state, idx_image, idx_chain, system, chain );

    // Copy std::vector<Eigen::Vector3> into one single Eigen::VectorX
    const int nos = system->nos;
    auto & image  = *system->spins;

    vectorfield grad( nos, { 0, 0, 0 } );
    vectorfield minimum_mode( nos, { 0, 0, 0 } );
    MatrixX hess( 3 * nos, 3 * nos );
    vectorfield force( nos, { 0, 0, 0 } );
    // std::vector<float> forces(3*nos);

    // The gradient force (unprojected)
    system->hamiltonian->Gradient( image, grad );
    Vectormath::set_c_a( 1, grad, grad, system->geometry->mask_unpinned );

    // Output
    for( unsigned int _i = 0; _i < nos; ++_i )
    {
        for( int dim = 0; dim < 3; ++dim )
        {
            f_grad[3 * _i + dim] = (float)-grad[_i][dim];
        }
    }

    // The Hessian (unprojected)
    system->hamiltonian->Hessian( image, hess );

    // Number of lowest modes to be calculated
    // NOTE THE ORDER OF THE MODES: the first eigenvalue is not necessarily the lowest for n>1
    int n_modes       = 6;
    int mode_positive = 0;
    mode_positive     = std::max( 0, std::min( n_modes - 1, mode_positive ) );

    Eigen::Ref<VectorX> image_3N = Eigen::Map<VectorX>( image[0].data(), 3 * nos );
    Eigen::Ref<VectorX> grad_3N  = Eigen::Map<VectorX>( grad[0].data(), 3 * nos );

    // The gradient (unprojected)
    system->hamiltonian->Gradient( image, grad );
    Vectormath::set_c_a( 1, grad, grad, system->geometry->mask_unpinned );

    // The Hessian (unprojected)
    system->hamiltonian->Hessian( image, hess );
// Remove blocks of pinned spins
#ifdef SPIRIT_ENABLE_PINNING
    for( int i = 0; i < nos; ++i )
    {
        for( int j = 0; j < nos; ++j )
        {
            if( ( !system->geometry->mask_unpinned[i] ) || ( !system->geometry->mask_unpinned[j] ) )
            {
                hess.block<3, 3>( 3 * i, 3 * j ).setZero();
            }
        }
    }
#endif // SPIRIT_ENABLE_PINNING

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Get the eigenspectrum
    MatrixX hessian_final = MatrixX::Zero( 2 * nos, 2 * nos );
    MatrixX basis_3Nx2N   = MatrixX::Zero( 3 * nos, 2 * nos );
    VectorX eigenvalues;
    MatrixX eigenvectors;
    bool successful = Eigenmodes::Hessian_Partial_Spectrum(
        system->mmf_parameters, image, grad, hess, n_modes, basis_3Nx2N, hessian_final, eigenvalues, eigenvectors );

    if( successful )
    {
        // Determine the mode to follow
        VectorX mode_3N;
        if( eigenvalues[0] < -1e-6 )
        {
            // Retrieve the minimum mode
            mode_3N = basis_3Nx2N * eigenvectors.col( 0 );
            for( int n = 0; n < nos; ++n )
                minimum_mode[n] = { mode_3N[3 * n], mode_3N[3 * n + 1], mode_3N[3 * n + 2] };
        }
        else
        {
            // Retrieve the chosen mode
            mode_3N = basis_3Nx2N * eigenvectors.col( mode_positive );
            for( int n = 0; n < nos; ++n )
                minimum_mode[n] = { mode_3N[3 * n], mode_3N[3 * n + 1], mode_3N[3 * n + 2] };
        }

        // Get the scalar product of mode and gradient
        scalar mode_grad = mode_3N.dot( grad_3N );
        // Get the angle between mode and gradient (in the tangent plane!)
        VectorX grad_tangent_3N = grad_3N - grad_3N.dot( image_3N ) * image_3N;
        scalar mode_grad_angle  = std::abs( mode_grad / ( mode_3N.norm() * grad_3N.norm() ) );

        // Make sure there is nothing wrong
        check_modes( image, grad, basis_3Nx2N, eigenvalues, eigenvectors, minimum_mode );

        // If the lowest eigenvalue is negative, we follow the minimum mode
        if( eigenvalues[0] < -1e-6 && mode_grad_angle > 1e-8 ) // -1e-6)// || switched2)
        {
            std::cerr << fmt::format(
                "negative region: {:<20}   angle = {:15.10f}   lambda*F = {:15.10f}", eigenvalues.transpose(),
                std::acos( std::min( mode_grad_angle, scalar( 1.0 ) ) ) * 180.0 / C::Pi, std::abs( mode_grad ) )
                      << std::endl;

            // Invert the gradient force along the minimum mode
            Manifoldmath::invert_parallel( grad, minimum_mode );

            // Copy out the forces
            Vectormath::set_c_a( -1, grad, force, system->geometry->mask_unpinned );
        }
        // Otherwise we follow some chosen mode, as long as it is not orthogonal to the gradient
        else if( mode_grad_angle > 1e-8 )
        {
            std::cerr << fmt::format(
                "positive region: {:<20}   angle = {:15.10f}   lambda*F = {:15.10f}", eigenvalues.transpose(),
                std::acos( std::min( mode_grad_angle, scalar( 1.0 ) ) ) * 180.0 / C::Pi, std::abs( mode_grad ) )
                      << std::endl;

            int sign = ( scalar( 0 ) < mode_grad ) - ( mode_grad < scalar( 0 ) );

            // Calculate the force
            // Vectormath::set_c_a(mode_grad, minimum_mode, force, system->geometry->mask_unpinned);
            Vectormath::set_c_a( sign, minimum_mode, force, system->geometry->mask_unpinned );

            // // Copy out the forces
            // Vectormath::set_c_a(1, grad, force, system->geometry->mask_unpinned);
        }
        else
        {
            if( std::abs( eigenvalues[0] ) > 1e-8 )
                std::cerr << fmt::format(
                    "bad region:        {:<20}   angle = {:15.10f}   lambda*F = {:15.10f}", eigenvalues.transpose(),
                    std::acos( std::min( mode_grad_angle, scalar( 1.0 ) ) ) * 180.0 / C::Pi, std::abs( mode_grad ) )
                          << std::endl;
            else
                std::cerr << fmt::format(
                    "zero region:       {:<20}   angle = {:15.10f}   lambda*F = {:15.10f}", eigenvalues.transpose(),
                    std::acos( std::min( mode_grad_angle, scalar( 1.0 ) ) ) * 180.0 / C::Pi, std::abs( mode_grad ) )
                          << std::endl;

            // Copy out the forces
            Vectormath::set_c_a( 1, grad, force, system->geometry->mask_unpinned );
        }
    }
    else
    {
        // Spectra was not successful in calculating an eigenvector
        Log( Log_Level::Error, Log_Sender::MMF, "Failed to calculate eigenvectors of the Hessian!" );
        Log( Log_Level::Info, Log_Sender::MMF, "Zeroing the MMF force..." );
        for( auto & x : force )
            x.setZero();
    }

    // Copy out the forces
    for( unsigned int _i = 0; _i < nos; ++_i )
    {
        for( int dim = 0; dim < 3; ++dim )
        {
            // gradient[3*_i+dim] = -grad[3*_i+dim];
            forces[3 * _i + dim] = (float)force[_i][dim];
            mode[3 * _i + dim]   = (float)minimum_mode[_i][dim];
        }
    }
}