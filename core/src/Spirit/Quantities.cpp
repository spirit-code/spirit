#include <Spirit/Quantities.h>
#include <Spirit/Geometry.h>
#include <data/State.hpp>
#include <engine/HTST.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>
#include <engine/Manifoldmath.hpp>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
//#include <unsupported/Eigen/CXX11/Tensor>
#include <GenEigsSolver.h>  // Also includes <MatOp/DenseGenMatProd.h>
#include <GenEigsRealShiftSolver.h>

void Quantity_Get_Magnetization(State * state,  float m[3], int idx_image, int idx_chain)
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        // image->Lock(); // Mutex locks in these functions may cause problems with the performance of UIs
        
        auto mag = Engine::Vectormath::Magnetization(*image->spins);
        image->M = Vector3{ mag[0], mag[1], mag[2] };

        // image->Unlock();
        
        for (int i=0; i<3; ++i) 
            m[i] = (float)mag[i];
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

float Quantity_Get_Topological_Charge(State * state, int idx_image, int idx_chain)
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        // image->Lock(); // Mutex locks in these functions may cause problems with the performance of UIs

        scalar charge = 0;
        int dimensionality = Geometry_Get_Dimensionality(state, idx_image, idx_chain);
        if (dimensionality == 2)
            charge = Engine::Vectormath::TopologicalCharge(*image->spins, 
                        image->geometry->positions, image->geometry->triangulation());

        // image->Unlock();
        
        return (float)charge;
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
        return 0;
    }
}




float Quantity_Get_HTST_Prefactor(State * state, int idx_image_minimum, int idx_image_sp, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image_minimum, image_sp;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image_minimum, idx_chain, image_minimum, chain);
    from_indices(state, idx_image_sp, idx_chain, image_sp, chain);

    return (float)Engine::HTST::Get_Prefactor(image_minimum, image_sp);
}


void Quantity_Get_Grad_Force_MinimumMode(State * state, float * f_grad, float * eval, float * mode, float * forces, int idx_image, int idx_chain)
{
    using namespace Engine;

    std::shared_ptr<Data::Spin_System> system;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, system, chain);

    // Copy std::vector<Eigen::Vector3> into one single Eigen::VectorX
    const int nos = system->nos;
    auto& image = *system->spins;

    vectorfield gradient(nos, {0,0,0});
    vectorfield minimum_mode(nos, {0,0,0});
    MatrixX hessian(3*nos,3*nos);
    vectorfield force(nos, {0,0,0});
    // std::vector<float> forces(3*nos);

    // The gradient force (unprojected)
    system->hamiltonian->Gradient(image, gradient);
    Vectormath::set_c_a(1, gradient, gradient, system->llg_parameters->pinning->mask_unpinned);


    // The Hessian (unprojected)
    system->hamiltonian->Hessian(image, hessian);
    for (int i = 0; i < nos; ++i)
    {
        for (int j = 0; j < nos; ++j)
        {
            if ( ! (system->llg_parameters->pinning->mask_unpinned[i] && system->llg_parameters->pinning->mask_unpinned[j]) )
                hessian.block<3, 3>(3*i, 3*j).setZero();
        }
    }

    // Handy mappings
    VectorX conf = Eigen::Map<VectorX>(image[0].data(), 3 * nos);
    VectorX grad = Eigen::Map<VectorX>(gradient[0].data(), 3 * nos);

    // Output
    for (unsigned int _i = 0; _i < nos; ++_i)
    {
        for (int dim=0; dim<3; ++dim)
        {
            f_grad[3*_i+dim] = (float)-gradient[_i][dim];
        }
    }



    // Calculate the final Hessian to use for the minimum mode
    MatrixX hessian_final = MatrixX::Zero(2*nos, 2*nos);
    // Manifoldmath::hessian_bordered(image, gradient, hessian, hessian_final);
    // Manifoldmath::hessian_projected(image, gradient, hessian, hessian_final);
    Manifoldmath::hessian_weingarten(image, gradient, hessian, hessian_final);
    // Manifoldmath::hessian_spherical(image, gradient, hessian, hessian_final);
    // Manifoldmath::hessian_covariant(image, gradient, hessian, hessian_final);

    Spectra::DenseGenMatProd<scalar> op(hessian_final);
    Spectra::GenEigsSolver< scalar, Spectra::SMALLEST_REAL, Spectra::DenseGenMatProd<scalar> > hessian_spectrum(&op, 1, 2*nos);
    hessian_spectrum.init();
    //		Compute the specified spectrum
    int nconv = hessian_spectrum.compute();
    if (hessian_spectrum.info() == Spectra::SUCCESSFUL)
    {
        // Calculate the Force
        // 		Retrieve the Eigenvalues
        VectorX evalues = hessian_spectrum.eigenvalues().real();
        eval[0] = evalues[0];
        scalar eval_lowest = evalues[0];
        // std::cerr << "region:         " << evalues << std::endl;
        //std::cerr << "spectra val: " << std::endl << hessian_spectrum.eigenvalues() << std::endl;
        //std::cerr << "spectra vec: " << std::endl << -reorder_vector(hessian_spectrum.eigenvectors().col(0).real()) << std::endl;

        // Create the Minimum Mode
        // Retrieve the Eigenvectors
        MatrixX evectors = hessian_spectrum.eigenvectors().real();
        Eigen::Ref<VectorX> evec_lowest_2N = evectors.col(0);

        // Extract the minimum mode (transform evec_lowest_2N back to 3N)
        MatrixX basis_3Nx2N = MatrixX::Zero(3*nos, 2*nos);
        Manifoldmath::tangent_basis_spherical(image, basis_3Nx2N); // Important to choose the right matrix here!
        VectorX evec_lowest_3N = basis_3Nx2N * evec_lowest_2N;
        for (int n=0; n<nos; ++n)
            minimum_mode[n] = {evec_lowest_3N[3*n], evec_lowest_3N[3*n+1], evec_lowest_3N[3*n+2]};
        
        VectorX emode = evec_lowest_3N;

        ////////////////////////////////////////////////////////////////
        // Check for complex numbers
        if (std::abs(hessian_spectrum.eigenvalues().imag()[0]) > 1e-8)
            std::cerr << "     >>>>>>>> WARNING  nonzero complex EW    WARNING" << std::endl; 
        for (int ispin=0; ispin<nos; ++ispin)
        {
            if (std::abs(hessian_spectrum.eigenvectors().col(0).imag()[0]) > 1e-8)
                std::cerr << "     >>>>>>>> WARNING  nonzero complex EV x  WARNING" << std::endl; 
            if (std::abs(hessian_spectrum.eigenvectors().col(0).imag()[1]) > 1e-8)
                std::cerr << "     >>>>>>>> WARNING  nonzero complex EV y  WARNING" << std::endl; 
            if (std::abs(hessian_spectrum.eigenvectors().col(0).imag()[2]) > 1e-8)
                std::cerr << "     >>>>>>>> WARNING  nonzero complex EV z  WARNING" << std::endl; 
        }
        ////////////////////////////////////////////////////////////////


        ////////////////////////////////////////////////////////////////
        // Get the gradient in 2N-representation
        Eigen::Ref<VectorX> grad_3N = Eigen::Map<VectorX>(gradient[0].data(), 3 * nos);
        VectorX grad_2N             = basis_3Nx2N.transpose() * grad_3N;
        // For one of the tests
        auto grad_tangential = gradient;
        Manifoldmath::project_tangential(grad_tangential, image);
        /////////
        // Norms
        scalar image_norm           = Manifoldmath::norm(image);
        scalar grad_norm            = Manifoldmath::norm(gradient);
        scalar grad_tangent_norm    = Manifoldmath::norm(grad_tangential);
        scalar mode_norm            = Manifoldmath::norm(minimum_mode);
        scalar mode_norm_2N         = evec_lowest_2N.norm();
        // Scalar products
        scalar mode_dot_image       = std::abs(Vectormath::dot(minimum_mode, image) / mode_norm); // mode should be orthogonal to image in 3N-space
        scalar mode_dot_grad        = std::abs(evec_lowest_3N.dot(grad_3N) / evec_lowest_3N.norm() / grad_3N.norm());
        scalar mode_dot_grad_2N     = std::abs(evec_lowest_2N.dot(grad_2N) / evec_lowest_2N.norm() / grad_2N.norm());
        // Do some more checks to ensure the mode fulfills our requirements
        bool bad_image_norm         = 1e-8  < std::abs( image_norm - std::sqrt((scalar)nos) ); // image norm should be sqrt(nos)
        bool bad_grad_norm          = 1e-8  > grad_norm;         // gradient should not be a zero vector
        bool bad_grad_tangent_norm  = 1e-8  > grad_tangent_norm; // gradient should not be a zero vector in tangent space
        bool bad_mode_norm          = 1e-8  > mode_norm;         // mode should not be a zero vector
        /////////
        bool bad_mode_dot_image     = 1e-10 < mode_dot_image;    // mode should be orthogonal to image in 3N-space
        bool bad_mode_dot_grad      = 1e-8  > mode_dot_grad;     // mode should not be orthogonal to gradient in 3N-space
        bool bad_mode_dot_grad_2N   = 1e-8  > mode_dot_grad_2N;  // mode should not be orthogonal to gradient in 2N-space
        // /////////
        // if ( bad_image_norm     || bad_mode_norm     || bad_grad_norm        || bad_grad_tangent_norm ||
        //         bad_mode_dot_image || bad_mode_dot_grad || bad_mode_dot_grad_2N )
        // {
        //     // scalar theta, phi;
        //     // Manifoldmath::spherical_from_cartesian(image[1], theta, phi);
        //     std::cerr << "-------------------------"                       << std::endl;
        //     std::cerr << "BAD MODE! evalue =      " << evalues[0]          << std::endl;
        //     // std::cerr << "image (theta,phi):      " << theta << " " << phi << std::endl;
        //     std::cerr << "image norm:             " << image_norm          << std::endl;
        //     std::cerr << "mode norm:              " << mode_norm           << std::endl;
        //     std::cerr << "mode norm 2N:           " << mode_norm_2N        << std::endl;
        //     std::cerr << "grad norm:              " << grad_norm           << std::endl;
        //     std::cerr << "grad norm tangential:   " << grad_tangent_norm   << std::endl;
        //     if (bad_image_norm)
        //         std::cerr << "   image norm is not equal to sqrt(nos): " << image_norm << std::endl;
        //     if (bad_mode_norm)
        //         std::cerr << "   mode norm is too small: "               << mode_norm  << std::endl;
        //     if (bad_grad_norm)
        //         std::cerr << "   gradient norm is too small: "           << grad_norm  << std::endl;
        //     if (bad_mode_dot_image)
        //     {
        //         std::cerr << "   mode NOT TANGENTIAL to SPINS: "         << mode_dot_image << std::endl;
        //         std::cerr << "             >>> check the (3N x 2N) spherical basis matrix" << std::endl;
        //     }
        //     if (bad_mode_dot_grad || bad_mode_dot_grad_2N)
        //     {
        //         std::cerr << "   mode is ORTHOGONAL to GRADIENT: 3N = " << mode_dot_grad << std::endl;
        //         std::cerr << "                              >>>  2N = " << mode_dot_grad_2N << std::endl;
        //     }
        //     std::cerr << "-------------------------" << std::endl;
        // }
        // ////////////////////////////////////////////////////////////////


        // Make sure the mode is a tangent vector
        Manifoldmath::project_tangential(minimum_mode, image);
        // Normalize the mode vector in 3N dimensions
        Manifoldmath::normalize(minimum_mode);


        // 		Check if the lowest eigenvalue is negative
        if (eval_lowest <= 0 && mode_dot_grad > 1e-8)// || switched2)
        {
            std::cerr << "negative region " << evalues.transpose() << std::endl;//<< "    lowest " << eval_lowest << std::endl;

            // Invert the gradient force along the minimum mode
            Manifoldmath::invert_parallel(gradient, minimum_mode);

            // Copy out the forces
            Vectormath::set_c_a(-1, gradient, force, system->llg_parameters->pinning->mask_unpinned);
        }
        // Otherwise we seek for the lowest nonzero eigenvalue
        else if (mode_dot_grad > 1e-8)
        {
            // Scalar product of mode and gradient
            scalar lambda_F = Vectormath::dot(minimum_mode, gradient);

            std::cerr << "positive region " << evalues.transpose() << "  lambda*F=" << lambda_F << std::endl;//<< "    lowest " << eval_lowest << std::endl;

            // Calculate the force
            Vectormath::set_c_a(lambda_F, minimum_mode, force, system->llg_parameters->pinning->mask_unpinned);

            // // Copy out the forces
            // Vectormath::set_c_a(1, grad, force, system->llg_parameters->pinning->mask_unpinned);
        }
        else
        {
            std::cerr << "bad region " << evalues.transpose() << std::endl;
            // Copy out the forces
            Vectormath::set_c_a(1, gradient, force, system->llg_parameters->pinning->mask_unpinned);
        }

        // Copy out the forces
        for (unsigned int _i = 0; _i < nos; ++_i)
        {
            for (int dim=0; dim<3; ++dim)
            {
                // gradient[3*_i+dim] = -grad[3*_i+dim];
                forces[3*_i+dim] = (float)force[_i][dim];
                mode[3*_i+dim] = (float)minimum_mode[_i][dim];
            }
        }
    }
    else
    {
        std::cerr << "DID NOT COMPUTE SPECTRA" << std::endl;
    }
}