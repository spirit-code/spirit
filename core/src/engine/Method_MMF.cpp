#include <Spirit_Defines.h>
#include <engine/Eigenmodes.hpp>
#include <engine/Manifoldmath.hpp>
#include <engine/Method_MMF.hpp>
#include <engine/Vectormath.hpp>
#include <io/IO.hpp>
#include <io/OVF_File.hpp>
#include <utility/Logging.hpp>
#include <utility/Version.hpp>

#include <cstring>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <fmt/format.h>

using Utility::Log_Level;
using Utility::Log_Sender;
namespace C = Utility::Constants;

namespace Engine
{

template<Solver solver>
Method_MMF<solver>::Method_MMF( std::shared_ptr<Data::Spin_System> system, int idx_chain )
        : Method_Solver<solver>( system->mmf_parameters, -1, idx_chain )
{
    this->systems = std::vector<std::shared_ptr<Data::Spin_System>>( 1, system );
    this->system  = system;
    this->noi     = this->systems.size();
    this->nos     = this->systems[0]->nos;

    switched1        = false;
    switched2        = false;
    this->SenderName = Utility::Log_Sender::MMF;

    // History
    this->history = std::map<std::string, std::vector<scalar>>{ { "max_torque", { this->max_torque } } };

    // We assume that the systems are not converged before the first iteration
    this->max_torque = system->mmf_parameters->force_convergence + 1.0;

    this->hessian = MatrixX( 3 * this->nos, 3 * this->nos );
    // Forces
    this->gradient     = vectorfield( this->nos, { 0, 0, 0 } );
    this->minimum_mode = vectorfield( this->nos, { 0, 0, 0 } );
    this->xi           = vectorfield( this->nos, { 0, 0, 0 } );

    // Last iteration
    this->spins_last = vectorfield( this->nos );
    this->Rx_last    = 0.0;

    // Force function
    // ToDo: move into parameters
    this->mm_function = "Spectra Matrix"; // "Spectra Matrix" "Spectra Prefactor" "Lanczos"

    this->mode_follow_previous = 0;

    // Create shared pointers to the method's systems' spin configurations
    this->configurations    = std::vector<std::shared_ptr<vectorfield>>( 1 );
    this->configurations[0] = this->system->spins;

    //---- Initialise Solver-specific variables
    this->Initialize();
}

template<Solver solver>
void Method_MMF<solver>::Calculate_Force(
    const std::vector<std::shared_ptr<vectorfield>> & configurations, std::vector<vectorfield> & forces )
{
    // if (this->mm_function == "Spectra Matrix")
    // {
    this->Calculate_Force_Spectra_Matrix( configurations, forces );
// }
// else if (this->mm_function == "Lanczos")
// {
//     this->Calculate_Force_Lanczos(configurations, forces);
// }
#ifdef SPIRIT_ENABLE_PINNING
    Vectormath::set_c_a( 1, forces[0], forces[0], this->systems[0]->geometry->mask_unpinned );
#endif // SPIRIT_ENABLE_PINNING
}

void check_modes(
    const vectorfield & image, const vectorfield & gradient, const MatrixX & tangent_basis, const VectorX & eigenvalues,
    const MatrixX & eigenvectors_2N, const vectorfield & minimum_mode )
{
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
    auto gradient_tangential = gradient;
    Manifoldmath::project_tangential( gradient_tangential, image );
    // Get the tangential gradient in 2N-representation
    Eigen::Ref<VectorX> gradient_tangent_3N = Eigen::Map<VectorX>( gradient_tangential[0].data(), 3 * nos );
    VectorX gradient_tangent_2N             = tangent_basis.transpose() * gradient_tangent_3N;
    // Eigenstuff
    scalar eval_lowest     = eigenvalues[0];
    VectorX evec_lowest_2N = eigenvectors_2N.col( 0 );
    VectorX evec_lowest_3N = tangent_basis * evec_lowest_2N;
    /////////
    // Norms
    scalar image_norm        = Manifoldmath::norm( image );
    scalar grad_norm         = Manifoldmath::norm( gradient );
    scalar grad_tangent_norm = Manifoldmath::norm( gradient_tangential );
    scalar mode_norm         = Manifoldmath::norm( minimum_mode );
    scalar mode_norm_2N      = evec_lowest_2N.norm();
    // Scalar products
    scalar mode_dot_image = std::abs(
        Vectormath::dot( minimum_mode, image ) / mode_norm ); // mode should be orthogonal to image in 3N-space
    scalar mode_grad_angle
        = std::abs( evec_lowest_3N.dot( gradient_tangent_3N ) / evec_lowest_3N.norm() / gradient_tangent_3N.norm() );
    scalar mode_grad_angle_2N
        = std::abs( evec_lowest_2N.dot( gradient_tangent_2N ) / evec_lowest_2N.norm() / gradient_tangent_2N.norm() );
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

template<Solver solver>
void Method_MMF<solver>::Calculate_Force_Spectra_Matrix(
    const std::vector<std::shared_ptr<vectorfield>> & configurations, std::vector<vectorfield> & forces )
{
    auto & image = *configurations[0];
    auto & force = forces[0];
    // auto& force_virtual = forces_virtual[0];
    auto & parameters = *this->systems[0]->mmf_parameters;

    const int nos = this->nos;

    // Number of lowest modes to be calculated
    int n_modes = parameters.n_modes;
    // Mode to follow in the positive region
    int mode_positive = parameters.n_mode_follow;
    mode_positive     = std::max( 0, std::min( n_modes - 1, mode_positive ) );
    // Mode to follow in the negative region
    int mode_negative = parameters.n_mode_follow;
    if( false ) // if (parameters.negative_follow_lowest)
        mode_negative = 0;
    mode_negative = std::max( 0, std::min( n_modes - 1, mode_negative ) );

    // The gradient (unprojected)
    this->systems[0]->hamiltonian->Gradient( image, gradient );
    Vectormath::set_c_a( 1, gradient, gradient, this->systems[0]->geometry->mask_unpinned );

    // The Hessian (unprojected)
    this->systems[0]->hamiltonian->Hessian( image, hessian );

    Eigen::Ref<VectorX> image_3N    = Eigen::Map<VectorX>( image[0].data(), 3 * nos );
    Eigen::Ref<VectorX> gradient_3N = Eigen::Map<VectorX>( gradient[0].data(), 3 * nos );

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Get the eigenspectrum
    MatrixX hessian_final = MatrixX::Zero( 2 * nos, 2 * nos );
    MatrixX basis_3Nx2N   = MatrixX::Zero( 3 * nos, 2 * nos );
    VectorX eigenvalues;
    MatrixX eigenvectors;
    bool successful = Eigenmodes::Hessian_Partial_Spectrum(
        this->parameters, image, gradient, hessian, n_modes, basis_3Nx2N, hessian_final, eigenvalues, eigenvectors );

    if( successful )
    {
        // TODO: if the mode that is followed in the positive region is not no. 0,
        //       the following won't work!!
        //       Need to save the mode_follow as a local variable and update it as necessary.
        //       Question: what to do when MMF is paused and re-started? Do we remember the
        //       correct mode, and if yes how?
        //       Note: it should be distinguished if we are starting in the positive or negative
        //       region. Probably some of this should go into the constructor.

        // Determine the mode to follow
        int mode_follow = mode_negative;
        if( eigenvalues[0] > -1e-6 )
            mode_follow = mode_positive;

        if( true ) // && eigenvalues[0] > -1e-6)
        {
            // Determine if we are still following the same mode and correct if not
            // std::abs(mode_2N_previous.dot(eigenvectors.col(mode_follow))) < 1e-2
            // std::abs(mode_2N_previous.dot(eigenvectors.col(itest)))       >= 1-1e-4
            if( mode_2N_previous.size() > 0 )
            {
                mode_follow          = mode_follow_previous;
                scalar mode_dot_mode = std::abs( mode_2N_previous.dot( eigenvectors.col( mode_follow ) ) );
                if( mode_dot_mode < 0.99 )
                {
                    // Need to look for our mode
                    std::cerr << fmt::format(
                        "Looking for previous mode, which used to be {}...", mode_follow_previous );
                    int ntest = 6;
                    // int start = std::max(0, mode_follow_previous-ntest);
                    // int stop  = std::min(n_modes, mode_follow_previous+ntest);
                    for( int itest = 0; itest < n_modes; ++itest )
                    {
                        scalar m_dot_m_test = std::abs( mode_2N_previous.dot( eigenvectors.col( itest ) ) );
                        if( m_dot_m_test > mode_dot_mode )
                        {
                            mode_follow   = itest;
                            mode_dot_mode = m_dot_m_test;
                        }
                    }
                    if( mode_follow != mode_follow_previous )
                        std::cerr << fmt::format( "Found mode no. {}", mode_follow ) << std::endl;
                    else
                        std::cerr << "Did not find a new mode..." << std::endl;
                }
            }

            // Save chosen mode as "previous" for next iteration
            mode_follow_previous = mode_follow;
            mode_2N_previous     = eigenvectors.col( mode_follow );
        }

        // Ref to correct mode
        Eigen::Ref<VectorX> mode_2N = eigenvectors.col( mode_follow );
        scalar mode_evalue          = eigenvalues[mode_follow];

        // Retrieve the chosen mode as vectorfield
        VectorX mode_3N = basis_3Nx2N * mode_2N;
        for( int n = 0; n < nos; ++n )
            this->minimum_mode[n] = { mode_3N[3 * n], mode_3N[3 * n + 1], mode_3N[3 * n + 2] };

        // Get the scalar product of mode and gradient
        scalar mode_grad = mode_3N.dot( gradient_3N );
        // Get the angle between mode and gradient (in the tangent plane!)
        VectorX graient_tangent_3N = gradient_3N - gradient_3N.dot( image_3N ) * image_3N;
        scalar mode_grad_angle     = std::abs( mode_grad / ( mode_3N.norm() * gradient_3N.norm() ) );

        // Make sure there is nothing wrong
        check_modes( image, gradient, basis_3Nx2N, eigenvalues, eigenvectors, minimum_mode );

        Manifoldmath::project_tangential( gradient, image );

        // Some debugging prints
        if( mode_evalue < -1e-6 && mode_grad_angle > 1e-8 ) // -1e-6)// || switched2)
        {
            std::cerr << fmt::format(
                "negative region: {:<65}   mode={}   angle = {:15.10f}   lambda*F = {:15.10f}", eigenvalues.transpose(),
                mode_follow, std::acos( std::min( mode_grad_angle, scalar( 1.0 ) ) ) * 180.0 / C::Pi,
                std::abs( mode_grad ) )
                      << std::endl;
        }
        else if( mode_grad_angle > 1e-8 )
        {
            std::cerr << fmt::format(
                "positive region: {:<65}   mode={}   angle = {:15.10f}   lambda*F = {:15.10f}", eigenvalues.transpose(),
                mode_follow, std::acos( std::min( mode_grad_angle, scalar( 1.0 ) ) ) * 180.0 / C::Pi,
                std::abs( mode_grad ) )
                      << std::endl;
        }
        else
        {
            if( std::abs( mode_evalue ) > 1e-8 )
            {
                std::cerr << fmt::format(
                    "bad region:      {:<65}   mode={}   angle = {:15.10f}   lambda*F = {:15.10f}",
                    eigenvalues.transpose(), mode_follow,
                    std::acos( std::min( mode_grad_angle, scalar( 1.0 ) ) ) * 180.0 / C::Pi, std::abs( mode_grad ) )
                          << std::endl;
            }
            else
            {
                std::cerr << fmt::format(
                    "zero region:     {:<65}   mode={}   angle = {:15.10f}   lambda*F = {:15.10f}",
                    eigenvalues.transpose(), mode_follow,
                    std::acos( std::min( mode_grad_angle, scalar( 1.0 ) ) ) * 180.0 / C::Pi, std::abs( mode_grad ) )
                          << std::endl;
            }
        }

        // TODO: parameter whether to *always* follow the minimum mode
        if( false )
        {
            // Invert the gradient force along the minimum mode
            Manifoldmath::invert_parallel( gradient, minimum_mode );

            // Copy out the forces
            Vectormath::set_c_a( -1, gradient, force, this->systems[0]->geometry->mask_unpinned );
        }
        else
        {
            if( eigenvalues[0] < -1e-6 && mode_grad_angle > 1e-8 ) // -1e-6)// || switched2)
            {
                // Invert the gradient force along the minimum mode
                Manifoldmath::invert_parallel( gradient, minimum_mode );

                // Copy out the forces
                Vectormath::set_c_a( -1, gradient, force, this->systems[0]->geometry->mask_unpinned );
            }
            else if( mode_grad_angle > 1e-8 )
            {
                // TODO: add switch between gradient and mode following for positive region
                if( false )
                {
                    // Calculate the force
                    // Vectormath::set_c_a(mode_grad, this->minimum_mode, force, this->systems[0]->geometry->mask_unpinned);
                    int sign = ( scalar( 0 ) < mode_grad ) - ( mode_grad < scalar( 0 ) );
                    Vectormath::set_c_a( sign, this->minimum_mode, force, this->systems[0]->geometry->mask_unpinned );
                }
                else
                {
                    // Copy out the forces
                    Vectormath::set_c_a( 1, gradient, force, this->systems[0]->geometry->mask_unpinned );
                }
            }
            else
            {
                if( std::abs( mode_evalue ) > 1e-8 )
                {
                    // Invert the gradient force along the minimum mode
                    Manifoldmath::invert_parallel( gradient, minimum_mode );

                    // Copy out the forces
                    Vectormath::set_c_a( -1, gradient, force, this->systems[0]->geometry->mask_unpinned );
                }
                else
                {
                    // Copy out the forces
                    Vectormath::set_c_a( 1, gradient, force, this->systems[0]->geometry->mask_unpinned );
                }
            }
        }
    }
    else
    {
        // Spectra was not successful in calculating an eigenvector
        Log( Log_Level::Error, Log_Sender::MMF, "Failed to calculate eigenvectors of the Hessian!" );
        Log( Log_Level::Info, Log_Sender::MMF, "Zeroing the MMF force..." );
        Vectormath::fill( force, Vector3{ 0, 0, 0 } );
    }
}

void printmatrix( MatrixX & m )
{
    std::cerr << m << std::endl;
}

// Check if the Forces are converged
template<Solver solver>
bool Method_MMF<solver>::Converged()
{
    if( this->max_torque < this->systems[0]->mmf_parameters->force_convergence )
        return true;
    return false;
}

template<Solver solver>
void Method_MMF<solver>::Hook_Pre_Iteration()
{
}

template<Solver solver>
void Method_MMF<solver>::Hook_Post_Iteration()
{
    // --- Convergence Parameter Update
    this->max_torque = 0;
    // Loop over images to calculate the maximum torques
    for( unsigned int img = 0; img < this->systems.size(); ++img )
    {
        auto fmax = this->MaxTorque_on_Image( *( this->systems[img]->spins ), this->forces_virtual[img] );
        if( fmax > 0 )
            this->max_torque = fmax;
        else
            this->max_torque = 0;
    }
}

template<Solver solver>
void Method_MMF<solver>::Save_Current( std::string starttime, int iteration, bool initial, bool final )
{
    // History save
    this->history["max_torque"].push_back( this->max_torque );

    // File save
    if( this->parameters->output_any )
    {
        // Convert indices to formatted strings
        auto s_img         = fmt::format( "{:0>2}", this->idx_image );
        int base           = (int)log10( this->parameters->n_iterations );
        std::string s_iter = fmt::format( "{:0>" + fmt::format( "{}", base ) + "}", iteration );

        std::string preSpinsFile;
        std::string preEnergyFile;
        std::string fileTag;

        if( this->parameters->output_file_tag == "<time>" )
            fileTag = starttime + "_";
        else if( this->parameters->output_file_tag != "" )
            fileTag = this->parameters->output_file_tag + "_";
        else
            fileTag = "";

        preSpinsFile  = this->parameters->output_folder + "/" + fileTag + "Image-" + s_img + "_Spins";
        preEnergyFile = this->parameters->output_folder + "/" + fileTag + "Image-" + s_img + "_Energy";

        // Function to write or append image and energy files
        auto writeOutputConfiguration
            = [this, preSpinsFile, preEnergyFile, iteration]( std::string suffix, bool append )
        {
            try
            {
                // File name and comment
                std::string spinsFile      = preSpinsFile + suffix + ".ovf";
                std::string output_comment = fmt::format(
                    "{} simulation ({} solver)\n# Desc:      Iteration: {}\n# Desc:      Maximum torque: {}",
                    this->Name(), this->SolverFullName(), iteration, this->max_torque );

                // File format
                IO::VF_FileFormat format = this->systems[0]->mmf_parameters->output_vf_filetype;

                // Spin Configuration
                auto & spins        = *this->systems[0]->spins;
                auto segment        = IO::OVF_Segment( *this->systems[0] );
                std::string title   = fmt::format( "SPIRIT Version {}", Utility::version_full );
                segment.title       = strdup( title.c_str() );
                segment.comment     = strdup( output_comment.c_str() );
                segment.valuedim    = 3;
                segment.valuelabels = strdup( "spin_x spin_y spin_z" );
                segment.valueunits  = strdup( "none none none" );
                if( append )
                    IO::OVF_File( spinsFile ).append_segment( segment, spins[0].data(), int( format ) );
                else
                    IO::OVF_File( spinsFile ).write_segment( segment, spins[0].data(), int( format ) );
            }
            catch( ... )
            {
                spirit_handle_exception_core( "LLG output failed" );
            }
        };

        auto writeOutputEnergy = [this, preSpinsFile, preEnergyFile, iteration]( std::string suffix, bool append )
        {
            bool normalize   = this->systems[0]->llg_parameters->output_energy_divide_by_nspins;
            bool readability = this->systems[0]->llg_parameters->output_energy_add_readability_lines;

            // File name
            std::string energyFile        = preEnergyFile + suffix + ".txt";
            std::string energyFilePerSpin = preEnergyFile + "-perSpin" + suffix + ".txt";

            // Energy
            if( append )
            {
                // Check if Energy File exists and write Header if it doesn't
                std::ifstream f( energyFile );
                if( !f.good() )
                    IO::Write_Energy_Header(
                        *this->systems[0], energyFile, { "iteration", "E_tot" }, true, normalize, readability );
                // Append Energy to File
                IO::Append_Image_Energy( *this->systems[0], iteration, energyFile, normalize, readability );
            }
            else
            {
                IO::Write_Energy_Header(
                    *this->systems[0], energyFile, { "iteration", "E_tot" }, true, normalize, readability );
                IO::Append_Image_Energy( *this->systems[0], iteration, energyFile, normalize, readability );
                if( this->systems[0]->mmf_parameters->output_energy_spin_resolved )
                {
                    // Gather the data
                    std::vector<std::pair<std::string, scalarfield>> contributions_spins( 0 );
                    this->systems[0]->UpdateEnergy();
                    this->systems[0]->hamiltonian->Energy_Contributions_per_Spin(
                        *this->systems[0]->spins, contributions_spins );
                    int datasize = ( 1 + contributions_spins.size() ) * this->systems[0]->nos;
                    scalarfield data( datasize, 0 );
                    for( int ispin = 0; ispin < this->systems[0]->nos; ++ispin )
                    {
                        scalar E_spin = 0;
                        int j         = 1;
                        for( auto & contribution : contributions_spins )
                        {
                            E_spin += contribution.second[ispin];
                            data[ispin + j] = contribution.second[ispin];
                            ++j;
                        }
                        data[ispin] = E_spin;
                    }

                    // Segment
                    auto segment = IO::OVF_Segment( *this->systems[0] );

                    std::string title   = fmt::format( "SPIRIT Version {}", Utility::version_full );
                    segment.title       = strdup( title.c_str() );
                    std::string comment = fmt::format( "Energy per spin. Total={}meV", this->systems[0]->E );
                    for( auto & contribution : this->systems[0]->E_array )
                        comment += fmt::format( ", {}={}meV", contribution.first, contribution.second );
                    segment.comment  = strdup( comment.c_str() );
                    segment.valuedim = 1 + this->systems[0]->E_array.size();

                    std::string valuelabels = "Total";
                    std::string valueunits  = "meV";
                    for( auto & pair : this->systems[0]->E_array )
                    {
                        valuelabels += fmt::format( " {}", pair.first );
                        valueunits += " meV";
                    }
                    segment.valuelabels = strdup( valuelabels.c_str() );

                    // File format
                    IO::VF_FileFormat format = this->systems[0]->llg_parameters->output_vf_filetype;

                    // open and write
                    IO::OVF_File( energyFilePerSpin ).write_segment( segment, data.data(), int( format ) );

                    Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
                         fmt::format( "Wrote spins to file \"{}\" with format {}", energyFilePerSpin, int( format ) ),
                         -1, -1 );
                }
            }
        };

        // Initial image before simulation
        if( initial && this->parameters->output_initial )
        {
            writeOutputConfiguration( "-initial", false );
            writeOutputEnergy( "-initial", false );
        }
        // Final image after simulation
        else if( final && this->parameters->output_final )
        {
            writeOutputConfiguration( "-final", false );
            writeOutputEnergy( "-final", false );
        }

        // Single file output
        if( this->systems[0]->mmf_parameters->output_configuration_step )
        {
            writeOutputConfiguration( "_" + s_iter, false );
        }
        if( this->systems[0]->mmf_parameters->output_energy_step )
        {
            writeOutputEnergy( "_" + s_iter, false );
        }

        // Archive file output (appending)
        if( this->systems[0]->mmf_parameters->output_configuration_archive )
        {
            writeOutputConfiguration( "-archive", true );
        }
        if( this->systems[0]->mmf_parameters->output_energy_archive )
        {
            writeOutputEnergy( "-archive", true );
        }

        // Save Log
        Log.Append_to_File();
    }
}

template<Solver solver>
void Method_MMF<solver>::Finalize()
{
    this->systems[0]->iteration_allowed = false;
}

// Method name as string
template<Solver solver>
std::string Method_MMF<solver>::Name()
{
    return "MMF";
}

// Template instantiations
template class Method_MMF<Solver::SIB>;
template class Method_MMF<Solver::Heun>;
template class Method_MMF<Solver::Depondt>;
template class Method_MMF<Solver::RungeKutta4>;
template class Method_MMF<Solver::LBFGS_OSO>;
template class Method_MMF<Solver::LBFGS_Atlas>;
template class Method_MMF<Solver::VP>;
template class Method_MMF<Solver::VP_OSO>;

} // namespace Engine
