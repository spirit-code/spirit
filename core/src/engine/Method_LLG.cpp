#include <Spirit_Defines.h>
#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <engine/Method_LLG.hpp>
#include <engine/Vectormath.hpp>
#include <io/IO.hpp>
#include <io/OVF_File.hpp>
#include <utility/Logging.hpp>
#include <utility/Version.hpp>

#include <ctime>
#include <iostream>

#include <fmt/format.h>

using namespace Utility;

namespace Engine
{

template<Solver solver>
Method_LLG<solver>::Method_LLG( std::shared_ptr<Data::Spin_System> system, int idx_img, int idx_chain )
        : Method_Solver<solver>( system->llg_parameters, idx_img, idx_chain ), picoseconds_passed( 0 )
{
    // Currently we only support a single image being iterated at once:
    this->systems    = std::vector<std::shared_ptr<Data::Spin_System>>( 1, system );
    this->SenderName = Utility::Log_Sender::LLG;

    this->noi = this->systems.size();
    this->nos = this->systems[0]->nos;

    // Forces
    this->forces                   = std::vector<vectorfield>( this->noi, vectorfield( this->nos ) );
    this->forces_virtual           = std::vector<vectorfield>( this->noi, vectorfield( this->nos ) );
    this->Gradient                 = std::vector<vectorfield>( this->noi, vectorfield( this->nos ) );
    this->xi                       = vectorfield( this->nos, { 0, 0, 0 } );
    this->s_c_grad                 = vectorfield( this->nos, { 0, 0, 0 } );
    this->temperature_distribution = scalarfield( this->nos, 0 );

    // We assume it is not converged before the first iteration
    this->force_converged = std::vector<bool>( this->noi, false );
    this->max_torque      = system->llg_parameters->force_convergence + 1.0;

    // History
    this->history = std::map<std::string, std::vector<scalar>>{ { "max_torque", { this->max_torque } },
                                                                { "E", { this->max_torque } },
                                                                { "M_z", { this->max_torque } } };

    // Create shared pointers to the method's systems' spin configurations
    this->configurations = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for( int i = 0; i < this->noi; ++i )
        this->configurations[i] = this->systems[i]->spins;

    // Allocate force array
    // this->force = std::vector<vectorfield>(this->noi, vectorfield(this->nos, Vector3::Zero()));	// [noi][3*nos]

    //---- Initialise Solver-specific variables
    this->Initialize();

    // Initial force calculation s.t. it does not seem to be already converged
    this->Prepare_Thermal_Field();
    this->Calculate_Force( this->configurations, this->forces );
    this->Calculate_Force_Virtual( this->configurations, this->forces, this->forces_virtual );
    // Post iteration hook to get forceMaxAbsComponent etc
    this->Hook_Post_Iteration();
}

template<Solver solver>
void Method_LLG<solver>::Prepare_Thermal_Field()
{
    auto & parameters = *this->systems[0]->llg_parameters;
    auto & geometry   = *this->systems[0]->geometry;
    auto & damping    = parameters.damping;

    if( parameters.temperature > 0 || parameters.temperature_gradient_inclination != 0 )
    {
        scalar epsilon = std::sqrt( 2 * damping * parameters.dt * Constants::gamma / Constants::mu_B * Constants::k_B )
                         / ( 1 + damping * damping );

        // PRNG gives Gaussian RN with width 1 -> scale by epsilon and sqrt(T/mu_s)
        auto distribution = std::normal_distribution<scalar>{ 0, 1 };

        // If we have a temperature gradient, we use the distribution (scalarfield)
        if( parameters.temperature_gradient_inclination != 0 )
        {
            // Calculate distribution
            Vectormath::get_gradient_distribution(
                geometry, parameters.temperature_gradient_direction, parameters.temperature,
                parameters.temperature_gradient_inclination, this->temperature_distribution, 0, 1e30 );

            // TODO: parallelization of this is actually not quite so trivial
            // #pragma omp parallel for
            for( unsigned int i = 0; i < this->xi.size(); ++i )
            {
                for( int dim = 0; dim < 3; ++dim )
                    this->xi[i][dim] = epsilon * std::sqrt( this->temperature_distribution[i] / geometry.mu_s[i] )
                                       * distribution( parameters.prng );
            }
        }
        // If we only have homogeneous temperature we do it more efficiently
        else if( parameters.temperature > 0 )
        {
            // TODO: parallelization of this is actually not quite so trivial
            // #pragma omp parallel for
            for( unsigned int i = 0; i < this->xi.size(); ++i )
            {
                for( int dim = 0; dim < 3; ++dim )
                    this->xi[i][dim] = epsilon * std::sqrt( parameters.temperature / geometry.mu_s[i] )
                                       * distribution( parameters.prng );
            }
        }
    }
}

template<Solver solver>
void Method_LLG<solver>::Calculate_Force(
    const std::vector<std::shared_ptr<vectorfield>> & configurations, std::vector<vectorfield> & forces )
{
    // Loop over images to calculate the total force on each Image
    for( unsigned int img = 0; img < this->systems.size(); ++img )
    {
        // Minus the gradient is the total Force here
        this->systems[img]->hamiltonian->Gradient_and_Energy( *configurations[img], Gradient[img], current_energy );

#ifdef SPIRIT_ENABLE_PINNING
        Vectormath::set_c_a( 1, Gradient[img], Gradient[img], this->systems[img]->geometry->mask_unpinned );
#endif // SPIRIT_ENABLE_PINNING

        // Copy out
        Vectormath::set_c_a( -1, Gradient[img], forces[img] );
    }
}

template<Solver solver>
void Method_LLG<solver>::Calculate_Force_Virtual(
    const std::vector<std::shared_ptr<vectorfield>> & configurations, const std::vector<vectorfield> & forces,
    std::vector<vectorfield> & forces_virtual )
{
    using namespace Utility;

    for( unsigned int i = 0; i < configurations.size(); ++i )
    {
        auto & image         = *configurations[i];
        auto & force         = forces[i];
        auto & force_virtual = forces_virtual[i];
        auto & parameters    = *this->systems[i]->llg_parameters;

        //////////
        // time steps
        scalar damping = parameters.damping;
        // dt = time_step [ps] * gyromagnetic ratio / mu_B / (1+damping^2) <- not implemented
        scalar dtg     = parameters.dt * Constants::gamma / Constants::mu_B / ( 1 + damping * damping );
        scalar sqrtdtg = dtg / std::sqrt( parameters.dt );
        // STT
        // - monolayer
        scalar a_j      = parameters.stt_magnitude;
        Vector3 s_c_vec = parameters.stt_polarisation_normal;
        // - gradient
        scalar b_j  = a_j;             // pre-factor b_j = u*mu_s/gamma (see bachelorthesis Constantin)
        scalar beta = parameters.beta; // non-adiabatic parameter of correction term
        Vector3 je  = s_c_vec;         // direction of current
        //////////

        // This is the force calculation as it should be for direct minimization
        // TODO: Also calculate force for VP solvers without additional scaling
        if( solver == Solver::LBFGS_OSO || solver == Solver::LBFGS_Atlas )
        {
            Vectormath::set_c_cross( 1.0, image, force, force_virtual );
        }
        else if( parameters.direct_minimization || solver == Solver::VP || solver == Solver::VP_OSO )
        {
            dtg = parameters.dt * Constants::gamma / Constants::mu_B;
            Vectormath::set_c_cross( dtg, image, force, force_virtual );
        }
        // Dynamics simulation
        else
        {
            auto & geometry = *this->systems[0]->geometry;

            Vectormath::set_c_a( dtg, force, force_virtual );
            Vectormath::add_c_cross( dtg * damping, image, force, force_virtual );
            Vectormath::scale( force_virtual, geometry.mu_s, true );

            // STT
            if( a_j > 0 )
            {
                if( parameters.stt_use_gradient )
                {
                    auto & boundary_conditions = this->systems[0]->hamiltonian->boundary_conditions;
                    // Gradient approximation for in-plane currents
                    Vectormath::directional_gradient(
                        image, geometry, boundary_conditions, je, s_c_grad ); // s_c_grad = (j_e*grad)*S
                    Vectormath::add_c_a(
                        dtg * a_j * ( damping - beta ), s_c_grad, force_virtual ); // TODO: a_j durch b_j ersetzen
                    Vectormath::add_c_cross(
                        dtg * a_j * ( 1 + beta * damping ), s_c_grad, image,
                        force_virtual ); // TODO: a_j durch b_j ersetzen
                    // Gradient in current richtung, daher => *(-1)
                }
                else
                {
                    // Monolayer approximation
                    Vectormath::add_c_a( -dtg * a_j * ( damping - beta ), s_c_vec, force_virtual );
                    Vectormath::add_c_cross( -dtg * a_j * ( 1 + beta * damping ), s_c_vec, image, force_virtual );
                }
            }

            // Temperature
            if( parameters.temperature > 0 || parameters.temperature_gradient_inclination != 0 )
            {
                Vectormath::add_c_a( 1, this->xi, force_virtual );
                Vectormath::add_c_cross( damping, image, this->xi, force_virtual );
            }
        }
// Apply Pinning
#ifdef SPIRIT_ENABLE_PINNING
        Vectormath::set_c_a( 1, force_virtual, force_virtual, this->systems[0]->geometry->mask_unpinned );
#endif // SPIRIT_ENABLE_PINNING
    }
}

template<Solver solver>
double Method_LLG<solver>::get_simulated_time()
{
    return this->picoseconds_passed;
}

template<Solver solver>
bool Method_LLG<solver>::Converged()
{
    // Check if all images converged
    return std::all_of( this->force_converged.begin(), this->force_converged.end(), []( bool b ) { return b; } );
}

template<Solver solver>
void Method_LLG<solver>::Hook_Pre_Iteration()
{
}

template<Solver solver>
void Method_LLG<solver>::Hook_Post_Iteration()
{
    // Increment the time counter (picoseconds)
    this->picoseconds_passed += this->systems[0]->llg_parameters->dt;

    // --- Convergence Parameter Update
    // Loop over images to calculate the maximum torques
    for( unsigned int img = 0; img < this->systems.size(); ++img )
    {
        this->force_converged[img] = false;
        // auto fmax = this->Force_on_Image_MaxAbsComponent(*(this->systems[img]->spins), this->forces_virtual[img]);
        auto fmax = this->MaxTorque_on_Image( *( this->systems[img]->spins ), this->forces_virtual[img] );

        if( fmax > 0 )
            this->max_torque = fmax;
        else
            this->max_torque = 0;
        if( fmax < this->systems[img]->llg_parameters->force_convergence )
            this->force_converged[img] = true;
    }

    // --- Image Data Update
    // Update the system's Energy
    // ToDo: copy instead of recalculating

    this->systems[0]->E = current_energy;

    // ToDo: How to update eff_field without numerical overhead?
    // systems[0]->effective_field = Gradient[0];
    // Vectormath::scale(systems[0]->effective_field, -1);
    Manifoldmath::project_tangential( this->forces[0], *this->systems[0]->spins );
    Vectormath::set_c_a( 1, this->forces[0], this->systems[0]->effective_field );
    // systems[0]->UpdateEffectiveField();

    // TODO: In order to update Rx with the neighbouring images etc., we need the state -> how to do this?

    // --- Renormalize Spins?
    // TODO: figure out specialization of members (Method_LLG should hold Parameters_Method_LLG)
    // if (this->parameters->renorm_sd) {
    //     try {
    //         //Vectormath::Normalize(3, s->nos, s->spins);
    //     }
    //     catch (Exception ex)
    // 	{
    //         if (ex == Exception::Division_by_zero)
    // 		{
    // 			Log(Utility::Log_Level::Warning, Utility::Log_Sender::LLG, "During Iteration Spin = (0,0,0) was
    // detected. Using Random Spin Array");
    //             //Utility::Configurations::Random(s, false);
    //         }
    //         else { throw(ex); }
    //     }

    // }//endif renorm_sd
}

template<Solver solver>
void Method_LLG<solver>::Finalize()
{
    this->systems[0]->iteration_allowed = false;
}

template<Solver solver>
void Method_LLG<solver>::Save_Current( std::string starttime, int iteration, bool initial, bool final )
{
    // History save
    this->history["max_torque"].push_back( this->max_torque );
    this->systems[0]->UpdateEnergy();
    this->history["E"].push_back( this->systems[0]->E );
    auto mag = Engine::Vectormath::Magnetization( *this->systems[0]->spins );
    this->history["M_z"].push_back( mag[2] );

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

        if( this->systems[0]->llg_parameters->output_file_tag == "<time>" )
            fileTag = starttime + "_";
        else if( this->systems[0]->llg_parameters->output_file_tag != "" )
            fileTag = this->systems[0]->llg_parameters->output_file_tag + "_";
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
                IO::VF_FileFormat format = this->systems[0]->llg_parameters->output_vf_filetype;

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
                    IO::OVF_File( spinsFile ).append_segment( segment, spins[0].data(), int( format ) );
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
                if( this->systems[0]->llg_parameters->output_energy_spin_resolved )
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
        if( this->systems[0]->llg_parameters->output_configuration_step )
        {
            writeOutputConfiguration( "_" + s_iter, false );
        }
        if( this->systems[0]->llg_parameters->output_energy_step )
        {
            writeOutputEnergy( "_" + s_iter, false );
        }

        // Archive file output (appending)
        if( this->systems[0]->llg_parameters->output_configuration_archive )
        {
            writeOutputConfiguration( "-archive", true );
        }
        if( this->systems[0]->llg_parameters->output_energy_archive )
        {
            writeOutputEnergy( "-archive", true );
        }

        // Save Log
        Log.Append_to_File();
    }
}

// Method name as string
template<Solver solver>
std::string Method_LLG<solver>::Name()
{
    return "LLG";
}

// Template instantiations
template class Method_LLG<Solver::SIB>;
template class Method_LLG<Solver::Heun>;
template class Method_LLG<Solver::Depondt>;
template class Method_LLG<Solver::RungeKutta4>;
template class Method_LLG<Solver::LBFGS_OSO>;
template class Method_LLG<Solver::LBFGS_Atlas>;
template class Method_LLG<Solver::VP>;
template class Method_LLG<Solver::VP_OSO>;

} // namespace Engine
