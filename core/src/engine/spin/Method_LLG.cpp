#include <Spirit/Spirit_Defines.h>
#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <engine/Vectormath.hpp>
#include <engine/spin/Method_LLG.hpp>
#include <io/IO.hpp>
#include <io/OVF_File.hpp>
#include <utility/Logging.hpp>
#include <utility/Version.hpp>

#include <fmt/format.h>

#include <ctime>

using namespace Utility;

namespace Engine
{

namespace Spin
{

template<Solver solver>
Method_LLG<solver>::Method_LLG( std::shared_ptr<system_t> system, int idx_img, int idx_chain )
        : Method_Solver<solver>( system->llg_parameters, idx_img, idx_chain ), picoseconds_passed( 0 )
{
    // Currently we only support a single image being iterated at once:
    this->systems    = std::vector<std::shared_ptr<system_t>>( 1, system );
    this->SenderName = Utility::Log_Sender::LLG;

    this->noi = this->systems.size();
    this->nos = this->systems[0]->nos; // assume all systems have the same size

    // Forces (assume all systems have the same size)
    this->forces         = std::vector( this->noi, vectorfield( this->nos, Vector3::Zero() ) );
    this->forces_virtual = std::vector( this->noi, vectorfield( this->nos, Vector3::Zero() ) );
    this->Gradient       = std::vector( this->noi, vectorfield( this->nos, Vector3::Zero() ) );
    this->common_methods = std::vector( this->noi, Common::Method_LLG<common_solver( solver )>( this->nos ) );

    // We assume it is not converged before the first iteration
    this->force_converged = std::vector<bool>( this->noi, false );
    this->max_torque      = system->llg_parameters->force_convergence + 1.0;

    // Create shared pointers to the method's systems' spin configurations
    this->configurations = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for( int i = 0; i < this->noi; ++i )
        this->configurations[i] = this->systems[i]->state;

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
    for( int i = 0; i < this->noi; ++i )
    {
        common_methods[i].Prepare_Thermal_Field(
            *this->systems[i]->llg_parameters, this->systems[i]->hamiltonian->get_geometry() );
    }
}

template<Solver solver>
void Method_LLG<solver>::Calculate_Force(
    const std::vector<std::shared_ptr<vectorfield>> & configurations, std::vector<vectorfield> & forces )
{
    // Loop over images to calculate the total force on each Image
    for( std::size_t img = 0; img < this->systems.size(); ++img )
    {
        // Minus the gradient is the total Force here
        this->systems[img]->hamiltonian->Gradient_and_Energy( *configurations[img], Gradient[img], current_energy );

#ifdef SPIRIT_ENABLE_PINNING
        Vectormath::set_c_a(
            1, Gradient[img], Gradient[img], this->systems[img]->hamiltonian->get_geometry().mask_unpinned );
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
    for( int i = 0; i < this->noi; ++i )
    {
        const auto & sys = *this->systems[i];
        common_methods[i].Virtual_Force_Spin(
            *sys.llg_parameters, sys.hamiltonian->get_geometry(), sys.hamiltonian->get_boundary_conditions(),
            *configurations[i], forces[i], forces_virtual[i] );
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
    return std::all_of( begin( force_converged ), end( force_converged ), []( bool b ) { return b; } );
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
    for( std::size_t img = 0; img < this->systems.size(); ++img )
    {
        this->force_converged[img] = false;
        Manifoldmath::project_tangential( this->forces_virtual[img], *( this->systems[img]->state ) );
        const scalar fmax = Vectormath::max_norm( this->forces_virtual[img] );

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

    this->systems[0]->E.total = current_energy;

    // ToDo: How to update eff_field without numerical overhead?
    // systems[0]->effective_field = Gradient[0];
    // Vectormath::scale(systems[0]->effective_field, -1);
    Manifoldmath::project_tangential( this->forces[0], *this->systems[0]->state );
    Vectormath::set_c_a( 1, this->forces[0], this->systems[0]->M.effective_field );
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
void Method_LLG<solver>::Message_Block_Step( std::vector<std::string> & block )
{
    if constexpr(
        solver == Solver::VP || solver == Solver::VP_OSO || solver == Solver::LBFGS_OSO
        || solver == Solver::LBFGS_Atlas )
        return;
    else if( !this->systems[0]->llg_parameters->direct_minimization )
        block.emplace_back( fmt::format( "    Simulated time:       {} ps", this->get_simulated_time() ) );
}

template<Solver solver>
void Method_LLG<solver>::Message_Block_End( std::vector<std::string> & block )
{
    if constexpr(
        solver == Solver::VP || solver == Solver::VP_OSO || solver == Solver::LBFGS_OSO
        || solver == Solver::LBFGS_Atlas )
        return;
    else if( !this->systems[0]->llg_parameters->direct_minimization )
        block.emplace_back( fmt::format( "    Simulated time:       {} ps", this->get_simulated_time() ) );
}

template<Solver solver>
void Method_LLG<solver>::Save_Current( std::string starttime, int iteration, bool initial, bool final )
{
    if( this->systems.empty() || this->systems[0] == nullptr )
        return;
    auto & sys = *this->systems[0];

    // History save
    this->history_iteration.push_back( this->iteration );
    this->history_max_torque.push_back( this->max_torque );
    this->history_energy.push_back( sys.E.total );

    // this->history["max_torque"].push_back( this->max_torque );
    // sys.UpdateEnergy();
    // this->history["E"].push_back( sys.E );
    // Removed magnetization, since at the moment it required a temporary allocation to compute
    // auto mag = Engine::Vectormath::Magnetization( *sys.spins );
    // this->history["M_z"].push_back( mag[2] );

    // File save
    if( this->parameters->output_any )
    {
        // Convert indices to formatted strings
        auto s_img         = fmt::format( "{:0>2}", this->idx_image );
        auto base          = static_cast<std::int32_t>( log10( this->parameters->n_iterations ) );
        std::string s_iter = fmt::format( fmt::runtime( "{:0>" + fmt::format( "{}", base ) + "}" ), iteration );

        std::string preSpinsFile;
        std::string preEnergyFile;
        std::string fileTag;

        if( sys.llg_parameters->output_file_tag == "<time>" )
            fileTag = starttime + "_";
        else if( sys.llg_parameters->output_file_tag != "" )
            fileTag = sys.llg_parameters->output_file_tag + "_";
        else
            fileTag = "";

        preSpinsFile  = this->parameters->output_folder + "/" + fileTag + "Image-" + s_img + "_Spins";
        preEnergyFile = this->parameters->output_folder + "/" + fileTag + "Image-" + s_img + "_Energy";

        // Function to write or append image and energy files
        auto writeOutputConfiguration = [this, &sys, preSpinsFile, iteration]( const std::string & suffix, bool append )
        {
            try
            {
                // File name and comment
                std::string spinsFile      = preSpinsFile + suffix + ".ovf";
                std::string output_comment = fmt::format(
                    "{} simulation ({} solver)\n# Desc:      Iteration: {}\n# Desc:      Maximum torque: {}",
                    this->Name(), this->SolverFullName(), iteration, this->max_torque );

                // File format
                IO::VF_FileFormat format = sys.llg_parameters->output_vf_filetype;

                // Spin Configuration
                auto & spins        = *sys.state;
                auto segment        = IO::OVF_Segment( sys.hamiltonian->get_geometry() );
                std::string title   = fmt::format( "SPIRIT Version {}", Utility::version_full );
                segment.title       = strdup( title.c_str() );
                segment.comment     = strdup( output_comment.c_str() );
                segment.valuedim    = IO::Spin::State::valuedim;
                segment.valuelabels = strdup( IO::Spin::State::valuelabels.data() );
                segment.valueunits  = strdup( IO::Spin::State::valueunits.data() );

                const IO::Spin::State::Buffer buffer( spins );
                if( append )
                    IO::OVF_File( spinsFile ).append_segment( segment, buffer.data(), static_cast<int>( format ) );
                else
                    IO::OVF_File( spinsFile ).write_segment( segment, buffer.data(), static_cast<int>( format ) );
            }
            catch( ... )
            {
                spirit_handle_exception_core( "LLG output failed" );
            }
        };

        IO::Flags flags;
        if( sys.llg_parameters->output_energy_divide_by_nspins )
            flags |= IO::Flag::Normalize_by_nos;
        if( sys.llg_parameters->output_energy_add_readability_lines )
            flags |= IO::Flag::Readability;
        auto writeOutputEnergy = [&sys, flags, preEnergyFile, iteration]( const std::string & suffix, bool append )
        {
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
                        sys.E, energyFile, { "iteration", "E_tot" }, IO::Flag::Contributions | flags );
                // Append Energy to File
                IO::Append_Image_Energy( sys.E, sys.hamiltonian->get_geometry(), iteration, energyFile, flags );
            }
            else
            {
                IO::Write_Energy_Header( sys.E, energyFile, { "iteration", "E_tot" }, IO::Flag::Contributions | flags );
                IO::Append_Image_Energy( sys.E, sys.hamiltonian->get_geometry(), iteration, energyFile, flags );
                if( sys.llg_parameters->output_energy_spin_resolved )
                {
                    // Gather the data
                    Data::vectorlabeled<scalarfield> contributions_spins( 0 );
                    sys.UpdateEnergy();
                    sys.hamiltonian->Energy_Contributions_per_Spin( *sys.state, sys.E.per_interaction_per_spin );

                    IO::Write_Image_Energy_Contributions(
                        sys.E, sys.hamiltonian->get_geometry(), energyFilePerSpin,
                        sys.llg_parameters->output_vf_filetype );
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
        if( sys.llg_parameters->output_configuration_step )
        {
            writeOutputConfiguration( "_" + s_iter, false );
        }
        if( sys.llg_parameters->output_energy_step )
        {
            writeOutputEnergy( "_" + s_iter, false );
        }

        // Archive file output (appending)
        if( sys.llg_parameters->output_configuration_archive )
        {
            writeOutputConfiguration( "-archive", true );
        }
        if( sys.llg_parameters->output_energy_archive )
        {
            writeOutputEnergy( "-archive", true );
        }

        // Save Log
        Log.Append_to_File();
    }
}

// Method name as string
template<Solver solver>
std::string_view Method_LLG<solver>::Name()
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

} // namespace Spin

} // namespace Engine
