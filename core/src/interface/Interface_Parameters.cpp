#include <interface/Interface_Parameters.h>
#include <interface/Interface_State.h>
#include <data/State.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Constants.hpp>

using namespace Utility;

/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Set Parameters ---------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

// Set LLG
void Parameters_Set_LLG_Time_Step(State *state, float dt, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    auto p = image->llg_parameters;
    // Translate from picoseconds to units of our SIB
    p->dt = dt*std::pow(10,-12)/Constants::mu_B*1.760859644*std::pow(10,11);

	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
        "Set LLG dt=" + std::to_string(dt), idx_image, idx_chain);
}

void Parameters_Set_LLG_Damping(State *state, float damping, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    auto p = image->llg_parameters;
    p->damping = damping;

	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
        "Set LLG damping=" + std::to_string(damping), idx_image, idx_chain);
}

void Parameters_Set_LLG_N_Iterations(State *state, int n_iterations, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    image->llg_parameters->n_iterations = n_iterations;

	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
        "Set LLG n_iterations=" + std::to_string(n_iterations), idx_image, idx_chain);
}

void Parameters_Set_LLG_N_Iterations_Log(State *state, int n_iterations_log, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    image->llg_parameters->n_iterations_log = n_iterations_log;

	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
        "Set LLG n_iterations_log=" + std::to_string(n_iterations_log), idx_image, idx_chain);
}

// Set GNEB
void Parameters_Set_GNEB_Spring_Constant(State *state, float spring_constant, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    auto p = chain->gneb_parameters;
    p->spring_constant = spring_constant;

	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
        "Set GNEB spring constant =" + std::to_string(spring_constant), idx_image, idx_chain);
}

void Parameters_Set_GNEB_Climbing_Falling(State *state, int image_type, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    chain->image_type[idx_image] = static_cast<Data::GNEB_Image_Type>(image_type);

	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
        "Set GNEB image type =" + std::to_string(image_type), idx_image, idx_chain);
}

void Parameters_Set_GNEB_N_Iterations(State *state, int n_iterations, int idx_chain)
{
	int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    chain->gneb_parameters->n_iterations = n_iterations;

	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
        "Set GNEB n_iterations=" + std::to_string(n_iterations), idx_image, idx_chain);
}

void Parameters_Set_GNEB_N_Iterations_Log(State *state, int n_iterations_log,  int idx_chain)
{
	int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    chain->gneb_parameters->n_iterations_log = n_iterations_log;

	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
        "Set GNEB n_iterations_log=" + std::to_string(n_iterations_log), idx_image, idx_chain);
}

/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Get Parameters ---------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

// Get LLG
void Parameters_Get_LLG_Time_Step(State *state, float * dt, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    auto p = image->llg_parameters;
    *dt = (float)(p->dt/std::pow(10, -12)*Constants::mu_B/1.760859644/std::pow(10, 11));

}

void Parameters_Get_LLG_Damping(State *state, float * damping, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    auto p = image->llg_parameters;
    *damping = (float)p->damping;
}

int Parameters_Get_LLG_N_Iterations(State *state, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    auto p = image->llg_parameters;
    return p->n_iterations;
}

int Parameters_Get_LLG_N_Iterations_Log(State *state, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    auto p = image->llg_parameters;
    return p->n_iterations_log;
}

// Get GNEB
void Parameters_Get_GNEB_Spring_Constant(State *state, float * spring_constant, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    auto p = chain->gneb_parameters;
    *spring_constant = (float)p->spring_constant;
}

void Parameters_Get_GNEB_Climbing_Falling(State *state, int * image_type, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    *image_type = (int)chain->image_type[idx_image];
}

int Parameters_Get_GNEB_N_Iterations(State *state, int idx_chain)
{
	int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    auto p = chain->gneb_parameters;
    return p->n_iterations;
}

int Parameters_Get_GNEB_N_Iterations_Log(State *state,int idx_chain)
{
	int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    auto p = chain->gneb_parameters;
    return p->n_iterations_log;
}

int Parameters_Get_GNEB_N_Energy_Interpolations(State *state, int idx_chain)
{
	int idx_image = -1;
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	auto p = chain->gneb_parameters;
	return p->n_E_interpolations;
}