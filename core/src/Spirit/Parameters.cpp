#include <Spirit/Parameters.h>
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

	image->Lock();
    auto p = image->llg_parameters;
    // Translate from picoseconds to units of our SIB
    p->dt = dt*std::pow(10,-12)/Constants::mu_B*1.760859644*std::pow(10,11);
	image->Unlock();

	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
        "Set LLG dt=" + std::to_string(dt), idx_image, idx_chain);
}

void Parameters_Set_LLG_Damping(State *state, float damping, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

	image->Lock();
    auto p = image->llg_parameters;
    p->damping = damping;
	image->Unlock();

	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
        "Set LLG damping=" + std::to_string(damping), idx_image, idx_chain);
}

void Parameters_Set_LLG_N_Iterations(State *state, int n_iterations, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

	image->Lock();
    image->llg_parameters->n_iterations = n_iterations;
	image->Unlock();

	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
        "Set LLG n_iterations=" + std::to_string(n_iterations), idx_image, idx_chain);
}

void Parameters_Set_LLG_N_Iterations_Log(State *state, int n_iterations_log, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

	image->Lock();
    image->llg_parameters->n_iterations_log = n_iterations_log;
	image->Unlock();

	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
        "Set LLG n_iterations_log=" + std::to_string(n_iterations_log), idx_image, idx_chain);
}

void Parameters_Set_LLG_STT(State *state, float magnitude, const float * normal, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

	image->Lock();

    // Magnitude
    image->llg_parameters->stt_magnitude = magnitude;
    // Normal
    image->llg_parameters->stt_polarisation_normal[0] = normal[0];
    image->llg_parameters->stt_polarisation_normal[1] = normal[1];
    image->llg_parameters->stt_polarisation_normal[2] = normal[2];
	if (image->llg_parameters->stt_polarisation_normal.norm() < 0.9)
	{
		image->llg_parameters->stt_polarisation_normal = { 0,0,1 };
		Log(Utility::Log_Level::Warning, Utility::Log_Sender::API, "s_c_vec = {0,0,0} replaced by {0,0,1}");
	}
	else image->llg_parameters->stt_polarisation_normal.normalize();

	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
        "Set spin current to " + std::to_string(magnitude) + ", direction (" + std::to_string(normal[0]) + "," + std::to_string(normal[1]) + "," + std::to_string(normal[2]) + ")", idx_image, idx_chain);

	image->Unlock();
}

void Parameters_Set_LLG_Temperature(State *state, float T, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

	image->Lock();

    image->llg_parameters->temperature = T;

	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
        "Set temperature to " + std::to_string(T), idx_image, idx_chain);

	image->Unlock();
}

// Set MC
void Parameters_Set_MC_Temperature(State *state, float T, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

	image->Lock();

    image->mc_parameters->temperature = T;

	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
        "Set temperature to " + std::to_string(T), idx_image, idx_chain);

	image->Unlock();
}

// Set GNEB
void Parameters_Set_GNEB_Spring_Constant(State *state, float spring_constant, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

	chain->Lock();
    auto p = chain->gneb_parameters;
    p->spring_constant = spring_constant;
	chain->Unlock();

	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
        "Set GNEB spring constant =" + std::to_string(spring_constant), idx_image, idx_chain);
}

void Parameters_Set_GNEB_Climbing_Falling(State *state, int image_type, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

	chain->Lock();
    chain->image_type[idx_image] = static_cast<Data::GNEB_Image_Type>(image_type);
	chain->Unlock();

	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
        "Set GNEB image type =" + std::to_string(image_type), idx_image, idx_chain);
}

void Parameters_Set_GNEB_N_Iterations(State *state, int n_iterations, int idx_chain)
{
	int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

	chain->Lock();
    chain->gneb_parameters->n_iterations = n_iterations;
	chain->Unlock();

	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
        "Set GNEB n_iterations=" + std::to_string(n_iterations), idx_image, idx_chain);
}

void Parameters_Set_GNEB_N_Iterations_Log(State *state, int n_iterations_log,  int idx_chain)
{
	int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

	chain->Lock();
    chain->gneb_parameters->n_iterations_log = n_iterations_log;
	chain->Unlock();

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

void Parameters_Get_LLG_STT(State *state, float * magnitude, float * normal, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    // Magnitude
    *magnitude = (float)image->llg_parameters->stt_magnitude;
    // Normal
    normal[0] = (float)image->llg_parameters->stt_polarisation_normal[0];
    normal[1] = (float)image->llg_parameters->stt_polarisation_normal[1];
    normal[2] = (float)image->llg_parameters->stt_polarisation_normal[2];
}

void Parameters_Get_LLG_Temperature(State *state, float * T, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    *T = (float)image->llg_parameters->temperature;
}

// Get MC
void Parameters_Get_MC_Temperature(State *state, float * T, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    *T = (float)image->mc_parameters->temperature;
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