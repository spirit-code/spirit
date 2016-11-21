#include "Interface_Hamiltonian.h"
#include "Interface_State.h"

#include "State.hpp"
#include "Spin_System.hpp"
#include "Spin_System_Chain.hpp"
#include "utility/Vectormath.hpp"

/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Set Parameters ---------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

void Hamiltonian_Set_Boundary_Conditions(State *state, const bool * periodical, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    image->hamiltonian->boundary_conditions[0] = periodical[0];
    image->hamiltonian->boundary_conditions[1] = periodical[1];
    image->hamiltonian->boundary_conditions[2] = periodical[2];
}

void Hamiltonian_Set_mu_s(State *state, float mu_s, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    if (image->hamiltonian->Name() == "Isotropic Heisenberg")
    {
        auto ham = (Engine::Hamiltonian_Isotropic*)image->hamiltonian.get();
        ham->mu_s = mu_s;
    }
    else if (image->hamiltonian->Name() == "Anisotropic Heisenberg")
    {
        auto ham = (Engine::Hamiltonian_Anisotropic*)image->hamiltonian.get();
        for (auto& m : ham->mu_s) m = mu_s;
    }
}

void Hamiltonian_Set_Field(State *state, float magnitude, const float * normal, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    if (image->hamiltonian->Name() == "Isotropic Heisenberg")
    {
        auto ham = (Engine::Hamiltonian_Isotropic*)image->hamiltonian.get();

        // Magnitude
        ham->external_field_magnitude = magnitude *  ham->mu_s * Utility::Vectormath::MuB();
        
        // Normal
        ham->external_field_normal[0] = normal[0];
        ham->external_field_normal[1] = normal[1];
        ham->external_field_normal[2] = normal[2];
        try {
            Utility::Vectormath::Normalize(ham->external_field_normal);
        }
        catch (Utility::Exception ex) {
            if (ex == Utility::Exception::Division_by_zero) {
                ham->external_field_normal[0] = 0.0;
                ham->external_field_normal[1] = 0.0;
                ham->external_field_normal[2] = 1.0;
                Log(Utility::Log_Level::Warning, Utility::Log_Sender::API, "B_vec = {0,0,0} replaced by {0,0,1}");
            }
            else
            {
                Log(Utility::Log_Level::Severe, Utility::Log_Sender::API, "Unknown Exception! Exiting...");
                exit(0);
            }
        }
    }
    else if (image->hamiltonian->Name() == "Anisotropic Heisenberg")
    {
        // TODO
    }
}

void Hamiltonian_Set_Exchange(State *state, int n_shells, const float* jij, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    if (image->hamiltonian->Name() == "Isotropic Heisenberg")
    {
        auto ham = (Engine::Hamiltonian_Isotropic*)image->hamiltonian.get();

        for (int i=0; i<n_shells; ++i)
        {
            ham->jij[i] = jij[i];
        }
    }
    else if (image->hamiltonian->Name() == "Anisotropic Heisenberg")
    {
        // TODO
    }
}

void Hamiltonian_Set_Anisotropy(State *state, float magnitude, const float * normal, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    if (image->hamiltonian->Name() == "Isotropic Heisenberg")
    {
        auto ham = (Engine::Hamiltonian_Isotropic*)image->hamiltonian.get();

        // Magnitude
        ham->anisotropy_magnitude = magnitude;
        // Normal
        ham->anisotropy_normal[0] = normal[0];
        ham->anisotropy_normal[1] = normal[1];
        ham->anisotropy_normal[2] = normal[2];
        try {
            Utility::Vectormath::Normalize(ham->anisotropy_normal);
        }
        catch (Utility::Exception ex) {
            if (ex == Utility::Exception::Division_by_zero) {
                ham->anisotropy_normal[0] = 0.0;
                ham->anisotropy_normal[1] = 0.0;
                ham->anisotropy_normal[2] = 1.0;
                Log(Utility::Log_Level::Warning, Utility::Log_Sender::API, "Aniso_vec = {0,0,0} replaced by {0,0,1}");
            }
            else
            {
                Log(Utility::Log_Level::Severe, Utility::Log_Sender::API, "Unknown Exception! Exiting...");
                exit(0);
            }
        }
    }
    else if (image->hamiltonian->Name() == "Anisotropic Heisenberg")
    {
        // TODO
    }
}

void Hamiltonian_Set_DMI(State *state, float dij, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    if (image->hamiltonian->Name() == "Isotropic Heisenberg")
    {
        auto ham = (Engine::Hamiltonian_Isotropic*)image->hamiltonian.get();

        ham->dij = dij;
    }
    else if (image->hamiltonian->Name() == "Isotropic Heisenberg")
    {

    }
}

void Hamiltonian_Set_BQE(State *state, float bij, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    if (image->hamiltonian->Name() == "Isotropic Heisenberg")
    {
        auto ham = (Engine::Hamiltonian_Isotropic*)image->hamiltonian.get();

        ham->bij = bij;
    }
    else if (image->hamiltonian->Name() == "Anisotropic Heisenberg")
    {

    }
}

void Hamiltonian_Set_FSC(State *state, float kijkl, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    if (image->hamiltonian->Name() == "Isotropic Heisenberg")
    {
        auto ham = (Engine::Hamiltonian_Isotropic*)image->hamiltonian.get();

        ham->kijkl = kijkl;
    }
    else if (image->hamiltonian->Name() == "Anisotropic Heisenberg")
    {

    }
}

void Hamiltonian_Set_STT(State *state, float magnitude, const float * normal, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    // Magnitude
    image->llg_parameters->stt_magnitude = magnitude;
    // Normal
    image->llg_parameters->stt_polarisation_normal[0] = normal[0];
    image->llg_parameters->stt_polarisation_normal[1] = normal[1];
    image->llg_parameters->stt_polarisation_normal[2] = normal[2];
    try {
        Utility::Vectormath::Normalize(image->llg_parameters->stt_polarisation_normal);
    }
    catch (Utility::Exception ex) {
        if (ex == Utility::Exception::Division_by_zero) {
            image->llg_parameters->stt_polarisation_normal[0] = 0.0;
            image->llg_parameters->stt_polarisation_normal[1] = 0.0;
            image->llg_parameters->stt_polarisation_normal[2] = 1.0;
            Log(Utility::Log_Level::Warning, Utility::Log_Sender::API, "s_c_vec = {0,0,0} replaced by {0,0,1}");
        }
        else
        {
            Log(Utility::Log_Level::Severe, Utility::Log_Sender::API, "Unknown Exception! Exiting...");
            exit(0);
        }
    }
}

void Hamiltonian_Set_Temperature(State *state, float T, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    image->llg_parameters->temperature = T;
}

/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Get Parameters ---------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

const char * Hamiltonian_Get_Name(State * state, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	return image->hamiltonian->Name().c_str();
}

void Hamiltonian_Get_Boundary_Conditions(State *state, bool * periodical, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    periodical[0] = image->hamiltonian->boundary_conditions[0];
    periodical[1] = image->hamiltonian->boundary_conditions[1];
    periodical[2] = image->hamiltonian->boundary_conditions[2];
}

void Hamiltonian_Get_mu_s(State *state, float * mu_s, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    if (image->hamiltonian->Name() == "Isotropic Heisenberg")
    {
        auto ham = (Engine::Hamiltonian_Isotropic*)image->hamiltonian.get();
        *mu_s = (float)ham->mu_s;
    }
    else if (image->hamiltonian->Name() == "Anisotropic Heisenberg")
    {
        auto ham = (Engine::Hamiltonian_Anisotropic*)image->hamiltonian.get();
        *mu_s = (float)ham->mu_s[0];
    }
}

void Hamiltonian_Get_Field(State *state, float * magnitude, float * normal, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    if (image->hamiltonian->Name() == "Isotropic Heisenberg")
    {
        auto ham = (Engine::Hamiltonian_Isotropic*)image->hamiltonian.get();

        // Magnitude
        *magnitude = (float)(ham->external_field_magnitude / ham->mu_s / Utility::Vectormath::MuB());
        
        // Normal
        normal[0] = (float)ham->external_field_normal[0];
        normal[1] = (float)ham->external_field_normal[1];
        normal[2] = (float)ham->external_field_normal[2];
    }
    else if (image->hamiltonian->Name() == "Anisotropic Heisenberg")
    {
        auto ham = (Engine::Hamiltonian_Anisotropic*)image->hamiltonian.get();

        // Magnitude
        *magnitude = (float)(ham->external_field_magnitude[0] / ham->mu_s[0] / Utility::Vectormath::MuB());

        // Normal
        normal[0] = (float)ham->external_field_normal[0][0];
        normal[1] = (float)ham->external_field_normal[1][0];
        normal[2] = (float)ham->external_field_normal[2][0];
    }
}

void Hamiltonian_Get_Exchange(State *state, int * n_shells, float * jij, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    if (image->hamiltonian->Name() == "Isotropic Heisenberg")
    {
        auto ham = (Engine::Hamiltonian_Isotropic*)image->hamiltonian.get();

        *n_shells = ham->n_neigh_shells;

        // Note the array needs to be correctly allocated beforehand!
        for (int i=0; i<*n_shells; ++i)
        {
            jij[i] = (float)ham->jij[i];
        }
    }
    else if (image->hamiltonian->Name() == "Anisotropic Heisenberg")
    {
        // TODO
    }
}

void Hamiltonian_Get_Anisotropy(State *state, float * magnitude, float * normal, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    if (image->hamiltonian->Name() == "Isotropic Heisenberg")
    {
        auto ham = (Engine::Hamiltonian_Isotropic*)image->hamiltonian.get();

        // Magnitude
        *magnitude = (float)ham->anisotropy_magnitude;
        
        // Normal
        normal[0] = (float)ham->anisotropy_normal[0];
        normal[1] = (float)ham->anisotropy_normal[1];
        normal[2] = (float)ham->anisotropy_normal[2];
    }
    else if (image->hamiltonian->Name() == "Anisotropic Heisenberg")
    {
        auto ham = (Engine::Hamiltonian_Anisotropic*)image->hamiltonian.get();

        // Magnitude
        *magnitude = (float)ham->anisotropy_magnitude[0];

        // Normal
        normal[0] = (float)ham->anisotropy_normal[0][0];
        normal[1] = (float)ham->anisotropy_normal[0][1];
        normal[2] = (float)ham->anisotropy_normal[0][2];
    }
}

void Hamiltonian_Get_DMI(State *state, float * dij, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    if (image->hamiltonian->Name() == "Isotropic Heisenberg")
    {
        auto ham = (Engine::Hamiltonian_Isotropic*)image->hamiltonian.get();

        *dij = (float)ham->dij;
    }
    else if (image->hamiltonian->Name() == "Anisotropic Heisenberg")
    {
        // TODO
    }
}

void Hamiltonian_Get_BQE(State *state, float * bij, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    if (image->hamiltonian->Name() == "Isotropic Heisenberg")
    {
        auto ham = (Engine::Hamiltonian_Isotropic*)image->hamiltonian.get();

        *bij = (float)ham->bij;
    }
    else if (image->hamiltonian->Name() == "Anisotropic Heisenberg")
    {
        // TODO
    }
}

void Hamiltonian_Get_FSC(State *state, float * kijkl, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    if (image->hamiltonian->Name() == "Isotropic Heisenberg")
    {
        auto ham = (Engine::Hamiltonian_Isotropic*)image->hamiltonian.get();

        *kijkl = (float)ham->kijkl;
    }
    else if (image->hamiltonian->Name() == "Anisotropic Heisenberg")
    {
        // TODO
    }
}

void Hamiltonian_Get_STT(State *state, float * magnitude, float * normal, int idx_image, int idx_chain)
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

void Hamiltonian_Get_Temperature(State *state, float * T, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    *T = (float)image->llg_parameters->temperature;
}