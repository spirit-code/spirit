#include "Spin_System.h"
#include "Spin_System_Chain.h"
#include "Vectormath.h"
#include "Interface_State.h"
#include "Interface_Hamiltonian.h"

/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Set Parameters ---------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

void Hamiltonian_Set_Boundary_Conditions(State *state, bool periodical_a, bool periodical_b, bool periodical_c)
{
    auto s = state->c->images[state->c->active_image];
    auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();

    s->hamiltonian->boundary_conditions[0] = periodical_a;
    s->hamiltonian->boundary_conditions[1] = periodical_b;
    s->hamiltonian->boundary_conditions[2] = periodical_c;
}

void Hamiltonian_Set_mu_s(State *state, float mu_s)
{
    auto s = state->c->images[state->c->active_image];
    auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();
    ham->mu_s = mu_s;
}

void Hamiltonian_Set_Field(State *state, float magnitude, float normal_x, float normal_y, float normal_z)
{
    auto s = state->c->images[state->c->active_image];
    auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();

    // Magnitude
    ham->external_field_magnitude = magnitude *  ham->mu_s * Utility::Vectormath::MuB();
    
    // Normal
    ham->external_field_normal[0] = normal_x;
    ham->external_field_normal[1] = normal_y;
    ham->external_field_normal[2] = normal_z;
    try {
        Utility::Vectormath::Normalize(ham->external_field_normal);
    }
    catch (Utility::Exception ex) {
        if (ex == Utility::Exception::Division_by_zero) {
            ham->external_field_normal[0] = 0.0;
            ham->external_field_normal[1] = 0.0;
            ham->external_field_normal[2] = 1.0;
            Utility::Log.Send(Utility::Log_Level::WARNING, Utility::Log_Sender::GUI, "B_vec = {0,0,0} replaced by {0,0,1}");
        }
        else { throw(ex); }
    }
}

void Hamiltonian_Set_Exchange(State *state, int n_shells, float* jij)
{
    auto s = state->c->images[state->c->active_image];
    auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();

    for (int i=0; i<n_shells; ++i)
    {
        ham->jij[i] = jij[i];
    }
}

void Hamiltonian_Set_DMI(State *state, float dij)
{
    auto s = state->c->images[state->c->active_image];
    auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();

    ham->dij = dij;
}

void Hamiltonian_Set_Anisotropy(State *state, float magnitude, float normal_x, float normal_y, float normal_z)
{
    auto s = state->c->images[state->c->active_image];
    auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();

    // Magnitude
    ham->anisotropy_magnitude = magnitude;
    // Normal
    ham->anisotropy_normal[0] = normal_x;
    ham->anisotropy_normal[1] = normal_y;
    ham->anisotropy_normal[2] = normal_z;
    try {
        Utility::Vectormath::Normalize(ham->anisotropy_normal);
    }
    catch (Utility::Exception ex) {
        if (ex == Utility::Exception::Division_by_zero) {
            ham->anisotropy_normal[0] = 0.0;
            ham->anisotropy_normal[1] = 0.0;
            ham->anisotropy_normal[2] = 1.0;
            Utility::Log.Send(Utility::Log_Level::WARNING, Utility::Log_Sender::GUI, "Aniso_vec = {0,0,0} replaced by {0,0,1}");
        }
        else { throw(ex); }
    }
}

void Hamiltonian_Set_STT(State *state, float magnitude, float normal_x, float normal_y, float normal_z)
{
    auto s = state->c->images[state->c->active_image];

    // Magnitude
    s->llg_parameters->stt_magnitude = magnitude;
    // Normal
    s->llg_parameters->stt_polarisation_normal[0] = normal_x;
    s->llg_parameters->stt_polarisation_normal[1] = normal_y;
    s->llg_parameters->stt_polarisation_normal[2] = normal_z;
    try {
        Utility::Vectormath::Normalize(s->llg_parameters->stt_polarisation_normal);
    }
    catch (Utility::Exception ex) {
        if (ex == Utility::Exception::Division_by_zero) {
            s->llg_parameters->stt_polarisation_normal[0] = 0.0;
            s->llg_parameters->stt_polarisation_normal[1] = 0.0;
            s->llg_parameters->stt_polarisation_normal[2] = 1.0;
            Utility::Log.Send(Utility::Log_Level::WARNING, Utility::Log_Sender::GUI, "s_c_vec = {0,0,0} replaced by {0,0,1}");
        }
        else { throw(ex); }
    }
}

void Hamiltonian_Set_Temperature(State *state, float T)
{
    auto s = state->c->images[state->c->active_image];
    s->llg_parameters->temperature = T;
}

/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Get Parameters ---------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

void Hamiltonian_Get_Boundary_Conditions(State *state, bool * periodical_a, bool * periodical_b, bool * periodical_c)
{
    auto s = state->c->images[state->c->active_image];
    auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();

    *periodical_a = s->hamiltonian->boundary_conditions[0];
    *periodical_b = s->hamiltonian->boundary_conditions[1];
    *periodical_c = s->hamiltonian->boundary_conditions[2];
}

void Hamiltonian_Get_mu_s(State *state, float * mu_s)
{
    auto s = state->c->images[state->c->active_image];
    auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();
    *mu_s = ham->mu_s;
}

void Hamiltonian_Get_Field(State *state, float * magnitude, float * normal_x, float * normal_y, float * normal_z)
{
    auto s = state->c->images[state->c->active_image];
    auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();

    // Magnitude
    *magnitude = ham->external_field_magnitude / ham->mu_s / Utility::Vectormath::MuB();
    
    // Normal
    *normal_x = ham->external_field_normal[0];
    *normal_y = ham->external_field_normal[1];
    *normal_z = ham->external_field_normal[2];
}

// TODO: do this correctly...
void Hamiltonian_Get_Exchange(State *state, int * n_shells, float * jij)
{
    auto s = state->c->images[state->c->active_image];
    auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();

    *n_shells = ham->n_neigh_shells;

    // for (int i=0; i<*n_shells; ++i)
    // {
    //     jij[i] = ham->jij[i];
    // }
}

void Hamiltonian_Get_DMI(State *state, float * dij)
{
    auto s = state->c->images[state->c->active_image];
    auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();

    *dij = ham->dij;
}

void Hamiltonian_Get_Anisotropy(State *state, float * magnitude, float * normal_x, float * normal_y, float * normal_z)
{
    auto s = state->c->images[state->c->active_image];
    auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();

    // Magnitude
    *magnitude = ham->anisotropy_magnitude;
    
    // Normal
    *normal_x = ham->anisotropy_normal[0];
    *normal_y = ham->anisotropy_normal[1];
    *normal_z = ham->anisotropy_normal[2];
}

void Hamiltonian_Get_STT(State *state, float * magnitude, float * normal_x, float * normal_y, float * normal_z)
{
    auto s = state->c->images[state->c->active_image];

    // Magnitude
    *magnitude = s->llg_parameters->stt_magnitude;
    // Normal
    *normal_x = s->llg_parameters->stt_polarisation_normal[0];
    *normal_y = s->llg_parameters->stt_polarisation_normal[1];
    *normal_z = s->llg_parameters->stt_polarisation_normal[2];
}

void Hamiltonian_Get_Temperature(State *state, float * T)
{
    auto s = state->c->images[state->c->active_image];
    *T = s->llg_parameters->temperature;
}