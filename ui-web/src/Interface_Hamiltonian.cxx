#include "Spin_System.h"
#include "Spin_System_Chain.h"
#include "Vectormath.h"
#include "Interface_State.h"
#include "Interface_Hamiltonian.h"

void Hamiltonian_Boundary_Conditions(State *state, bool periodical_a, bool periodical_b, bool periodical_c)
{
    auto s = state->c->images[state->c->active_image];
    auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();

    // Boundary conditions
    s->hamiltonian->boundary_conditions[0] = periodical_a;
    s->hamiltonian->boundary_conditions[1] = periodical_b;
    s->hamiltonian->boundary_conditions[2] = periodical_c;
}

void Hamiltonian_mu_s(State *state, float mu_s)
{
    auto s = state->c->images[state->c->active_image];
    auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();
    ham->mu_s = mu_s;
}

void Hamiltonian_Field(State *state, float magnitude, float normal_x, float normal_y, float normal_z)
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

void Hamiltonian_Exchange(State *state, int n_shells, float* jij)
{
    auto s = state->c->images[state->c->active_image];
    auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();

    for (int i=0; i<n_shells; ++i)
    {
        ham->jij[i] = jij[i];
    }
}

void Hamiltonian_DMI(State *state, float dij)
{
    auto s = state->c->images[state->c->active_image];
    auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();

    ham->dij = dij;
}

void Hamiltonian_Anisotropy(State *state, float magnitude, float normal_x, float normal_y, float normal_z)
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

void Hamiltonian_STT(State *state, float magnitude, float normal_x, float normal_y, float normal_z)
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

void Hamiltonian_Temperature(State *state, float T)
{
    auto s = state->c->images[state->c->active_image];
    s->llg_parameters->temperature = T;
}