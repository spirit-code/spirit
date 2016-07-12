#include "Spin_System.h"
#include "Spin_System_Chain.h"
#include "Vectormath.h"
#include "Interface_Globals.h"
#include "Interface_Hamiltonian.h"

void Hamiltonian_Boundary_Conditions(bool periodical_a, bool periodical_b, bool periodical_c)
{
    auto s = c->images[c->active_image];
    auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();

    // Boundary conditions
    s->hamiltonian->boundary_conditions[0] = periodical_a;
    s->hamiltonian->boundary_conditions[1] = periodical_b;
    s->hamiltonian->boundary_conditions[2] = periodical_c;
}

void Hamiltonian_mu_s(float mu_s)
{
    auto s = c->images[c->active_image];
    auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();
    ham->mu_s = mu_s;
}

void Hamiltonian_Field(float magnitude, float normal_x, float normal_y, float normal_z)
{
    auto s = c->images[c->active_image];
    auto ham = (Engine::Hamiltonian_Isotropic*)s->hamiltonian.get();

    //		magnitude
    ham->external_field_magnitude = magnitude *  ham->mu_s * Utility::Vectormath::MuB();
    
    //		normal
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

void Hamiltonian_Exchange(float mu_s)
{
    
}

void Hamiltonian_DMI(float mu_s)
{
    
}

void Hamiltonian_Anisotropy(float mu_s)
{
    
}

void Hamiltonian_STT(float mu_s)
{
    
}

void Hamiltonian_Temperature(float mu_s)
{
    
}