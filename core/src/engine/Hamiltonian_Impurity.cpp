#include <engine/Hamiltonian_Impurity.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Neighbours.hpp>
#include <data/Spin_System.hpp>
#include <utility/Constants.hpp>

#include <algorithm>

#include <Eigen/Dense>
#include <Eigen/Core>


using namespace Data;
using namespace Utility;
namespace C = Utility::Constants;
using Engine::Vectormath::check_atom_type;
using Engine::Vectormath::idx_from_pair;
using Engine::Vectormath::idx_from_tupel;


namespace Engine
{
    Hamiltonian_Impurity::Hamiltonian_Impurity(
        scalar external_field_magnitude, Vector3 external_field_normal,
        intfield anisotropy_indices, scalarfield anisotropy_magnitudes, vectorfield anisotropy_normals,
        pairfield exchange_pairs, scalarfield exchange_magnitudes,
        pairfield dmi_pairs, scalarfield dmi_magnitudes, vectorfield dmi_normals,
        std::shared_ptr<Data::Geometry> geometry,
        intfield boundary_conditions
    ) :
        Hamiltonian(boundary_conditions),
        geometry(geometry),
        external_field_magnitude(external_field_magnitude * C::mu_B), external_field_normal(external_field_normal),
        anisotropy_indices(anisotropy_indices), anisotropy_magnitudes(anisotropy_magnitudes), anisotropy_normals(anisotropy_normals),
        exchange_pairs(exchange_pairs), exchange_magnitudes(exchange_magnitudes),
        dmi_pairs(dmi_pairs), dmi_magnitudes(dmi_magnitudes), dmi_normals(dmi_normals)
    {
        // Generate interaction pairs, constants etc.
        this->Update_Energy_Contributions();
    }

    void Hamiltonian_Impurity::Update_Energy_Contributions()
    {
    }

    void Hamiltonian_Impurity::E_Zeeman(const vectorfield & spins, scalarfield & Energy)
    {
        const int N = geometry->n_cell_atoms;
        const int N_cells = geometry->n_cells_total;
        auto& mu_s = this->geometry->mu_s;

        for( int ispin = N*N_cells; ispin < geometry->nos; ++ispin )
            Energy[ispin] -= mu_s[ispin] * this->external_field_magnitude * this->external_field_normal.dot(spins[ispin]);
    }

    void Hamiltonian_Impurity::E_Anisotropy(const vectorfield & spins, scalarfield & Energy)
    {
        // TODO
        // const int N = geometry->n_cell_atoms;
        // const int N_cells = geometry->n_cells_total;

        // for( int iani = 0; iani < anisotropy_indices.size(); ++iani )
        // {
        //     int ispin = N*N_cells + anisotropy_indices[iani];
        //     Energy[ispin] -= this->anisotropy_magnitudes[iani] * std::pow(anisotropy_normals[iani].dot(spins[ispin]), 2.0);
        // }
    }

    void Hamiltonian_Impurity::E_Exchange(const vectorfield & spins, scalarfield & Energy)
    {
        // TODO
        // const int N = geometry->n_cell_atoms;
        // const int N_cells = geometry->n_cells_total;

        // #pragma omp parallel for
        // for( unsigned int icell = 0; icell < geometry->n_cells_total; ++icell )
        // {
        //     for( unsigned int i_pair = 0; i_pair < exchange_pairs.size(); ++i_pair )
        //     {
        //         int ispin = N*N_cells + exchange_pairs[i_pair].i + icell*geometry->n_cell_atoms;
        //         int jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, exchange_pairs[i_pair]);
        //         if( jspin >= 0 )
        //         {
        //             Energy[ispin] -= 0.5 * exchange_magnitudes[i_pair] * spins[ispin].dot(spins[jspin]);
        //             #ifndef SPIRIT_USE_OPENMP
        //             Energy[jspin] -= 0.5 * exchange_magnitudes[i_pair] * spins[ispin].dot(spins[jspin]);
        //             #endif
        //         }
        //     }
        // }
    }

    void Hamiltonian_Impurity::E_DMI(const vectorfield & spins, scalarfield & Energy)
    {
        // TODO
        // #pragma omp parallel for
        // for( unsigned int icell = 0; icell < geometry->n_cells_total; ++icell )
        // {
        //     for( unsigned int i_pair = 0; i_pair < dmi_pairs.size(); ++i_pair )
        //     {
        //         int ispin = dmi_pairs[i_pair].i + icell*geometry->n_cell_atoms;
        //         int jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, dmi_pairs[i_pair]);
        //         if( jspin >= 0 )
        //         {
        //             Energy[ispin] -= 0.5 * dmi_magnitudes[i_pair] * dmi_normals[i_pair].dot(spins[ispin].cross(spins[jspin]));
        //             #ifndef SPIRIT_USE_OPENMP
        //             Energy[jspin] -= 0.5 * dmi_magnitudes[i_pair] * dmi_normals[i_pair].dot(spins[ispin].cross(spins[jspin]));
        //             #endif
        //         }
        //     }
        // }
    }


    scalar Hamiltonian_Impurity::Energy_Single_Spin(int ispin, const vectorfield & spins)
    {
        // TODO
        scalar Energy = 0;
        return Energy;
    }


    void Hamiltonian_Impurity::Gradient_Zeeman(vectorfield & gradient)
    {
        const int N = geometry->n_cell_atoms;
        const int N_cells = geometry->n_cells_total;
        auto& mu_s = this->geometry->mu_s;

        for( int ispin = N*N_cells; ispin < geometry->nos; ++ispin )
            gradient[ispin] -= mu_s[ispin] * this->external_field_magnitude * this->external_field_normal;
    }

    void Hamiltonian_Impurity::Gradient_Anisotropy(const vectorfield & spins, vectorfield & gradient)
    {
        // TODO
        // const int N = geometry->n_cell_atoms;

        // #pragma omp parallel for
        // for( int icell = 0; icell < geometry->n_cells_total; ++icell )
        // {
        //     for( int iani = 0; iani < anisotropy_indices.size(); ++iani )
        //     {
        //         int ispin = icell*N + anisotropy_indices[iani];
        //         if( check_atom_type(this->geometry->atom_types[ispin]) )
        //             gradient[ispin] -= 2.0 * this->anisotropy_magnitudes[iani] * this->anisotropy_normals[iani] * anisotropy_normals[iani].dot(spins[ispin]);
        //     }
        // }
    }

    void Hamiltonian_Impurity::Gradient_Exchange(const vectorfield & spins, vectorfield & gradient)
    {
        // TODO
        // #pragma omp parallel for
        // for( int icell = 0; icell < geometry->n_cells_total; ++icell )
        // {
        //     for( unsigned int i_pair = 0; i_pair < exchange_pairs.size(); ++i_pair )
        //     {
        //         int ispin = exchange_pairs[i_pair].i + icell*geometry->n_cell_atoms;
        //         int jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, exchange_pairs[i_pair]);
        //         if( jspin >= 0 )
        //         {
        //             gradient[ispin] -= exchange_magnitudes[i_pair] * spins[jspin];
        //             #ifndef SPIRIT_USE_OPENMP
        //             gradient[jspin] -= exchange_magnitudes[i_pair] * spins[ispin];
        //             #endif
        //         }
        //     }
        // }
    }

    void Hamiltonian_Impurity::Gradient_DMI(const vectorfield & spins, vectorfield & gradient)
    {
        // TODO
        // #pragma omp parallel for
        // for( int icell = 0; icell < geometry->n_cells_total; ++icell )
        // {
        //     for( unsigned int i_pair = 0; i_pair < dmi_pairs.size(); ++i_pair )
        //     {
        //         int ispin = dmi_pairs[i_pair].i + icell*geometry->n_cell_atoms;
        //         int jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, dmi_pairs[i_pair]);
        //         if( jspin >= 0 )
        //         {
        //             gradient[ispin] -= dmi_magnitudes[i_pair] * spins[jspin].cross(dmi_normals[i_pair]);
        //             #ifndef SPIRIT_USE_OPENMP
        //             gradient[jspin] += dmi_magnitudes[i_pair] * spins[ispin].cross(dmi_normals[i_pair]);
        //             #endif
        //         }
        //     }
        // }
    }


    void Hamiltonian_Impurity::Hessian(const vectorfield & spins, MatrixX & hessian)
    {
        // TODO
        // int nos = spins.size();
        // const int N = geometry->n_cell_atoms;

        // // --- Single Spin elements
        // #pragma omp parallel for
        // for( int icell = 0; icell < geometry->n_cells_total; ++icell )
        // {
        //     for( int iani = 0; iani < anisotropy_indices.size(); ++iani )
        //     {
        //         int ispin = icell*N + anisotropy_indices[iani];
        //         if( check_atom_type(this->geometry->atom_types[ispin]) )
        //         {
        //             for( int alpha = 0; alpha < 3; ++alpha )
        //             {
        //                 for ( int beta = 0; beta < 3; ++beta )
        //                 {
        //                     int i = 3 * ispin + alpha;
        //                     int j = 3 * ispin + alpha;
        //                     hessian(i, j) += -2.0 * this->anisotropy_magnitudes[iani] *
        //                                             this->anisotropy_normals[iani][alpha] *
        //                                             this->anisotropy_normals[iani][beta];
        //                 }
        //             }
        //         }
        //     }
        // }

        // // --- Spin Pair elements
        // // Exchange
        // #pragma omp parallel for
        // for( int icell = 0; icell < geometry->n_cells_total; ++icell )
        // {
        //     for( unsigned int i_pair = 0; i_pair < exchange_pairs.size(); ++i_pair )
        //     {
        //         int ispin = exchange_pairs[i_pair].i + icell*geometry->n_cell_atoms;
        //         int jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, exchange_pairs[i_pair]);
        //         if( jspin >= 0 )
        //         {
        //             for( int alpha = 0; alpha < 3; ++alpha )
        //             {
        //                 int i = 3 * ispin + alpha;
        //                 int j = 3 * jspin + alpha;

        //                 hessian(i, j) += -exchange_magnitudes[i_pair];
        //                 #ifndef SPIRIT_USE_OPENMP
        //                 hessian(j, i) += -exchange_magnitudes[i_pair];
        //                 #endif
        //             }
        //         }
        //     }
        // }

        // // DMI
        // #pragma omp parallel for
        // for( int icell = 0; icell < geometry->n_cells_total; ++icell )
        // {
        //     for( unsigned int i_pair = 0; i_pair < dmi_pairs.size(); ++i_pair )
        //     {
        //         int ispin = dmi_pairs[i_pair].i + icell*geometry->n_cell_atoms;
        //         int jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, dmi_pairs[i_pair]);
        //         if( jspin >= 0 )
        //         {
        //             int i = 3*ispin;
        //             int j = 3*jspin;

        //             hessian(i+2, j+1) +=  dmi_magnitudes[i_pair] * dmi_normals[i_pair][0];
        //             hessian(i+1, j+2) += -dmi_magnitudes[i_pair] * dmi_normals[i_pair][0];
        //             hessian(i, j+2)   +=  dmi_magnitudes[i_pair] * dmi_normals[i_pair][1];
        //             hessian(i+2, j)   += -dmi_magnitudes[i_pair] * dmi_normals[i_pair][1];
        //             hessian(i+1, j)   +=  dmi_magnitudes[i_pair] * dmi_normals[i_pair][2];
        //             hessian(i, j+1)   += -dmi_magnitudes[i_pair] * dmi_normals[i_pair][2];

        //             #ifndef SPIRIT_USE_OPENMP
        //             hessian(j+1, i+2) +=  dmi_magnitudes[i_pair] * dmi_normals[i_pair][0];
        //             hessian(j+2, i+1) += -dmi_magnitudes[i_pair] * dmi_normals[i_pair][0];
        //             hessian(j+2, i)   +=  dmi_magnitudes[i_pair] * dmi_normals[i_pair][1];
        //             hessian(j, i+2)   += -dmi_magnitudes[i_pair] * dmi_normals[i_pair][1];
        //             hessian(j, i+1)   +=  dmi_magnitudes[i_pair] * dmi_normals[i_pair][2];
        //             hessian(j+1, i)   += -dmi_magnitudes[i_pair] * dmi_normals[i_pair][2];
        //             #endif
        //         }
        //     }
        // }
    }

    // Hamiltonian name as string
    static const std::string name = "Impurity";
    const std::string& Hamiltonian_Impurity::Name() { return name; }
}