#ifndef SPIRIT_USE_CUDA

#include <engine/Hamiltonian_Micromagnetic.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Neighbours.hpp>
#include <data/Spin_System.hpp>
#include <utility/Constants.hpp>
#include <algorithm>
#include <complex>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "FFT.hpp"
#include <cstdio>
#include <utility/Custom_Field.hpp>
using namespace Data;
using namespace Utility;
namespace C = Utility::Constants_Micromagnetic;
using Engine::Vectormath::idx_from_pair;
using Engine::Vectormath::tupel_from_idx_2;


namespace Engine
{
    Hamiltonian_Micromagnetic::Hamiltonian_Micromagnetic(
        scalarfield external_field_magnitude, vectorfield external_field_normal,
        intfield n_anisotropies, std::vector<std::vector<scalar>>  anisotropy_magnitudes, std::vector<std::vector<Vector3>> anisotropy_normals,
        scalarfield  exchange_stiffness,
        scalarfield  dmi,
        std::shared_ptr<Data::Geometry> geometry,
        int spatial_gradient_order,
        intfield boundary_conditions,
        Vector3 cell_sizes,
        scalarfield Ms, int region_num
    ) : Hamiltonian(boundary_conditions), spatial_gradient_order(spatial_gradient_order), geometry(geometry),
        external_field_magnitude(external_field_magnitude), external_field_normal(external_field_normal),
        n_anisotropies(n_anisotropies), anisotropy_magnitudes(anisotropy_magnitudes), anisotropy_normals(anisotropy_normals),
        exchange_stiffness(exchange_stiffness), dmi(dmi),
        cell_sizes(cell_sizes), Ms(Ms), region_num(region_num)
    {
        // Generate interaction pairs, constants etc.
        regions=intfield(geometry->nos,0);

        Regionvalues test;

        regions_book = regionbook(region_num);
        for (int i=0; i<region_num; i++)
        {
            test.external_field_magnitude = external_field_magnitude[i];
            test.external_field_normal = external_field_normal[i];
            test.Ms = Ms[i];
            test.Dmi = dmi[i];
            test.Aexch = exchange_stiffness[i];
            test.n_anisotropies = n_anisotropies[i];

            for (int j=0; j < n_anisotropies[i]; j++)
            {
                test.anisotropy_magnitudes[j] = anisotropy_magnitudes[i][j];
                test.anisotropy_normals[j] = anisotropy_normals[i][j];
            }

            this->regions_book[i] = test;
        }

        exchange_table = std::vector<std::vector<scalar>>(region_num, std::vector<scalar>(region_num, 0));
        for (int i=0;i<region_num;i++)
        {
            for (int j=0;j<region_num;j++)
            {
                exchange_table[i][j] = (this->regions_book[i].Aexch + this->regions_book[j].Aexch) / 2;
            }
        }

        exchange_tensor <<  exchange_stiffness[0], 0, 0,
                            0, exchange_stiffness[0], 0, 
                            0, 0, exchange_stiffness[0];
        std::cout << "\n\nexchange_tensor\n" << exchange_tensor << "\n\n";

        dmi_tensor <<   dmi[0], 0, 0,
                        0, dmi[0], 0, 
                        0, 0, dmi[0];
        std::cout << "\n\ndmi_tensor\n" << dmi_tensor << "\n\n";

        std::cout << "\n\nMs " << Ms[0] << "\n\n";
        std::cout << "\n\nexternal field magnitude " << external_field_magnitude[0] << "\n\n";
        std::cout << "\n\nexternal field normal " << external_field_normal[0].transpose() << "\n\n";

        this->Update_Interactions();
    }

    void Hamiltonian_Micromagnetic::Update_Interactions()
    {
        #if defined(SPIRIT_USE_OPENMP)
        // When parallelising (cuda or openmp), we need all neighbours per spin
        const bool use_redundant_neighbours = true;
        #else
        // When running on a single thread, we can ignore redundant neighbours
        const bool use_redundant_neighbours = false;
        #endif

        Pair t{0,0,{0,0,0}};
        neigh = pairfield(18, t);

        neigh[0].translations = {1, 0, 0};
        neigh[1].translations = {-1, 0, 0};
        neigh[2].translations = {0, 1, 0};
        neigh[3].translations = {0, -1, 0};
        neigh[4].translations = {0, 0, 1};
        neigh[5].translations = {0, 0, -1};
        neigh[6].translations = {1, 1, 0};
        neigh[7].translations = {-1, -1, 0};
        neigh[8].translations = {1, -1, 0};
        neigh[9].translations = {-1, 1, 0};
        neigh[10].translations = {1, 0, 1};
        neigh[11].translations = {-1, 0, -1};
        neigh[12].translations = {1, 0, -1};
        neigh[13].translations = {-1, 0, 1};
        neigh[14].translations = {0, 1, 1};
        neigh[15].translations = {0, -1, -1};
        neigh[16].translations = {0, 1, -1};
        neigh[17].translations = {0, -1, 1};

        this->spatial_gradient = field<Matrix3>(geometry->nos, Matrix3::Zero());

        // Update, which terms still contribute
        this->Update_Energy_Contributions();
        // Dipole-dipole (FFT)
        this->Prepare_DDI();
        external_field=vectorfield(geometry->nos,Vector3{0,0,0});

    }

    void Hamiltonian_Micromagnetic::Update_Energy_Contributions()
    {
        #ifndef SPIRIT_LOW_MEMORY
            this->energy_contributions_per_spin = std::vector<std::pair<std::string, scalarfield>>(0);
            this->gradient_contributions_per_spin = std::vector<std::pair<std::string, vectorfield>>(0);
            //this->idx_zeeman = 0;

            // External field
            //if( this->external_field_magnitude > 0 )
            //{
            this->energy_contributions_per_spin.push_back({"Zeeman", scalarfield(0)});
            this->gradient_contributions_per_spin.push_back({"Zeeman", vectorfield(0)});
            this->idx_zeeman = this->energy_contributions_per_spin.size()-1;
            //}
            //else
                //this->idx_zeeman = -1;
            // TODO: Anisotropy
            // if( ... )
            // {
            this->energy_contributions_per_spin.push_back({"Anisotropy", scalarfield(0) });
            this->gradient_contributions_per_spin.push_back({"Anisotropy", vectorfield(0) });
            this->idx_anisotropy = this->energy_contributions_per_spin.size()-1;
            // }
            // else
                //this->idx_anisotropy = -1;
            // TODO: Exchange
            // if( ... )
            // {
            this->energy_contributions_per_spin.push_back({"Exchange", scalarfield(0) });
            this->gradient_contributions_per_spin.push_back({"Exchange", vectorfield(0) });
            this->idx_exchange = this->energy_contributions_per_spin.size()-1;
            // }
            // else
                //this->idx_exchange = -1;
            // TODO: DMI
            // if( ... )
            // {
            this->energy_contributions_per_spin.push_back({"DMI", scalarfield(0) });
            this->gradient_contributions_per_spin.push_back({"DMI", vectorfield(0) });
            this->idx_dmi = this->energy_contributions_per_spin.size()-1;
            // }
            // else
                //this->idx_dmi = -1;
            // TODO: DDI
            // if( ... )
            // {
            this->energy_contributions_per_spin.push_back({"DDI", scalarfield(0) });
            this->gradient_contributions_per_spin.push_back({"DDI", vectorfield(0) });
            this->idx_ddi = this->energy_contributions_per_spin.size()-1;

            int nos = geometry->nos;
            for( auto& contrib : this->gradient_contributions_per_spin )
            {
                // Allocate if not already allocated
                if (contrib.second.size() != nos) contrib.second = vectorfield(nos, Vector3{0, 0, 0});
                // Otherwise set to zero
                else Vectormath::fill(contrib.second, Vector3{0, 0, 0});
            }
        #endif

        #ifdef SPIRIT_LOW_MEMORY
            // Energy contributions
            temp_energies=scalarfield(this->geometry->nos,0);
            energy_array=std::vector<std::pair<std::string, scalar>> (0);
            this->energy_array.push_back({"Zeeman", 0});
            this->idx_zeeman = this->energy_array.size()-1;
            this->energy_array.push_back({"Anisotropy", 0});
            this->idx_anisotropy = this->energy_array.size()-1;
            this->energy_array.push_back({"Exchange", 0});
            this->idx_exchange = this->energy_array.size()-1;
            this->energy_array.push_back({"DMI", 0 });
            this->idx_dmi = this->energy_array.size()-1;
            this->energy_array.push_back({"DDI", 0 });
            this->idx_ddi = this->energy_array.size()-1;
        #endif
    }

    void Hamiltonian_Micromagnetic::Energy_Contributions_per_Spin(const vectorfield & spins, std::vector<std::pair<std::string, scalarfield>> & contributions)
    {
        #ifndef SPIRIT_LOW_MEMORY
            if( contributions.size() != this->energy_contributions_per_spin.size() )
            {
                contributions = this->energy_contributions_per_spin;
            }

            int nos = spins.size();
            for( auto& contrib : contributions )
            {
                // Allocate if not already allocated
                if (contrib.second.size() != nos) contrib.second = scalarfield(nos, 0);
                // Otherwise set to zero
                //else Vectormath::fill(contrib.second, 0);
            }
            // External field
            if( this->idx_zeeman >=0 )     Energy_Set(spins, contributions[this->idx_zeeman].second, this->gradient_contributions_per_spin[this->idx_zeeman].second);

            // Anisotropy
            if( this->idx_anisotropy >=0 ) Energy_Set(spins, contributions[this->idx_anisotropy].second, this->gradient_contributions_per_spin[this->idx_anisotropy].second);

            // Exchange
            if( this->idx_exchange >=0 )   Energy_Set(spins, contributions[this->idx_exchange].second, this->gradient_contributions_per_spin[this->idx_exchange].second);
            // DMI
            if( this->idx_dmi >=0 )        Energy_Set(spins, contributions[this->idx_dmi].second, this->gradient_contributions_per_spin[this->idx_dmi].second);
            // DDI
            if( this->idx_ddi >=0 )        Energy_Set(spins, contributions[this->idx_ddi].second, this->gradient_contributions_per_spin[this->idx_ddi].second);
        #endif
        #ifdef SPIRIT_LOW_MEMORY
            // Energy already set in Gradient
        #endif
    }

    void Hamiltonian_Micromagnetic::Energy_Set(const vectorfield & spins, scalarfield & Energy, vectorfield & gradient)
    {
        #pragma omp parallel for
        for( int icell = 0; icell < geometry->n_cells_total; ++icell )
        {
            Energy[icell] = 0.5 * regions_book[regions[icell]].Ms * gradient[icell].dot(spins[icell]);
        }
    }
    #ifdef SPIRIT_LOW_MEMORY
        scalar Hamiltonian_Micromagnetic::Energy_Low_Memory(const vectorfield & spins, vectorfield & gradient)
        {
            #pragma omp parallel for
            for( int icell = 0; icell < geometry->n_cells_total; ++icell )
            {
                temp_energies[icell] = 0.5 * regions_book[regions[icell]].Ms * gradient[icell].dot(spins[icell]);
            }
            return Vectormath::sum(temp_energies);
        }
    #endif
    scalar Hamiltonian_Micromagnetic::Energy_Single_Spin(int ispin, const vectorfield & spins)
    {
        scalar Energy = 0;
        return Energy;
    }

    void Hamiltonian_Micromagnetic::Gradient(const vectorfield & spins, vectorfield & gradient)
    {
         // Set to zero
        Vectormath::fill(gradient, {0,0,0});
        this->Spatial_Gradient(spins);
        #ifndef SPIRIT_LOW_MEMORY

            Gradient_Zeeman(this->gradient_contributions_per_spin[this->idx_zeeman].second);
            Gradient_Anisotropy(spins, this->gradient_contributions_per_spin[this->idx_anisotropy].second);
            Gradient_Exchange(spins, this->gradient_contributions_per_spin[this->idx_exchange].second);
            Gradient_DMI(spins, this->gradient_contributions_per_spin[this->idx_dmi].second);
            Gradient_DDI(spins, this->gradient_contributions_per_spin[this->idx_ddi].second);

            Vectormath::add_c_a(1, this->gradient_contributions_per_spin[this->idx_zeeman].second, gradient);
            Vectormath::add_c_a(1, this->gradient_contributions_per_spin[this->idx_anisotropy].second, gradient);
            Vectormath::add_c_a(1, this->gradient_contributions_per_spin[this->idx_exchange].second, gradient);
            Vectormath::add_c_a(1, this->gradient_contributions_per_spin[this->idx_dmi].second, gradient);
            Vectormath::add_c_a(1, this->gradient_contributions_per_spin[this->idx_ddi].second, gradient);
        #endif
        #ifdef SPIRIT_LOW_MEMORY
            scalar temp=0;
            scalar temp1=0;
            Gradient_Zeeman(gradient);
            temp=Energy_Low_Memory(spins, gradient);
            this->energy_array[this->idx_zeeman].second=temp-temp1;
            temp1=temp;
            Gradient_Anisotropy(spins, gradient);
            temp=Energy_Low_Memory(spins, gradient);
            this->energy_array[this->idx_anisotropy].second=temp-temp1;
            //printf("%f\n", temp-temp1);
            temp1=temp;
            Gradient_Exchange(spins, gradient);
            temp=Energy_Low_Memory(spins, gradient);
            this->energy_array[this->idx_exchange].second=temp-temp1;
            //printf("%f\n", temp-temp1);
            temp1=temp;
            Gradient_DMI(spins, gradient);
            temp=Energy_Low_Memory(spins, gradient);
            this->energy_array[this->idx_dmi].second=temp-temp1;
            //printf("%f\n", temp-temp1);
            temp1=temp;
            Gradient_DDI(spins, gradient);
            temp=Energy_Low_Memory(spins, gradient);
            this->energy_array[this->idx_ddi].second=temp-temp1;
            //printf("%f\n", temp-temp1);
        #endif

    }

    void Hamiltonian_Micromagnetic::Gradient_Zeeman(vectorfield & gradient)
    {
        scalar m0 = (4 * C::Pi) * 1e-7;
        Utility::Custom_Field::CustomField(geometry->n_cells_total, this->geometry->positions.data(),this->geometry->center, picoseconds_passed, external_field.data());
        #pragma omp parallel for
        for( int icell = 0; icell < geometry->n_cells_total; ++icell )
        {
            #ifndef SPIRIT_LOW_MEMORY
                gradient[icell] -= m0*regions_book[regions[icell]].Ms * regions_book[regions[icell]].external_field_magnitude*regions_book[regions[icell]].external_field_normal;
                gradient[icell] -= m0*regions_book[regions[icell]].Ms * external_field[icell];
            #endif
            #ifdef SPIRIT_LOW_MEMORY
                gradient[icell] = m0*regions_book[regions[icell]].Ms * regions_book[regions[icell]].external_field_magnitude*regions_book[regions[icell]].external_field_normal;
                gradient[icell] -= m0*regions_book[regions[icell]].Ms * external_field[icell];
            #endif
        }
    }

    void Hamiltonian_Micromagnetic::Gradient_Anisotropy(const vectorfield & spins, vectorfield & gradient)
    {
        #pragma omp parallel for
        for( int icell = 0; icell < geometry->n_cells_total; ++icell )
        {
            #ifndef SPIRIT_LOW_MEMORY
                gradient[icell][0]=0;
                gradient[icell][1]=0;
                gradient[icell][2]=0;
            #endif
            for (int i = 0; i < regions_book[regions[icell]].n_anisotropies; i++)
            {
                gradient[icell] -= 2.0 * regions_book[regions[icell]].anisotropy_magnitudes[i] / regions_book[regions[icell]].Ms * regions_book[regions[icell]].anisotropy_normals[i] * regions_book[regions[icell]].anisotropy_normals[i].dot(spins[icell]);
                //gradient[ispin] = -2.0 * anisotropy_magnitude / Ms * anisotropy_normal * (anisotropy_normal[0]*spins[ispin][0]+anisotropy_normal[1]*spins[ispin][1]+anisotropy_normal[2]*spins[ispin][2]);
                //gradient[ispin] -= 2.0 * 500000 / Ms * temp2 * temp2.dot(spins[ispin]);
                //gradient[ispin] += 2.0 * anisotropy_mag / Ms * ((pow(temp2.dot(spins[ispin]),2)+ pow(temp3.dot(spins[ispin]), 2))*(temp1.dot(spins[ispin])*temp1)+ (pow(temp1.dot(spins[ispin]), 2) + pow(temp3.dot(spins[ispin]), 2))*(temp2.dot(spins[ispin])*temp2)+(pow(temp1.dot(spins[ispin]),2)+ pow(temp2.dot(spins[ispin]), 2))*(temp3.dot(spins[ispin])*temp3));
                //gradient[ispin] -= 2.0 * 500000 / Ms * ((pow(temp2.dot(spins[ispin]), 2) + pow(temp3.dot(spins[ispin]), 2))*(temp1.dot(spins[ispin])*temp1) + (pow(temp1.dot(spins[ispin]), 2) + pow(temp3.dot(spins[ispin]), 2))*(temp2.dot(spins[ispin])*temp2));

            }
        }
    }
    void Hamiltonian_Micromagnetic::Gradient_Exchange(const vectorfield & spins, vectorfield & gradient)
    {
        #pragma omp parallel for
        for( int icell = 0; icell < geometry->n_cells_total; ++icell )
        {
            int ispin = icell;//basically id of a cell
            #ifndef SPIRIT_LOW_MEMORY
                gradient[icell][0]=0;
                gradient[icell][1]=0;
                gradient[icell][2]=0;
            #endif

            for (unsigned int i = 0; i < 3; ++i)
            {
                int ispin_plus  = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[2 * i]);
                int ispin_minus = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[2 * i + 1]);

                if (ispin_plus == -1) {
                    ispin_plus = ispin;
                }
                if (ispin_minus == -1) {
                    ispin_minus = ispin;
                }
                scalar Aexch = 0.25 * (2 * regions_book[regions[ispin]].Aexch + regions_book[regions[ispin_plus]].Aexch + regions_book[regions[ispin_minus]].Aexch);

                gradient[ispin] -= 2 * Aexch / regions_book[regions[ispin]].Ms * (spins[ispin_plus] - 2 * spins[ispin] + spins[ispin_minus]) / (cell_sizes[i] * cell_sizes[i]);
            }
        }
    }

    void Hamiltonian_Micromagnetic::Spatial_Gradient(const vectorfield & spins)
    {
        #pragma omp parallel for
        for( int icell = 0; icell < geometry->n_cells_total; ++icell )
        {
            int ispin = icell;//basically id of a cell
            for (unsigned int i = 0; i < 3; ++i)
            {
                int ispin_plus = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms,geometry->atom_types, neigh[2 * i]);
                int ispin_minus = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms,geometry->atom_types, neigh[2 * i + 1]);
                if (ispin_plus == -1) {
                    ispin_plus = ispin;
                }
                if (ispin_minus == -1) {
                    ispin_minus = ispin;
                }
                spatial_gradient[ispin](0, i) = (spins[ispin_plus][0] - spins[ispin_minus][0]) / (cell_sizes[i]) / 2;
                spatial_gradient[ispin](1, i) = (spins[ispin_plus][1] - spins[ispin_minus][1]) / (cell_sizes[i]) / 2;
                spatial_gradient[ispin](2, i) = (spins[ispin_plus][2] - spins[ispin_minus][2]) / (cell_sizes[i]) / 2;
            }
        }
    }

    void Hamiltonian_Micromagnetic::Gradient_DMI(const vectorfield & spins, vectorfield & gradient)
    {
        Matrix3 dmi_tensor;
        dmi_tensor(0,0)=0;
        dmi_tensor(0,1)=0;
        dmi_tensor(0,2)=0;
        dmi_tensor(1,0)=0;
        dmi_tensor(1,1)=0;
        dmi_tensor(1,2)=0;
        dmi_tensor(2,0)=0;
        dmi_tensor(2,1)=0;
        dmi_tensor(2,2)=0;
        #pragma omp parallel for
        for( int icell = 0; icell < geometry->n_cells_total; ++icell )
        {
            int ispin = icell;//basically id of a cell
            dmi_tensor(0,1)=-regions_book[regions[ispin]].Dmi;
            dmi_tensor(1,0)=regions_book[regions[ispin]].Dmi;
            #ifndef SPIRIT_LOW_MEMORY
                gradient[icell][0]=0;
                gradient[icell][1]=0;
                gradient[icell][2]=0;
            #endif
            for (unsigned int i = 0; i < 3; ++i)
            {
                gradient[ispin][0] += 2 * dmi_tensor(1, i) /regions_book[regions[ispin]].Ms * spatial_gradient[ispin](2, i) - 2 * dmi_tensor(2, i) /regions_book[regions[ispin]].Ms * spatial_gradient[ispin](1, i);
                gradient[ispin][1] += 2 * dmi_tensor(2, i) /regions_book[regions[ispin]].Ms * spatial_gradient[ispin](0, i) - 2 * dmi_tensor(0, i) /regions_book[regions[ispin]].Ms * spatial_gradient[ispin](2, i);
                gradient[ispin][2] += 2 * dmi_tensor(0, i) /regions_book[regions[ispin]].Ms * spatial_gradient[ispin](1, i) - 2 * dmi_tensor(1, i) /regions_book[regions[ispin]].Ms * spatial_gradient[ispin](0, i);
            }
        }
    }


    void Hamiltonian_Micromagnetic::Gradient_DDI(const vectorfield & spins, vectorfield & gradient)
    {
        this->Gradient_DDI_FFT(spins, gradient);
    }
    void Hamiltonian_Micromagnetic::Gradient_DDI_Cutoff(const vectorfield & spins, vectorfield & gradient)
    {
        // TODO
    }

    void Hamiltonian_Micromagnetic::Gradient_DDI_FFT(const vectorfield & spins, vectorfield & gradient)
    {
        #pragma omp parallel for
        for( int icell = 0; icell < geometry->n_cells_total; ++icell )
        {
            mult_spins[icell]=spins[icell]*regions_book[regions[icell]].Ms/minMs;
        }
        FFT_Spins(mult_spins);
        auto& ft_D_matrices = transformed_dipole_matrices;

        auto& ft_spins = fft_plan_spins.cpx_ptr;

        auto& res_iFFT = fft_plan_reverse.real_ptr;
        auto& res_mult = fft_plan_reverse.cpx_ptr;

        int number_of_mults = it_bounds_pointwise_mult[0] * it_bounds_pointwise_mult[1] * it_bounds_pointwise_mult[2];
        //#ifndef SPIRIT_LOW_MEMORY

        int Na = geometry->n_cells[0];
        int Nb = geometry->n_cells[1];
        int Nc = geometry->n_cells[2];
        /*#endif
        #ifdef SPIRIT_LOW_MEMORY
            CU_Mult_Spins << <(geometry->nos + 1023) / 1024, 1024 >> > (minMs, geometry->n_cells_total, regions_book.data(),regions.data(),spins.data(), spins.data());
            CU_CHECK_AND_SYNC();
            FFT_Spins(spins);
        #endif*/
        //FFT_Spins(spins);

        // TODO: also parallelize over i_b1
        // Loop over basis atoms (i.e sublattices) and add contribution of each sublattice
        int idx_b1, idx_b2, idx_d;

        // Workaround for compability with intel compiler
        const int c_n_cell_atoms = geometry->n_cell_atoms;
        const int * c_it_bounds_pointwise_mult = it_bounds_pointwise_mult.data();
        // Loop over basis atoms (i.e sublattices)
        #pragma omp parallel for collapse(5)
        for( int i_b1 = 0; i_b1 <c_n_cell_atoms; ++i_b1 )
        {
            for( int c = 0; c < c_it_bounds_pointwise_mult[2]; ++c )
            {
                for( int b = 0; b < c_it_bounds_pointwise_mult[1]; ++b )
                {
                    for( int a = 0; a < c_it_bounds_pointwise_mult[0]; ++a )
                    {
                        for( int i_b2 = 0; i_b2 < c_n_cell_atoms; ++i_b2 )
                        {    //std::cout<<c<<"\n";
                            // Look up at which position the correct D-matrices are saved
                            int& b_inter = inter_sublattice_lookup[i_b1 + i_b2 * geometry->n_cell_atoms];

                            idx_b2 = i_b2 * spin_stride.basis + a * spin_stride.a + b * spin_stride.b + c * spin_stride.c;
                            idx_b1 = i_b1 * spin_stride.basis + a * spin_stride.a + b * spin_stride.b + c * spin_stride.c;
                            idx_d  = b_inter * dipole_stride.basis + a * dipole_stride.a + b * dipole_stride.b + c * dipole_stride.c;

                            auto& fs_x = ft_spins[idx_b2                       ];
                            auto& fs_y = ft_spins[idx_b2 + 1 * spin_stride.comp];
                            auto& fs_z = ft_spins[idx_b2 + 2 * spin_stride.comp];

                            auto& fD_xx = ft_D_matrices[idx_d                    ];
                            auto& fD_xy = ft_D_matrices[idx_d + 1 * dipole_stride.comp];
                            auto& fD_xz = ft_D_matrices[idx_d + 2 * dipole_stride.comp];
                            auto& fD_yy = ft_D_matrices[idx_d + 3 * dipole_stride.comp];
                            auto& fD_yz = ft_D_matrices[idx_d + 4 * dipole_stride.comp];
                            auto& fD_zz = ft_D_matrices[idx_d + 5 * dipole_stride.comp];

                            FFT::addTo(res_mult[idx_b1 + 0 * spin_stride.comp], FFT::mult3D(fD_xx, fD_xy, fD_xz, fs_x, fs_y, fs_z), i_b2 == 0);
                            FFT::addTo(res_mult[idx_b1 + 1 * spin_stride.comp], FFT::mult3D(fD_xy, fD_yy, fD_yz, fs_x, fs_y, fs_z), i_b2 == 0);
                            FFT::addTo(res_mult[idx_b1 + 2 * spin_stride.comp], FFT::mult3D(fD_xz, fD_yz, fD_zz, fs_x, fs_y, fs_z), i_b2 == 0);
                        }
                    }
                }// end iteration over padded lattice cells
            }// end iteration over second sublattice
        }

        // Inverse Fourier Transform
        FFT::batch_iFour_3D(fft_plan_reverse);

        int nos = it_bounds_write_gradients[0] * it_bounds_write_gradients[1] * it_bounds_write_gradients[2] * it_bounds_write_gradients[3];
        int tupel[4];
        int idx_pad;
        #pragma omp parallel for
        for( int idx_orig = 0; idx_orig < nos; ++idx_orig )
        {

            tupel_from_idx_2(idx_orig, tupel, it_bounds_write_gradients.data(), 4); //tupel now is {ib, a, b, c}
            idx_pad = tupel[0] * spin_stride.basis + tupel[1] * spin_stride.a + tupel[2] * spin_stride.b + tupel[3] * spin_stride.c;
            //printf("%d %f %f\n", idx_orig, resiFFT[this->idx_pad],gradient[this->idx_orig][0]);
            #ifndef SPIRIT_LOW_MEMORY
                gradient[idx_orig][0] = -res_iFFT[idx_pad]*minMs*1e-7/(sublattice_size);
                gradient[idx_orig][1] = -res_iFFT[idx_pad + 1 * spin_stride.comp]*minMs*1e-7/(sublattice_size);
                gradient[idx_orig][2] = -res_iFFT[idx_pad + 2 * spin_stride.comp]*minMs*1e-7/(sublattice_size);
            #endif
            #ifdef SPIRIT_LOW_MEMORY
                gradient[idx_orig][0] -= res_iFFT[idx_pad]*minMs*1e-7/(sublattice_size);
                gradient[idx_orig][1] -= res_iFFT[idx_pad + 1 * spin_stride.comp]*minMs*1e-7/(sublattice_size);
                gradient[idx_orig][2] -= res_iFFT[idx_pad + 2 * spin_stride.comp]*minMs*1e-7/(sublattice_size);
            #endif
        }
        /*#ifdef SPIRIT_LOW_MEMORY
            CU_Div_Spins << <(geometry->nos + 1023) / 1024, 1024 >> > (minMs, geometry->n_cells_total, regions_book.data(),regions.data(),spins.data(),spins.data());
            CU_CHECK_AND_SYNC();
        #endif*/
    }//end Field_DipoleDipole

    void Hamiltonian_Micromagnetic::Gradient_DDI_Direct(const vectorfield & spins, vectorfield & gradient)
        {
            int tupel1[3];
            int tupel2[3];
            int sublattice_size = it_bounds_write_dipole[0] * it_bounds_write_dipole[1] * it_bounds_write_dipole[2];

            scalar mult = 1 / (4 * C::Pi);
            scalar m0 = (4 * C::Pi) * 1e-7;
            int img_a = boundary_conditions[0] == 0 ? 0 : ddi_n_periodic_images[0];
            int img_b = boundary_conditions[1] == 0 ? 0 : ddi_n_periodic_images[1];
            int img_c = boundary_conditions[2] == 0 ? 0 : ddi_n_periodic_images[2];

            for (int idx1 = 0; idx1 < geometry->nos; idx1++)
            {
                gradient[idx1][0]=0;
                gradient[idx1][1]=0;
                gradient[idx1][2]=0;
                double kk=0;
                for (int idx2 = 0; idx2 < geometry->nos; idx2++)
                {
                    int a1 = idx1%(it_bounds_write_spins[1]);
                    int b1 = ((int)(idx1/it_bounds_write_spins[1]))%(it_bounds_write_spins[2]);
                    int c1 = (int)idx1/(it_bounds_write_spins[1]*it_bounds_write_spins[2]);
                    int a2 = idx2%(it_bounds_write_spins[1]);
                    int b2 = ((int)(idx2/it_bounds_write_spins[1]))%(it_bounds_write_spins[2]);
                    int c2 = (int)idx2/(it_bounds_write_spins[1]*it_bounds_write_spins[2]);
                    /*int a_idx = a < n_cells[0] ? a : a - iteration_bounds[0];
                    int b_idx = b < n_cells[1] ? b : b - iteration_bounds[1];
                    int c_idx = c < n_cells[2] ? c : c - iteration_bounds[2];*/
                    int a_idx = a1-a2;
                    int b_idx = b1-b2;
                    int c_idx = c1-c2;
                    if ((a_idx==0) && (b_idx==0) && (c_idx==0)){
                        //printf("test\n");
                        //continue;
                    }
                    //printf("%d %d %d\n", a_idx,b_idx,c_idx);
                    /*if ((a_idx==20) || (b_idx==20) || (c_idx==1)){
                        //printf("test1\n");
                        //if (c_idx!=1)
                            //printf("%d %d %d %d\n", a_idx, b_idx, c_idx,  dipole_stride.comp);
                        continue;
                    }*/
                    //scalar delta[3] = { 3,3,0.3 };
                    //int idx = b_inter * dipole_stride.basis + a * dipole_stride.a + b * dipole_stride.b + c * dipole_stride.c;
                    scalar Dxx = 0, Dxy = 0, Dxz = 0, Dyy = 0, Dyz = 0, Dzz = 0;
                    //asa
                    for (int i = 0; i < 2; i++) {
                        for (int j = 0; j < 2; j++) {
                            for (int k = 0; k < 2; k++) {
                                double r = sqrt((a_idx + i - 0.5f)*(a_idx + i - 0.5f)*cell_sizes[0]* cell_sizes[0] + (b_idx + j - 0.5f)*(b_idx + j-0.5f)*cell_sizes[1] * cell_sizes[1] + (c_idx + k - 0.5f)*(c_idx + k - 0.5f)*cell_sizes[2] * cell_sizes[2]);
                                Dxx += mult * pow(-1.0f, i + j + k) * atan(((c_idx + k-0.5f) * (b_idx + j - 0.5f) * cell_sizes[1]*cell_sizes[2]/cell_sizes[0] / r / (a_idx + i - 0.5f)));
                                //fft_dipole_inputs[this->idx + 1 * dipole_stride.comp] += -mult * pow(-1.0f, i + j + k) * log(abs(((c_idx + k - 0.5f)* cell_sizes[2] + r)/((c_idx + k - 0.5f)* cell_sizes[2] - r)));
                                //fft_dipole_inputs[this->idx + 2 * dipole_stride.comp] += -mult * pow(-1.0f, i + j + k) * log(abs(((b_idx + j - 0.5f)* cell_sizes[1] + r)/((b_idx + j - 0.5f)* cell_sizes[1] - r)));
                                Dxy -= mult * pow(-1.0f, i + j + k) * log((((c_idx + k - 0.5f)* cell_sizes[2] + r)));
                                Dxz -= mult * pow(-1.0f, i + j + k) * log((((b_idx + j - 0.5f)* cell_sizes[1] + r)));

                                Dyy += mult * pow(-1.0f, i + j + k) * atan(((a_idx + i-0.5f) * (c_idx + k - 0.5f) * cell_sizes[2]*cell_sizes[0]/cell_sizes[1] / r / (b_idx + j - 0.5f)));
                                //fft_dipole_inputs[this->idx + 4 * dipole_stride.comp] += -mult * pow(-1.0f, i + j + k) * log(abs(((a_idx + i - 0.5f)* cell_sizes[0] + r)/((a_idx + i - 0.5f)* cell_sizes[0] - r)));
                                Dyz -= mult * pow(-1.0f, i + j + k) * log((((a_idx + i - 0.5f)* cell_sizes[0] + r)));
                                Dzz += mult * pow(-1.0f, i + j + k) * atan(((b_idx + j-0.5f) * (a_idx + i - 0.5f) * cell_sizes[0]*cell_sizes[1]/cell_sizes[2] / r / (c_idx + k - 0.5f)));

                            }
                        }
                    }/*
                    Dxx=Nii(a_idx*delta[0],b_idx*delta[1],c_idx*delta[2],delta[0],delta[1],delta[2]);
                    Dxy=Nij(a_idx*delta[0],b_idx*delta[1],c_idx*delta[2],delta[0],delta[1],delta[2]);
                    Dxz=Nij(a_idx*delta[0],c_idx*delta[2], b_idx*delta[1],delta[0],delta[2],delta[1]);
                    Dyy=Nii(b_idx*delta[1],a_idx*delta[0],c_idx*delta[2],delta[1],delta[0],delta[2]);
                    Dyz=Nij(b_idx*delta[1],c_idx*delta[2], b_idx*delta[1],delta[1],delta[2],delta[0]);
                    Dzz=Nii(c_idx*delta[2],a_idx*delta[0],b_idx*delta[1],delta[2],delta[0],delta[1]);*/
                    kk+=Dxx;
                    /*gradient[idx1][0] -= (Dxx * spins[idx2][0] + Dxy * spins[idx2][1] + Dxz * spins[idx2][2]) * Ms*m0;
                    gradient[idx1][1] -= (Dxy * spins[idx2][0] + Dyy * spins[idx2][1] + Dyz * spins[idx2][2]) * Ms*m0;
                    gradient[idx1][2] -= (Dxz * spins[idx2][0] + Dyz * spins[idx2][1] + Dzz * spins[idx2][2]) * Ms*m0;*/
                }

            }
        }

    void Hamiltonian_Micromagnetic::FFT_Spins(const vectorfield & spins)
    {
        int nos = it_bounds_write_spins[0] * it_bounds_write_spins[1] * it_bounds_write_spins[2] * it_bounds_write_spins[3];
        int tupel[4];
        int idx_pad;
        #pragma omp parallel for
        for( int idx_orig = 0; idx_orig < geometry->n_cells_total; ++idx_orig )
        {
            tupel_from_idx_2(idx_orig, tupel, it_bounds_write_spins.data(), 4); //tupel now is {ib, a, b, c}
            idx_pad = tupel[0] * spin_stride.basis + tupel[1] * spin_stride.a + tupel[2] * spin_stride.b + tupel[3] * spin_stride.c;
            fft_plan_spins.real_ptr[idx_pad] = spins[idx_orig][0];
            fft_plan_spins.real_ptr[idx_pad + 1 * spin_stride.comp] = spins[idx_orig][1];
            fft_plan_spins.real_ptr[idx_pad + 2 * spin_stride.comp] = spins[idx_orig][2];
            //printf("%f %f\n",fft_spin_inputs[this->idx_pad], fft_spin_inputs[this->idx_pad+30]);
        }
        FFT::batch_Four_3D(fft_plan_spins);
    }

    void Hamiltonian_Micromagnetic::FFT_Dipole_Matrices(FFT::FFT_Plan & fft_plan_dipole, int img_a, int img_b, int img_c)
    {
        auto& fft_dipole_inputs = fft_plan_dipole.real_ptr;

        field<int> img = {
                            img_a,
                            img_b,
                            img_c
        };

        // Work around to make bravais vectors and cell_atoms available to GPU as they are currently saves as std::vectors and not fields ...
        auto translation_vectors = field<Vector3>();
        auto cell_atom_translations = field<Vector3>();

        for (int i = 0; i < 3; i++)
            translation_vectors.push_back(geometry->lattice_constant * geometry->bravais_vectors[i]);

        for (int i = 0; i < geometry->n_cell_atoms; i++)
            cell_atom_translations.push_back(geometry->positions[i]);

        int tupel[3];
        int sublattice_size = it_bounds_write_dipole[0] * it_bounds_write_dipole[1] * it_bounds_write_dipole[2];
        //prefactor of ddi interaction
        //scalar mult = 2.0133545*1e-28 * 0.057883817555 * 0.057883817555 / (4 * 3.141592653589793238462643383279502884197169399375105820974 * 1e-30);
        //scalar mult = 1 / (4 * 3.141592653589793238462643383279502884197169399375105820974);
        scalar mult = 1;
        //#pragma omp parallel for
        for( int idx0 = 0; idx0 < sublattice_size; ++idx0 )
        {
            tupel_from_idx_2(idx0, tupel, it_bounds_write_dipole.data(), 3); // tupel now is {a, b, c}
            auto& a = tupel[0];
            auto& b = tupel[1];
            auto& c = tupel[2];
            //std::cout<<a<<" "<<b <<" "<<c<<"\n";
            /*if ((a>198)||(b>198)||(c>198)){
                printf("%d %d %d\n", a,b,c);
            }*/
            /*int a_idx = a < n_cells[0] ? a : a - iteration_bounds[0];
            int b_idx = b < n_cells[1] ? b : b - iteration_bounds[1];
            int c_idx = c < n_cells[2] ? c : c - iteration_bounds[2];*/
            /*int a_idx = a +1 - (int)iteration_bounds[0]/2;
            int b_idx = b +1- (int)iteration_bounds[1]/2;
            int c_idx = c +1- (int)iteration_bounds[2]/2;*/
            int a_idx = a < geometry->n_cells[0] ? a : a - it_bounds_write_dipole[0];
            int b_idx = b < geometry->n_cells[1] ? b : b - it_bounds_write_dipole[1];
            int c_idx = c < geometry->n_cells[2] ? c : c - it_bounds_write_dipole[2];

            int idx = a * dipole_stride.a + b * dipole_stride.b + c * dipole_stride.c;
            //std::cout<<it_bounds_write_dipole[0]<<"\n";
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    for (int k = 0; k < 2; k++) {
                        double r = sqrt((a_idx + i - 0.5f)*(a_idx + i - 0.5f)*cell_sizes[0]* cell_sizes[0] + (b_idx + j - 0.5f)*(b_idx + j-0.5f)*cell_sizes[1] * cell_sizes[1] + (c_idx + k - 0.5f)*(c_idx + k - 0.5f)*cell_sizes[2] * cell_sizes[2]);
                        fft_dipole_inputs[idx] += mult * pow(-1.0f, i + j + k) * atan(((c_idx + k-0.5f) * (b_idx + j - 0.5f) * cell_sizes[1]*cell_sizes[2]/cell_sizes[0] / r / (a_idx + i - 0.5f)));
                        //fft_dipole_inputs[idx + 1 * dipole_stride.comp] += -mult * pow(-1.0f, i + j + k) * log(abs(((c_idx + k - 0.5f)* cell_sizes[2] + r)/((c_idx + k - 0.5f)* cell_sizes[2] - r)));
                        //fft_dipole_inputs[idx + 2 * dipole_stride.comp] += -mult * pow(-1.0f, i + j + k) * log(abs(((b_idx + j - 0.5f)* cell_sizes[1] + r)/((b_idx + j - 0.5f)* cell_sizes[1] - r)));
                        fft_dipole_inputs[idx + 1 * dipole_stride.comp] -= mult * pow(-1.0f, i + j + k) * log((((c_idx + k - 0.5f)* cell_sizes[2] + r)));
                        fft_dipole_inputs[idx + 2 * dipole_stride.comp] -= mult * pow(-1.0f, i + j + k) * log((((b_idx + j - 0.5f)* cell_sizes[1] + r)));

                        fft_dipole_inputs[idx + 3 * dipole_stride.comp] += mult * pow(-1.0f, i + j + k) * atan(((a_idx + i-0.5f) * (c_idx + k - 0.5f) * cell_sizes[2]*cell_sizes[0]/cell_sizes[1] / r / (b_idx + j - 0.5f)));
                        //fft_dipole_inputs[idx + 4 * dipole_stride.comp] += -mult * pow(-1.0f, i + j + k) * log(abs(((a_idx + i - 0.5f)* cell_sizes[0] + r)/((a_idx + i - 0.5f)* cell_sizes[0] - r)));
                        fft_dipole_inputs[idx + 4 * dipole_stride.comp] -= mult * pow(-1.0f, i + j + k) * log((((a_idx + i - 0.5f)* cell_sizes[0] + r)));
                        fft_dipole_inputs[idx + 5 * dipole_stride.comp] += mult * pow(-1.0f, i + j + k) * atan(((b_idx + j-0.5f) * (a_idx + i - 0.5f) * cell_sizes[0]*cell_sizes[1]/cell_sizes[2] / r / (c_idx + k - 0.5f)));

                    }
                }
            }
                //if (fft_dipole_inputs[this->idx]<-0.03)
        }
        FFT::batch_Four_3D(fft_plan_dipole);
    }
    void Hamiltonian_Micromagnetic::Prepare_DDI()
    {
        Clean_DDI();
        mult_spins=vectorfield(geometry->nos,Vector3{0,0,1});
        minMs=regions_book[0].Ms;
        for(int i=0;i<region_num;i++){
            if (regions_book[i].Ms<minMs) minMs=regions_book[i].Ms;
        }
        n_cells_padded.resize(3);
        n_cells_padded[0] = (geometry->n_cells[0] > 1) ? 2 * geometry->n_cells[0] : 1;
        n_cells_padded[1] = (geometry->n_cells[1] > 1) ? 2 * geometry->n_cells[1] : 1;
        n_cells_padded[2] = (geometry->n_cells[2] > 1) ? 2 * geometry->n_cells[2] : 1;
        sublattice_size = n_cells_padded[0] * n_cells_padded[1] * n_cells_padded[2];
        //printf("111 %d %d %d\n", n_cells_padded[0],n_cells_padded[1],n_cells_padded[2]);

        inter_sublattice_lookup.resize(geometry->n_cell_atoms * geometry->n_cell_atoms);

        //we dont need to transform over length 1 dims
        std::vector<int> fft_dims;
        for (int i = 2; i >= 0; i--) //notice that reverse order is important!
        {
            if (n_cells_padded[i] > 1)
                fft_dims.push_back(n_cells_padded[i]);
        }

        //Count how many distinct inter-lattice contributions we need to store
        n_inter_sublattice = 0;
        for (int i = 0; i < geometry->n_cell_atoms; i++)
        {
            for (int j = 0; j < geometry->n_cell_atoms; j++)
            {
                if (i != 0 && i == j) continue;
                n_inter_sublattice++;
            }
        }
        //printf("lex%d %d %d\n", n_inter_sublattice, fft_dims[0],fft_dims[1]);
        //Set the iteration bounds for the nested for loops that are flattened in the kernels
        it_bounds_write_spins = { geometry->n_cell_atoms,
                                      geometry->n_cells[0],
                                      geometry->n_cells[1],
                                      geometry->n_cells[2] };

        it_bounds_write_dipole = { n_cells_padded[0],
                                      n_cells_padded[1],
                                      n_cells_padded[2] };

        it_bounds_pointwise_mult = { (n_cells_padded[0] / 2 + 1), // due to redundancy in real fft
                                      n_cells_padded[1],
                                      n_cells_padded[2] };

        it_bounds_write_gradients = { geometry->n_cell_atoms,
                                      geometry->n_cells[0],
                                      geometry->n_cells[1],
                                      geometry->n_cells[2] };

        FFT::FFT_Plan fft_plan_dipole = FFT::FFT_Plan(fft_dims, false, 6 * n_inter_sublattice, sublattice_size);
        fft_plan_spins = FFT::FFT_Plan(fft_dims, false, 3 * geometry->n_cell_atoms, sublattice_size);
        fft_plan_reverse = FFT::FFT_Plan(fft_dims, true, 3 * geometry->n_cell_atoms, sublattice_size);

        field<int*> temp_s = { &spin_stride.comp, &spin_stride.basis, &spin_stride.a, &spin_stride.b, &spin_stride.c };
        field<int*> temp_d = { &dipole_stride.comp, &dipole_stride.basis, &dipole_stride.a, &dipole_stride.b, &dipole_stride.c };;
        FFT::get_strides(temp_s, { 3, this->geometry->n_cell_atoms, n_cells_padded[0], n_cells_padded[1], n_cells_padded[2] });
        FFT::get_strides(temp_d, { 6, n_inter_sublattice, n_cells_padded[0], n_cells_padded[1], n_cells_padded[2] });
        /*
        //perform FFT of dipole matrices
        int img_a = boundary_conditions[0] == 0 ? 0 : ddi_n_periodic_images[0];
        int img_b = boundary_conditions[1] == 0 ? 0 : ddi_n_periodic_images[1];
        int img_c = boundary_conditions[2] == 0 ? 0 : ddi_n_periodic_images[2];

        FFT_Dipole_Matrices(fft_plan_dipole, img_a, img_b, img_c); */
        FFT_Dipole_Matrices(fft_plan_dipole, 0, 0, 0);

        transformed_dipole_matrices = std::move(fft_plan_dipole.cpx_ptr);
    }//end prepare

    void Hamiltonian_Micromagnetic::Clean_DDI()
    {
        fft_plan_spins = FFT::FFT_Plan();
        fft_plan_reverse = FFT::FFT_Plan();
    }

    void Hamiltonian_Micromagnetic::Hessian(const vectorfield & spins, MatrixX & hessian)
    {
    }


    // Hamiltonian name as string
    static const std::string name = "Micromagnetic";
    const std::string& Hamiltonian_Micromagnetic::Name() { return name; }
}

#endif

