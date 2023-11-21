#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_DDI_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_DDI_HPP

#include <engine/FFT.hpp>
#include <engine/interaction/ABC.hpp>

namespace Engine
{

enum class DDI_Method
{
    FFT    = SPIRIT_DDI_METHOD_FFT,
    FMM    = SPIRIT_DDI_METHOD_FMM,
    Cutoff = SPIRIT_DDI_METHOD_CUTOFF,
    None   = SPIRIT_DDI_METHOD_NONE
};

namespace Interaction
{

class DDI : public Interaction::Base<DDI>
{
public:
    DDI( Hamiltonian * hamiltonian, Engine::DDI_Method ddi_method, intfield n_periodic_images, bool pb_zero_padding,
         scalar cutoff_radius ) noexcept;
    DDI( Hamiltonian * hamiltonian, Engine::DDI_Method ddi_method, const Data::DDI_Data & ddi_data ) noexcept;

    void setParameters(
        DDI_Method ddi_method, const intfield & n_periodic_images, bool pb_zero_padding, scalar cutoff_radius )
    {
        this->method                = ddi_method;
        this->ddi_n_periodic_images = n_periodic_images;
        this->ddi_cutoff_radius     = cutoff_radius;
        this->ddi_pb_zero_padding   = pb_zero_padding;
        hamiltonian->onInteractionChanged();
    }
    void getParameters(
        DDI_Method & ddi_method, intfield & n_periodic_images, bool & pb_zero_padding, scalar & cutoff_radius ) const
    {
        ddi_method        = this->method;
        n_periodic_images = this->ddi_n_periodic_images;
        cutoff_radius     = this->ddi_cutoff_radius;
        pb_zero_padding   = this->ddi_pb_zero_padding;
    }

    bool is_contributing() const override;

    void Energy_per_Spin( const vectorfield & spins, scalarfield & energy ) override;
    void Hessian( const vectorfield & spins, MatrixX & hessian ) override;
    void Sparse_Hessian( const vectorfield & spins, std::vector<triplet> & hessian ) override;

    void Gradient( const vectorfield & spins, vectorfield & gradient ) override;

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    scalar Energy_Single_Spin( int ispin, const vectorfield & spins ) override;

    // Interaction name as string
    static constexpr std::string_view name          = "DDI";
    static constexpr std::optional<int> spin_order_ = 2;

protected:
    void updateFromGeometry( const Data::Geometry * geometry ) override;

private:
    DDI_Method method;
    intfield ddi_n_periodic_images;
    bool ddi_pb_zero_padding;
    //      ddi cutoff variables
    scalar ddi_cutoff_radius;
    pairfield ddi_pairs;
    scalarfield ddi_magnitudes;
    vectorfield ddi_normals;

    void Energy_per_Spin_Direct( const vectorfield & spins, scalarfield & energy );
    void Energy_per_Spin_Cutoff( const vectorfield & spins, scalarfield & energy );
    void Energy_per_Spin_FFT( const vectorfield & spins, scalarfield & energy );

    void Gradient_Direct( const vectorfield & spins, vectorfield & gradient );
    void Gradient_Cutoff( const vectorfield & spins, vectorfield & gradient );
    void Gradient_FFT( const vectorfield & spins, vectorfield & gradient );

    // Preparations for DDI-Convolution Algorithm
    void Prepare_DDI();
    void Clean_DDI();

    // Plans for FT / rFT
    FFT::FFT_Plan fft_plan_spins;
    FFT::FFT_Plan fft_plan_reverse;

    field<FFT::FFT_cpx_type> transformed_dipole_matrices;

    bool save_dipole_matrices = false;
    field<FFT::FFT_real_type> dipole_matrices;

    // Number of inter-sublattice contributions
    int n_inter_sublattice;
    // At which index to look up the inter-sublattice D-matrices
    field<int> inter_sublattice_lookup;

    // Lengths of padded system
    field<int> n_cells_padded;
    // Total number of padded spins per sublattice
    int sublattice_size;

    FFT::StrideContainer spin_stride;
    FFT::StrideContainer dipole_stride;

    // Calculate the FT of the padded D-matrics
    void FFT_Dipole_Matrices( FFT::FFT_Plan & fft_plan_dipole, int img_a, int img_b, int img_c );
    // Calculate the FT of the padded spins
    void FFT_Spins( const vectorfield & spins, FFT::FFT_Plan & fft_plan ) const;

    // Bounds for nested for loops. Only important for the CUDA version
    field<int> it_bounds_pointwise_mult;
    field<int> it_bounds_write_gradients;
    field<int> it_bounds_write_spins;
    field<int> it_bounds_write_dipole;
};

} // namespace Interaction

} // namespace Engine
#endif
