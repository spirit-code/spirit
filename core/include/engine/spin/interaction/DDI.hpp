#pragma once
#ifndef SPIRIT_CORE_ENGINE_INTERACTION_DDI_HPP
#define SPIRIT_CORE_ENGINE_INTERACTION_DDI_HPP

#include <engine/FFT.hpp>
#include <engine/spin/interaction/Functor_Prototypes.hpp>
#include <utility/Constants.hpp>

namespace Engine
{

namespace Spin
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

struct DDI
{
    using state_t = vectorfield;

    struct Data
    {
        DDI_Method method    = DDI_Method::None;
        scalar cutoff_radius = 0.0;
        bool pb_zero_padding = false;
        intfield n_periodic_images{};

        Data() = default;
        Data( DDI_Method method, scalar cutoff_radius, bool pb_zero_padding, intfield n_periodic_images )
                : method( method ),
                  cutoff_radius( cutoff_radius ),
                  pb_zero_padding( pb_zero_padding ),
                  n_periodic_images( std::move( n_periodic_images ) ) {};
    };

    struct Cache
    {
        pairfield pairs{};
        scalarfield magnitudes{};
        vectorfield normals{};

        const ::Data::Geometry * geometry    = nullptr;
        const intfield * boundary_conditions = nullptr;

        // Plans for FT / rFT
        FFT::FFT_Plan fft_plan_spins   = FFT::FFT_Plan();
        FFT::FFT_Plan fft_plan_reverse = FFT::FFT_Plan();

        field<FFT::FFT_cpx_type> transformed_dipole_matrices{};

        bool save_dipole_matrices = false;
        field<FFT::FFT_real_type> dipole_matrices{};

        // Number of inter-sublattice contributions
        int n_inter_sublattice{};
        // At which index to look up the inter-sublattice D-matrices
        field<int> inter_sublattice_lookup{};

        // Lengths of padded system
        field<int> n_cells_padded{};
        // Total number of padded spins per sublattice
        int sublattice_size{};

        FFT::StrideContainer spin_stride{};
        FFT::StrideContainer dipole_stride{};

        // Bounds for nested for loops. Only important for the CUDA version
        field<int> it_bounds_pointwise_mult{};
        field<int> it_bounds_write_gradients{};
        field<int> it_bounds_write_spins{};
        field<int> it_bounds_write_dipole{};
    };

    static bool is_contributing( const Data & data, const Cache & )
    {
        return data.method != DDI_Method::None;
    };

    static void applyGeometry(
        const ::Data::Geometry & geometry, const intfield & boundary_conditions, const Data & data, Cache & cache );

    using Energy             = Functor::NonLocal::Energy_Functor<Functor::NonLocal::DataRef<DDI>>;
    using Gradient           = Functor::NonLocal::Gradient_Functor<Functor::NonLocal::DataRef<DDI>>;
    using Hessian            = Functor::NonLocal::Hessian_Functor<Functor::NonLocal::DataRef<DDI>>;
    using Energy_Single_Spin = Functor::NonLocal::Energy_Single_Spin_Functor<Functor::NonLocal::DataRef<DDI>>;
    using Energy_Total       = Functor::NonLocal::Reduce_Functor<Energy>;

    static std::size_t Sparse_Hessian_Size_per_Cell( const Data & data, const Cache & )
    {
        if( data.method == DDI_Method::None )
            return 0;
        else
            return 9;
    };

    // Interaction name as string
    static constexpr std::string_view name = "DDI";

    static constexpr bool local = false;
};

template<>
template<typename Callable>
void DDI::Hessian::operator()( const vectorfield & spins, Callable & hessian ) const
{
    namespace C = Utility::Constants;
    if( !is_contributing )
        return;

    if( cache.geometry == nullptr || cache.boundary_conditions == nullptr )
        // TODO: turn this into an error
        return;

    const auto & geometry = *cache.geometry;
    const auto nos        = spins.size();

    // Tentative Dipole-Dipole (only works for open boundary conditions)
    if( data.method != DDI_Method::None )
    {
        static constexpr scalar mult = C::mu_0 * C::mu_B * C::mu_B / ( 4 * C::Pi * 1e-30 );
        for( unsigned int idx1 = 0; idx1 < nos; idx1++ )
        {
            for( unsigned int idx2 = 0; idx2 < nos; idx2++ )
            {
                auto diff = geometry.positions[idx2] - geometry.positions[idx1];
                scalar d = diff.norm(), d3 = 0, d5 = 0;
                scalar Dxx = 0, Dxy = 0, Dxz = 0, Dyy = 0, Dyz = 0, Dzz = 0;
                if( d > 1e-10 )
                {
                    d3 = d * d * d;
                    d5 = d * d * d * d * d;
                    Dxx += mult * ( 3 * diff[0] * diff[0] / d5 - 1 / d3 );
                    Dxy += mult * 3 * diff[0] * diff[1] / d5; // same as Dyx
                    Dxz += mult * 3 * diff[0] * diff[2] / d5; // same as Dzx
                    Dyy += mult * ( 3 * diff[1] * diff[1] / d5 - 1 / d3 );
                    Dyz += mult * 3 * diff[1] * diff[2] / d5; // same as Dzy
                    Dzz += mult * ( 3 * diff[2] * diff[2] / d5 - 1 / d3 );
                }

                const int i = 3 * idx1;
                const int j = 3 * idx2;

                hessian( i + 0, j + 0, -geometry.mu_s[idx1] * geometry.mu_s[idx2] * ( Dxx ) );
                hessian( i + 1, j + 0, -geometry.mu_s[idx1] * geometry.mu_s[idx2] * ( Dxy ) );
                hessian( i + 2, j + 0, -geometry.mu_s[idx1] * geometry.mu_s[idx2] * ( Dxz ) );
                hessian( i + 0, j + 1, -geometry.mu_s[idx1] * geometry.mu_s[idx2] * ( Dxy ) );
                hessian( i + 1, j + 1, -geometry.mu_s[idx1] * geometry.mu_s[idx2] * ( Dyy ) );
                hessian( i + 2, j + 1, -geometry.mu_s[idx1] * geometry.mu_s[idx2] * ( Dyz ) );
                hessian( i + 0, j + 2, -geometry.mu_s[idx1] * geometry.mu_s[idx2] * ( Dxz ) );
                hessian( i + 1, j + 2, -geometry.mu_s[idx1] * geometry.mu_s[idx2] * ( Dyz ) );
                hessian( i + 2, j + 2, -geometry.mu_s[idx1] * geometry.mu_s[idx2] * ( Dzz ) );
            }
        }
    }

    // // TODO: Dipole-Dipole
    // for (unsigned int i_pair = 0; i_pair < this->DD_indices.size(); ++i_pair)
    // {
    //     // indices
    //     int idx_1 = DD_indices[i_pair][0];
    //     int idx_2 = DD_indices[i_pair][1];
    //     // prefactor
    //     scalar prefactor = 0.0536814951168
    //         * mu_s[idx_1] * mu_s[idx_2]
    //         / std::pow(DD_magnitude[i_pair], 3);
    //     // components
    //     for (int alpha = 0; alpha < 3; ++alpha)
    //     {
    //         for (int beta = 0; beta < 3; ++beta)
    //         {
    //             int idx_h = idx_1 + alpha*nos + 3 * nos*(idx_2 + beta*nos);
    //             if (alpha == beta)
    //                 hessian[idx_h] += prefactor;
    //             hessian[idx_h] += -3.0*prefactor*DD_normal[i_pair][alpha] * DD_normal[i_pair][beta];
    //         }
    //     }
    // }
};

} // namespace Interaction

} // namespace Spin

} // namespace Engine
#endif
