#pragma once
#ifndef SPIRIT_CORE_ENGINE_METHOD_TS_SAMPLING_HPP
#define SPIRIT_CORE_ENGINE_METHOD_TS_SAMPLING_HPP

#include <engine/Method_MC.hpp>
#include <deque>


namespace Engine
{
/*
    Transition plane sampling
*/
class Method_TS_Sampling : public Method_MC
{
    public:
    Method_TS_Sampling( std::shared_ptr<Data::Spin_System> system, int idx_img, int idx_chain );

    void Set_Transition_Plane_Normal( vectorfield & spins_minimum, vectorfield & unstable_mode );
    void Compute_Transition_Plane_Normal( );

    std::string Name() override;

    private:
    void Iteration() override;
    // void Displace_Spin(int ispin, vectorfield & spins_new, std::uniform_real_distribution<scalar> & distribution, std::vector<int> & changed_indices, vectorfield & old_spins) override;

    vectorfield transition_plane_normal;
    std::uniform_int_distribution<> distribution_idx;
    std::deque<int> rejected;
};

}
#endif