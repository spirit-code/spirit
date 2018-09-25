#pragma once
#ifndef DATA_SPIN_SYSTEM_CHAIN_H
#define DATA_SPIN_SYSTEM_CHAIN_H

#include "Spirit_Defines.h"
#include <data/Spin_System.hpp>
#include <data/Parameters_Method_GNEB.hpp>

namespace Data
{
    enum class GNEB_Image_Type
    {
        Normal,
        Climbing,
        Falling,
        Stationary
    };

    struct HTST_Info
    {
        // Relevant images
        std::shared_ptr<Spin_System> minimum;
        std::shared_ptr<Spin_System> saddle_point;

        // Eigenmodes
        scalarfield eigenvalues_min;
        // vectorfield eigenvectors_min;
        scalarfield eigenvalues_sp;
        // vectorfield eigenvectors_sp;
        scalarfield perpendicular_velocity;

        // Prefactor constituents
        scalar temperature_exponent;
        scalar me;
        scalar Omega_0;
        scalar s;
        scalar volume_min;
        scalar volume_sp;
        scalar prefactor_dynamical;
        scalar prefactor;
    };

    class Spin_System_Chain
    {
    public:
        // Constructor
        Spin_System_Chain(std::vector<std::shared_ptr<Spin_System>> images, std::shared_ptr<Data::Parameters_Method_GNEB> gneb_parameters, bool iteration_allowed = false);

        // For multithreading
        void Lock() const;
        void Unlock() const;

        int noi;	// Number of Images
        std::vector<std::shared_ptr<Spin_System>> images;
        int idx_active_image;

        // Parameters for GNEB Iterations
        std::shared_ptr<Data::Parameters_Method_GNEB> gneb_parameters;

        // Are we allowed to iterate on this chain or do a singleshot?
        bool iteration_allowed;
        bool singleshot_allowed;

        // Climbing and falling images
        std::vector<GNEB_Image_Type> image_type;

        // Reaction coordinates of images in the chain
        std::vector<scalar> Rx;

        // Reaction coordinates of interpolated points
        std::vector<scalar> Rx_interpolated;

        // Total Energy of the spin systems and interpolated values
        std::vector<scalar> E_interpolated;
        std::vector<std::vector<scalar>> E_array_interpolated;

        // If a prefactor calculation is performed on the chain, we keep the results
        HTST_Info htst_info;

    private:
        // Mutex for thread-safety
        mutable std::mutex mutex;
    };
}
#endif