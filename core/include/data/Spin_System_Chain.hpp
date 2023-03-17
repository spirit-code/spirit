#pragma once
#ifndef SPIRIT_CORE_DATA_SPIN_SYSTEM_CHAIN_HPP
#define SPIRIT_CORE_DATA_SPIN_SYSTEM_CHAIN_HPP

#include "Spirit_Defines.h"
#include <Spirit/Parameters_GNEB.h>
#include <data/Parameters_Method_GNEB.hpp>
#include <data/Spin_System.hpp>

namespace Data
{

enum class GNEB_Image_Type
{
    Normal     = GNEB_IMAGE_NORMAL,
    Climbing   = GNEB_IMAGE_CLIMBING,
    Falling    = GNEB_IMAGE_FALLING,
    Stationary = GNEB_IMAGE_STATIONARY
};

struct HTST_Info
{
    bool sparse = false;

    // Relevant images
    std::shared_ptr<Spin_System> minimum;
    std::shared_ptr<Spin_System> saddle_point;

    // Eigenmodes
    int n_eigenmodes_keep          = 0;
    VectorX eigenvalues_min        = VectorX( 0 );
    MatrixX eigenvectors_min       = MatrixX( 0, 0 );
    VectorX eigenvalues_sp         = VectorX( 0 );
    MatrixX eigenvectors_sp        = MatrixX( 0, 0 );
    VectorX perpendicular_velocity = VectorX( 0 );
    scalar det_min                 = 0;
    scalar det_sp                  = 0;

    // Prefactor constituents
    scalar temperature_exponent = 0;
    scalar me                   = 0;
    scalar Omega_0              = 0;
    scalar s                    = 0;
    scalar volume_min           = 0;
    scalar volume_sp            = 0;
    scalar prefactor_dynamical  = 0;
    scalar prefactor            = 0;
};

class Spin_System_Chain
{
public:
    // Constructor
    Spin_System_Chain(
        std::vector<std::shared_ptr<Spin_System>> images, std::shared_ptr<Data::Parameters_Method_GNEB> gneb_parameters,
        bool iteration_allowed = false );

    // For multithreading
    void Lock() noexcept;
    void Unlock() noexcept;

    int noi; // Number of Images
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
    // FIFO mutex for thread-safety
    Utility::OrderedLock ordered_lock;
};

} // namespace Data

#endif