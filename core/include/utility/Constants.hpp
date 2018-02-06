#pragma once
#ifndef UTILITY_CONSTANTS_H
#define UTILITY_CONSTANTS_H

namespace Utility
{
    // Constants by convention:
    //      Energy scale: Millielctronvolts
    //      Time scale: Picoseconds
    //      Magnetic fields scale: Tesla
    namespace Constants
    {
        // The Bohr Magneton [meV/T]
        double const mu_B = 0.057883817555;

        // The vacuum permeability [T^2 m^3 / meV]
        double const mu_0 = 2.0133545*1e-28;

        // The Boltzmann constant [meV/K]
        double const k_B  = 0.08617330350;

        // Planck constant [meV*ps/rad]
        double const hbar = 0.6582119514;

        // Millirydberg [mRy/meV]
        double const mRy = 1.0/13.605693009;

        // Gyromagnetic ratio of electron [rad/(ps*T)] 
        // Also gives the Larmor precession frequency for electron
        double const gamma = 0.1760859644;

        // Electron g-factor [unitless]
        double const g_e = 2.00231930436182;

        // Pi [rad]
        double const Pi = 3.14159265358979323846;
    }

    // Constants_mRy by convention:
    //      Energy scale: Millirydberg
    //      Time scale: Picoseconds
    //      Magnetic fields scale: Tesla
    namespace Constants_mRy
    {
        // The Bohr Magneton [mRy/T]
        double const mu_B = Constants::mu_B / Constants::mRy;

        // The Boltzmann constant [mRy/K]
        double const k_B  = Constants::k_B  / Constants::mRy;

        // Planck constant [mRy*ps/rad]
        double const hbar = Constants::hbar / Constants::mRy;

        // Millielectronvolt [meV/mRy]
        double const meV = 1.0 / Constants::mRy;

        // Gyromagnetic ratio of electron [rad/(ps*T)]
        double const gamma = 0.1760859644;

        // Electron g-factor [unitless]
        double const g_e = 2.00231930436182;

        // Pi [rad]
        double const Pi = 3.14159265358979323846;
    }
}

#endif