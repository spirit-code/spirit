#pragma once
#ifndef UTILITY_CONSTANTS_H
#define UTILITY_CONSTANTS_H

namespace Utility
{
    // Constants by convention:
    //      Spatial scale: Angstrom
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

        // Gyromagnetic ratio of electron [rad/(ps*T)]
        // Also gives the Larmor precession frequency for electron
        double const gamma = 0.1760859644;

        // Electron (Landé) g-factor = gamma * hbar / mu_B [unitless]
        double const g_e = 2.00231930436182;

        // Millirydberg [mRy/meV]
        double const mRy = 1.0/13.605693009;

        // erg [erg/meV]
        double const erg = 6.2415091*1e14;

        // Joule [Joule/meV]
        double const Joule = 6.2415091*1e21;
        // Pi [rad]
        double const Pi = 3.141592653589793238462643383279502884197169399375105820974;

        // Pi/2 [rad]
        double const Pi_2 = 1.570796326794896619231321691639751442098584699687552910487;
    }

    // Constants_mRy by convention:
    //      Spatial scale: Angstrom
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

        // Gyromagnetic ratio of electron [rad/(ps*T)]
        double const gamma = Constants::gamma;

        // Electron g-factor [unitless]
        double const g_e = Constants::g_e;

        // Millielectronvolt [meV/mRy]
        double const meV = 1.0 / Constants::mRy;

        // Joule [Joule/mRy]
        double const mRy = Constants::Joule / Constants::mRy;

        // erg [erg/mRy]
        double const erg = Constants::erg / Constants::mRy;

        // Pi [rad]
        double const Pi = Constants::Pi;

        // Pi/2 [rad]
        double const Pi_2 = Constants::Pi_2;
    }

    // Constants by micromagnetic convention (SI units):
    //     Spatial scale: meters
    //     Energy scale: Joule
    //     Time scale: seconds
    //     Magnetic fields scale: Tesla

    namespace Constants_Micromagnetic
    {
        // The Bohr Magneton [Joule/T]
        double const mu_B = Constants::mu_B * Constants::Joule;

        // The vacuum permeability [T^2 m^3 / Joule]
        double const mu_0 = Constants::mu_0 / Constants::Joule;

        // The Boltzmann constant [J/K]
        double const k_B  = Constants::k_B * Constants::Joule;

        // Planck constant [J*s/rad]
        double const hbar = Constants::hbar * Constants::Joule * 1e-12;

        // Gyromagnetic ratio of electron [rad/(s*T)]
        double const gamma = Constants::gamma * 1e+12;

        // Electron (Landé) g-factor = gamma * hbar / mu_B [unitless]
        double const g_e = Constants::g_e;

        // meV [meV/Joule]
        double const meV = 1.0 / Constants::Joule;

        // Millirydberg [mRy/Joule]
        double const mRy = Constants::mRy / Constants::Joule;

        // erg [erg/Joule]
        double const erg = Constants::erg / Constants::Joule;

        // Pi [rad]
        double const Pi = Constants::Pi;

        // Pi/2 [rad]
        double const Pi_2 = Constants::Pi_2;
    }
}

#endif