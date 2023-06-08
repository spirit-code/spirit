#pragma once
#ifndef SPIRIT_CORE_IO_DATAWRITER_HPP
#define SPIRIT_CORE_IO_DATAWRITER_HPP

#include <data/Geometry.hpp>
#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>

namespace Spirit::IO
{

void Write_Neighbours_Exchange( const Data::Spin_System & system, const std::string & filename );

void Write_Neighbours_DMI( const Data::Spin_System & system, const std::string & filename );

void Write_Energy_Header(
    const Data::Spin_System & system, const std::string & filename, const std::vector<std::string> && firstcolumns,
    bool contributions = true, bool normalize_by_nos = true, bool readability_toggle = true );

// Appends the Energy of a spin system with energy contributions (without header)
void Append_Image_Energy(
    const Data::Spin_System & system, int iteration, const std::string & filename, bool normalize_by_nos = true,
    bool readability_toggle = true );

// Save energy contributions of a spin system
void Write_Image_Energy(
    const Data::Spin_System & system, const std::string & filename, bool normalize_by_nos = true,
    bool readability_toggle = true );

// Saves Energies of all images with header and contributions
void Write_Chain_Energies(
    const Data::Spin_System_Chain & chain, int iteration, const std::string & filename, bool normalize_by_nos = true,
    bool readability_toggle = true );

// Saves the Energies interpolated by the GNEB method
void Write_Chain_Energies_Interpolated(
    const Data::Spin_System_Chain & chain, const std::string & filename, bool normalize_by_nos = true,
    bool readability_toggle = true );

} // namespace Spirit::IO

#endif
