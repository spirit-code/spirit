#pragma once
#ifndef SPIRIT_CORE_IO_DATAWRITER_HPP
#define SPIRIT_CORE_IO_DATAWRITER_HPP

#include <data/Geometry.hpp>
#include <data/State.hpp>
#include <io/Flags.hpp>

namespace IO
{

void Write_Neighbours_Exchange( const State::system_t & system, const std::string & filename );

void Write_Neighbours_DMI( const State::system_t & system, const std::string & filename );

void Write_Energy_Header(
    const State::system_t & system, const std::string & filename, const std::vector<std::string> && firstcolumns,
    Flags flags = Flag::Contributions | Flag::Normalize_by_nos | Flag::Readability );

// Appends the Energy of a spin system with energy contributions (without header)
void Append_Image_Energy(
    const State::system_t & system, int iteration, const std::string & filename,
    Flags flags = Flag::Readability | Flag::Normalize_by_nos );

// Save energy contributions of a spin system
void Write_Image_Energy(
    const State::system_t & system, const std::string & filename,
    Flags flags = Flag::Readability | Flag::Normalize_by_nos );

// Saves Energies of all images with header and contributions
void Write_Chain_Energies(
    const State::chain_t & chain, int iteration, const std::string & filename,
    Flags flags = Flag::Readability | Flag::Normalize_by_nos );
// Saves the Energies interpolated by the GNEB method
void Write_Chain_Energies_Interpolated(
    const State::chain_t & chain, const std::string & filename,
    Flags flags = Flag::Readability | Flag::Normalize_by_nos );

} // namespace IO

#endif
