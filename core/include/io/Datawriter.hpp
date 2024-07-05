#pragma once
#ifndef SPIRIT_CORE_IO_DATAWRITER_HPP
#define SPIRIT_CORE_IO_DATAWRITER_HPP

#include <data/Geometry.hpp>
#include <data/State.hpp>
#include <io/Flags.hpp>

namespace IO
{

void Write_Neighbours_Exchange(
    const Engine::Spin::Interaction::Exchange::Cache & cache, const std::string & filename );

void Write_Neighbours_DMI( const Engine::Spin::Interaction::DMI::Cache & cache, const std::string & filename );

void Write_Energy_Header(
    const Data::System_Energy & E, const std::string & filename, const std::vector<std::string> && firstcolumns,
    Flags flags = Flag::Contributions | Flag::Normalize_by_nos | Flag::Readability );

// Appends the Energy of a spin system with energy contributions (without header)
void Append_Image_Energy(
    const Data::System_Energy & E, const Data::Geometry & geometry, int iteration, const std::string & filename,
    Flags flags = Flag::Readability | Flag::Normalize_by_nos );

// Save energy contributions of a spin system
void Write_Image_Energy(
    const Data::System_Energy & E, const Data::Geometry & geometry, const std::string & filename,
    Flags flags = Flag::Readability | Flag::Normalize_by_nos );

// Saves Energies of all images with header and contributions
template<typename ChainType>
void Write_Chain_Energies(
    const ChainType & chain, int iteration, const std::string & filename,
    Flags flags = Flag::Readability | Flag::Normalize_by_nos );
// Saves the Energies interpolated by the GNEB method
template<typename ChainType>
void Write_Chain_Energies_Interpolated(
    const ChainType & chain, const std::string & filename, Flags flags = Flag::Readability | Flag::Normalize_by_nos );

// Save interaction-resolved energy contributions of a spin system
void Write_Image_Energy_Contributions(
    const Data::System_Energy & E, const Data::Geometry & geometry, const std::string & filename,
    IO::VF_FileFormat format );

} // namespace IO

#endif
