#pragma once
#ifndef SPIRIT_CORE_IO_DATAPARSER_HPP
#define SPIRIT_CORE_IO_DATAPARSER_HPP

#include <data/Geometry.hpp>
#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <data/State.hpp>
#include <io/Fileformat.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/IO.hpp>

#include <string_view>

namespace IO
{

namespace Spin
{

using StateType = vectorfield;

namespace State
{

static constexpr const int valuedim           = 3;
static constexpr std::string_view valuelabels = "spin_x spin_y spin_z";
static constexpr std::string_view valueunits  = "none none none";

template<typename T>
class Buffer;

} // namespace State

void Read_NonOVF_System_Configuration(
    StateType & spins, Data::Geometry & geometry, const int nos, const int idx_image_infile, const std::string & file );

} // namespace Spin

namespace State = IO::Spin::State;
using IO::Spin::Read_NonOVF_System_Configuration;

void Check_NonOVF_Chain_Configuration(
    std::shared_ptr<::State::chain_t> chain, const std::string & file, int start_image_infile, int end_image_infile,
    const int insert_idx, int & noi_to_add, int & noi_to_read, const int idx_chain );

void Basis_from_File(
    const std::string & basis_file, Data::Basis_Cell_Composition & cell_composition, std::vector<Vector3> & cell_atoms,
    std::size_t & n_cell_atoms ) noexcept;

void Defects_from_File(
    const std::string & defects_file, int & n_defects, field<Site> & defect_sites, intfield & defect_types ) noexcept;

void Pinned_from_File(
    const std::string & pinned_file, int & n_pinned, field<Site> & pinned_sites, vectorfield & pinned_spins ) noexcept;

} // namespace IO

#include <io/Dataparser.inl>

#endif
