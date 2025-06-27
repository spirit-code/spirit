#pragma once

#include <data/Geometry.hpp>
#include <engine/spin/Hamiltonian.hpp>
#include <io/Fileformat.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/IO.hpp>

namespace IO
{

auto Gaussian_from_TOML( const toml::table & tbl, std::vector<std::string> & parameter_log )
    -> Engine::Spin::Interaction::Gaussian::Data;

auto Zeeman_from_TOML( const toml::table & tbl, std::vector<std::string> & parameter_log )
    -> Engine::Spin::Interaction::Zeeman::Data;

auto Anisotropy_from_TOML(
    const toml::table & tbl, const Data::Geometry & geometry, std::vector<std::string> & parameter_log )
    -> std::pair<Engine::Spin::Interaction::Anisotropy::Data, Engine::Spin::Interaction::Cubic_Anisotropy::Data>;

auto Biaxial_Anisotropy_from_TOML(
    const toml::table & tbl, const Data::Geometry & geometry,
    std::vector<std::string> & parameter_log ) -> Engine::Spin::Interaction::Biaxial_Anisotropy::Data;

auto Pair_Interactions_from_TOML(
    const toml::table & tbl, const Data::Geometry & geometry, std::vector<std::string> & parameter_log )
    -> std::pair<Engine::Spin::Interaction::Exchange::Data, Engine::Spin::Interaction::DMI::Data>;

auto Quadruplets_from_TOML(
    const toml::table & tbl, const Data::Geometry & geometry,
    std::vector<std::string> & parameter_log ) -> Engine::Spin::Interaction::Quadruplet::Data;

auto DDI_from_TOML( const toml::table & tbl, const Data::Geometry & geometry, std::vector<std::string> & parameter_log )
    -> Engine::Spin::Interaction::DDI::Data;

} // namespace IO
