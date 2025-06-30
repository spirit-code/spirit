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
auto Gaussian_to_TOML( const Engine::Spin::Interaction::Gaussian::Data * data ) -> toml::table;

auto Zeeman_from_TOML( const toml::table & tbl, std::vector<std::string> & parameter_log )
    -> Engine::Spin::Interaction::Zeeman::Data;
auto Zeeman_to_TOML( const Engine::Spin::Interaction::Zeeman::Data * data ) -> toml::table;

auto Anisotropy_from_TOML(
    const toml::table & tbl, const Data::Geometry & geometry, std::vector<std::string> & parameter_log )
    -> std::pair<Engine::Spin::Interaction::Anisotropy::Data, Engine::Spin::Interaction::Cubic_Anisotropy::Data>;
auto Anisotropy_to_TOML(
    const Engine::Spin::Interaction::Anisotropy::Data * uniaxial,
    const Engine::Spin::Interaction::Cubic_Anisotropy::Data * cubic ) -> toml::table;

auto Biaxial_Anisotropy_from_TOML(
    const toml::table & tbl, const Data::Geometry & geometry,
    std::vector<std::string> & parameter_log ) -> Engine::Spin::Interaction::Biaxial_Anisotropy::Data;
auto Biaxial_Anisotropy_to_TOML( const Engine::Spin::Interaction::Biaxial_Anisotropy::Data * data ) -> toml::table;

auto Pair_Interactions_from_TOML(
    const toml::table & tbl, const Data::Geometry & geometry, std::vector<std::string> & parameter_log )
    -> std::pair<Engine::Spin::Interaction::Exchange::Data, Engine::Spin::Interaction::DMI::Data>;
auto Pair_Interactions_to_TOML(
    const Engine::Spin::Interaction::Exchange::Cache * exchange,
    const Engine::Spin::Interaction::DMI::Cache * dmi ) -> toml::table;

auto Quadruplets_from_TOML(
    const toml::table & tbl, const Data::Geometry & geometry,
    std::vector<std::string> & parameter_log ) -> Engine::Spin::Interaction::Quadruplet::Data;
auto Quadruplets_to_TOML( const Engine::Spin::Interaction::Quadruplet::Data * data ) -> toml::table;

auto DDI_from_TOML( const toml::table & tbl, const Data::Geometry & geometry, std::vector<std::string> & parameter_log )
    -> Engine::Spin::Interaction::DDI::Data;
auto DDI_to_TOML( const Engine::Spin::Interaction::DDI::Data * data ) -> toml::table;

} // namespace IO
