#pragma once
#ifndef SPIRIT_CORE_IO_CONFIGPARSER_HPP
#define SPIRIT_CORE_IO_CONFIGPARSER_HPP

#include <data/Geometry.hpp>
#include <data/Parameters_Method_GNEB.hpp>
#include <data/Parameters_Method_LLG.hpp>
#include <data/Parameters_Method_MMF.hpp>
#include <data/State.hpp>
#include <engine/spin/Hamiltonian.hpp>
#include <io/Hamiltonian.hpp>

#include <toml++/toml.hpp>

namespace IO
{

/*
 * Note that due to the modular structure of the input parsers, input may be given in one or in separate files.
 * Input may be given incomplete. In this case a log entry is created and default values are used.
 */

void Log_from_TOML( const toml::table &, bool force_quiet = false );
auto Spin_System_from_TOML( const toml::table & ) -> std::unique_ptr<::State::system_t>;
auto Geometry_from_TOML( const toml::table & ) -> Data::Geometry;

auto Parameters_Method_LLG_from_TOML( const toml::table & ) -> std::unique_ptr<Data::Parameters_Method_LLG>;
auto Parameters_Method_MC_from_TOML( const toml::table & ) -> std::unique_ptr<Data::Parameters_Method_MC>;
auto Parameters_Method_GNEB_from_TOML( const toml::table & ) -> std::unique_ptr<Data::Parameters_Method_GNEB>;
auto Parameters_Method_EMA_from_TOML( const toml::table & ) -> std::unique_ptr<Data::Parameters_Method_EMA>;
auto Parameters_Method_MMF_from_TOML( const toml::table & ) -> std::unique_ptr<Data::Parameters_Method_MMF>;

template<typename T>
auto toml_transform( const toml::node & node ) noexcept -> std::optional<T>;

template<typename T>
auto toml_transform( toml::node_view<const toml::node> node ) noexcept -> std::optional<T>;

template<typename Container>
auto toml_array_from_container( const Container & iterable ) -> toml::array;

} // namespace IO

#include <io/Configparser.inl>

#endif
