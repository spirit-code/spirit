#pragma once
#ifndef SPIRIT_CORE_IO_CONFIGWRITER_HPP
#define SPIRIT_CORE_IO_CONFIGWRITER_HPP

#include <data/Geometry.hpp>
#include <data/Parameters_Method_GNEB.hpp>
#include <data/Parameters_Method_LLG.hpp>
#include <data/Parameters_Method_MMF.hpp>
#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <engine/Hamiltonian_Gaussian.hpp>
#include <engine/Hamiltonian_Heisenberg.hpp>

namespace IO
{

void Folders_to_Config(
    const std::string & config_file, const std::shared_ptr<Data::Parameters_Method_LLG> parameters_llg,
    const std::shared_ptr<Data::Parameters_Method_MC> parameters_mc,
    const std::shared_ptr<Data::Parameters_Method_GNEB> parameters_gneb,
    const std::shared_ptr<Data::Parameters_Method_MMF> parameters_mmf );

void Log_Levels_to_Config( const std::string & config_file );

void Geometry_to_Config( const std::string & config_file, const std::shared_ptr<Data::Geometry> geometry );

void Parameters_Method_LLG_to_Config(
    const std::string & config_file, const std::shared_ptr<Data::Parameters_Method_LLG> parameters );

void Parameters_Method_MC_to_Config(
    const std::string & config_file, const std::shared_ptr<Data::Parameters_Method_MC> parameters );

void Parameters_Method_GNEB_to_Config(
    const std::string & config_file, const std::shared_ptr<Data::Parameters_Method_GNEB> parameters );

void Parameters_Method_MMF_to_Config(
    const std::string & config_file, const std::shared_ptr<Data::Parameters_Method_MMF> parameters );

void Hamiltonian_to_Config(
    const std::string & config_file, const std::shared_ptr<Engine::Hamiltonian> hamiltonian,
    const std::shared_ptr<Data::Geometry> geometry );

void Hamiltonian_Heisenberg_to_Config(
    const std::string & config_file, std::shared_ptr<Engine::Hamiltonian> hamiltonian,
    const std::shared_ptr<Data::Geometry> geometry );

void Hamiltonian_Gaussian_to_Config(
    const std::string & config_file, const std::shared_ptr<Engine::Hamiltonian> hamiltonian );

} // namespace IO

#endif