#pragma once

#include <data/Geometry.hpp>
#include <engine/spin/Hamiltonian.hpp>
#include <io/Fileformat.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/IO.hpp>

namespace IO
{

void Gaussian_from_Config(
    const std::string & config_file_name, std::vector<std::string> & parameter_log, scalarfield & amplitude,
    scalarfield & width, vectorfield & center );

void Zeeman_from_Config(
    const std::string & config_file_name, std::vector<std::string> & parameter_log, scalar & magnitude,
    Vector3 & normal );

void Anisotropy_from_Config(
    const std::string & config_file_name, const Data::Geometry & geometry, std::vector<std::string> & parameter_log,
    intfield & uniaxial_indices, scalarfield & uniaxial_magnitudes, vectorfield & uniaxial_normals,
    intfield & cubic_indices, scalarfield & cubic_magnitudes );

void Biaxial_Anisotropy_from_Config(
    const std::string & config_file_name, const Data::Geometry & geometry, std::vector<std::string> & parameter_log,
    intfield & indices, field<PolynomialBasis> & polynomial_bases, field<unsigned int> & polynomial_site_p,
    field<PolynomialTerm> & polynomial_terms );

void Pair_Interactions_from_Pairs_from_Config(
    const std::string & config_file_name, const Data::Geometry & geometry, std::vector<std::string> & parameter_log,
    pairfield & exchange_pairs, scalarfield & exchange_magnitudes, pairfield & dmi_pairs, scalarfield & dmi_magnitudes,
    vectorfield & dmi_normals );

void Pair_Interactions_from_Shells_from_Config(
    const std::string & config_file_name, const Data::Geometry & geometry, std::vector<std::string> & parameter_log,
    scalarfield & exchange_magnitudes, scalarfield & dmi_magnitudes, int & dm_chirality );

void Quadruplets_from_Config(
    const std::string & config_file_name, const Data::Geometry & geometry, std::vector<std::string> & parameter_log,
    quadrupletfield & quadruplets, scalarfield & quadruplet_magnitudes );

void DDI_from_Config(
    const std::string & config_file_name, const Data::Geometry & geometry, std::vector<std::string> & parameter_log,
    Engine::Spin::DDI_Method & ddi_method, intfield & ddi_n_periodic_images, bool & ddi_pb_zero_padding,
    scalar & ddi_radius );

} // namespace IO
