
#pragma once
#ifndef IO_DATAPARSER_H
#define IO_DATAPARSER_H

#include <data/Geometry.hpp>
#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <data/Parameters_Method_LLG.hpp>
#include <data/Parameters_Method_GNEB.hpp>
#include <data/Parameters_Method_MMF.hpp>
#include <engine/Hamiltonian_Heisenberg_Neighbours.hpp>
#include <engine/Hamiltonian_Heisenberg_Pairs.hpp>
#include <engine/Hamiltonian_Gaussian.hpp>
#include <io/IO.hpp>
#include <io/Fileformat.hpp>
#include <io/Filter_File_Handle.hpp>

namespace IO
{
    void Read_ColumnVector_Configuration( Filter_File_Handle& myfile, const char delimiter,
                                          const int stride, vectorfield& vf,
                                          const Data::Geometry& geometry );
    void Read_Spin_Configuration_CSV( std::shared_ptr<Data::Spin_System> s, const std::string file );
    void Read_Spin_Configuration( std::shared_ptr<Data::Spin_System> s, const std::string file, 
                                  VF_FileFormat format = VF_FileFormat::SPIRIT_CSV_POS_SPIN );
    void Read_SpinChain_Configuration( std::shared_ptr<Data::Spin_System_Chain> c, 
                                       const std::string file );
    void Anisotropy_from_File( const std::string anisotropyFile, 
                               const std::shared_ptr<Data::Geometry> geometry, int& n_indices,
                               intfield& anisotropy_index, scalarfield& anisotropy_magnitude, 
                               vectorfield& anisotropy_normal );
    void Pairs_from_File( const std::string pairsFile, 
                          const std::shared_ptr<Data::Geometry> geometry, int& nop,
                          pairfield& exchange_pairs, scalarfield& exchange_magnitudes,
                          pairfield& dmi_pairs, scalarfield& dmi_magnitudes, 
                          vectorfield& dmi_normals );
    void Triplets_from_File( const std::string tripletsFile, 
                                const std::shared_ptr<Data::Geometry> geometry, int& noq,
                                tripletfield& triplets, scalarfield& triplet_magnitudes, scalarfield& triplet_magnitudes2 );
    void Quadruplets_from_File( const std::string quadrupletsFile, 
                                const std::shared_ptr<Data::Geometry> geometry, int& noq,
                                quadrupletfield& quadruplets, scalarfield& quadruplet_magnitudes );
    void Defects_from_File( const std::string defectsFile, int& n_defects,
                            intfield& defect_indices, intfield & defect_types );
    void Pinned_from_File( const std::string pinnedFile, int& n_pinned,
                           intfield& pinned_indices, vectorfield& pinned_spins );
    // Read data from OVF file format
    void Read_From_OVF( vectorfield& vf, const Data::Geometry& geometry, std::string inputfilename, 
                        VF_FileFormat format );
    // Reads the OVF binary data to the vectorfield vf
    void OVF_Read_Binary( Filter_File_Handle& myfile, const int ovf_binary_length, 
                          const std::array<int, 3>& ovf_xyz_nodes, vectorfield& vf );
    // Checking the initial check values of OVF binary data
    bool OVF_Check_Binary_Initial_Values( Filter_File_Handle& myfile, const int ovf_binary_length );
    // Read the OVF text data to the vectorfield vf 
    void OVF_Read_Text( Filter_File_Handle& myfile, const Data::Geometry& geometry, vectorfield& vf );
};// end namespace IO
#endif