#pragma once
#ifndef IO_DATAWRITER_H
#define IO_DATAWRITER_H

#include <data/Geometry.hpp>
#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>

namespace IO
{
    // ================================== General ==================================
    // NOTE: This must be the first function called for every SPIRIT data file since it is the one
    // that implements the Dump/Append mechanism. All the other functions are just appending data.
    void Write_SPIRIT_Version( const std::string filename, bool append );
    // =========================== Saving Configurations ===========================
    // Write/Append spin positions to file
    void Write_Spin_Positions( const Data::Geometry& geometry, 
                               const std::string filename, VF_FileFormat format,
                               const std::string comment, bool append = false );
    // Write/Append Spin_System's spin configurations to file
    void Write_Spin_Configuration( const vectorfield& vf, const Data::Geometry& geometry, 
                                   const std::string filename, VF_FileFormat format, 
                                   const std::string comment, bool append = false );
    // Write/Append Spin_System_Chain's spin configurations to file
    void Write_Chain_Spin_Configuration( const std::shared_ptr<Data::Spin_System_Chain>& c, 
                                         const std::string filename, VF_FileFormat format,
                                         const std::string comment, bool append = false );
    
    // Saves any SPIRIT format
    void Save_To_SPIRIT( const vectorfield & vf, const Data::Geometry & geometry, 
                         const std::string filename, VF_FileFormat format, 
                         const std::string comment );
    // Saves any OVF format 
    void Save_To_OVF( const vectorfield& vf, const Data::Geometry& geometry, std::string filename, 
                      VF_FileFormat format, const std::string comment );
    // Writes the OVF bin data
    void Write_OVF_bin_data( const vectorfield& vf, const Data::Geometry& geometry, 
                             const std::string filename, VF_FileFormat format );
    // Writes the OVF text data
    void Write_OVF_text_data( const vectorfield& vf, const Data::Geometry& geometry,
                              std::string& output_to_file );

    // =========================== Saving Energies ===========================
    void Write_Energy_Header( const Data::Spin_System& s, const std::string filename, 
                              std::vector<std::string> firstcolumns={"iteration", "E_tot"}, 
                              bool contributions=true, bool normalize_nos=true );
    // Appends the Energy of a spin system with energy contributions (without header)
    void Append_Image_Energy( const Data::Spin_System& s, const int iteration, 
                               const std::string filename, bool normalize_nos=true );
    // Save energy contributions of a spin system
    void Write_Image_Energy( const Data::Spin_System& system, const std::string filename, 
                              bool normalize_by_nos=true );
    // Save energy contributions of a spin system per spin
    void Write_Image_Energy_per_Spin( const Data::Spin_System & s, const std::string filename, 
                                       bool normalize_nos=true );

    // Saves Energies of all images with header and contributions
    void Write_Chain_Energies( const Data::Spin_System_Chain& c, const int iteration, 
                               const std::string filename, bool normalize_nos=true );
    // Saves the Energies interpolated by the GNEB method
    void Write_Chain_Energies_Interpolated( const Data::Spin_System_Chain& c, 
                                            const std::string filename, bool normalize_nos=true );

    // =========================== Saving Forces ===========================
    // Saves the forces on an image chain
    void Write_System_Force( const Data::Spin_System& s, const std::string filename );
    // Saves the forces on an image chain
    void Write_Chain_Forces( const Data::Spin_System_Chain& c, const std::string filename );
};
#endif