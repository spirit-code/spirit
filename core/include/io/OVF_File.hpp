#pragma once
#ifndef SPIRIT_CORE_IO_OVFFILE_HPP
#define SPIRIT_CORE_IO_OVFFILE_HPP

#include <data/Spin_System.hpp>

#include <ovf.h>

namespace IO
{

struct OVF_Segment : ::ovf_segment
{
    OVF_Segment();
    OVF_Segment( const Data::Spin_System & system );
    ~OVF_Segment();
};

struct OVF_File : ::ovf_file
{
    OVF_File( const std::string & filename, bool should_exist = false );
    ~OVF_File();

    const char * latest_message();

    void read_segment_header( int index, ::ovf_segment & segment );
    void read_segment_data( int index, const ::ovf_segment & segment, float * data );
    void read_segment_data( int index, const ::ovf_segment & segment, double * data );
    void write_segment( const ::ovf_segment & segment, float * data, int format = OVF_FORMAT_BIN );
    void write_segment( const ::ovf_segment & segment, double * data, int format = OVF_FORMAT_BIN );
    void append_segment( const ::ovf_segment & segment, float * data, int format = OVF_FORMAT_BIN );
    void append_segment( const ::ovf_segment & segment, double * data, int format = OVF_FORMAT_BIN );
};

} // namespace IO

#endif