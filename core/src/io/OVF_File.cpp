#include <io/OVF_File.hpp>
#include <utility/Exception.hpp>

namespace IO
{

OVF_Segment::OVF_Segment()
{
    ovf_segment_initialize( this );
}

OVF_Segment::OVF_Segment( const Data::Geometry & geometry )
{
    ovf_segment_initialize( this );

    this->valuedim      = 0;
    this->valuelabels   = const_cast<char *>( "" );
    this->valueunits    = const_cast<char *>( "" );
    this->meshtype      = const_cast<char *>( "rectangular" );
    this->meshunit      = const_cast<char *>( "nm" );
    this->n_cells[0]    = geometry.n_cells[0] * geometry.n_cell_atoms;
    this->n_cells[1]    = geometry.n_cells[1];
    this->n_cells[2]    = geometry.n_cells[2];
    this->N             = geometry.nos;
    this->bounds_min[0] = geometry.bounds_min[0] * 0.1;
    this->bounds_min[1] = geometry.bounds_min[1] * 0.1;
    this->bounds_min[2] = geometry.bounds_min[2] * 0.1;
    this->bounds_max[0] = geometry.bounds_max[0] * 0.1;
    this->bounds_max[1] = geometry.bounds_max[1] * 0.1;
    this->bounds_max[2] = geometry.bounds_max[2] * 0.1;
    this->origin[0]     = 0;
    this->origin[1]     = 0;
    this->origin[2]     = 0;
    this->step_size[0]  = geometry.lattice_constant * geometry.bravais_vectors[0][0] * 0.1;
    this->step_size[1]  = geometry.lattice_constant * geometry.bravais_vectors[1][1] * 0.1;
    this->step_size[2]  = geometry.lattice_constant * geometry.bravais_vectors[2][2] * 0.1;
}

OVF_Segment::~OVF_Segment()
{
    // TODO: create ovf_segment_free
    // free(this->title);
    // free(this->comment);
    // free(this->meshunit);
    // free(this->meshtype);
    // free(this->valuelabels);
    // free(this->valueunits);
    // free(this->meshunit);
    // free(this->n_cells);
    // free(this->step_size);
    // free(this->bounds_min);
    // free(this->bounds_max);
    // free(this->origin);
}

OVF_File::OVF_File( const std::string & filename, bool should_exist )
{
    ovf_file_initialize( this, filename.c_str() );

    if( !this->found && should_exist )
    {
        spirit_throw(
            Utility::Exception_Classifier::Bad_File_Content, Utility::Log_Level::Error,
            fmt::format(
                "Unable open file \"{}\", are you sure it exists? Message: {}", filename, this->latest_message() ) );
    }
}

OVF_File::~OVF_File()
{
    ovf_close( this );
}

const char * OVF_File::latest_message()
{
    return ovf_latest_message( this );
}

void OVF_File::read_segment_header( int index, ovf_segment & segment )
{
    if( ovf_read_segment_header( this, index, &segment ) != OVF_OK )
    {
        spirit_throw(
            Utility::Exception_Classifier::Bad_File_Content, Utility::Log_Level::Error,
            fmt::format(
                "OVF header of segment {}/{} in file \"{}\" could not be parsed. Message: {}", index + 1,
                this->n_segments, this->file_name, this->latest_message() ) );
    }
}

void OVF_File::read_segment_data( int index, const ovf_segment & segment, float * data )
{
    if( ovf_read_segment_data_4( this, index, &segment, data ) != OVF_OK )
    {
        spirit_throw(
            Utility::Exception_Classifier::Bad_File_Content, Utility::Log_Level::Error,
            fmt::format(
                "OVF data segment {}/{} in file \"{}\" could not be parsed. Message: {}", index + 1, this->n_segments,
                this->file_name, this->latest_message() ) );
    }
}

void OVF_File::read_segment_data( int index, const ovf_segment & segment, double * data )
{
    if( ovf_read_segment_data_8( this, index, &segment, data ) != OVF_OK )
    {
        spirit_throw(
            Utility::Exception_Classifier::Bad_File_Content, Utility::Log_Level::Error,
            fmt::format(
                "OVF data segment {}/{} in file \"{}\" could not be parsed. Message: {}", index + 1, this->n_segments,
                this->file_name, this->latest_message() ) );
    }
}

void OVF_File::write_segment( const ovf_segment & segment, const float * data, int format )
{
    if( ovf_write_segment_4( this, &segment, data, format ) != OVF_OK )
    {
        spirit_throw(
            Utility::Exception_Classifier::Bad_File_Content, Utility::Log_Level::Error,
            fmt::format( "Unable to write OVF file \"{}\". Message: {}", this->file_name, this->latest_message() ) );
    }
}

void OVF_File::write_segment( const ovf_segment & segment, const double * data, int format )
{
    if( ovf_write_segment_8( this, &segment, data, format ) != OVF_OK )
    {
        spirit_throw(
            Utility::Exception_Classifier::Bad_File_Content, Utility::Log_Level::Error,
            fmt::format( "Unable to write OVF file \"{}\". Message: {}", this->file_name, this->latest_message() ) );
    }
}

void OVF_File::append_segment( const ovf_segment & segment, const float * data, int format )
{
    if( ovf_append_segment_4( this, &segment, data, format ) != OVF_OK )
    {
        spirit_throw(
            Utility::Exception_Classifier::Bad_File_Content, Utility::Log_Level::Error,
            fmt::format(
                "Unable to append segment to OVF file \"{}\". Message: {}", this->file_name, this->latest_message() ) );
    }
}

void OVF_File::append_segment( const ovf_segment & segment, const double * data, int format )
{
    if( ovf_append_segment_8( this, &segment, data, format ) != OVF_OK )
    {
        spirit_throw(
            Utility::Exception_Classifier::Bad_File_Content, Utility::Log_Level::Error,
            fmt::format(
                "Unable to append segment to OVF file \"{}\". Message: {}", this->file_name, this->latest_message() ) );
    }
}

} // namespace IO
