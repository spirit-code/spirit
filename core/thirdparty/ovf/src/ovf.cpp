#include "ovf.h"
#include <detail/helpers.hpp>
#include <detail/parse.hpp>
#include <detail/write.hpp>
#include <fmt/format.h>


void ovf_file_initialize(struct ovf_file * ovf_file_ptr, const char * filename)
try
{
    // Initialize the struct
    ovf_file_ptr->file_name  = strdup(filename);
    ovf_file_ptr->version    = 0,
    ovf_file_ptr->found      = false;
    ovf_file_ptr->is_ovf     = false;
    ovf_file_ptr->n_segments = 0;
    ovf_file_ptr->_state     = new parser_state;

    // Check if the file exists
    std::fstream filestream( filename );
    ovf_file_ptr->found = filestream.is_open();
    filestream.close();

    // Parse the overall header and do the initial parse of segments
    if( ovf_file_ptr->found )
        ovf::detail::parse::initial(*ovf_file_ptr);
}
catch( ... )
{
}


struct ovf_file * ovf_open(const char * filename)
try
{
    // Initialize the struct
    struct ovf_file * ovf_file_ptr = new ovf_file;
    ovf_file_initialize(ovf_file_ptr, filename);
    return ovf_file_ptr;
}
catch( ... )
{
    return nullptr;
}


void ovf_segment_initialize(struct ovf_segment * ovf_segment_ptr)
try
{
    ovf_segment_ptr->title              = const_cast<char *>("");
    ovf_segment_ptr->comment            = const_cast<char *>("");
    ovf_segment_ptr->valuedim           = 0;
    ovf_segment_ptr->valueunits         = const_cast<char *>("");
    ovf_segment_ptr->valuelabels        = const_cast<char *>("");
    ovf_segment_ptr->meshtype           = const_cast<char *>("");
    ovf_segment_ptr->meshunit           = const_cast<char *>("");
    ovf_segment_ptr->pointcount         = 0;
    ovf_segment_ptr->n_cells[0]         = 0;
    ovf_segment_ptr->n_cells[1]         = 0;
    ovf_segment_ptr->n_cells[2]         = 0;
    ovf_segment_ptr->N                  = 0;
    ovf_segment_ptr->step_size[0]       = 0;
    ovf_segment_ptr->step_size[1]       = 0;
    ovf_segment_ptr->step_size[2]       = 0;
    ovf_segment_ptr->bounds_min[0]      = 0;
    ovf_segment_ptr->bounds_min[1]      = 0;
    ovf_segment_ptr->bounds_min[2]      = 0;
    ovf_segment_ptr->bounds_max[0]      = 0;
    ovf_segment_ptr->bounds_max[1]      = 0;
    ovf_segment_ptr->bounds_max[2]      = 0;
    ovf_segment_ptr->lattice_constant   = 0;
    ovf_segment_ptr->origin[0]          = 0;
    ovf_segment_ptr->origin[1]          = 0;
    ovf_segment_ptr->origin[2]          = 0;
}
catch( ... )
{
}


struct ovf_segment * ovf_segment_create()
try
{
    struct ovf_segment * ovf_segment_ptr = new ovf_segment;
    ovf_segment_initialize(ovf_segment_ptr);
    return ovf_segment_ptr;
}
catch( ... )
{
    return nullptr;
}


bool check_segment(const ovf_segment * segment)
try
{
    if( !segment->title )
        return false;

    if( !segment->comment )
        return false;

    if( !segment->title )
        return false;

    return true;
}
catch( ... )
{
    return false;
}


int ovf_read_segment_header(struct ovf_file * ovf_file_ptr, int index, struct ovf_segment *segment)
try
{
    if( !ovf_file_ptr )
        return OVF_ERROR;

    if( !segment )
    {
        ovf_file_ptr->_state->message_latest =
            "libovf ovf_read_segment_header: invalid segment pointer";
        return OVF_ERROR;
    }

    if( !ovf_file_ptr->found )
    {
        ovf_file_ptr->_state->message_latest = fmt::format(
            "libovf ovf_read_segment_header: file \'{}\' does not exist...",
            ovf_file_ptr->file_name);
        return OVF_ERROR;
    }

    if( !ovf_file_ptr->is_ovf )
    {
        ovf_file_ptr->_state->message_latest = fmt::format(
            "libovf ovf_read_segment_header: file \'{}\' is not ovf...",
            ovf_file_ptr->file_name);
        return OVF_ERROR;
    }

    if( index < 0 )
    {
        ovf_file_ptr->_state->message_latest = fmt::format(
            "libovf ovf_read_segment_header: invalid index ({}) < 0...",
            index, ovf_file_ptr->n_segments, ovf_file_ptr->file_name);
        return OVF_ERROR;
    }

    if( index >= ovf_file_ptr->n_segments )
    {
        ovf_file_ptr->_state->message_latest = fmt::format(
            "libovf ovf_read_segment_header: index ({}) >= n_segments ({}) of file \'{}\'...",
            index, ovf_file_ptr->n_segments, ovf_file_ptr->file_name);
        return OVF_ERROR;
    }
    int retcode = ovf::detail::parse::segment_header( *ovf_file_ptr, index, *segment );
    if (retcode != OVF_OK)
        ovf_file_ptr->_state->message_latest += "\novf_read_segment_header failed.";
    return retcode;
}
catch( ... )
{
    return OVF_ERROR;
}


int ovf_read_segment_data_4(struct ovf_file *ovf_file_ptr, int index, const struct ovf_segment *segment, float *data)
try
{
    if( !ovf_file_ptr )
        return OVF_ERROR;

    if( !segment )
    {
        ovf_file_ptr->_state->message_latest =
            "libovf ovf_read_segment_data_4: invalid segment pointer";
        return OVF_ERROR;
    }

    if( !check_segment(segment) )
    {
        ovf_file_ptr->_state->message_latest =
            "libovf ovf_read_segment_data_4: segment not correctly initialized";
        return OVF_ERROR;
    }

    if( !data )
    {
        ovf_file_ptr->_state->message_latest =
            "libovf ovf_read_segment_data_4: invalid data pointer";
        return OVF_ERROR;
    }

    if( !ovf_file_ptr->found )
    {
        ovf_file_ptr->_state->message_latest = fmt::format(
            "libovf ovf_read_segment_data_4: file \'{}\' does not exist...",
            ovf_file_ptr->file_name);
        return OVF_ERROR;
    }

    if( !ovf_file_ptr->is_ovf )
    {
        ovf_file_ptr->_state->message_latest = fmt::format(
            "libovf ovf_read_segment_data_4: file \'{}\' is not ovf...",
            ovf_file_ptr->file_name);
        return OVF_ERROR;
    }

    if( index >= ovf_file_ptr->n_segments )
    {
        ovf_file_ptr->_state->message_latest = fmt::format(
            "libovf ovf_read_segment_data_4: index ({}) >= n_segments ({}) of file \'{}\'...",
            index, ovf_file_ptr->n_segments, ovf_file_ptr->file_name);
        return OVF_ERROR;
    }

    int retcode = ovf::detail::parse::segment_data(*ovf_file_ptr, index, *segment, data);
    if( retcode != OVF_OK )
        ovf_file_ptr->_state->message_latest += "\novf_read_segment_data_4 failed.";
    return retcode;
}
catch( ... )
{
    return OVF_ERROR;
}


int ovf_read_segment_data_8(struct ovf_file *ovf_file_ptr, int index, const struct ovf_segment *segment, double *data)
try
{
    if( !ovf_file_ptr )
        return OVF_ERROR;

    if( !segment )
    {
        ovf_file_ptr->_state->message_latest =
            "libovf ovf_read_segment_data_8: invalid segment pointer";
        return OVF_ERROR;
    }

    if( !check_segment(segment) )
    {
        ovf_file_ptr->_state->message_latest =
            "libovf ovf_read_segment_data_8: segment not correctly initialized";
        return OVF_ERROR;
    }

    if( !data )
    {
        ovf_file_ptr->_state->message_latest =
            "libovf ovf_read_segment_data_8: invalid data pointer";
        return OVF_ERROR;
    }

    if( !ovf_file_ptr->found )
    {
        ovf_file_ptr->_state->message_latest = fmt::format(
            "libovf ovf_read_segment_8: file \'{}\' does not exist...",
            ovf_file_ptr->file_name);
        return OVF_ERROR;
    }

    if( !ovf_file_ptr->is_ovf )
    {
        ovf_file_ptr->_state->message_latest = fmt::format(
            "libovf ovf_read_segment_8: file \'{}\' is not ovf...",
            ovf_file_ptr->file_name);
        return OVF_ERROR;
    }

    if( index >= ovf_file_ptr->n_segments )
    {
        ovf_file_ptr->_state->message_latest = fmt::format(
            "libovf ovf_read_segment_8: index ({}) >= n_segments ({}) of file \'{}\'...",
            index, ovf_file_ptr->n_segments, ovf_file_ptr->file_name);
        return OVF_ERROR;
    }

    int retcode = ovf::detail::parse::segment_data(*ovf_file_ptr, index, *segment, data);
    if (retcode != OVF_OK)
        ovf_file_ptr->_state->message_latest += "\novf_read_segment_data_8 failed.";
    return retcode;
}
catch( ... )
{
    return OVF_ERROR;
}


int ovf_write_segment_4(struct ovf_file *ovf_file_ptr, const struct ovf_segment *segment, float *data, int format)
try
{
    if( !ovf_file_ptr )
        return OVF_ERROR;

    if( !segment )
    {
        ovf_file_ptr->_state->message_latest =
            "libovf ovf_write_segment_4: invalid segment pointer";
        return OVF_ERROR;
    }

    if( !check_segment(segment) )
    {
        ovf_file_ptr->_state->message_latest =
            "libovf ovf_write_segment_4: segment not correctly initialized";
        return OVF_ERROR;
    }

    if( !data )
    {
        ovf_file_ptr->_state->message_latest =
            "libovf ovf_write_segment_4: invalid data pointer";
        return OVF_ERROR;
    }

    if( format == OVF_FORMAT_BIN8 ||
        format == OVF_FORMAT_BIN4 )
        format = OVF_FORMAT_BIN;

    if( format != OVF_FORMAT_BIN  &&
        format != OVF_FORMAT_TEXT &&
        format != OVF_FORMAT_CSV  )
    {
        ovf_file_ptr->_state->message_latest =
            fmt::format("libovf ovf_write_segment_4: invalid format \'{}\'...", format);
        return OVF_ERROR;
    }

    int retcode = ovf::detail::write::segment(ovf_file_ptr, segment, data, false, format);
    if (retcode != OVF_OK)
        ovf_file_ptr->_state->message_latest += "\novf_write_segment_4 failed.";
    return retcode;
}
catch( ... )
{
    return OVF_ERROR;
}


int ovf_write_segment_8(struct ovf_file *ovf_file_ptr, const struct ovf_segment *segment, double *data, int format)
try
{
    if( !ovf_file_ptr )
        return OVF_ERROR;

    if( !segment )
    {
        ovf_file_ptr->_state->message_latest =
            "libovf ovf_write_segment_8: invalid segment pointer";
        return OVF_ERROR;
    }

    if( !check_segment(segment) )
    {
        ovf_file_ptr->_state->message_latest =
            "libovf ovf_write_segment_8: segment not correctly initialized";
        return OVF_ERROR;
    }

    if( !data )
    {
        ovf_file_ptr->_state->message_latest =
            "libovf ovf_write_segment_8: invalid data pointer";
        return OVF_ERROR;
    }

    if( format == OVF_FORMAT_BIN8 ||
        format == OVF_FORMAT_BIN4 )
        format = OVF_FORMAT_BIN;

    if( format != OVF_FORMAT_BIN  &&
        format != OVF_FORMAT_TEXT &&
        format != OVF_FORMAT_CSV  )
    {
        ovf_file_ptr->_state->message_latest =
            fmt::format("libovf ovf_write_segment_8: invalid format \'{}\'...", format);
        return OVF_ERROR;
    }

    int retcode = ovf::detail::write::segment(ovf_file_ptr, segment, data, false, format);
    if (retcode != OVF_OK)
        ovf_file_ptr->_state->message_latest += "\novf_write_segment_8 failed.";
    return retcode;
}
catch( ... )
{
    return OVF_ERROR;
}


int ovf_append_segment_4(struct ovf_file *ovf_file_ptr, const struct ovf_segment *segment, float *data, int format)
try
{
    if( !ovf_file_ptr )
        return OVF_ERROR;

    if( !segment )
    {
        ovf_file_ptr->_state->message_latest =
            "libovf ovf_append_segment_4: invalid segment pointer";
        return OVF_ERROR;
    }

    if( !check_segment(segment) )
    {
        ovf_file_ptr->_state->message_latest =
            "libovf ovf_append_segment_4: segment not correctly initialized";
        return OVF_ERROR;
    }

    if( !data )
    {
        ovf_file_ptr->_state->message_latest =
            "libovf ovf_append_segment_4: invalid data pointer";
        return OVF_ERROR;
    }

    if( ovf_file_ptr->found && !ovf_file_ptr->is_ovf )
    {
        ovf_file_ptr->_state->message_latest = "libovf ovf_append_segment_4: file is not ovf...";
        return OVF_ERROR;
    }

    if( format == OVF_FORMAT_BIN8 ||
        format == OVF_FORMAT_BIN4 )
        format = OVF_FORMAT_BIN;

    if( format != OVF_FORMAT_BIN  &&
        format != OVF_FORMAT_TEXT &&
        format != OVF_FORMAT_CSV  )
    {
        ovf_file_ptr->_state->message_latest =
            fmt::format("libovf ovf_append_segment_4: invalid format \'{}\'...", format);
        return OVF_ERROR;
    }

    bool append = ovf_file_ptr->found;
    int retcode = ovf::detail::write::segment(ovf_file_ptr, segment, data, append, format);
    if (retcode != OVF_OK)
        ovf_file_ptr->_state->message_latest += "\novf_append_segment_4 failed.";
    return retcode;
}
catch( ... )
{
    return OVF_ERROR;
}


int ovf_append_segment_8(struct ovf_file *ovf_file_ptr, const struct ovf_segment *segment, double *data, int format)
try
{
    if( !ovf_file_ptr )
        return OVF_ERROR;

    if( !segment )
    {
        ovf_file_ptr->_state->message_latest =
            "libovf ovf_append_segment_8: invalid segment pointer";
        return OVF_ERROR;
    }

    if( !check_segment(segment) )
    {
        ovf_file_ptr->_state->message_latest =
            "libovf ovf_append_segment_8: segment not correctly initialized";
        return OVF_ERROR;
    }

    if( !data )
    {
        ovf_file_ptr->_state->message_latest =
            "libovf ovf_append_segment_8: invalid data pointer";
        return OVF_ERROR;
    }

    if( format == OVF_FORMAT_BIN8 ||
        format == OVF_FORMAT_BIN4 )
        format = OVF_FORMAT_BIN;

    if( ovf_file_ptr->found && !ovf_file_ptr->is_ovf )
    {
        ovf_file_ptr->_state->message_latest = "libovf ovf_append_segment_8: file is not ovf...";
        return OVF_ERROR;
    }

    if( format != OVF_FORMAT_BIN  &&
        format != OVF_FORMAT_TEXT &&
        format != OVF_FORMAT_CSV  )
    {
        ovf_file_ptr->_state->message_latest =
            fmt::format("libovf ovf_append_segment_8: invalid format \'{}\'...", format);
        return OVF_ERROR;
    }

    bool append = ovf_file_ptr->found;
    int retcode = ovf::detail::write::segment(ovf_file_ptr, segment, data, append, format);
    if (retcode != OVF_OK)
        ovf_file_ptr->_state->message_latest += "\novf_append_segment_8 failed.";
    return retcode;
}
catch( ... )
{
    return OVF_ERROR;
}


const char * ovf_latest_message(struct ovf_file *ovf_file_ptr)
try
{
    if( !ovf_file_ptr )
        return "";

    ovf_file_ptr->_state->message_out = ovf_file_ptr->_state->message_latest;
    ovf_file_ptr->_state->message_latest = "";
    return ovf_file_ptr->_state->message_out.c_str();
}
catch( ... )
{
    return "";
}


int ovf_close(struct ovf_file * ovf_file_ptr)
try
{
    if( !ovf_file_ptr )
        return OVF_ERROR;
    if( !ovf_file_ptr->_state )
        return OVF_ERROR;
    delete(ovf_file_ptr->_state);
    return OVF_OK;
}
catch( ... )
{
    return OVF_ERROR;
}