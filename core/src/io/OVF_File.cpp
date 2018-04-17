#include <io/OVF_File.hpp>

#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

#include <engine/Vectormath.hpp>

using namespace Utility;

namespace IO
{
    File_OVF::File_OVF( std::string filename, VF_FileFormat format ) : 
        filename(filename), format(format)
    {
        this->isOVF = false;
        this->output_to_file = "";
        this->output_to_file.reserve( int( 0x08000000 ) );  // reserve 128[MByte]
        this->sender = Log_Sender::IO;
        this->n_segments = -1;

        // the datatype_out is used when writing an OVF file
        if ( this->format == VF_FileFormat::OVF_BIN8 ) 
            this->datatype_out = "Binary 8";
        else if ( this->format == VF_FileFormat::OVF_BIN4 ) 
            this->datatype_out = "Binary 4";
        else if( this->format == VF_FileFormat::OVF_TEXT ) 
            this->datatype_out = "Text";
        else if( this->format == VF_FileFormat::OVF_CSV ) 
            this->datatype_out = "CSV";

        this->ifile = NULL;
        this->n_segments = 0;
        this->n_segments_pos = 0;
        this->n_segments_as_str = "";
        this->version = "";
        this->title = "";
        this->meshunit = "";
        this->meshtype = "";
        this->valueunits = "";
        this->datatype_in = "";
        this->max = Vector3(0,0,0);
        this->min = Vector3(0,0,0);
        this->pointcount = -1;
        this->base = { Vector3(0,0,0), Vector3(0,0,0), Vector3(0,0,0) };
        this->stepsize = Vector3(0,0,0);
        this->sender = Log_Sender::IO;

        // check if the file exists
        std::fstream file( filename );
        this->file_exists = file.is_open();
        file.close();
                
        // if the file exists check the version
        if ( this->file_exists ) check_version();
           
        // if the file has the OVF header get the number and the positions of the segments
        if ( this->isOVF )
        {
            read_n_segments_from_top_header();

            int n_seg = count_and_locate_segments();
            
            // compare with n_segments in top header
            if( this->n_segments != n_seg )
                spirit_throw( Utility::Exception_Classifier::Bad_File_Content, 
                              Utility::Log_Level::Error, fmt::format( "OVF Segment number "
                              "in header ({0}) is different from the number of segments "
                              "({1}) in file", this->n_segments, n_seg ) );
        }  
    }
   
    void File_OVF::check_version()
    {

        this->ifile = std::unique_ptr<Filter_File_Handle>( 
                            new Filter_File_Handle( this->filename, this->comment_tag ) ); 
        
        // check if the file has an OVF top header
        if ( this->ifile->Read_Single( this->version, "# OOMMF OVF", false ) )
        {
            // check the OVF version
            if( this->version != "2.0" && this->version != "2" )
            {
                spirit_throw( Utility::Exception_Classifier::Bad_File_Content, 
                              Utility::Log_Level::Error,
                              fmt::format( "OVF {0} is not supported", this->version ) );
            } 
            this->isOVF = true;
        }
        else
        {
            this->isOVF = false;
        }
        
        this->ifile = NULL;
    }

    void File_OVF::read_header()
    {
        try
        {
            ifile->Read_String( this->title, "# Title:" );
            ifile->Read_Single( this->meshunit, "# meshunit:" );
            ifile->Require_Single( this->valuedim, "# valuedim:" );
            ifile->Read_String( this->valueunits, "# valueunits:" );
            ifile->Read_String( this->valueunits, "# valuelabels:" );
            
            ifile->Read_Single( this->min.x(), "# xmin:" );
            ifile->Read_Single( this->min.y(), "# ymin:" );
            ifile->Read_Single( this->min.z(), "# zmin:" );
            ifile->Read_Single( this->max.x(), "# xmax:" );
            ifile->Read_Single( this->max.y(), "# ymax:" );
            ifile->Read_Single( this->max.z(), "# zmax:" );
            
            ifile->Require_Single( this->meshtype, "# meshtype:" );
            
            if( this->meshtype != "rectangular" && this->meshtype != "irregular" )
            {
                spirit_throw(Utility::Exception_Classifier::Bad_File_Content, Utility::Log_Level::Error,
                    "Mesh type must be either \"rectangular\" or \"irregular\"");
            }
            
            // Emit Header to Log
            auto lvl = Log_Level::Debug;
            
            Log( lvl, this->sender, fmt::format( "# OVF title               = {}", this->title ) );
            Log( lvl, this->sender, fmt::format( "# OVF values dimensions   = {}", this->valuedim ) );
            Log( lvl, this->sender, fmt::format( "# OVF meshunit            = {}", this->meshunit ) );
            Log( lvl, this->sender, fmt::format( "# OVF xmin                = {}", this->min.x() ) );
            Log( lvl, this->sender, fmt::format( "# OVF ymin                = {}", this->min.y() ) );
            Log( lvl, this->sender, fmt::format( "# OVF zmin                = {}", this->min.z() ) );
            Log( lvl, this->sender, fmt::format( "# OVF xmax                = {}", this->max.x() ) );
            Log( lvl, this->sender, fmt::format( "# OVF ymax                = {}", this->max.y() ) );
            Log( lvl, this->sender, fmt::format( "# OVF zmax                = {}", this->max.z() ) );
            
            // For different mesh types
            if( this->meshtype == "rectangular" )
            {
                ifile->Read_Vector3( this->base[0], "# xbase:", true );
                ifile->Read_Vector3( this->base[1], "# ybase:", true );
                ifile->Read_Vector3( this->base[2], "# zbase:", true );
                
                ifile->Require_Single( this->stepsize.x(), "# xstepsize:" );
                ifile->Require_Single( this->stepsize.y(), "# ystepsize:" );
                ifile->Require_Single( this->stepsize.z(), "# zstepsize:" );
                
                ifile->Require_Single( this->nodes[0], "# xnodes:" );
                ifile->Require_Single( this->nodes[1], "# ynodes:" );
                ifile->Require_Single( this->nodes[2], "# znodes:" );
                
                // Write to Log
                Log( lvl, this->sender, fmt::format( "# OVF meshtype <{}>", this->meshtype ) );
                Log( lvl, this->sender, fmt::format( "# xbase      = {:.8}", this->base[0] ) );
                Log( lvl, this->sender, fmt::format( "# ybase      = {:.8}", this->base[1] ) );
                Log( lvl, this->sender, fmt::format( "# zbase      = {:.8}", this->base[2] ) );
                Log( lvl, this->sender, fmt::format( "# xstepsize  = {:.8f}", this->stepsize.x() ) );
                Log( lvl, this->sender, fmt::format( "# ystepsize  = {:.8f}", this->stepsize.y() ) );
                Log( lvl, this->sender, fmt::format( "# zstepsize  = {:.8f}", this->stepsize.z() ) );
                Log( lvl, this->sender, fmt::format( "# xnodes     = {}", this->nodes[0] ) );
                Log( lvl, this->sender, fmt::format( "# ynodes     = {}", this->nodes[1] ) );
                Log( lvl, this->sender, fmt::format( "# znodes     = {}", this->nodes[2] ) );
            }
            
            // Check mesh type
            if ( this->meshtype == "irregular" )
            {
                ifile->Require_Single( this->pointcount, "# pointcount:" );
                
                // Write to Log
                Log( lvl, this->sender, fmt::format( "# OVF meshtype <{}>", this->meshtype ) );
                Log( lvl, this->sender, fmt::format( "# OVF point count = {}", this->pointcount ) );
            }
        }
        catch (...) 
        {
            spirit_rethrow( fmt::format("Failed to read OVF file \"{}\".", this->filename) );
        }
    }
        
    void File_OVF::check_geometry( const Data::Geometry& geometry )
    {
        try
        {
            // Check that nos is smaller or equal to the nos of the current image
            int nos = this->nodes[0] * this->nodes[1] * this->nodes[2];
            if ( nos > geometry.nos )
                spirit_throw(Utility::Exception_Classifier::Bad_File_Content, 
                    Utility::Log_Level::Error,"NOS of the OVF file is greater than the NOS in the "
                    "current image");
            
            // Check if the geometry of the ovf file is the same with the one of the current image
            if ( this->nodes[0] != geometry.n_cells[0] ||
                 this->nodes[1] != geometry.n_cells[1] ||
                 this->nodes[2] != geometry.n_cells[2] )
            {
                Log(Log_Level::Warning, this->sender, fmt::format("The geometry of the OVF file "
                    "does not much the geometry of the current image") );
            }
        }
        catch (...) 
        {
            spirit_rethrow( fmt::format("Failed to read OVF file \"{}\".", this->filename) );
        }
    }
        
    void File_OVF::read_data( vectorfield& vf )
    {
        try
        {
            auto lvl = Log_Level::Debug;
            
            // Raw data representation
            ifile->Read_String( this->datatype_in, "# Begin: Data" );
            std::istringstream repr( this->datatype_in );
            repr >> this->datatype_in;
            if( this->datatype_in == "binary" ) 
                repr >> this->binary_length;
            else
                this->binary_length = 0;
            
            Log( lvl, this->sender, fmt::format( "# OVF data representation = {}", this->datatype_in ) );
            Log( lvl, this->sender, fmt::format( "# OVF binary length       = {}", this->binary_length ) );
            
            // Check that representation and binary length valures are ok
            if( this->datatype_in != "text" && 
                this->datatype_in != "binary" &&
                this->datatype_in != "csv" )
            {
                spirit_throw( Utility::Exception_Classifier::Bad_File_Content, 
                              Utility::Log_Level::Error, "Data representation must be "
                              "either \"text\", \"binary\" or \"csv\"");
            }
            
            if( this->datatype_in == "binary" && 
                 this->binary_length != 4 && this->binary_length != 8  )
            {
                spirit_throw( Exception_Classifier::Bad_File_Content, Log_Level::Error,
                              "Binary representation can be either \"binary 8\" or \"binary 4\"");
            }
            
            // Read the data
            if( this->datatype_in == "binary" )
                read_data_bin( vf );
            else if( this->datatype_in == "text" )
                read_data_txt( vf );
            else if( this->datatype_in == "csv" )
                read_data_txt( vf, "," );
        }
        catch (...) 
        {
            spirit_rethrow( fmt::format("Failed to read OVF file \"{}\".", filename) );
        }
    }
   

    bool File_OVF::check_binary_values()
    {
        try
        {
            // create initial check values for the binary data (see OVF specification)
            const double ref_8b = *reinterpret_cast<const double *>( &this->test_hex_8b );
            double read_8byte = 0;
            
            const float ref_4b = *reinterpret_cast<const float *>( &this->test_hex_4b );
            float read_4byte = 0;
            
            // check the validity of the initial check value read with the reference one
            if ( this->binary_length == 4 )
            {    
                ifile->myfile->read( reinterpret_cast<char *>( &read_4byte ), sizeof(float) );
                if ( read_4byte != ref_4b ) 
                {
                    spirit_throw( Exception_Classifier::Bad_File_Content, Log_Level::Error,
                                  "OVF initial check value of binary data is inconsistent" );
                }
            }
            else if ( this->binary_length == 8 )
            {
                ifile->myfile->read( reinterpret_cast<char *>( &read_8byte ), sizeof(double) );
                if ( read_8byte != ref_8b )
                {
                    spirit_throw( Exception_Classifier::Bad_File_Content, Log_Level::Error,
                                  "OVF initial check value of binary data is inconsistent" );
                }
            }
            
            return true;
        }
        catch (...)
        {
            spirit_rethrow( "Failed to check OVF initial binary value" );
            return false;
        }
    
    
    }

    void File_OVF::read_data_bin( vectorfield& vf )
    {
        try
        {        
            // Set the input stream indicator to the end of the line describing the data block
            ifile->iss.seekg( std::ios::end );
            
            // Check if the initial check value of the binary data is valid
            if( !check_binary_values() )
                spirit_throw( Exception_Classifier::Bad_File_Content, Log_Level::Error,
                              "The OVF initial binary value could not be read correctly");
            
            // Comparison of datum size compared to scalar type
            if ( this->binary_length == 4 )
            {
                int vectorsize = 3 * sizeof(float);
                float buffer[3];
                int index;
                for( int k=0; k<this->nodes[2]; k++ )
                {
                    for( int j=0; j<this->nodes[1]; j++ )
                    {
                        for( int i=0; i<this->nodes[0]; i++ )
                        {
                            index = i + j*this->nodes[0] + k*this->nodes[0]*this->nodes[1];
                            
                            ifile->myfile->read(reinterpret_cast<char *>(&buffer[0]), vectorsize);
                            
                            vf[index][0] = static_cast<scalar>(buffer[0]);
                            vf[index][1] = static_cast<scalar>(buffer[1]);
                            vf[index][2] = static_cast<scalar>(buffer[2]);
                        }
                    }
                }
                
            }
            else if (this->binary_length == 8)
            {
                int vectorsize = 3 * sizeof(double);
                double buffer[3];
                int index;
                for (int k = 0; k<this->nodes[2]; k++)
                {
                    for (int j = 0; j<this->nodes[1]; j++)
                    {
                        for (int i = 0; i<this->nodes[0]; i++)
                        {
                            index = i + j*this->nodes[0] + k*this->nodes[0] * this->nodes[1];
                            
                            ifile->myfile->read(reinterpret_cast<char *>(&buffer[0]), vectorsize);
                            
                            vf[index][0] = static_cast<scalar>(buffer[0]);
                            vf[index][1] = static_cast<scalar>(buffer[1]);
                            vf[index][2] = static_cast<scalar>(buffer[2]);
                        }
                    }
                }
            }
        }
        catch (...)
        {
            spirit_rethrow( "Failed to read OVF binary data" );
        }
    }
    
    void File_OVF::read_data_txt( vectorfield& vf, const std::string& delimiter )
    {
        try
        { 
            int nos = this->nodes[0] * this->nodes[1] * this->nodes[2];
            
            for (int i=0; i<nos; i++)
            {
                this->ifile->GetLine( delimiter );
                
                this->ifile->iss >> vf[i][0];
                this->ifile->iss >> vf[i][1];
                this->ifile->iss >> vf[i][2];
            }
        }
        catch (...)
        {
            spirit_rethrow( "Failed to check OVF initial binary value" );
        }
    }

    void File_OVF::write_top_header()
    {
        this->output_to_file += fmt::format( "# OOMMF OVF 2.0\n" );
        this->output_to_file += fmt::format( this->empty_line );
       
        // initialize n_segments to zero
        this->n_segments = 0;
        // convert n_segments to string
        std::string n_segments_str = std::to_string( this->n_segments );
        // calculate padding's length
        int padding_length = this->n_segments_str_digits - n_segments_str.length(); 
        // create padding string 
        std::string padding( padding_length, '0' );
        // write padding plus n_segments
        this->output_to_file += fmt::format( "# Segment count: {}\n", padding + n_segments_str );
        
        Dump_to_File( this->output_to_file, this->filename );  // Dump to file
        this->output_to_file = "";  // reset output string buffer
    }

    void File_OVF::write_data_bin( const vectorfield& vf )
    {
        // float test value
        const float ref_4b = *reinterpret_cast<const float *>( &this->test_hex_4b );
        
        // double test value
        const double ref_8b = *reinterpret_cast<const double *>( &this->test_hex_8b );
        
        if( format == VF_FileFormat::OVF_BIN8 )
        {
            this->output_to_file += std::string( reinterpret_cast<const char *>(&ref_8b),
                sizeof(double) );
            
            // in case that scalar is 4bytes long
            if (sizeof(scalar) == sizeof(float))
            {
                double buffer[3];
                for (unsigned int i=0; i<vf.size(); i++)
                {
                    buffer[0] = static_cast<double>(vf[i][0]);
                    buffer[1] = static_cast<double>(vf[i][1]);
                    buffer[2] = static_cast<double>(vf[i][2]);
                    this->output_to_file += std::string( reinterpret_cast<char *>(buffer), 
                        sizeof(buffer) );
                }
            } 
            else
            {
                for (unsigned int i=0; i<vf.size(); i++)
                    this->output_to_file += 
                        std::string( reinterpret_cast<const char *>(&vf[i]), 3*sizeof(double) );
            }
        }
        else if( format == VF_FileFormat::OVF_BIN4 )
        {
            this->output_to_file += std::string( reinterpret_cast<const char *>(&ref_4b),
                sizeof(float) );
            
            // in case that scalar is 8bytes long
            if (sizeof(scalar) == sizeof(double))
            {
                float buffer[3];
                for (unsigned int i=0; i<vf.size(); i++)
                {
                    buffer[0] = static_cast<float>(vf[i][0]);
                    buffer[1] = static_cast<float>(vf[i][1]);
                    buffer[2] = static_cast<float>(vf[i][2]);
                    this->output_to_file += std::string( reinterpret_cast<char *>(buffer), 
                        sizeof(buffer) );
                }
            } 
            else
            {
                for (unsigned int i=0; i<vf.size(); i++)
                    this->output_to_file += 
                        std::string( reinterpret_cast<const char *>(&vf[i]), 3*sizeof(float) );
            }
        }
    }

    void File_OVF::write_data_txt( const vectorfield& vf, const std::string& delimiter )
    {
        for (int iatom = 0; iatom < vf.size(); ++iatom)
        {
            this->output_to_file += fmt::format( "{:20.10f}{} {:20.10f}{} {:20.10f}{}\n", 
                                                  vf[iatom][0], delimiter, 
                                                  vf[iatom][1], delimiter,
                                                  vf[iatom][2], delimiter );
        }
    }

    void File_OVF::increment_n_segments()
    {
        try
        {
            std::fstream file( this->filename ); 
       
            // update n_segments
            this->n_segments++;
            
            // convert updated n_segment into padded string
            std::string new_n_str = std::to_string( this->n_segments );
            std::string::size_type new_n_len = new_n_str.length();

            std::string::size_type padding_len = this->n_segments_str_digits - new_n_len;
            std::string padding( padding_len, '0' ); 

            // n_segments_pos is the end of the line that contains '#segment count' (after '\n')
            std::ios::off_type offset = this->n_segments_str_digits + 1;
            
            // go to the beginning '#segment count' value position
            file.seekg( this->n_segments_pos );
            file.seekg( (-1)*offset, std::ios::cur );

            // replace n_segments value in the stream
            file << ( padding + new_n_str );

            file.close();
        }
        catch( ... )
        {
            spirit_rethrow( fmt::format("Failed to increment n_segments in OVF file \"{}\".", 
                            this->filename) );
        }
    }

    void File_OVF::read_n_segments_from_top_header()
    {
        try
        {
            this->ifile = std::unique_ptr<Filter_File_Handle>( 
                                new Filter_File_Handle( this->filename, this->comment_tag ) ); 
           
            // get the number of segments from the initial keyword
            ifile->Require_Single( this->n_segments, "# segment count:" ); 

            // get the number of segment as string
            ifile->Read_String( this->n_segments_as_str, "# segment count:" );

            // save the file position indicator in case we have to increment n_segment
            this->n_segments_pos = this->ifile->GetPosition();

            // TODO: what will happen if the n_segments does not have padding?
            
            // close the file
            this->ifile = NULL;
        }
        catch( ... )
        {
            spirit_rethrow( fmt::format("Failed to read OVF file \"{}\".", this->filename) );
        }

    }

    int File_OVF::count_and_locate_segments()
    {
        try
        {
            this->ifile = std::unique_ptr<Filter_File_Handle>( 
                                new Filter_File_Handle( this->filename, this->comment_tag ) ); 

            // get the number of segments from the occurrences of "# Begin: Segment"
            int n_begin_segment = 0;
            
            std::ios::pos_type end = this->ifile->GetPosition( std::ios::end ); 
            
            // NOTE: the keyword to find must be lower case since the Filter File Handle 
            // converts the content of the input file to lower case automatically
            while( ifile->Find( "# begin: segment" ) )
            {
                std::ios::pos_type pos = this->ifile->GetPosition(); 
                this->segment_fpos.push_back( pos );
                ifile->SetLimits( pos, end );

                ++n_begin_segment;
            }
           
            // find the very last keyword of the file
            this->segment_fpos.push_back( end );

            // reset limits
            ifile->ResetLimits();
            
            // close the file
            this->ifile = NULL;

            return n_begin_segment;
        }
        catch( ... )
        {
            spirit_rethrow( fmt::format("Failed to read OVF file \"{}\".", this->filename) );
        }
    }
    
// Public methods ------------------------------------------------------------------------------

    bool File_OVF::is_OVF()
    {
        return this->isOVF;
    }
    
    int File_OVF::get_n_segments()
    {
        return this->n_segments;
    }

    void File_OVF::read_segment( vectorfield& vf, const Data::Geometry& geometry, 
                                 const int idx_seg )
    {
        try
        {
            if ( !this->file_exists )
            {
                spirit_throw( Exception_Classifier::File_not_Found, Log_Level::Warning, 
                              fmt::format( "The file \"{}\" does not exist", filename ) );
            } 
            else if ( this->n_segments = 0 )
            {
                spirit_throw( Exception_Classifier::Bad_File_Content, Log_Level::Warning, 
                              fmt::format( "File \"{}\" is empty", filename ) );
            }
            else
            {
                // open the file
                this->ifile = std::unique_ptr<Filter_File_Handle>( 
                                    new Filter_File_Handle( this->filename, this->comment_tag ) ); 
                
                // NOTE: seg_idx.max = segment_fpos.size - 2
                if ( idx_seg >= ( this->segment_fpos.size() - 1 ) )
                    spirit_throw( Exception_Classifier::Input_parse_failed, Log_Level::Error,
                                  "OVF error while choosing segment - index out of bounds" );

                this->ifile->SetLimits( this->segment_fpos[idx_seg], 
                                        this->segment_fpos[idx_seg+1] );
           
                read_header();
                check_geometry( geometry );
                read_data( vf );

                // close the file
                this->ifile = NULL;
            }
        }
        catch( ... )
        {
            spirit_rethrow( fmt::format("Failed to read OVF file \"{}\".", this->filename) );
        }
    }
    
    void File_OVF::write_segment( const vectorfield& vf, const Data::Geometry& geometry,
                                  const std::string comment, const bool append )
    {
        try
        {
            this->output_to_file.reserve( int( 0x08000000 ) );  // reserve 128[MByte]
         
            // If we are not appending or the file does not exists we need to write the top header
            // and to turn the file_exists attribute to true so we can append more segments
            if ( !append || !this->file_exists ) 
            {
                write_top_header();
                read_n_segments_from_top_header();  // finds the file position of n_segments
                this->file_exists = true; 
            }
            
            this->output_to_file += fmt::format( this->empty_line );
            this->output_to_file += fmt::format( "# Begin: Segment\n" );
            this->output_to_file += fmt::format( "# Begin: Header\n" );
            this->output_to_file += fmt::format( this->empty_line );
            
            this->output_to_file += fmt::format( "# Title: SPIRIT Version {}\n", 
                                                 Utility::version_full );
            this->output_to_file += fmt::format( this->empty_line );
            
            this->output_to_file += fmt::format( "# Desc: {}\n", this->comment );
            this->output_to_file += fmt::format( this->empty_line );
            
            // The value dimension is always 3 since we are writting Vector3-data
            this->output_to_file += fmt::format( "# valuedim: {} ##Value dimension\n", 3 );
            this->output_to_file += fmt::format( "# valueunits: None None None\n" );
            this->output_to_file +=
                fmt::format("# valuelabels: spin_x_component spin_y_component "
                            "spin_z_component \n");
            this->output_to_file += fmt::format( this->empty_line );
            
            this->output_to_file += fmt::format( "## Fundamental mesh measurement unit. "
                                                 "Treated as a label:\n" );
            this->output_to_file += fmt::format( "# meshunit: unspecified\n" );
            this->output_to_file += fmt::format( this->empty_line );
            
            this->output_to_file += fmt::format( "# xmin: {}\n", geometry.bounds_min[0] );
            this->output_to_file += fmt::format( "# ymin: {}\n", geometry.bounds_min[1] );
            this->output_to_file += fmt::format( "# zmin: {}\n", geometry.bounds_min[2] );
            this->output_to_file += fmt::format( "# xmax: {}\n", geometry.bounds_max[0] );
            this->output_to_file += fmt::format( "# ymax: {}\n", geometry.bounds_max[1] );
            this->output_to_file += fmt::format( "# zmax: {}\n", geometry.bounds_max[2] );
            this->output_to_file += fmt::format( this->empty_line );
            
            // TODO: Spirit does not support irregular geometry yet. Write ONLY rectangular mesh
            this->output_to_file += fmt::format( "# meshtype: rectangular\n" );
           
            // Bravais Lattice
            this->output_to_file += fmt::format( "# xbase: {} {} {}\n", 
                                                 geometry.bravais_vectors[0][0], 
                                                 geometry.bravais_vectors[0][1],
                                                 geometry.bravais_vectors[0][2] );
            this->output_to_file += fmt::format( "# ybase: {} {} {}\n",
                                                 geometry.bravais_vectors[1][0], 
                                                 geometry.bravais_vectors[1][1],
                                                 geometry.bravais_vectors[1][2] );
            this->output_to_file += fmt::format( "# zbase: {} {} {}\n",
                                                 geometry.bravais_vectors[2][0], 
                                                 geometry.bravais_vectors[2][1],
                                                 geometry.bravais_vectors[2][2] );
            
            this->output_to_file += fmt::format( "# xstepsize: {}\n", 
                                        geometry.lattice_constant * geometry.bravais_vectors[0][0] );
            this->output_to_file += fmt::format( "# ystepsize: {}\n", 
                                        geometry.lattice_constant * geometry.bravais_vectors[1][1] );
            this->output_to_file += fmt::format( "# zstepsize: {}\n", 
                                        geometry.lattice_constant * geometry.bravais_vectors[2][2] );
            
            this->output_to_file += fmt::format( "# xnodes: {}\n", geometry.n_cells[0] );
            this->output_to_file += fmt::format( "# ynodes: {}\n", geometry.n_cells[1] );
            this->output_to_file += fmt::format( "# znodes: {}\n", geometry.n_cells[2] );
            this->output_to_file += fmt::format( this->empty_line );
            
            this->output_to_file += fmt::format( "# End: Header\n" );
            this->output_to_file += fmt::format( this->empty_line );
            
            // Data
            this->output_to_file += fmt::format( "# Begin: Data {}\n", this->datatype_out );
            
            if ( this->format == VF_FileFormat::OVF_BIN8 || format == VF_FileFormat::OVF_BIN4 )
                write_data_bin( vf );
            else if ( this->format == VF_FileFormat::OVF_TEXT )
                write_data_txt( vf );
            else if ( this->format == VF_FileFormat::OVF_CSV )
                write_data_txt( vf, "," );
            
            this->output_to_file += fmt::format( "# End: Data {}\n", this->datatype_out );
            
            this->output_to_file += fmt::format( "# End: Segment\n" );
            
            // Append the #End keywords
            Append_String_to_File( this->output_to_file, this->filename );
            
            // reset output string buffer
            this->output_to_file = "";  
            
            // Increment the n_segments after succesfully appending the segment body to the file
            increment_n_segments(); 
        }
        catch( ... )
        {
            spirit_rethrow( fmt::format("Failed to write OVF file \"{}\".", this->filename) );
        }
    }
} // end of namespace IO
