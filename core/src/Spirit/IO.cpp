#include <Spirit/Chain.h>
#include <Spirit/Configurations.h>
#include <Spirit/IO.h>
#include <Spirit/State.h>

#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <data/State.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/IO.hpp>
#include <io/OVF_File.hpp>
#include <memory>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>
#include <utility/Version.hpp>

#include <fmt/format.h>

#include <memory>
#include <string>

// Helper function
std::string Get_Extension( const char * file )
{
    std::string filename( file );
    std::string::size_type n = filename.rfind( '.' );
    if( n != std::string::npos )
        return filename.substr( n );
    else
        return std::string( "" );
}

/*----------------------------------------------------------------------------------------------- */
/*--------------------------------- From Config File -------------------------------------------- */
/*----------------------------------------------------------------------------------------------- */

int IO_System_From_Config( State * state, const char * file, int idx_image, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    // Create System (and lock it)
    std::shared_ptr<State::system_t> system = IO::Spin_System_from_Config( std::string( file ) );
    system->lock();

    // Filter for unacceptable differences to other systems in the chain
    for( int i = 0; i < chain->noi; ++i )
    {
        if( chain->images[i]->nos != system->nos )
            return 0;
        // Currently the SettingsWidget does not support different images being isotropic AND
        // anisotropic at the same time
        if( chain->images[i]->hamiltonian->Name() != system->hamiltonian->Name() )
            return 0;
    }

    // Set System
    image->lock();
    try
    {
        *image = *system;
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image, idx_chain );
    }
    image->unlock();

    // Initial configuration
    scalar defaultPos[3]  = { 0, 0, 0 };
    scalar defaultRect[3] = { -1, -1, -1 };
    Configuration_Random( state, defaultPos, defaultRect, -1, -1, false, false, idx_image, idx_chain );

    // Return success
    return 1;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

/*----------------------------------------------------------------------------------------------- */
/*------------------------------------- Geometry ------------------------------------------------ */
/*----------------------------------------------------------------------------------------------- */

void IO_Positions_Write(
    State * state, const char * filename, int format, const char * comment, int idx_image, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    // Write the data
    image->lock();
    try
    {
        if( Get_Extension( filename ) != ".ovf" )
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
                 fmt::format(
                     "The file \"{}\" is written in OVF format but has different extension. "
                     "It is recommend to use the appropriate \".ovf\" extension",
                     filename ),
                 idx_image, idx_chain );

        // Helper variables
        const auto & geometry = image->hamiltonian->get_geometry();
        auto fileformat       = (IO::VF_FileFormat)format;

        switch( fileformat )
        {
            case IO::VF_FileFormat::OVF_BIN:
            case IO::VF_FileFormat::OVF_BIN4:
            case IO::VF_FileFormat::OVF_BIN8:
            case IO::VF_FileFormat::OVF_TEXT:
            case IO::VF_FileFormat::OVF_CSV:
            {
                auto segment = IO::OVF_Segment( image->hamiltonian->get_geometry() );

                std::string title   = fmt::format( "SPIRIT Version {}", Utility::version_full );
                segment.title       = strdup( title.c_str() );
                segment.comment     = strdup( comment );
                segment.valuedim    = 3;
                segment.valuelabels = strdup( "position_x position_y position_z" );
                segment.valueunits  = strdup( "none none none" );

                // Open and write
                IO::OVF_File( filename ).write_segment( segment, geometry.positions[0].data(), format );

                Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
                     fmt::format( "Wrote positions to file \"{}\" in {} format", filename, str( fileformat ) ),
                     idx_image, idx_chain );

                break;
            }
            default:
            {
                spirit_throw(
                    Utility::Exception_Classifier::Bad_File_Content, Utility::Log_Level::Error,
                    fmt::format( "Invalid file format index {}", format ) );
            }
        }
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image, idx_chain );
    }
    image->unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

/*----------------------------------------------------------------------------------------------- */
/*-------------------------------------- Images ------------------------------------------------- */
/*----------------------------------------------------------------------------------------------- */

int IO_N_Images_In_File( State * state, const char * filename, int idx_image, int idx_chain ) noexcept
try
{
    auto file = IO::OVF_File( filename );

    if( file.is_ovf )
        return file.n_segments;
    else
    {
        Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
             fmt::format( "File \"{}\" is not OVF. Cannot measure number of images.", filename ), idx_image,
             idx_chain );
        return -1;
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return -1;
}

void IO_Image_Read(
    State * state, const char * filename, int idx_image_infile, int idx_image_inchain, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image_inchain, idx_chain );

    image->lock();

    try
    {
        const std::string extension = Get_Extension( filename );

        // Helper variables
        auto & system_state = *image->state;
        // TODO: eliminate this copy
        auto geometry = image->hamiltonian->get_geometry();

        // Open
        auto file = IO::OVF_File( filename, true );

        if( !file.is_ovf )
        {
            Log( Utility::Log_Level::Error, Utility::Log_Sender::API,
                 fmt::format(
                     "File \"{}\" does not seem to be in valid OVF format. Message: {}. "
                     "Will try to read as data column text format file.",
                     filename, file.latest_message() ),
                 idx_image_inchain, idx_chain );

            IO::Read_NonOVF_System_Configuration( system_state, geometry, image->nos, idx_image_infile, filename );
            image->hamiltonian->set_geometry( geometry );
            image->unlock();
            return;
        }

        // Segment header
        auto segment = IO::OVF_Segment();

        // Read header
        file.read_segment_header( idx_image_infile, segment );

        ////////////////////////////////////////////////////////
        // TODO: CHECK GEOMETRY AND WARN IF IT DOES NOT MATCH //
        ////////////////////////////////////////////////////////

        if( segment.N < image->nos )
        {
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
                 fmt::format(
                     "OVF file \"{}\": segment {}/{} contains only {} spins while the system contains {}.", filename,
                     idx_image_infile + 1, file.n_segments, segment.N, image->nos ),
                 idx_image_inchain, idx_chain );
            segment.N = std::min( segment.N, image->nos );
        }
        else if( segment.N > image->nos )
        {
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
                 fmt::format(
                     "OVF file \"{}\": segment {}/{} contains {} spins while the system contains only {}. "
                     "Reading only part of the segment data.",
                     filename, idx_image_infile + 1, file.n_segments, segment.N, image->nos ),
                 idx_image_inchain, idx_chain );
            segment.N = std::min( segment.N, image->nos );
        }

        if( segment.valuedim != IO::State::valuedim )
        {
            spirit_throw(
                Utility::Exception_Classifier::Bad_File_Content, Utility::Log_Level::Error,
                fmt::format(
                    "Segment {}/{} in OVF file \"{}\" should have {} columns, but only has {}. Will not read.",
                    idx_image_infile + 1, file.n_segments, filename, IO::State::valuedim, segment.valuedim ) );
        }

        // Read data
        file.read_segment_data( idx_image_infile, segment, system_state[0].data() );
        using Engine::Field;
        using Engine::get;

        vectorfield & spins = get<Field::Spin>( system_state );
        for( std::size_t ispin = 0; ispin < spins.size(); ++ispin )
        {
            if( spins[ispin].norm() < 1e-5 )
            {
                spins[ispin] = { 0, 0, 1 };
// In case of spin vector close to zero we have a vacancy
#ifdef SPIRIT_ENABLE_DEFECTS
                geometry.atom_types[ispin] = -1;
#endif
            }
            else
                spins[ispin].normalize();
        }

#ifdef SPIRIT_ENABLE_DEFECTS
        image->hamiltonian->set_geometry( geometry );
#endif

        Log( Utility::Log_Level::Info, Utility::Log_Sender::API, fmt::format( "Read image from file \"{}\"", filename ),
             idx_image_inchain, idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image_inchain, idx_chain );
    }
    image->unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image_inchain, idx_chain );
}

void IO_Image_Write(
    State * state, const char * filename, int format, const char * comment, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    // Write the data
    image->lock();

    try
    {
        if( Get_Extension( filename ) != ".ovf" )
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
                 fmt::format(
                     "The file \"{}\" is written in OVF format but has different extension. "
                     "It is recommend to use the appropriate \".ovf\" extension",
                     filename ),
                 idx_image, idx_chain );

        switch( auto fileformat = static_cast<IO::VF_FileFormat>( format ) )
        {
            case IO::VF_FileFormat::OVF_BIN:
            case IO::VF_FileFormat::OVF_BIN4:
            case IO::VF_FileFormat::OVF_BIN8:
            case IO::VF_FileFormat::OVF_TEXT:
            case IO::VF_FileFormat::OVF_CSV:
            {
                auto segment              = IO::OVF_Segment( image->hamiltonian->get_geometry() );
                const auto & system_state = *image->state;

                std::string title   = fmt::format( "SPIRIT Version {}", Utility::version_full );
                segment.title       = strdup( title.c_str() );
                segment.comment     = strdup( comment );
                segment.valuedim    = IO::State::valuedim;
                segment.valuelabels = strdup( IO::State::valuelabels.data() );
                segment.valueunits  = strdup( IO::State::valueunits.data() );

                // Open and write
                const auto buffer = IO::State::Buffer( system_state );
                IO::OVF_File( filename ).write_segment( segment, buffer.data(), format );

                Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
                     fmt::format( "Wrote spins to file \"{}\" in {} format", filename, str( fileformat ) ), idx_image,
                     idx_chain );

                break;
            }
            default:
            {
                spirit_throw(
                    Utility::Exception_Classifier::Bad_File_Content, Utility::Log_Level::Error,
                    fmt::format( "Invalid file format index {}", format ) );
            }
        }
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image, idx_chain );
    }
    image->unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void IO_Image_Append(
    State * state, const char * filename, int format, const char * comment, int idx_image, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    // Write the data
    image->lock();

    try
    {
        auto fileformat = static_cast<IO::VF_FileFormat>( format );
        switch( fileformat )
        {
            case IO::VF_FileFormat::OVF_BIN:
            case IO::VF_FileFormat::OVF_BIN4:
            case IO::VF_FileFormat::OVF_BIN8:
            case IO::VF_FileFormat::OVF_TEXT:
            case IO::VF_FileFormat::OVF_CSV:
            {
                // Open
                auto file = IO::OVF_File( filename );

                // Check if the file was OVF
                if( file.found && !file.is_ovf )
                {
                    spirit_throw(
                        Utility::Exception_Classifier::Bad_File_Content, Utility::Log_Level::Error,
                        fmt::format( "Cannot append to non-OVF file \"{}\"", filename ) );
                }

                auto segment              = IO::OVF_Segment( image->hamiltonian->get_geometry() );
                const auto & system_state = *image->state;

                std::string title   = fmt::format( "SPIRIT Version {}", Utility::version_full );
                segment.title       = strdup( title.c_str() );
                segment.comment     = strdup( comment );
                segment.valuedim    = IO::State::valuedim;
                segment.valuelabels = strdup( IO::State::valuelabels.data() );
                segment.valueunits  = strdup( IO::State::valueunits.data() );

                // Open and write
                const auto buffer = IO::State::Buffer( system_state );
                IO::OVF_File( filename ).write_segment( segment, buffer.data(), format );

                break;
            }
            default:
            {
                spirit_throw(
                    Utility::Exception_Classifier::Bad_File_Content, Utility::Log_Level::Error,
                    fmt::format( "Invalid file format index {}", format ) );
            }
        }
        Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
             fmt::format( "Appended spins to file \"{}\" in {} format", filename, str( fileformat ) ), idx_image,
             idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image, idx_chain );
    }
    image->unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

/*----------------------------------------------------------------------------------------------- */
/*-------------------------------------- Chains ------------------------------------------------- */
/*----------------------------------------------------------------------------------------------- */

void IO_Chain_Read(
    State * state, const char * filename, int start_image_infile, int end_image_infile, int insert_idx,
    int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, insert_idx, idx_chain );

    chain->lock();
    bool success = false;

    // Read the data
    try
    {
        const std::string extension = Get_Extension( filename );

        auto & images = chain->images;
        int noi       = chain->noi;

        if( insert_idx < 0 )
            insert_idx = 0;

        if( insert_idx > noi )
        {
            Log( Utility::Log_Level::Error, Utility::Log_Sender::API,
                 fmt::format(
                     "IO_Chain_Read: Tried to start reading chain on invalid index"
                     "(insert_idx={}, but chain has {} images)",
                     insert_idx, noi ),
                 insert_idx, idx_chain );
        }

        // Open
        IO::OVF_File file( filename, true );

        if( file.is_ovf )
        {
            int noi_infile = file.n_segments;

            if( start_image_infile < 0 )
                start_image_infile = 0;

            if( end_image_infile < 0 )
                end_image_infile = noi_infile - 1;

            // Check if the ending image is valid otherwise set it to the last image infile
            if( end_image_infile < start_image_infile || end_image_infile >= noi_infile )
            {
                Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
                     fmt::format(
                         "IO_Chain_Read: specified invalid reading range (start_image_infile={}, end_image_infile={})."
                         " Set to read entire file \"{}\" ({} images).",
                         start_image_infile, end_image_infile, filename, noi_infile ),
                     insert_idx, idx_chain );
                end_image_infile = noi_infile - 1;
            }

            if( start_image_infile >= noi_infile )
            {
                spirit_throw(
                    Utility::Exception_Classifier::Bad_File_Content, Utility::Log_Level::Error,
                    fmt::format(
                        "Specified starting index {}, but file \"{}\" contains only {} images.", start_image_infile,
                        filename, noi_infile ) );
            }

            // If the idx of the starting image is valid
            int noi_to_read = end_image_infile - start_image_infile + 1;

            int noi_to_add = noi_to_read - ( noi - insert_idx );

            // Add the images if you need that
            if( noi_to_add > 0 )
            {
                chain->unlock();
                Chain_Image_to_Clipboard( state, noi - 1 );
                Chain_Set_Length( state, noi + noi_to_add );
                chain->lock();
            }

            // Read the images
            for( int i = insert_idx; i < noi_to_read; i++ )
            {
                auto & system_state = *images[i]->state;
                // Segment header
                auto segment = IO::OVF_Segment();
#ifdef SPIRIT_ENABLE_DEFECTS
                // TODO: eluminate this copy
                auto geometry = images[i]->hamiltonian->get_geometry();
#endif
                // Read header
                file.read_segment_header( start_image_infile, segment );

                ////////////////////////////////////////////////////////
                // TODO: CHECK GEOMETRY AND WARN IF IT DOES NOT MATCH //
                ////////////////////////////////////////////////////////

                if( segment.N < image->nos )
                {
                    Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
                         fmt::format(
                             "OVF file \"{}\": segment {}/{} contains only {} spins while the system contains {}.",
                             filename, start_image_infile + 1, file.n_segments, segment.N, image->nos ),
                         i, idx_chain );
                    segment.N = std::min( segment.N, image->nos );
                }
                else if( segment.N > image->nos )
                {
                    Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
                         fmt::format(
                             "OVF file \"{}\": segment {}/{} contains {} spins while the system contains only {}. "
                             "Reading only part of the segment data.",
                             filename, start_image_infile + 1, file.n_segments, segment.N, image->nos ),
                         i, idx_chain );
                    segment.N = std::min( segment.N, image->nos );
                }

                if( segment.valuedim != IO::State::valuedim )
                {
                    spirit_throw(
                        Utility::Exception_Classifier::Bad_File_Content, Utility::Log_Level::Error,
                        fmt::format(
                            "Segment {}/{} in OVF file \"{}\" should have {} columns, but only has {}. Will not read.",
                            start_image_infile + 1, file.n_segments, filename, IO::State::valuedim,
                            segment.valuedim ) );
                }

                // Read data
                file.read_segment_data( start_image_infile, segment, system_state[0].data() );
                auto & spins = Engine::get<Engine::Field::Spin>( system_state );
                for( unsigned int ispin = 0; ispin < spins.size(); ++ispin )
                {
                    if( spins[ispin].norm() < 1e-5 )
                    {
                        spins[ispin] = { 0, 0, 1 };
// In case of spin vector close to zero we have a vacancy
#ifdef SPIRIT_ENABLE_DEFECTS
                        geometry.atom_types[ispin] = -1;
#endif
                    }
                    else
                        spins[ispin].normalize();
                }

                start_image_infile++;
#ifdef SPIRIT_ENABLE_DEFECTS
                images[i]->hamiltonian->set_geometry( geometry );
#endif
            }

            success = true;
        }
        else
        {
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
                 fmt::format( "IO_Chain_Read: File \"{}\" seems to not be OVF. Trying to read column data", filename ),
                 insert_idx, idx_chain );

            int noi_to_add  = 0;
            int noi_to_read = 0;

            IO::Check_NonOVF_Chain_Configuration(
                chain, filename, start_image_infile, end_image_infile, insert_idx, noi_to_add, noi_to_read, idx_chain );
            // Add the images if you need that
            if( noi_to_add > 0 )
            {
                chain->unlock();
                Chain_Image_to_Clipboard( state, noi - 1 );
                Chain_Set_Length( state, noi + noi_to_add );
                chain->lock();
            }

            // Read the images
            if( noi_to_read > 0 )
            {
                for( int i = insert_idx; i < noi_to_read; i++ )
                {
                    // TODO: eliminate this copy
                    auto geometry = chain->images[i]->hamiltonian->get_geometry();
                    IO::Read_NonOVF_System_Configuration(
                        *chain->images[i]->state, geometry, chain->images[i]->nos, start_image_infile, filename );
                    chain->images[i]->hamiltonian->set_geometry( geometry );
                    start_image_infile++;
                }
                success = true;
            }
        }
    }
    catch( ... )
    {
        spirit_handle_exception_api( insert_idx, idx_chain );
    }

    chain->unlock();

    if( success )
    {
        // Update llg simulation information array size
        for( int i = state->method_image.size(); i < chain->noi; ++i )
            state->method_image.push_back( std::shared_ptr<Engine::Method>() );

        // Update state
        State_Update( state );

        // Update array lengths
        Chain_Setup_Data( state, idx_chain );

        Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
             fmt::format( "IO_Chain_Read: Read chain from file \"{}\"", filename ), insert_idx, idx_chain );
    }
}
catch( ... )
{
    spirit_handle_exception_api( insert_idx, idx_chain );
}

void IO_Chain_Write( State * state, const char * filename, int format, const char * comment, int idx_chain ) noexcept
try
{
    int idx_image = 0;

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    // Read the data
    chain->lock();
    try
    {
        auto fileformat = (IO::VF_FileFormat)format;

        switch( fileformat )
        {
            case IO::VF_FileFormat::OVF_BIN:
            case IO::VF_FileFormat::OVF_BIN4:
            case IO::VF_FileFormat::OVF_BIN8:
            case IO::VF_FileFormat::OVF_TEXT:
            case IO::VF_FileFormat::OVF_CSV:
            {
                auto & images = chain->images;

                // Open
                auto file = IO::OVF_File( filename );

                auto segment              = IO::OVF_Segment( image->hamiltonian->get_geometry() );
                const auto & system_state = *image->state;

                std::string title       = fmt::format( "SPIRIT Version {}", Utility::version_full );
                segment.title           = strdup( title.c_str() );
                std::string comment_str = "";
                comment_str             = fmt::format( "Image {} of {}. {}", 1, chain->noi, comment );
                segment.comment         = strdup( comment_str.c_str() );
                segment.valuedim        = IO::State::valuedim;
                segment.valuelabels     = strdup( IO::State::valuelabels.data() );
                segment.valueunits      = strdup( IO::State::valueunits.data() );

                // Open and write
                const IO::State::Buffer buffer( system_state );
                file.write_segment( segment, buffer.data(), format );

                for( int i = 1; i < chain->noi; i++ )
                {
                    comment_str     = fmt::format( "Image {} of {}. {}", i + 1, chain->noi, comment );
                    segment.comment = strdup( comment_str.c_str() );
                    file.append_segment( segment, IO::State::Buffer( *images[i]->state ).data(), int( fileformat ) );
                }

                break;
            }
            default:
            {
                spirit_throw(
                    Utility::Exception_Classifier::Bad_File_Content, Utility::Log_Level::Error,
                    fmt::format( "Invalid file format index {}", format ) );
            }
        }

        Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
             fmt::format( "Wrote chain to file \"{}\" in {} format", filename, str( fileformat ) ), idx_image,
             idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image, idx_chain );
    }
    chain->unlock();
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

void IO_Chain_Append( State * state, const char * filename, int format, const char * comment, int idx_chain ) noexcept
try
{
    int idx_image = 0;

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    // Read the data
    chain->lock();
    try
    {
        auto fileformat = (IO::VF_FileFormat)format;
        switch( fileformat )
        {
            case IO::VF_FileFormat::OVF_BIN:
            case IO::VF_FileFormat::OVF_BIN4:
            case IO::VF_FileFormat::OVF_BIN8:
            case IO::VF_FileFormat::OVF_TEXT:
            case IO::VF_FileFormat::OVF_CSV:
            {
                // Open
                auto file = IO::OVF_File( filename );

                // Check if the file was OVF
                if( file.found && !file.is_ovf )
                {
                    spirit_throw(
                        Utility::Exception_Classifier::Bad_File_Content, Utility::Log_Level::Error,
                        fmt::format( "Cannot append to non-OVF file \"{}\"", filename ) );
                }

                auto segment        = IO::OVF_Segment( image->hamiltonian->get_geometry() );
                auto & system_state = *image->state;

                std::string title       = fmt::format( "SPIRIT Version {}", Utility::version_full );
                segment.title           = strdup( title.c_str() );
                std::string comment_str = "";

                comment_str         = fmt::format( "Image {} of {}. {}", 0, chain->noi, comment );
                segment.comment     = strdup( comment_str.c_str() );
                segment.valuedim    = IO::State::valuedim;
                segment.valuelabels = strdup( IO::State::valuelabels.data() );
                segment.valueunits  = strdup( IO::State::valueunits.data() );

                // Write
                for( int i = 0; i < chain->noi; i++ )
                {
                    comment_str     = fmt::format( "Image {} of {}. {}", i, chain->noi, comment );
                    segment.comment = strdup( comment_str.c_str() );
                    const IO::State::Buffer buffer( system_state );
                    file.write_segment( segment, buffer.data(), int( fileformat ) );
                }

                break;
            }
            default:
            {
                spirit_throw(
                    Utility::Exception_Classifier::Bad_File_Content, Utility::Log_Level::Error,
                    fmt::format( "Invalid file format index {}", format ) );
            }
        }
        Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
             fmt::format( "Wrote chain to file \"{}\" in {} format", filename, str( fileformat ) ), idx_image,
             idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image, idx_chain );
    }
    chain->unlock();
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

/*----------------------------------------------------------------------------------------------- */
/*--------------------------------------- Data -------------------------------------------------- */
/*----------------------------------------------------------------------------------------------- */

// Interactions Pairs
void IO_Image_Write_Neighbours_Exchange( State * state, const char * file, int idx_image, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    // Write the data
    if( const auto * cache = image->hamiltonian->cache<Engine::Spin::Interaction::Exchange>(); cache != nullptr )
        IO::Write_Neighbours_Exchange( *cache, std::string( file ) );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void IO_Image_Write_Neighbours_DMI( State * state, const char * file, int idx_image, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    // Write the data
    if( const auto * cache = image->hamiltonian->cache<Engine::Spin::Interaction::DMI>(); cache != nullptr )
        IO::Write_Neighbours_DMI( *cache, std::string( file ) );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

// IO_Energies_Spins_Save
void IO_Image_Write_Energy_per_Spin(
    State * state, const char * filename, int format, int idx_image, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    // Write the data
    image->lock();

    auto & system = *image;
    auto & spins  = *image->state;

    // Gather the data
    system.UpdateEnergy();
    system.hamiltonian->Energy_Contributions_per_Spin( spins, image->E.per_interaction_per_spin );
    int dataperspin = 1 + image->E.per_interaction_per_spin.size();
    int datasize    = dataperspin * system.nos;
    scalarfield data( datasize, 0 );
    for( int ispin = 0; ispin < system.nos; ++ispin )
    {
        scalar E_spin = 0;
        int j         = 1;
        for( auto & contribution : image->E.per_interaction_per_spin )
        {
            E_spin += contribution.second[ispin];
            data[ispin * dataperspin + j] = contribution.second[ispin];
            ++j;
        }
        data[ispin * dataperspin] = E_spin;
    }

    try
    {
        if( Get_Extension( filename ) != ".ovf" )
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
                 fmt::format(
                     "The file \"{}\" is written in OVF format but has different extension. "
                     "It is recommend to use the appropriate \".ovf\" extension",
                     filename ),
                 idx_image, idx_chain );

        auto fileformat = (IO::VF_FileFormat)format;
        switch( fileformat )
        {
            case IO::VF_FileFormat::OVF_BIN:
            case IO::VF_FileFormat::OVF_BIN4:
            case IO::VF_FileFormat::OVF_BIN8:
            case IO::VF_FileFormat::OVF_TEXT:
            case IO::VF_FileFormat::OVF_CSV:
            {
                auto segment = IO::OVF_Segment( image->hamiltonian->get_geometry() );

                std::string title   = fmt::format( "SPIRIT Version {}", Utility::version_full );
                segment.title       = strdup( title.c_str() );
                std::string comment = fmt::format( "Energy per spin. Total={}meV", system.E.total );
                for( auto & contribution : system.E.per_interaction )
                    comment += fmt::format( ", {}={}meV", contribution.first, contribution.second );
                segment.comment  = strdup( comment.c_str() );
                segment.valuedim = 1 + system.E.per_interaction.size();

                std::string valuelabels = "Total";
                std::string valueunits  = "meV";
                for( auto & pair : system.E.per_interaction )
                {
                    valuelabels += fmt::format( " {}", pair.first );
                    valueunits += " meV";
                }
                segment.valuelabels = strdup( valuelabels.c_str() );

                // Open
                auto file = IO::OVF_File( filename );
                // Write
                file.write_segment( segment, data.data(), format );

                Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
                     fmt::format( "Wrote spins to file \"{}\" in {} format", filename, str( fileformat ) ), idx_image,
                     idx_chain );

                break;
            }
            default:
            {
                spirit_throw(
                    Utility::Exception_Classifier::Bad_File_Content, Utility::Log_Level::Error,
                    fmt::format( "Invalid file format index {}", format ) );
            }
        }
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image, idx_chain );
    }

    image->unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

// IO_Energy_Spins_Save
void IO_Image_Write_Energy( State * state, const char * file, int idx_image, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    // Write the data
    IO::Write_Image_Energy( image->E, image->hamiltonian->get_geometry(), std::string( file ) );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

// IO_Energies_Save
void IO_Chain_Write_Energies( State * state, const char * file, int idx_chain ) noexcept
try
{
    int idx_image = -1;

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    // Write the data
    IO::Write_Chain_Energies( *chain, idx_chain, std::string( file ) );
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

// IO_Energies_Interpolated_Save
void IO_Chain_Write_Energies_Interpolated( State * state, const char * file, int idx_chain ) noexcept
try
{
    int idx_image = -1;

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    // Write the data
    IO::Write_Chain_Energies_Interpolated( *chain, std::string( file ) );
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

/*----------------------------------------------------------------------------------------------- */
/*-------------------------------------- Eigenmodes --------------------------------------------- */
/*----------------------------------------------------------------------------------------------- */

void IO_Eigenmodes_Read( State * state, const char * filename, int idx_image_inchain, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image_inchain, idx_chain );

    // Read the data
    image->lock();
    try
    {
        const std::string extension = Get_Extension( filename );

        auto & spins = *image->state;

        // Open
        auto file = IO::OVF_File( filename, true );

        if( !file.is_ovf )
        {
            spirit_throw(
                Utility::Exception_Classifier::Bad_File_Content, Utility::Log_Level::Error,
                fmt::format(
                    "IO_Eigenmodes_Read only supports the OVF file format, "
                    "but \"{}\" does not seem to be valid OVF format. Message: {}",
                    filename, file.latest_message() ) );
        }

        // If the modes buffer's size is not the same as the n_segments then resize
        int n_eigenmodes = file.n_segments - 1;
        if( image->modes.size() != n_eigenmodes )
        {
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::IO,
                 fmt::format(
                     "Resizing eigenmode buffer because the number of modes in the OVF file ({}) "
                     "is different from the buffer size ({})",
                     n_eigenmodes, image->modes.size() ) );
            image->modes.resize( n_eigenmodes );
            image->eigenvalues.resize( n_eigenmodes );
        }

        Log( Utility::Log_Level::Debug, Utility::Log_Sender::IO,
             fmt::format( "Reading {} eigenmodes from file \"{}\"", n_eigenmodes, filename ) );

        ////////// Read in the eigenvalues
        // Segment header
        auto segment = IO::OVF_Segment();
        // Read header
        file.read_segment_header( 0, segment );
        // Check
        if( segment.valuedim != 1 )
        {
            spirit_throw(
                Utility::Exception_Classifier::Bad_File_Content, Utility::Log_Level::Error,
                fmt::format(
                    "Eigenvalue segment of OVF file \"{}\" should have 1 column, but has {}. Will not read.", filename,
                    segment.valuedim ) );
        }
        // Read data
        file.read_segment_data( 0, segment, image->eigenvalues.data() );

        ////////// Read in the eigenmodes
        for( int idx = 0; idx < n_eigenmodes; idx++ )
        {
            // If the mode buffer is created by resizing then it needs to be allocated
            if( !image->modes[idx].has_value() )
                image->modes[idx].emplace( vectorfield( spins.size(), Vector3{ 1, 0, 0 } ) );

            // Read header
            file.read_segment_header( idx + 1, segment );

            ////////////////////////////////////////////////////////
            // TODO: CHECK GEOMETRY AND WARN IF IT DOES NOT MATCH //
            ////////////////////////////////////////////////////////

            if( segment.N < image->nos )
            {
                Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
                     fmt::format(
                         "OVF file \"{}\": segment {}/{} contains only {} vectors while the system contains {}.",
                         filename, idx + 1, file.n_segments, segment.N, image->nos ),
                     idx_image_inchain, idx_chain );
                segment.N = std::min( segment.N, image->nos );
            }
            else if( segment.N > image->nos )
            {
                Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
                     fmt::format(
                         "OVF file \"{}\": segment {}/{} contains {} vectors while the system contains only {}. "
                         "Reading only part of the segment data.",
                         filename, idx + 1, file.n_segments, segment.N, image->nos ),
                     idx_image_inchain, idx_chain );
                segment.N = std::min( segment.N, image->nos );
            }

            if( segment.valuedim != 3 )
            {
                spirit_throw(
                    Utility::Exception_Classifier::Bad_File_Content, Utility::Log_Level::Error,
                    fmt::format(
                        "OVF file \"{}\" should have 3 columns, but only has {}. Will not read.", filename,
                        segment.valuedim ) );
            }

            // Read data
            file.read_segment_data( idx + 1, segment, ( *image->modes[idx] )[0].data() );
        }

        // If the modes vector was reseized adjust the n_modes value
        if( image->modes.size() != image->ema_parameters->n_modes )
            image->ema_parameters->n_modes = image->modes.size();

        // Print the first eigenvalues
        int n_log_eigenvalues = ( n_eigenmodes > 50 ) ? 50 : n_eigenmodes;

        Log( Utility::Log_Level::Info, Utility::Log_Sender::IO,
             fmt::format(
                 "The first {} eigenvalues are: {}", n_log_eigenvalues,
                 fmt::join( image->eigenvalues.begin(), image->eigenvalues.begin() + n_log_eigenvalues, ", " ) ) );
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image_inchain, idx_chain );
    }
    image->unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image_inchain, idx_chain );
}

void IO_Eigenmodes_Write(
    State * state, const char * filename, int format, const char * comment, int idx_image, int idx_chain ) noexcept
try
{

    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    // Write the data
    image->lock();
    try
    {
        if( Get_Extension( filename ) != ".ovf" )
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
                 fmt::format(
                     "The file \"{}\" is written in OVF format but has different extension. "
                     "It is recommend to use the appropriate \".ovf\" extension",
                     filename ),
                 idx_image, idx_chain );

        auto fileformat = (IO::VF_FileFormat)format;
        switch( fileformat )
        {
            case IO::VF_FileFormat::OVF_BIN:
            case IO::VF_FileFormat::OVF_BIN4:
            case IO::VF_FileFormat::OVF_BIN8:
            case IO::VF_FileFormat::OVF_TEXT:
            case IO::VF_FileFormat::OVF_CSV:
            {
                auto segment      = IO::OVF_Segment( image->hamiltonian->get_geometry() );
                std::string title = fmt::format( "SPIRIT Version {}", Utility::version_full );
                segment.title     = strdup( title.c_str() );

                // Determine number of modes
                int n_modes = 0;
                for( int i = 0; i < image->modes.size(); i++ )
                    if( image->modes[i].has_value() )
                        ++n_modes;

                // Open
                auto file = IO::OVF_File( filename );

                /////// Eigenspectrum
                std::string comment_str = fmt::format( "{}\n# Desc: eigenspectrum of {} eigenmodes", comment, n_modes );
                segment.comment         = strdup( comment_str.c_str() );
                segment.valuedim        = 1;
                segment.valuelabels     = strdup( "eigenvalue" );
                segment.valueunits      = strdup( "meV" );
                segment.meshunit        = strdup( "none" );
                segment.n_cells[0]      = n_modes;
                segment.n_cells[1]      = 1;
                segment.n_cells[2]      = 1;
                segment.N               = n_modes;
                segment.step_size[0]    = 0;
                segment.step_size[1]    = 0;
                segment.step_size[2]    = 0;
                segment.bounds_max[0]   = 0;
                segment.bounds_max[1]   = 0;
                segment.bounds_max[2]   = 0;

                // Write
                file.write_segment( segment, image->eigenvalues.data(), format );

                /////// Eigenmodes
                segment             = IO::OVF_Segment( image->hamiltonian->get_geometry() );
                segment.valuedim    = 3;
                segment.valuelabels = strdup( "mode_x mode_y mode_z" );
                segment.valueunits  = strdup( "none none none" );
                for( int i = 0; i < n_modes; i++ )
                {
                    comment_str = fmt::format(
                        "{}\n# Desc: eigenmode {}/{}, eigenvalue = {}", comment, i + 1, n_modes,
                        image->eigenvalues[i] );
                    segment.comment = strdup( comment_str.c_str() );

                    // Write
                    file.append_segment( segment, ( *image->modes[i] )[0].data(), format );
                }

                Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
                     fmt::format( "Wrote eigenmodes to file \"{}\" in {} format", filename, str( fileformat ) ),
                     idx_image, idx_chain );

                break;
            }
            default:
            {
                spirit_throw(
                    Utility::Exception_Classifier::Bad_File_Content, Utility::Log_Level::Error,
                    fmt::format( "Invalid file format index {}", format ) );
            }
        }
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image, idx_chain );
    }
    image->unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}
