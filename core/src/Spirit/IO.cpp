#include <Spirit/State.h>
#include <Spirit/IO.h>
#include <Spirit/Chain.h>
#include <Spirit/Configurations.h>

#include <data/State.hpp>
#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <io/IO.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/OVF_File.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

#include <fmt/format.h>

#include <memory>
#include <string>

/////// TODO: make use of file format specifications
/////// TODO: implement remaining functions

// helper function
std::string Get_Extension( const char *file )
{
    std::string filename(file);
    std::string::size_type n = filename.rfind('.');
    if ( n != std::string::npos ) return filename.substr(n); 
    else return std::string(""); 
}

/*----------------------------------------------------------------------------------------------- */
/*--------------------------------- From Config File -------------------------------------------- */
/*----------------------------------------------------------------------------------------------- */

int IO_System_From_Config(State * state, const char * file, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        // Create System (and lock it)
        std::shared_ptr<Data::Spin_System> system = IO::Spin_System_from_Config(std::string(file));
        system->Lock();
        
        // Filter for unacceptable differences to other systems in the chain
        for (int i = 0; i < chain->noi; ++i)
        {
            if (state->active_chain->images[i]->nos != system->nos) return 0;
            // Currently the SettingsWidget does not support different images being isotropic AND 
            // anisotropic at the same time
            if (state->active_chain->images[i]->hamiltonian->Name() != system->hamiltonian->Name()) 
                return 0;
        }
        
        // Set System
        image->Lock();
        try
        {
            *image = *system;
        }
        catch( ... )
        {
            spirit_handle_exception_api(idx_image, idx_chain);
        }
        image->Unlock();
        
        // Initial configuration
        float defaultPos[3] = {0,0,0}; 
        float defaultRect[3] = {-1,-1,-1};
        Configuration_Random(state, defaultPos, defaultRect, -1, -1, false, false, idx_image, idx_chain);
        
        // Return success
        return 1;
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
        return 0;
    }
}

/*----------------------------------------------------------------------------------------------- */
/*------------------------------------- Geometry ------------------------------------------------ */
/*----------------------------------------------------------------------------------------------- */

void IO_Positions_Write( State * state, const char *file, int format, 
                         const char *comment, int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        // Write the data
        image->Lock();
        try
        {
            IO::Write_Positions( *image->geometry, file, IO::VF_FileFormat(format), comment, false );
        }
        catch( ... )
        {
            spirit_handle_exception_api(idx_image, idx_chain);
        }
        image->Unlock();
        
        Log( Utility::Log_Level::Info, Utility::Log_Sender::API, fmt::format( "Wrote positions to file "
                "{} with format {}", file, format ), idx_image, idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}


/*----------------------------------------------------------------------------------------------- */
/*-------------------------------------- Images ------------------------------------------------- */
/*----------------------------------------------------------------------------------------------- */

int IO_N_Images_In_File( State * state, const char *file, int idx_image, int idx_chain ) noexcept
{
    try
    {   
        // We choose not to fetch the corrent indices since it is not necessary for the behavior
        // of that specific function
        
        IO::File_OVF file_ovf( file );
       
        if ( file_ovf.is_OVF() )
        {
            return file_ovf.get_n_segments();
        } 
        else
        {
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
                 fmt::format( "File {} is not OVF. Cannot measure number of images.", file ), 
                 idx_image, idx_chain );
            return -1;
        }
    }
    catch( ... )
    {
        spirit_handle_exception_api( idx_image, idx_chain );
    }
}

void IO_Image_Read( State *state, const char *file, int idx_image_infile, 
                    int idx_image_inchain, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image_inchain, idx_chain, image, chain );
       
        image->Lock();

        try
        {
            const std::string extension = Get_Extension( file );
            
            // helper variables
            auto& spins = *image->spins;
            auto& geometry = *image->geometry;
            
            if ( extension == ".ovf" || extension == ".txt" || extension == ".csv" || 
                 extension == "" )
            {
                // Create an OVF object
                IO::File_OVF file_ovf( file );

                if ( file_ovf.is_OVF() ) 
                {
                    file_ovf.read_segment( spins, geometry, idx_image_infile );
                } 
                else
                {
                    Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
                         fmt::format("File {} is not OVF. Trying to read column data", file ), 
                         idx_image_inchain, idx_chain );
                    
                    IO::Read_NonOVF_Spin_Configuration( spins, image->nos, 
                                                        idx_image_infile, file ); 
                }

                Log( Utility::Log_Level::Info, Utility::Log_Sender::API, fmt::format( "Read "
                     "image from file {}", file ), idx_image_inchain, idx_chain );
            }
            else
            {
                Log( Utility::Log_Level::Error, Utility::Log_Sender::API,
                     fmt::format("File {} does not have a supported file extension", file ),
                     idx_image_inchain, idx_chain );
            }
        }
        catch( ... )
        {
            spirit_handle_exception_api(idx_image_inchain, idx_chain);
        }
        
        image->Unlock();

    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image_inchain, idx_chain);
    }
}

void IO_Image_Write( State *state, const char *file, int format, const char* comment, 
                     int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        // Write the data
        image->Lock();
        
        try
        {
            if ( Get_Extension(file) != ".ovf" )
                Log( Utility::Log_Level::Warning, Utility::Log_Sender::API, fmt::format( "The "
                     "file {} is in OVF format but has a different extension. It is recommend "
                     "to use the appropriate \".ovf\" extension", file ), idx_image, idx_chain );

            IO::Write_Spin_Configuration( *image->spins, *image->geometry, std::string( file ), 
                                          (IO::VF_FileFormat)format, std::string( comment ), 
                                          false );
            
            Log( Utility::Log_Level::Info, Utility::Log_Sender::API, fmt::format( "Wrote spins "
                 "to file {} with format {}", file, format ), idx_image, idx_chain );
        }
        catch( ... )
        {
            spirit_handle_exception_api(idx_image, idx_chain);
        }

        image->Unlock();
        
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

void IO_Image_Append( State *state, const char *file, int format, const char * comment, 
                      int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        // Write the data
        image->Lock();
        
        try
        {
            if ( Get_Extension(file) != ".ovf" )
            {
                Log( Utility::Log_Level::Error, Utility::Log_Sender::API, fmt::format( "The file "
                     "{} is in OVF format but has a different extension", file ), 
                     idx_image, idx_chain );
            }
            else
            {
                IO::Write_Spin_Configuration( *image->spins, *image->geometry, std::string( file ), 
                                              (IO::VF_FileFormat)format, std::string( comment ), 
                                              true );
                Log( Utility::Log_Level::Info, Utility::Log_Sender::API, fmt::format( "Appended "
                     "spins to file {} with format {}", file, format ), idx_image, idx_chain );
            }
        }
        catch( ... )
        {
            spirit_handle_exception_api(idx_image, idx_chain);
        }

        image->Unlock();

    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}


/*----------------------------------------------------------------------------------------------- */
/*-------------------------------------- Chains ------------------------------------------------- */
/*----------------------------------------------------------------------------------------------- */

void IO_Chain_Read( State *state, const char *file, int starting_image, 
                    int ending_image, int insert_idx, int idx_chain ) noexcept
{
    int idx_image = -1;

    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;

        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );

        // Read the data
        chain->Lock();
        
        bool success = false;
        
        try
        {
            const std::string extension = Get_Extension( file );  
           
            // helper variables
            auto& images = chain->images;
            int noi = chain->noi; 

            if ( insert_idx < 0 || insert_idx > noi )
            {

                Log( Utility::Log_Level::Error, Utility::Log_Sender::API,
                     fmt::format( "Invalid insert_idx {}. State has {} noi", insert_idx, noi ), 
                     insert_idx, idx_chain );
            }
            else if ( extension == ".ovf" || extension == ".txt" || 
                      extension == ".csv" || extension == "" )
            {
                // Create an OVF object
                IO::File_OVF file_ovf( file );
                
                if ( file_ovf.is_OVF() ) 
                {
                    int noi_infile = file_ovf.get_n_segments();
                   
                    // Check if the ending image is valid otherwise set it to the last image infile
                    if ( ending_image < starting_image || ending_image >= noi_infile )
                    {
                        ending_image = noi_infile - 1;
                        Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
                             fmt::format( "Invalid ending_image. Value was set to the last image "
                             "of the file"), starting_image, idx_chain );
                    }

                    // If the idx of the starting image is valid
                    if ( starting_image < noi_infile )
                    {
                        int noi_to_read = ending_image - starting_image + 1;
                       
                        int noi_to_add = noi_to_read - ( noi - insert_idx );
       
                        // Add the images if you need that
                        if ( noi_to_add > 0 ) 
                        { 
                            chain->Unlock();
                            Chain_Image_to_Clipboard( state, noi-1 );
                            for (int i=0; i<noi_to_add; i++) Chain_Push_Back( state );
                            chain->Lock(); 
                        } 

                        // Read the images
                        for (int i=insert_idx; i<noi_to_read; i++)
                        {
                            file_ovf.read_segment( *images[i]->spins, *images[i]->geometry, 
                                                   starting_image );
                            starting_image++;
                        }
                        
                        success = true;
                    }
                    else
                    {
                        Log( Utility::Log_Level::Error, Utility::Log_Sender::API,
                             fmt::format( "Invalid starting_idx. File {} has {} noi", file, 
                             noi_infile ), insert_idx, idx_chain );
                    }
                } 
                else
                {
                    Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
                         fmt::format( "File {} is not OVF. Trying to read column data", file ), 
                         insert_idx, idx_chain );
                   
                    //// TODO: Fix arguments - rename function in its source
                    //IO::Read_NonOVF_SpinChain_Configuration( spins, image->nos, 
                                                             //idx_image_infile, file ); 
                    success = true; 
                }
            }
            else
            {
                Log( Utility::Log_Level::Error, Utility::Log_Sender::API,
                     fmt::format( "File {} does not have a supported file extension", file ),
                     insert_idx, idx_chain );
            }
        }
        catch( ... )
        {
            spirit_handle_exception_api(idx_image, idx_chain);
        }
        
        chain->Unlock();
        
        if ( success )
        {
            // Update llg simulation information array size
            if ((int)state->method_image[idx_chain].size() < chain->noi)
            {
                for (int i=state->method_image[idx_chain].size(); i < chain->noi; ++i)
                    state->method_image[idx_chain].push_back( 
                        std::shared_ptr<Engine::Method>( ) );
            }

            // Update state
            State_Update(state);

            // Update array lengths
            Chain_Setup_Data(state, idx_chain);

            Log( Utility::Log_Level::Info, Utility::Log_Sender::API, fmt::format( "Read "
                 "chain from file {}", file ), starting_image, idx_chain );
        } 
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

void IO_Chain_Write( State *state, const char *file, int format, const char* comment, 
                     int idx_chain ) noexcept
{
    int idx_image = -1;

    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;

        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );

        // Read the data
        chain->Lock();
        try
        {
            IO::Write_Chain_Spin_Configuration( chain, std::string(file), (IO::VF_FileFormat)format, 
                                                std::string(comment), false );
        }
        catch( ... )
        {
            spirit_handle_exception_api(idx_image, idx_chain);
        }
        chain->Unlock();

        Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
                fmt::format("Wrote chain to file {} with format {}", file, format), 
                idx_image, idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

void IO_Chain_Append( State *state, const char *file, int format, const char* comment, 
                     int idx_chain ) noexcept
{
    int idx_image = -1;

    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        // Read the data
        chain->Lock();
        try
        {
            IO::Write_Chain_Spin_Configuration( chain, std::string(file), (IO::VF_FileFormat)format, 
                                                std::string(comment), true );
        }
        catch( ... )
        {
            spirit_handle_exception_api(idx_image, idx_chain);
        }
        chain->Unlock();

        Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
                fmt::format("Wrote chain to file {} with format {}", file, format), 
                idx_image, idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

/*----------------------------------------------------------------------------------------------- */
/*--------------------------------------- Data -------------------------------------------------- */
/*----------------------------------------------------------------------------------------------- */

//IO_Energies_Spins_Save
void IO_Image_Write_Energy_per_Spin(State * state, const char * file, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        // Write the data
        IO::Write_Image_Energy_per_Spin(*image, std::string(file));
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

//IO_Energy_Spins_Save
void IO_Image_Write_Energy(State * state, const char * file, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        // Write the data
        IO::Write_Image_Energy(*image, std::string(file));
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

//IO_Energies_Save
void IO_Chain_Write_Energies(State * state, const char * file, int idx_chain) noexcept
{
    int idx_image = -1;
    
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        // Write the data
        IO::Write_Chain_Energies(*chain, idx_chain, std::string(file));
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

//IO_Energies_Interpolated_Save
void IO_Chain_Write_Energies_Interpolated(State * state, const char * file, int idx_chain) noexcept
{
    int idx_image = -1;
    
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        // Write the data
        IO::Write_Chain_Energies_Interpolated(*chain, std::string(file));
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}
