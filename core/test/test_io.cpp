#include <catch.hpp>
#include <io/IO.hpp>
#include <Spirit/State.h>
#include <Spirit/Configurations.h>
#include <Spirit/System.h>

const char inputfile[] = "core/test/input/fd_neighbours.cfg";

TEST_CASE( "IO", "[io]" )
{    
    // TODO: Diferent OVF test for text, 8 and 4 byte raw data
    
    auto state = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete );
    
    // files to be written
    std::vector<std::pair< std::string, int >>  filetypes { 
        { "core/test/io_test_files/test_ovf.ovf",     IO_Fileformat_OVF_bin8         },
        //{ "core/test/io_test_files/csv.data",         IO_Fileformat_CSV         },
        { "core/test/io_test_files/csv_pos.data",     IO_Fileformat_CSV_Pos     },
        //{ "core/test/io_test_files/regular_pos.data", IO_Fileformat_Regular_Pos },  
        { "core/test/io_test_files/regular.data",     IO_Fileformat_Regular     } };
    
    // buffer variables for better readability
    const char *filename;
    int filetype;
    
    for ( auto file: filetypes )
    {
        filename = file.first.c_str();      // get the filename from pair
        filetype = file.second;             // fet the filetype from pair
        
        // Log the filename
        INFO( "IO file " + file.first );
        
        // set config to minus z and write the system out
        Configuration_MinusZ( state.get() );
        IO_Image_Write( state.get(), filename, filetype );
        
        // set config to plus z and read the previously saved system
        Configuration_PlusZ( state.get() );
        IO_Image_Read( state.get(), filename, filetype );
        
        // make sure that the read in has the same nos
        int nos = System_Get_NOS( state.get() );
        REQUIRE( nos == 4 );
        
        // assure that the system read in corresponds to config minus z
        scalar* data = System_Get_Spin_Directions( state.get() );
        scalar eps  = 1e-8;
        
        for (int i=0; i<nos; i++)
        {
            REQUIRE( fabs( data[i*3] - 0 ) < eps );
            REQUIRE( fabs( data[i*3+1] - 0 ) < eps );
            REQUIRE( fabs( data[i*3+2] - (-1) ) < eps );
        }
    }
}