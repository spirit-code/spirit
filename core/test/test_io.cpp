#include <catch.hpp>
#include <io/IO.hpp>
#include <Spirit/State.h>
#include <Spirit/Configurations.h>
#include <Spirit/System.h>
#include <Spirit/Chain.h>
#include <utility>
#include <vector>

const char inputfile[] = "core/test/input/fd_neighbours.cfg";

TEST_CASE( "IO", "[io]" )
{    
    // TODO: Diferent OVF test for text, 8 and 4 byte raw data
    
    auto state = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete );
    
    // files to be written
    std::vector<std::pair< std::string, int >>  filetypes { 
        { "core/test/io_test_files/image_regular.data",     IO_Fileformat_Regular     }, 
        //{ "core/test/io_test_files/image_regular_pos.data", IO_Fileformat_Regular_Pos },  
        //{ "core/test/io_test_files/image_csv.data",         IO_Fileformat_CSV         },
        { "core/test/io_test_files/image_csv_pos.data",     IO_Fileformat_CSV_Pos     },
        //{ "core/test/io_test_files/image_ovf_txt.ovf",      IO_Fileformat_OVF_text    },
        { "core/test/io_test_files/image_ovf_bin_4.ovf",    IO_Fileformat_OVF_bin4    },
        { "core/test/io_test_files/image_ovf_bin_8.ovf",    IO_Fileformat_OVF_bin8    } };
    
    // buffer variables for better readability
    const char *filename;
    int filetype;
    
    for ( auto file: filetypes )
    {
        filename = file.first.c_str();      // get the filename from pair
        filetype = file.second;             // fet the filetype from pair
        
        // Log the filename
        INFO( "IO image " + file.first );
        
        // set config to minus z and write the system out
        Configuration_MinusZ( state.get() );
        IO_Image_Write( state.get(), filename, filetype, "io test" );
        IO_Image_Append( state.get(), filename, filetype, "io test" );
        
        // set config to plus z and read the previously saved system
        Configuration_PlusZ( state.get() );
        IO_Image_Read( state.get(), filename, filetype );
        
        // make sure that the read in has the same nos
        int nos = System_Get_NOS( state.get() );
        REQUIRE( nos == 4 );
        
        // assure that the system read in corresponds to config minus z
        scalar* data = System_Get_Spin_Directions( state.get() );
        
        for (int i=0; i<nos; i++)
        {
            REQUIRE( data[i*3] == Approx( 0 ) );
            REQUIRE( data[i*3+1] == Approx( 0 ) );
            REQUIRE( data[i*3+2] == Approx( -1 ) );
        }
    }
    
    // Energy and Energy per Spin
    //IO_Image_Write_Energy_per_Spin( state.get(), "core/test/io_test_files/E_per_spin.data"  );
    IO_Image_Write_Energy( state.get(), "core/test/io_test_files/Energy.data" );
}

TEST_CASE( "IO-CHAIN", "[io-chain]" )
{    
    // TODO: Diferent OVF test for text, 8 and 4 byte raw data
    
    auto state = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete );
    
    // create 2 additional images
    Chain_Image_to_Clipboard( state.get() );
    Chain_Insert_Image_Before( state.get() );
    Chain_Insert_Image_Before( state.get() );
    
    Chain_Jump_To_Image( state.get(), 0 );
    Configuration_MinusZ( state.get() );
    
    Chain_Jump_To_Image( state.get(), 1 );
    Configuration_Random( state.get() );
    
    Chain_Jump_To_Image( state.get(), 2 );
    Configuration_PlusZ( state.get() );
    
    // files to be written
    std::vector<std::pair< std::string, int >>  filetypes { 
        { "core/test/io_test_files/chain_regular.data",     IO_Fileformat_Regular     }, 
        { "core/test/io_test_files/chain_regular_pos.data", IO_Fileformat_Regular_Pos },  
        { "core/test/io_test_files/chain_csv.data",         IO_Fileformat_CSV         },
        { "core/test/io_test_files/chain_csv_pos.data",     IO_Fileformat_CSV_Pos     } };
    
    // buffer variables for better readability
    const char *filename;
    int filetype;
    
    for ( auto file: filetypes )
    {
        filename = file.first.c_str();      // get the filename from pair
        filetype = file.second;             // fet the filetype from pair
        
        // Log the filename
        INFO( "IO chain" + file.first );
        IO_Chain_Write( state.get(), filename, filetype );      // this must be overwritten
        IO_Chain_Write( state.get(), filename, filetype );
        IO_Chain_Append( state.get(), filename, filetype );
    }
}