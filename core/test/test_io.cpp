#include <catch.hpp>
#include <io/IO.hpp>
#include <Spirit/State.h>
#include <Spirit/Configurations.h>
#include <Spirit/System.h>
#include <Spirit/Chain.h>
#include <Spirit/Simulation.h>
#include <utility>
#include <vector>
#include <iostream>
#include <algorithm>
#include <string>

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
        { "core/test/io_test_files/image_ovf_txt.ovf",      IO_Fileformat_OVF_text    },
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

TEST_CASE( "IO-EIGENMODE-WRITE", "[io-ema]" )
{
    auto state = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete );

    // files to be written
    std::vector<std::pair<std::string,int>> filetypes { 
        //{ "core/test/io_test_files/eigenmode_regular.data",     IO_Fileformat_Regular     }, 
        //{ "core/test/io_test_files/eigenmode_regular_pos.data", IO_Fileformat_Regular_Pos },  
        //{ "core/test/io_test_files/eigenmode_csv.data",         IO_Fileformat_CSV         },
        //{ "core/test/io_test_files/eigenmode_csv_pos.data",     IO_Fileformat_CSV_Pos     },
        { "core/test/io_test_files/eigenmode_ovf_txt.ovf",      IO_Fileformat_OVF_text    },
        { "core/test/io_test_files/eigenmode_ovf_bin_4.ovf",    IO_Fileformat_OVF_bin4    },
        { "core/test/io_test_files/eigenmode_ovf_bin_8.ovf",    IO_Fileformat_OVF_bin8    } };
    
    // buffer variables for better readability
    const char *filename;
    int filetype;
    
    for ( auto file: filetypes )
    {
        filename = file.first.c_str();      // get the filename from pair
        filetype = file.second;             // fet the filetype from pair
        
        // Log the filename
        INFO( "IO eigenmodes " + file.first );
        
        Configuration_Skyrmion( state.get(), 5, 1, -90, false, false, false);
        Simulation_Calculate_Eigenmodes( state.get());
        IO_Eigenmodes_Write( state.get(), filename, filetype); 
    }
}

TEST_CASE( "IO-CHAIN-WRITE", "[io-chain]" )
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
        IO_Chain_Write( state.get(), filename, filetype );
    }
}

TEST_CASE( "IO-CHAIN-READ", "[io-chain]" )
{
    auto state = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete );
    
    std::vector<std::pair< std::string, int >>  filetypes { 
        { "core/test/io_test_files/chain_regular.data",     IO_Fileformat_Regular     }, 
        //{ "core/test/io_test_files/chain_regular_pos.data", IO_Fileformat_Regular_Pos },  
        //{ "core/test/io_test_files/chain_csv.data",         IO_Fileformat_CSV         },
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
        IO_Chain_Read( state.get(), filename, filetype );
        
        // Now the state must have 3 images
        REQUIRE( Chain_Get_NOI( state.get() ) == 3 );
        
        // create a name for printing the read in chain for validation purposes
        std::string cur_name = file.first;                    // get current name
        cur_name.erase( cur_name.end()-5, cur_name.end() );     // strip ending (".data")
        std::string validation_filename = cur_name + "_validate_reading.data"; // expand name
        
        // write the read in chain for visual inspection
        IO_Chain_Write( state.get(), validation_filename.c_str(), filetype );
    }
}

TEST_CASE( "IO-OVF-CAPITALIZATION", "[io-ovf]")
{
    // That test is checking that the IO_Image_Read() would deal properly with capitalization for 
    // every file that uses Filter_File_Handle(). We are testing the OVF_TEXT format. For that 
    // reason first (1) we have to parse in the io_test_file, convert every char to upper and then
    // rewrite it. Then (2) if we try to read in the capilized file test should NOT fail.
    
    // 1. Create the upper case file
    
    std::ifstream ifile( "core/test/io_test_files/image_ovf_txt.ovf", std::ios::in );
    std::ofstream ofile( "core/test/io_test_files/image_ovf_txt_CAP.ovf", std::ios::out );
    std::string line;
    
    while( std::getline( ifile,line ) )
    {
        std::transform( line.begin(), line.end(), line.begin(), ::toupper );
        ofile << line << std::endl;
    }
    
    ifile.close();
    ofile.close();
    
    // 2. Read the upper case file
    
    auto state = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete );
    
    IO_Image_Read( state.get(), "core/test/io_test_files/image_ovf_txt_CAP.ovf", 
                   IO_Fileformat_OVF_text );
                   
   scalar* data = System_Get_Spin_Directions( state.get() );
   
   // make sure that the read in has the same nos
   int nos = System_Get_NOS( state.get() );
   REQUIRE( nos == 4 );
   
   for (int i=0; i<nos; i++)
   {
       REQUIRE( data[i*3] == Approx( 0 ) );
       REQUIRE( data[i*3+1] == Approx( 0 ) );
       REQUIRE( data[i*3+2] == Approx( -1 ) );
   }
}
