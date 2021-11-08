#include <Spirit/Chain.h>
#include <Spirit/Configurations.h>
#include <Spirit/Simulation.h>
#include <Spirit/State.h>
#include <Spirit/System.h>

#include <io/IO.hpp>

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <catch.hpp>

const char inputfile[] = "core/test/input/fd_pairs.cfg";

TEST_CASE( "IO", "[io]" )
{
    auto state = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete );

    // files to be written
    std::vector<std::pair<std::string, int>> filetypes{
        { "core/test/io_test_files/image_ovf_txt.ovf", IO_Fileformat_OVF_text },
        { "core/test/io_test_files/image_ovf_bin_4.ovf", IO_Fileformat_OVF_bin4 },
        { "core/test/io_test_files/image_ovf_bin_8.ovf", IO_Fileformat_OVF_bin8 },
        { "core/test/io_test_files/image_ovf_csv.ovf", IO_Fileformat_OVF_csv },
    };

    // buffer variables for better readability
    const char * filename;
    int filetype;

    for( auto file : filetypes )
    {
        filename = file.first.c_str(); // get the filename from pair
        filetype = file.second;        // fet the filetype from pair

        // Log the filename
        INFO( "IO image " + file.first );

        // set config to minus z and write the system out
        Configuration_MinusZ( state.get() );
        IO_Image_Write( state.get(), filename, filetype, "io test" );

        // set config to plus z and read the previously saved system
        Configuration_PlusZ( state.get() );
        IO_Image_Read( state.get(), filename );

        // make sure that the read in has the same nos
        int nos = System_Get_NOS( state.get() );
        REQUIRE( nos == 4 );

        // assure that the system read in corresponds to config minus z
        scalar * data = System_Get_Spin_Directions( state.get() );

        for( int i = 0; i < nos; i++ )
        {
            REQUIRE( data[i * 3] == Approx( 0 ) );
            REQUIRE( data[i * 3 + 1] == Approx( 0 ) );
            REQUIRE( data[i * 3 + 2] == Approx( -1 ) );
        }
    }

    // Energy and Energy per Spin
    // IO_Image_Write_Energy_per_Spin( state.get(), "core/test/io_test_files/E_per_spin.data"  );
    IO_Image_Write_Energy( state.get(), "core/test/io_test_files/Energy.data" );
}

TEST_CASE( "IO-EIGENMODE-WRITE", "[io-ema]" )
{
    auto state = std::shared_ptr<State>( State_Setup( "core/test/input/fd_pairs.cfg" ), State_Delete );

    // files to be written
    std::vector<std::pair<std::string, int>> filetypes{
        //{ "core/test/io_test_files/eigenmode_regular.data",     IO_Fileformat_Regular     },
        //{ "core/test/io_test_files/eigenmode_regular_pos.data", IO_Fileformat_Regular_Pos },
        //{ "core/test/io_test_files/eigenmode_csv.data",         IO_Fileformat_CSV         },
        //{ "core/test/io_test_files/eigenmode_csv_pos.data",     IO_Fileformat_CSV_Pos     },
        { "core/test/io_test_files/eigenmode_ovf_txt.ovf", IO_Fileformat_OVF_text },
        { "core/test/io_test_files/eigenmode_ovf_bin_4.ovf", IO_Fileformat_OVF_bin4 },
        { "core/test/io_test_files/eigenmode_ovf_bin_8.ovf", IO_Fileformat_OVF_bin8 },
    };

    // buffer variables for better readability
    const char * filename;
    int filetype;

    for( auto file : filetypes )
    {
        filename = file.first.c_str(); // get the filename from pair
        filetype = file.second;        // fet the filetype from pair

        // Log the filename
        INFO( "IO eigenmodes " + file.first );

        Configuration_Skyrmion( state.get(), 5, 1, -90, false, false, false );
        System_Update_Eigenmodes( state.get() );
        IO_Eigenmodes_Write( state.get(), filename, filetype );
    }
}

TEST_CASE( "IO-CHAIN-WRITE", "[io-chain]" )
{
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
    std::vector<std::pair<std::string, int>> filetypes{
        { "core/test/io_test_files/chain_ovf_txt.ovf", IO_Fileformat_OVF_text },
        { "core/test/io_test_files/chain_ovf_bin_4.ovf", IO_Fileformat_OVF_bin4 },
        { "core/test/io_test_files/chain_ovf_bin_8.ovf", IO_Fileformat_OVF_bin8 },
        { "core/test/io_test_files/chain_ovf_csv.ovf", IO_Fileformat_OVF_csv },
    };

    // buffer variables for better readability
    const char * filename;
    int filetype;

    for( auto file : filetypes )
    {
        filename = file.first.c_str(); // get the filename from pair
        filetype = file.second;        // fet the filetype from pair

        // Log the filename
        INFO( "IO chain" + file.first );
        IO_Chain_Write( state.get(), filename, filetype );
    }
}

TEST_CASE( "IO-CHAIN-READ", "[io-chain]" )
{
    auto state = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete );

    std::vector<std::pair<std::string, int>> filetypes{
        { "core/test/io_test_files/chain_ovf_bin_4.ovf", IO_Fileformat_OVF_bin4 },
        { "core/test/io_test_files/chain_ovf_bin_8.ovf", IO_Fileformat_OVF_bin8 },
        { "core/test/io_test_files/chain_ovf_csv.ovf", IO_Fileformat_OVF_csv },
        { "core/test/io_test_files/chain_ovf_txt.ovf", IO_Fileformat_OVF_text }
    };

    // buffer variables for better readability
    const char * filename;
    int filetype;

    for( auto file : filetypes )
    {
        filename = file.first.c_str(); // get the filename from pair
        filetype = file.second;        // fet the filetype from pair

        // Log the filename
        INFO( "IO chain" + file.first );
        IO_Chain_Read( state.get(), filename );

        // Now the state must have 3 images
        int noi = Chain_Get_NOI( state.get() );
        REQUIRE( noi == 3 );

        // Get nos. Each image must have the same nos
        int nos = System_Get_NOS( state.get() );

        scalar * data;

        // Image 0 must have all the configurations to minus Z

        Chain_Jump_To_Image( state.get(), 0 );
        data = System_Get_Spin_Directions( state.get() );
        for( int i = 0; i < nos; i++ )
        {
            REQUIRE( data[i * 3] == Approx( 0 ) );
            REQUIRE( data[i * 3 + 1] == Approx( 0 ) );
            REQUIRE( data[i * 3 + 2] == Approx( -1 ) );
        }

        // Image 1 must have all the configurations at random orientation - we cannot test

        // Image 2 must have all the configurations to plus Z
        Chain_Jump_To_Image( state.get(), 2 );
        data = System_Get_Spin_Directions( state.get() );
        for( int i = 0; i < nos; i++ )
        {
            REQUIRE( data[i * 3] == Approx( 0 ) );
            REQUIRE( data[i * 3 + 1] == Approx( 0 ) );
            REQUIRE( data[i * 3 + 2] == Approx( 1 ) );
        }

        // Before testing the next filetype remove noi-1 images from the system
        for( int i = 0; i < ( noi - 1 ); i++ )
            Chain_Pop_Back( state.get() );
        REQUIRE( Chain_Get_NOI( state.get() ) == 1 );
    }
}

TEST_CASE( "IO-OVF-CAPITALIZATION", "[io-ovf]" )
{
    // That test is checking that the IO_Image_Read() would deal properly with capitalization for
    // every file that uses Filter_File_Handle(). We are testing the OVF_TEXT format. For that
    // reason first (1) we have to parse in the io_test_file, convert every char to upper and then
    // rewrite it. Then (2) if we try to read in the capilized file test should NOT fail.

    // 1. Create the upper case file

    std::ifstream ifile( "core/test/io_test_files/image_ovf_txt.ovf", std::ios::in );
    std::ofstream ofile( "core/test/io_test_files/image_ovf_txt_CAP.ovf", std::ios::out );
    std::string line;

    while( std::getline( ifile, line ) )
    {
        std::transform( line.begin(), line.end(), line.begin(), ::toupper );
        ofile << line << std::endl;
    }

    ifile.close();
    ofile.close();

    // 2. Read the upper case file

    auto state = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete );

    IO_Image_Read( state.get(), "core/test/io_test_files/image_ovf_txt_CAP.ovf" );

    scalar * data = System_Get_Spin_Directions( state.get() );

    // make sure that the read in has the same nos
    int nos = System_Get_NOS( state.get() );
    REQUIRE( nos == 4 );

    for( int i = 0; i < nos; i++ )
    {
        REQUIRE( data[i * 3] == Approx( 0 ) );
        REQUIRE( data[i * 3 + 1] == Approx( 0 ) );
        REQUIRE( data[i * 3 + 2] == Approx( -1 ) );
    }
}

TEST_CASE( "IO-READ-TXT-AND-CSV", "[io-txt-csv]" )
{
    // Checks if the API can handle properly raw .txt or .csv files. Any line starting by '#' must
    // be discarded. If the format is spins components ( s_x s_y s_z ) in columns or csv format
    // must be detected and handled automatically

    std::vector<std::pair<std::string, std::string>> filetypes{
        { "core/test/io_test_files/image_ovf_txt.ovf", "core/test/io_test_files/image_ovf_txt.txt" },
        { "core/test/io_test_files/image_ovf_txt.ovf", "core/test/io_test_files/image_ovf_no_extension" },
        { "core/test/io_test_files/image_ovf_csv.ovf", "core/test/io_test_files/image_ovf_csv.csv" },
        { "core/test/io_test_files/chain_ovf_txt.ovf", "core/test/io_test_files/chain_ovf_txt.txt" },
        { "core/test/io_test_files/chain_ovf_txt.ovf", "core/test/io_test_files/chain_ovf_no_extension" },
        { "core/test/io_test_files/chain_ovf_csv.ovf", "core/test/io_test_files/chain_ovf_csv.csv" },
    };

    // from (*.ovf filetype), to (*.new filetype), dump for dumping the first line "# OOMMF OVF..."
    std::string from, to, dump;

    for( auto pairs : filetypes )
    {
        from = pairs.first;
        to   = pairs.second;

        INFO( "IO read non OVF files " + to );

        // Create a file with different extension by copying an ovf

        std::ifstream ifile( from );
        std::ofstream ofile( to );

        // dump first line to invalidate OVF format
        std::getline( ifile, dump );

        // copy
        ofile << ifile.rdbuf();

        ifile.close();
        ofile.close();
    }

    for( int i = 0; i < 3; i++ )
    {
        to = filetypes[i].second;

        auto state = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete );

        IO_Image_Read( state.get(), to.c_str() );

        // make sure that the read in has the same nos
        int nos = System_Get_NOS( state.get() );
        REQUIRE( nos == 4 );

        // assure that the system read in corresponds to config minus z
        scalar * data = System_Get_Spin_Directions( state.get() );

        for( int j = 0; j < nos; j++ )
        {
            REQUIRE( data[j * 3] == Approx( 0 ) );
            REQUIRE( data[j * 3 + 1] == Approx( 0 ) );
            REQUIRE( data[j * 3 + 2] == Approx( -1 ) );
        }
    }

    for( int i = 3; i < 6; i++ )
    {
        to = filetypes[i].second;

        auto state = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete );

        IO_Chain_Read( state.get(), to.c_str() );

        // make sure that the read in has the same nos
        int noi = Chain_Get_NOI( state.get() );
        REQUIRE( noi == 3 );

        // Before testing the next filetype remove noi-1 images from the system
        for( int i = 0; i < ( noi - 1 ); i++ )
            Chain_Pop_Back( state.get() );
        REQUIRE( Chain_Get_NOI( state.get() ) == 1 );
    }
}

TEST_CASE( "IO-OVF-N_SEGMENTS", "[io-OVF-n_segments]" )
{
    // Checks if the OVF_File object will return the correct number of segments in an OVF file
    // by calling the API function. In case that the file is not OVF type the function must return
    // -1 and Log an appropriate warning.

    std::vector<std::pair<std::string, int>> filenames{
        { "core/test/io_test_files/image_ovf_txt.ovf", 1 },
        { "core/test/io_test_files/image_ovf_txt.txt", -1 },
        { "core/test/io_test_files/chain_ovf_txt.ovf", 3 },
    };

    auto state = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete );

    std::string file;
    int noi_read;
    int noi_known;

    for( auto pair : filenames )
    {
        file      = pair.first;
        noi_known = pair.second;

        INFO( file )

        noi_read = IO_N_Images_In_File( state.get(), file.c_str() );

        REQUIRE( noi_known == noi_read );
    }
}

TEST_CASE( "IO-INTERACTION-PAIRS", "[io-interactions-pairs]" )
{
    auto state = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete );

    IO_Image_Write_Neighbours_Exchange( state.get(), "core/test/io_test_files/neighbours_J.dat" );
    IO_Image_Write_Neighbours_DMI( state.get(), "core/test/io_test_files/neighbours_DMI.dat" );
}
