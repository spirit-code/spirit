#include <catch.hpp>
#include <io/IO.hpp>
#include <Spirit/State.h>
#include <Spirit/Configurations.h>
#include <Spirit/System.h>

const char inputfile[] = "core/test/input/fd_neighbours.cfg";

TEST_CASE( "IO", "[io]" )
{    
    // TODO: Diferent OVF test for text, 8 and 4 byte raw data
    
    SECTION( "OVF format" )
    {        
        auto state = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete );
        
        Configuration_MinusZ( state.get() );
        
        IO_Image_Write( state.get(), "core/test/io_test_files/test_ovf.ovf", IO_Fileformat_OVF );
        
        Configuration_PlusZ( state.get() );
        
        IO_Image_Read( state.get(), "core/test/io_test_files/test_ovf.ovf", IO_Fileformat_OVF );
        
        int nos = System_Get_NOS( state.get() );
        REQUIRE( nos == 4 );
        
        scalar* data = System_Get_Spin_Directions( state.get() );
        scalar eps  = 1e-8;
        
        for (int i=0; i<nos; i++)
        {
            REQUIRE( fabs( data[i*3] - 0 ) < eps );
            REQUIRE( fabs( data[i*3+1] - 0 ) < eps );
            REQUIRE( fabs( data[i*3+2] - (-1) ) < eps );
        }
    }
    
    SECTION( "SPIRIT Regular" )
    {
        SECTION( "Regural spin" )
        {
            auto state = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete );
            
            Configuration_MinusZ( state.get() );
            
            IO_Image_Write( state.get(), "core/test/io_test_files/regular.data", 
                            IO_Fileformat_Regular );
            
            Configuration_PlusZ( state.get() );
            
            IO_Image_Read( state.get(), "core/test/io_test_files/regular.data", 
                           IO_Fileformat_Regular );
            
            int nos = System_Get_NOS( state.get() );
            REQUIRE( nos == 4 );
            
            scalar* data = System_Get_Spin_Directions( state.get() );
            scalar eps  = 1e-8;
            
            for (int i=0; i<nos; i++)
            {
                REQUIRE( fabs( data[i*3] - 0 ) < eps );
                REQUIRE( fabs( data[i*3+1] - 0 ) < eps );
                REQUIRE( fabs( data[i*3+2] - (-1) ) < eps );
            }
        }
        
        SECTION( "Regular position spin" )
        {
            auto state = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete );
            
            Configuration_MinusZ( state.get() );
            
            IO_Image_Write( state.get(), "core/test/io_test_files/regular_pos.data", 
                            IO_Fileformat_Regular_Pos );
            
            Configuration_PlusZ( state.get() );
            
            IO_Image_Read( state.get(), "core/test/io_test_files/regular_pos.data", 
                           IO_Fileformat_Regular_Pos );
            
            int nos = System_Get_NOS( state.get() );
            REQUIRE( nos == 4 );
            
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
    
    // TODO: uncomment and extend when the implementation of proper CSV IO would be ready
    
    // SECTION( "SPIRIT CSV" )
    // {
    //     SECTION( "CSV spin" )
    //     {
    //         auto state = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete );
    // 
    //         Configuration_MinusZ( state.get() );
    //
    //         IO_Image_Write( state.get(), "core/test/io_test_files/csv.data", IO_Fileformat_CSV );
    //         
    //         Configuration_PlusZ( state.get() );
    //         
    //         IO_Image_Read( state.get(), "core/test/io_test_files/csv.data", IO_Fileformat_CSV );
    //         
    //         int nos = System_Get_NOS( state.get() );
    //         REQUIRE( nos == 4 );
    //         
    //         scalar* data = System_Get_Spin_Directions( state.get() );
    //         scalar eps  = 1e-8;
    //         
    //         for (int i=0; i<nos; i++)
    //         {
    //             REQUIRE( fabs( data[i*3] - 0 ) < eps );
    //             REQUIRE( fabs( data[i*3+1] - 0 ) < eps );
    //             REQUIRE( fabs( data[i*3+2] - (-1) ) < eps );
    //         } 
    //     }
    //     
    //     SECTION( "CSV position spin")
    //     {
    //         auto state = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete );
    // 
    //         Configuration_MinusZ( state.get() );
    //
    //         IO_Image_Write( state.get(), "core/test/io_test_files/csv_pos.data", 
    //                         IO_Fileformat_CSV_Pos );
    //         
    //         Configuration_PlusZ( state.get() );
    //         
    //         IO_Image_Read( state.get(), "core/test/io_test_files/csv_pos2.data", 
    //                        IO_Fileformat_CSV_Pos );
    //         
    //         int nos = System_Get_NOS( state.get() );
    //         REQUIRE( nos == 4 );
    //         
    //         scalar* data = System_Get_Spin_Directions( state.get() );
    //         scalar eps  = 1e-8;
    //         
    //         for (int i=0; i<nos; i++)
    //         {
    //             REQUIRE( fabs( data[i*3] - 0 ) < eps );
    //             REQUIRE( fabs( data[i*3+1] - 0 ) < eps );
    //             REQUIRE( fabs( data[i*3+2] - (-1) ) < eps );
    //         } 
    //     }
    // }
}