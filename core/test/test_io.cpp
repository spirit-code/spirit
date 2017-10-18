#include <catch.hpp>
#include <io/IO.hpp>
#include <Spirit/State.h>
#include <Spirit/Configurations.h>
#include <Spirit/System.h>

auto inputfile = "core/test/input/fd_neighbours.cfg";

TEST_CASE( "IO", "[io]" )
{
    auto state = std::shared_ptr<State>( State_Setup( inputfile ), State_Delete );
    
    Configuration_MinusZ( state.get() );
    IO_Image_Write( state.get(), "core/test/io_test_files/test_ovf", IO_Fileformat_OVF );
    
    Configuration_PlusZ( state.get() );
    
    IO_Image_Read( state.get(), "core/test/io_test_files/test_ovf.ovf", IO_Fileformat_OVF );
    
    //IO::Read_Spin_Configuration(image, "core/test/io_test_files/test_ovf", IO_Fileformat_OVF );
    
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