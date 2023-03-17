#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

int main( int argc, char * argv[] )
{
    // Set the number of decimals printed in catch output
    Catch::StringMaker<float>::precision  = 12;
    Catch::StringMaker<double>::precision = 12;

    return Catch::Session().run( argc, argv );
}