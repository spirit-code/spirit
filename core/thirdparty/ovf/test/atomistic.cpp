#include <catch.hpp>

#include <ovf.h>
#include <fmt/format.h>

#include <iostream>
#include <cstring>

void ovf_test_write(std::string filename, std::string meshtype, int ovf_extension_format)
{
    INFO( fmt::format("Testing 'ovf_test_write' with filename '{}', meshtype '{}', ovf_extension_format '{}'", filename, meshtype, ovf_extension_format) );

    auto create_test_segment = [&]()
    {
        // segment header
        auto segment = ovf_segment_create();
        segment->title   = strdup("ovf test title - write");
        segment->comment = strdup("test write");
        segment->valuedim   = 3;

        segment->n_cells[0] = 2;
        segment->n_cells[1] = 1;
        segment->n_cells[2] = 1;

        if(meshtype == "rectangular")
            segment->n_cells[1] = 2;

        // Fill in atomistic values
        segment->bravaisa[0] = 1;
        segment->bravaisa[1] = 0;
        segment->bravaisa[2] = 0;

        segment->bravaisb[0] = 0;
        segment->bravaisb[1] = 1;
        segment->bravaisb[2] = 0;

        segment->bravaisc[0] = 0;
        segment->bravaisc[1] = 0;
        segment->bravaisc[2] = 1;

        segment->pointcount = 4;

        segment->ncellpoints = 2;
        float basis[6] = {
                            0,0.1,0.2,
                            2,3,4
                        };

        segment->basis = (float *) malloc( segment->ncellpoints * 3 * sizeof(float) );
        for( int ib=0; ib < segment->ncellpoints; ib++)
        {
            for(int i=0; i<3; i++)
            {
                segment->basis[ib*3 + i] = basis[ib*3 + i];
            }
        }

        segment->N = 4;
        segment->meshtype = strdup(meshtype.c_str());
        return segment;
    };

    auto create_dummy_segment = [&]()
    {
        // Append second
        auto segment = ovf_segment_create();
        segment->title   = strdup("ovf test title - append");
        segment->comment = strdup("test append");
        segment->valuedim = 3;
        segment->n_cells[0] = 2;
        segment->n_cells[1] = 2;
        segment->n_cells[2] = 1;
        segment->N = 4;
        return segment;
    };

    // Create and write test_segment
    auto segment = create_test_segment();
    // data
    std::vector<double> field(3*segment->N, 1);
    field[0] = 3;
    field[3] = 2;
    field[6] = 1;
    field[9] = 0;

    // open
    auto file = ovf_open(filename.c_str());
    file->ovf_extension_format = ovf_extension_format; // Set the flag for the atomistic extension

    // write
    int success = ovf_write_segment_8(file, segment, field.data(), OVF_FORMAT_TEXT);
    if( OVF_OK != success )
        std::cerr << ovf_latest_message(file) << std::endl;

    if( ovf_extension_format == OVF_EXTENSION_FORMAT_OVF && meshtype=="lattice" ) // do not allow lattice for non AOVF extension
    {
        REQUIRE( success == OVF_ERROR );
    } else {
        REQUIRE( success == OVF_OK );
    }
    // close
    ovf_close(file);


    // Create and append dummy_segment
    // data
    segment = create_dummy_segment();
    std::vector<double> field_dummy(3*segment->N, 1);
    field_dummy[0] = 6;
    field_dummy[3] = 4;
    field_dummy[6] = 2;
    field_dummy[9] = 0;

    // open
    file = ovf_open(filename.c_str());
    // write
    success = ovf_append_segment_8(file, segment, field.data(), OVF_FORMAT_CSV);
    if( OVF_OK != success )
        std::cerr << ovf_latest_message(file) << std::endl;
    REQUIRE( success == OVF_OK );
    // close
    ovf_close(file);


    // Create and append test_segment
    // data
    segment = create_test_segment();
    // open
    file = ovf_open(filename.c_str());
    // write
    success = ovf_append_segment_8(file, segment, field.data(), OVF_FORMAT_CSV);
    if( OVF_OK != success )
        std::cerr << ovf_latest_message(file) << std::endl;

    if( ovf_extension_format == OVF_EXTENSION_FORMAT_OVF && meshtype=="lattice" ) // do not allow lattice for non AOVF extension
    {
        REQUIRE( success == OVF_ERROR );
    } else {
        REQUIRE( success == OVF_OK );
    }

    // close
    ovf_close(file);

}


void ovf_test_read(std::string filename, std::string meshtype, int ovf_extension_format)
{
    INFO( fmt::format("Testing 'ovf_test_read' with filename '{}', meshtype '{}', ovf_extension_format '{}'", filename, meshtype, ovf_extension_format) );

    auto verify_test_segment = [&](int index)
    {
        // open
        auto file = ovf_open(filename.c_str());
        std::cerr << ovf_latest_message(file) << std::endl;
        REQUIRE( file->found == true );
        REQUIRE( file->is_ovf == true );
        REQUIRE( file->n_segments == 3);
        REQUIRE( file->ovf_extension_format == ovf_extension_format );

        // segment header
        auto segment = ovf_segment_create();

        // read header
        int success = ovf_read_segment_header(file, index, segment);
        if( OVF_OK != success )
            std::cerr << ovf_latest_message(file) << std::endl;
        REQUIRE( success == OVF_OK );

        REQUIRE( segment->N == 4 );
        REQUIRE( std::string(segment->meshtype) == meshtype );

        if(std::string(segment->meshtype) == "lattice")
        {
            REQUIRE( segment->ncellpoints == 2);
            REQUIRE( segment->basis[0] == 0);

            REQUIRE( segment->basis[1] == 0.1f);
            REQUIRE( segment->basis[2] == 0.2f);
            REQUIRE( segment->basis[3] == 2);
            REQUIRE( segment->basis[4] == 3);
            REQUIRE( segment->basis[5] == 4);

            REQUIRE( segment->bravaisa[0] == 1);
            REQUIRE( segment->bravaisa[1] == 0);
            REQUIRE( segment->bravaisa[2] == 0);

            REQUIRE( segment->bravaisb[0] == 0);
            REQUIRE( segment->bravaisb[1] == 1);
            REQUIRE( segment->bravaisb[2] == 0);

            REQUIRE( segment->bravaisc[0] == 0);
            REQUIRE( segment->bravaisc[1] == 0);
            REQUIRE( segment->bravaisc[2] == 1);
        }
        // data
        std::vector<float> field(3*segment->N);

        // read data
        success = ovf_read_segment_data_4(file, index, segment, field.data());
        if( OVF_OK != success )
            std::cerr << ovf_latest_message(file) << std::endl;
        REQUIRE( success == OVF_OK );

        REQUIRE( field[0] == 3 );
        REQUIRE( field[1] == 1 );
        REQUIRE( field[3] == 2 );
        REQUIRE( field[6] == 1 );
        REQUIRE( field[9] == 0 );

        // close
        ovf_close(file);
    };

    verify_test_segment(0);
    verify_test_segment(2);
}

TEST_CASE( "Atomistic Write", "[write]" )
{
    const char * testfile = "testfile_cpp.aovf";

    SECTION( "write OVF" )
    {
        ovf_test_write( "test_file_ovf_rectangular_cpp.ovf", "rectangular", OVF_EXTENSION_FORMAT_OVF );
        ovf_test_write( "test_file_ovf_irregular_cpp.ovf",   "irregular",   OVF_EXTENSION_FORMAT_OVF );
        ovf_test_write( "test_file_ovf_lattice_cpp.ovf",     "lattice",     OVF_EXTENSION_FORMAT_OVF ); // Should print some error message
    }

    SECTION( "write AOVF" )
    {
        ovf_test_write( "test_file_atomistic_rectangular_cpp.aovf", "rectangular", OVF_EXTENSION_FORMAT_AOVF );
        ovf_test_write( "test_file_atomistic_lattice_cpp.aovf",     "lattice",     OVF_EXTENSION_FORMAT_AOVF );
        ovf_test_write( "test_file_atomistic_irregular_cpp.aovf",   "irregular",   OVF_EXTENSION_FORMAT_AOVF );
    }

    SECTION( "write AOVF_COMP" )
    {
        ovf_test_write( "test_file_atomistic_comp_rectangular_cpp.aovf", "rectangular", OVF_EXTENSION_FORMAT_AOVF_COMP );
        ovf_test_write( "test_file_atomistic_comp_lattice_cpp.aovf",     "lattice",     OVF_EXTENSION_FORMAT_AOVF_COMP );
        ovf_test_write( "test_file_atomistic_comp_irregular_cpp.aovf",   "irregular",  OVF_EXTENSION_FORMAT_AOVF_COMP );
    }
}

TEST_CASE( "Atomistic Read", "[read]" )
{
    SECTION( "read OVF" )
    {
        ovf_test_write( "test_file_ovf_rectangular_cpp.ovf", "rectangular", OVF_EXTENSION_FORMAT_OVF );
        ovf_test_write( "test_file_ovf_irregular_cpp.ovf",   "irregular",   OVF_EXTENSION_FORMAT_OVF );
        // ovf_test_write( "test_file_ovf_lattice_cpp.ovf",     "lattice",     OVF_EXTENSION_FORMAT_OVF ); // should not be written at all
    }

    SECTION( "read AOVF" )
    {
        ovf_test_read( "test_file_atomistic_rectangular_cpp.aovf",  "rectangular", OVF_EXTENSION_FORMAT_AOVF );
        ovf_test_read( "test_file_atomistic_lattice_cpp.aovf",      "lattice",     OVF_EXTENSION_FORMAT_AOVF );
        ovf_test_read( "test_file_atomistic_irregular_cpp.aovf",    "irregular",   OVF_EXTENSION_FORMAT_AOVF );
    }

    SECTION( "read AOVF_COMP" )
    {
        ovf_test_read( "test_file_atomistic_comp_rectangular_cpp.aovf", "rectangular", OVF_EXTENSION_FORMAT_AOVF_COMP );
        ovf_test_read( "test_file_atomistic_comp_lattice_cpp.aovf",     "lattice",     OVF_EXTENSION_FORMAT_AOVF_COMP );
        ovf_test_read( "test_file_atomistic_comp_irregular_cpp.aovf",   "irregular",   OVF_EXTENSION_FORMAT_AOVF_COMP );
    }
}