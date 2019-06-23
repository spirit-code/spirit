#include <catch.hpp>

#include <ovf.h>

#include <iostream>

TEST_CASE( "NonExistent", "[nonexistent]" )
{
    auto file = ovf_open("nonexistent.ovf");
    REQUIRE( file->found == false );
    REQUIRE( file->is_ovf == false );
    REQUIRE( file->n_segments == 0 );
    ovf_segment segment;
    int success = ovf_read_segment_header(file, 0, &segment);
    REQUIRE( success == OVF_ERROR );
    ovf_close(file);
}

TEST_CASE( "Write", "[write]" )
{
    const char * testfile = "testfile_cpp.ovf";

    SECTION( "write" )
    {
        // segment header
        auto segment = ovf_segment_create();
        segment->title = const_cast<char *>("ovf test title - write");
        segment->comment = const_cast<char *>("test write");
        segment->valuedim = 3;
        segment->n_cells[0] = 2;
        segment->n_cells[1] = 2;
        segment->n_cells[2] = 1;
        segment->N = 4;

        // data
        std::vector<double> field(3*segment->N, 1);
        field[0] = 3;
        field[3] = 2;
        field[6] = 1;
        field[9] = 0;

        // open
        auto file = ovf_open(testfile);

        // write
        int success = ovf_write_segment_8(file, segment, field.data(), OVF_FORMAT_TEXT);
        if( OVF_OK != success )
            std::cerr << ovf_latest_message(file) << std::endl;
        REQUIRE( success == OVF_OK );

        // close
        ovf_close(file);
    }

    SECTION( "append" )
    {
        // segment header
        auto segment = ovf_segment_create();
        segment->title = const_cast<char *>("ovf test title - append");
        segment->comment = const_cast<char *>("test append");
        segment->valuedim = 3;
        segment->n_cells[0] = 2;
        segment->n_cells[1] = 2;
        segment->n_cells[2] = 1;
        segment->N = 4;

        // data
        std::vector<double> field(3*segment->N, 1);
        field[0] = 6;
        field[3] = 4;
        field[6] = 2;
        field[9] = 0;

        // open
        auto file = ovf_open(testfile);

        // write
        int success = ovf_append_segment_8(file, segment, field.data(), OVF_FORMAT_CSV);
        if( OVF_OK != success )
            std::cerr << ovf_latest_message(file) << std::endl;
        REQUIRE( success == OVF_OK );

        // close
        ovf_close(file);
    }

    SECTION( "append irregular" )
    {
        // segment header
        auto segment = ovf_segment_create();
        segment->title = const_cast<char *>("ovf test title - append irregular mesh");
        segment->comment = const_cast<char *>("an irregular mesh has different keywords than a rectangular one");
        segment->valuedim = 3;
        segment->meshtype = const_cast<char *>("irregular");
        segment->pointcount = 4;

        // data
        std::vector<double> field(3*segment->pointcount, 1);
        field[0] = 6;
        field[3] = 4;
        field[6] = 2;
        field[9] = 0;

        // open
        auto file = ovf_open(testfile);

        // write
        int success = ovf_append_segment_8(file, segment, field.data(), OVF_FORMAT_CSV);
        if( OVF_OK != success )
            std::cerr << ovf_latest_message(file) << std::endl;
        REQUIRE( success == OVF_OK );

        // close
        ovf_close(file);
    }
}

TEST_CASE( "Read", "[read]" )
{
    const char * testfile = "testfile_cpp.ovf";

    SECTION( "first segment" )
    {
        // open
        auto file = ovf_open(testfile);
        REQUIRE( file->found == true );
        REQUIRE( file->is_ovf == true );
        REQUIRE( file->n_segments == 3 );
        int index = 0;

        // segment header
        auto segment = ovf_segment_create();

        // read header
        int success = ovf_read_segment_header(file, index, segment);
        if( OVF_OK != success )
            std::cerr << ovf_latest_message(file) << std::endl;
        REQUIRE( success == OVF_OK );
        REQUIRE( segment->N == 4 );
        REQUIRE( std::string(segment->meshtype) == "rectangular" );

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
    }

    SECTION( "second segment" )
    {
        // open
        auto file = ovf_open(testfile);
        REQUIRE( file->n_segments == 3 );
        int index = 1;

        // segment header
        auto segment = ovf_segment_create();

        // read header
        int success = ovf_read_segment_header(file, index, segment);
        if( OVF_OK != success )
            std::cerr << ovf_latest_message(file) << std::endl;
        REQUIRE( success == OVF_OK );
        REQUIRE( segment->N == 4 );
        REQUIRE( std::string(segment->meshtype) == "rectangular" );

        // data
        std::vector<double> field(3*segment->N);

        // read data
        success = ovf_read_segment_data_8(file, index, segment, field.data());
        if( OVF_OK != success )
            std::cerr << ovf_latest_message(file) << std::endl;
        REQUIRE( success == OVF_OK );
        REQUIRE( field[0] == 6 );
        REQUIRE( field[1] == 1 );
        REQUIRE( field[3] == 4 );
        REQUIRE( field[6] == 2 );
        REQUIRE( field[9] == 0 );

        // close
        ovf_close(file);
    }

    SECTION( "third segment" )
    {
        // open
        auto file = ovf_open(testfile);
        REQUIRE( file->n_segments == 3 );
        int index = 2;

        // segment header
        auto segment = ovf_segment_create();

        // read header
        int success = ovf_read_segment_header(file, index, segment);
        if( OVF_OK != success )
            std::cerr << ovf_latest_message(file) << std::endl;
        REQUIRE( success == OVF_OK );
        REQUIRE( segment->N == 4 );
        REQUIRE( std::string(segment->meshtype) == "irregular" );
        REQUIRE( segment->pointcount == 4 );

        // data
        std::vector<double> field(3*segment->N);

        // read data
        success = ovf_read_segment_data_8(file, index, segment, field.data());
        if( OVF_OK != success )
            std::cerr << ovf_latest_message(file) << std::endl;
        REQUIRE( success == OVF_OK );
        REQUIRE( field[0] == 6 );
        REQUIRE( field[1] == 1 );
        REQUIRE( field[3] == 4 );
        REQUIRE( field[6] == 2 );
        REQUIRE( field[9] == 0 );

        // close
        ovf_close(file);
    }
}