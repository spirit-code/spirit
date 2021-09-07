#include <catch.hpp>
#include <ovf.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <ovf.h>

#include <vector>
#include <string>
#include <utility>

#include <random>
#include <fmt/format.h>
#include <algorithm>
#include <stdlib.h>
#include <time.h>

enum class VERSION
{
    OVF,
    AOVF,
    AOVF_COMP,
    RANDOM
};

enum class MESHTYPE
{
    IRREGULAR,
    RECTANGULAR,
    LATTICE,
    RANDOM
};

// Helper struct to create a "weirdly" formatted OVF file
struct test_ovf_file
{
    int random_seed;
    bool shuffle = true;
    int n_whitespace = 4;
    int n_skippable_lines = 4;

    std::mt19937 g;

    // Some information we will need for the tests
    bool should_be_valid;
    VERSION version;
    MESHTYPE meshtype;

    std::string file_string = "";
    bool use_magic_char_comment = true;

    using kw_val_pair = std::pair<std::string, std::string>;

    std::vector< kw_val_pair > pairs;
    std::vector< kw_val_pair > pairs_lattice;
    std::vector< kw_val_pair > pairs_irregular;
    std::vector< kw_val_pair > pairs_rectangular;

    test_ovf_file(VERSION version, MESHTYPE meshtype, int random_seed) : version(version), meshtype(meshtype), random_seed(random_seed)
    {
        srand(random_seed);

        pairs_irregular =
        {
            {"meshtype", "irregular"},
            {"pointcount", "4"}
        };

        pairs_lattice =
        {
            {"meshtype", "lattice"},
            {"anodes", "2"},
            {"bnodes", "1"},
            {"cnodes", "1"},
            {"bravaisa", "1 0   0"},
            {"bravaisb", "0   1   0"},
            {"bravaisc", "0   0   1"},
            {"basis", "\n# 0   0  0\n# 0.1 0.1   0.1"},
            {"ncellpoints", "2"}
        };

        pairs_rectangular =
        {
            {"meshtype", "rectangular"},
            {"xbase", "2"},
            {"ybase", "1"},
            {"zbase", "1"},
            {"xstepsize", "0"},
            {"ystepsize", "0"},
            {"zstepsize", "0"},
            {"xnodes", "4"},
            {"ynodes", "1"},
            {"znodes", "1"}
        };

        pairs =
        {
            {"xmin", "0"},
            {"ymin", "0"},
            {"zmin", "0"},
            {"xmax", "0"},
            {"ymax", "0"},
            {"zmax", "0"},
            {"Title", "my a title"},
            {"Desc", "the description"},
            {"valuedim", "  3"},
            {"valueunits", " eV  "},
            {"valuelabels", "   "},
            {"meshunit", " nm  "}
        };
    }

    std::string random_whitespace()
    {
        if( n_whitespace < 1)
            return "";

        std::string result;
        for (int i=0; i<(rand()%n_whitespace); i++)
            result += " ";

        return std::move(result);
    }

    std::string comment()
    {
        std::string result;

        result += "##";
        if(use_magic_char_comment)
        {
            if(rand()%2 == 0)
            {
                result += "%";
            }
        }
        result += " " + random_whitespace() + "comment" + random_whitespace();

        return std::move(result);
    }

    std::string line_end()
    {
        std::string result;
        result += random_whitespace() + "\n";
        return std::move(result);
    }

    void append_skippable_lines( )
    {
        if( n_skippable_lines < 1)
            return;
        for (int i=0; i<(rand()%n_skippable_lines); i++)
        {
            if(rand()%2 == 0)
                file_string += comment() + line_end();
            if(rand()%2 == 0)
                file_string += "#" + line_end();
        }
    }

    std::string separate_with_whitespace(const std::vector<std::string> & in)
    {
        std::string result;
        for(auto & s : in)
        {
            result += random_whitespace( ) + s;
        }
        return std::move(result);
    }

    void append_version()
    {
        if( version == VERSION::OVF)
        {
            file_string += "#" + separate_with_whitespace({"OOMMF OVF ", "2.0"}) + line_end();
        } else if( version == VERSION::AOVF )
        {
            file_string += "#" + separate_with_whitespace({"AOVF ", "1.0"}) + line_end();
        } else if( version == VERSION::AOVF_COMP )
        {
            file_string += "#" + separate_with_whitespace({"OOMMF OVF ", "2.0"}) + line_end();
            file_string += "##%" + separate_with_whitespace({"AOVF ", "1.0"}) + line_end();
            use_magic_char_comment = false;
        }
    }

    void append_segment_count()
    {
        file_string += "# " + separate_with_whitespace({"Segment count: ", "000001"}) + line_end();
    }

    void append_begin_segment()
    {
        file_string += "#" + separate_with_whitespace({"Begin: ", "Segment"}) + line_end();
    }

    void append_end_segment()
    {
        file_string += "#" + separate_with_whitespace({"End: ", "Segment"}) + line_end();
    }

    void append_begin_header()
    {
        file_string += "#" + separate_with_whitespace({"Begin: ", "Header"}) + line_end();
    }

    void append_end_header()
    {
        file_string += "#" + separate_with_whitespace({"End: ", "Header"}) + line_end();
    }

    void append_pairs()
    {
        std::vector<kw_val_pair> file_pairs(0); // pairs with # prefix
        std::vector<kw_val_pair> magic_file_pairs(0); // pairs with ##% prefix

        // Insert all the pairs that always have to be present
        file_pairs.insert(file_pairs.end(), pairs.begin(), pairs.end());

        // if(meshtype == MESHTYPE::RANDOM);
        //     meshtype = static_cast<MESHTYPE>( rand() % (int(MESHTYPE::RANDOM)) );

        if(meshtype == MESHTYPE::LATTICE)
        {
            if(version == VERSION::AOVF_COMP)
            {
                file_pairs.push_back({"meshtype", "rectangular"});
                file_pairs.insert(file_pairs.end(), pairs_rectangular.begin(), pairs_rectangular.end());
            }
            file_pairs.insert(file_pairs.end(), pairs_lattice.begin(), pairs_lattice.end());
        } else if(meshtype == MESHTYPE::RECTANGULAR)
        {
            file_pairs.insert(file_pairs.end(), pairs_rectangular.begin(), pairs_rectangular.end());
        } else if(meshtype == MESHTYPE::IRREGULAR)
        {
            file_pairs.insert(file_pairs.end(), pairs_irregular.begin(), pairs_irregular.end());
        }

        // std::random_device rd;
        g = std::mt19937(random_seed);

        if(shuffle)
            std::shuffle(file_pairs.begin(), file_pairs.end(), g);

        for(auto & p : file_pairs)
        {
            std::string prefix = "#";

            if( meshtype==MESHTYPE::LATTICE && version==VERSION::AOVF_COMP)
            {
                if( std::find( pairs_lattice.begin(), pairs_lattice.end(), p ) != pairs_lattice.end() )
                {
                    prefix += "#%";
                }
            }

            if(p.first == std::string("basis"))
            {
                file_string += prefix + "basis: " + line_end() + prefix + "0  0 0" + line_end() + prefix + "0  0 0"
                + line_end();
            } else {
                file_string += prefix + separate_with_whitespace({p.first + ":", p.second}) + line_end();
            }
            append_skippable_lines();
        }
    }

    void append_data()
    {
        file_string +=
        "# Begin: Data Text\n"
        "0.000000000000        0.000000000000        0.000000000000\n"
        "3.000000000000        2.000000000000        1.000000000000\n"
        "0.000000000000        0.000000000000        0.000000000000\n"
        "0.000000000000        0.000000000000        0.000000000000\n"
        "# End: Data Text\n";
    }

    void write(std::string filename)
    {
        file_string.clear();
        append_version();
        // append_skippable_lines();
        append_segment_count();
        append_begin_segment();
        append_skippable_lines();
        append_begin_header();
        append_skippable_lines();
        append_pairs();
        append_skippable_lines();
        append_end_header();
        append_skippable_lines();
        append_data();
        append_skippable_lines();
        append_end_segment();

        std::ofstream out;
        out.open(filename);
        out << file_string;
        out.close();
    }

};


// open
void test_ovf_read(std::string filename, const test_ovf_file & test_file)
{
    INFO(fmt::format("Testing 'ovf_test_read' with filename '{}', meshtype '{}', ovf_extension_format '{}'", filename, int(test_file.meshtype), int(test_file.version)));

    auto file = ovf_open(filename.c_str());
    std::cerr << ovf_latest_message(file) << std::endl;
    REQUIRE( file->found == true );
    REQUIRE( file->is_ovf == true );
    REQUIRE( file->n_segments == 1);

    // segment header
    auto segment = ovf_segment_create();

    // read header
    int success = ovf_read_segment_header(file, 0, segment);
    if( OVF_OK != success )
        std::cerr << ovf_latest_message(file) << std::endl;
    REQUIRE( success == OVF_OK );
}

TEST_CASE("READ")
{
    test_ovf_file file = test_ovf_file(VERSION::OVF, MESHTYPE::RECTANGULAR, 1337);

    file.n_skippable_lines = 4;
    file.n_whitespace = 4;
    file.shuffle = true;

    for(int i=0; i<3; i++)
    {
        // OVF 2.0
        file.meshtype = MESHTYPE::RECTANGULAR;
        file.write("test_weird_ovf_rectangular.ovf");
        test_ovf_read("test_weird_ovf_rectangular.ovf", file);

        file.meshtype = MESHTYPE::IRREGULAR;
        file.write("test_weird_ovf_irregular.ovf");
        test_ovf_read("test_weird_ovf_irregular.ovf", file);

        // AOVF
        file.version = VERSION::AOVF;
        file.meshtype = MESHTYPE::RECTANGULAR;
        file.write("test_weird_aovf_rectangular.ovf");
        test_ovf_read("test_weird_aovf_rectangular.ovf", file);

        file.meshtype = MESHTYPE::IRREGULAR;
        file.write("test_weird_aovf_irregular.ovf");
        test_ovf_read("test_weird_aovf_irregular.ovf", file);

        file.meshtype = MESHTYPE::LATTICE;
        file.write("test_weird_aovf_lattice.ovf");
        test_ovf_read("test_weird_aovf_lattice.ovf", file);

        // AOVF_COMP
        file.version = VERSION::AOVF_COMP;
        file.meshtype = MESHTYPE::RECTANGULAR;
        file.write("test_weird_caovf_rectangular.ovf");
        test_ovf_read("test_weird_caovf_rectangular.ovf", file);

        file.meshtype = MESHTYPE::IRREGULAR;
        file.write("test_weird_caovf_irregular.ovf");
        test_ovf_read("test_weird_caovf_irregular.ovf", file);

        file.meshtype = MESHTYPE::LATTICE;
        file.write("test_weird_caovf_lattice.ovf");
        test_ovf_read("test_weird_caovf_lattice.ovf", file);
    }
}