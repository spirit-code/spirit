#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
//#include <sstream>
#include "catch.hpp"

#include <chaiscript/chaiscript.hpp>
#include <chaiscript/chaiscript_stdlib.hpp>
#include "../include/chaiscript/extras/string_id.hpp"

#include <iostream>

TEST_CASE( "string_id functions work", "[string_id]" ) {
  auto stdlib = chaiscript::Std_Lib::library();
  auto string_idlib = chaiscript::extras::string_id::bootstrap();

  chaiscript::ChaiScript chai(stdlib);
  chai.add(string_idlib);

  foonathan::string_id::default_database database;
  chai.add(chaiscript::var(std::ref(database)), "database");

  chai.eval(R""(
    var id = string_id("Test0815", database);
  )"");

  using namespace foonathan::string_id::literals;
  CHECK(chai.eval<const foonathan::string_id::string_id &>("id") == "Test0815");
}


