#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <string>
#include "catch.hpp"

#include <chaiscript/chaiscript.hpp>
#include <chaiscript/chaiscript_stdlib.hpp>
#include "../include/chaiscript/extras/string_methods.hpp"

TEST_CASE( "string_methods functions work", "[string_methods]" ) {
  // Create the ChaiScript environment with stdlib available.
  auto stdlib = chaiscript::Std_Lib::library();
  chaiscript::ChaiScript chai(stdlib);

  // Add the string_methods module.
  auto stringmethods = chaiscript::extras::string_methods::bootstrap();
  chai.add(chaiscript::bootstrap::standard_library::vector_type<std::vector<std::string>>("StringVector"));
  chai.add(stringmethods);

  // replace(string, string)
  CHECK(chai.eval<std::string>("\"Hello World!\".replace(\"Hello\", \"Goodbye\")") == "Goodbye World!");

  // replace(char, char)
  CHECK(chai.eval<std::string>("\"Hello World!\".replace('l', 'r')") == "Herro Worrd!");

  // trim()
  CHECK(chai.eval<std::string>("\"   Hello World!    \".trim()") == "Hello World!");
  CHECK(chai.eval<std::string>("\"   Hello World!    \".trimStart()") == "Hello World!    ");
  CHECK(chai.eval<std::string>("\"   Hello World!    \".trimEnd()") == "   Hello World!");

  // split()
  CHECK(chai.eval<std::string>("\"Hello,World,How,Are,You\".split(\",\")[1]") == "World");
  CHECK(chai.eval<std::string>("split(\"Hello,World,How,Are,You\", \",\")[1]") == "World");

  // toLowerCase()
  CHECK(chai.eval<std::string>("\"HeLLO WoRLD!\".toLowerCase()") == "hello world!");

  // toUpperCase()
  CHECK(chai.eval<std::string>("\"Hello World!\".toUpperCase()") == "HELLO WORLD!");

  // includes()
  CHECK(chai.eval<bool>("\"Hello World!\".includes(\"orl\")") == true);
  CHECK(chai.eval<bool>("\"Hello World!\".includes(\"Not Included\")") == false);
  CHECK(chai.eval<bool>("\"Hello World!\".includes('l')") == true);
  CHECK(chai.eval<bool>("\"Hello World!\".includes('a')") == false);
}
