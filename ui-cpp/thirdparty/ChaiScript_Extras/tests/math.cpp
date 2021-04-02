#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
//#include <sstream>
#include "catch.hpp"

#include <chaiscript/chaiscript.hpp>
#include <chaiscript/chaiscript_stdlib.hpp>
#include "../include/chaiscript/extras/math.hpp"

#include <iostream>

TEST_CASE( "Math functions work", "[math]" ) {
  auto mathlib = chaiscript::extras::math::bootstrap();

  auto stdlib = chaiscript::Std_Lib::library();
  chaiscript::ChaiScript chai(stdlib);
  chai.add(mathlib);

  // TRIG FUNCTIONS
  CHECK(chai.eval<double>("cos(0.5)") == cos(0.5));
  CHECK(chai.eval<double>("sin(0.5)") == sin(0.5));
  CHECK(chai.eval<double>("tan(0.5)") == tan(0.5));

  CHECK(chai.eval<double>("acos(0.5)") == acos(0.5));
  CHECK(chai.eval<double>("asin(0.5)") == asin(0.5));
  CHECK(chai.eval<double>("atan(0.5)") == atan(0.5));
  CHECK(chai.eval<double>("atan2(0.5, 0.5)") == atan2(0.5, 0.5));

  // HYPERBOLIC FUNCTIONS
  CHECK(chai.eval<double>("cosh(0.5)") == cosh(0.5));
  CHECK(chai.eval<double>("sinh(0.5)") == sinh(0.5));
  CHECK(chai.eval<double>("tanh(0.5)") == tanh(0.5));

  CHECK(chai.eval<double>("acosh(1.0)") == acosh(1.0));
  CHECK(chai.eval<double>("asinh(0.5)") == asinh(0.5));
  CHECK(chai.eval<double>("atanh(0.5)") == Approx(atanh(0.5)));

  // EXPONENTIAL AND LOGARITHMIC FUNCTIONS
  CHECK(chai.eval<double>("exp(0.5)") == exp(0.5));
  int exp = 0; CHECK(chai.eval<double>("var exp = 2\nfrexp(0.5, exp)") == frexp(0.5, &exp));
  CHECK(chai.eval<double>("ldexp(0.5, 2)") == ldexp(0.5, 2));
  CHECK(chai.eval<double>("log(0.5)") == log(0.5));
  CHECK(chai.eval<double>("log10(0.5)") == log10(0.5));
  double ipart = 0.5; CHECK(chai.eval<double>("var ipart = 0.5\nmodf(0.5, ipart)") == modf(0.5, &ipart));
  CHECK(chai.eval<double>("exp2(0.5)") == exp2(0.5));
  CHECK(chai.eval<double>("expm1(0.5)") == expm1(0.5));
  CHECK(chai.eval<int>("ilogb(0.5)") == ilogb(0.5));
  CHECK(chai.eval<double>("log1p(0.5)") == log1p(0.5));
  CHECK(chai.eval<double>("log2(0.5)") == log2(0.5));
  CHECK(chai.eval<double>("logb(0.5)") == ilogb(0.5));
  CHECK(chai.eval<double>("scalbn(0.5, 2)") == scalbn(0.5, 2));
  CHECK(chai.eval<double>("scalbln(0.5, 2l)") == scalbln(0.5, 2));

  // POWER FUNCTIONS
  CHECK(chai.eval<double>("pow(0.5, 3.0)") == pow(0.5, 3.0));
  CHECK(chai.eval<double>("sqrt(0.5)") == sqrt(0.5));
  CHECK(chai.eval<double>("cbrt(0.5)") == cbrt(0.5));
  CHECK(chai.eval<double>("hypot(0.5, 0.5)") == hypot(0.5, 0.5));

  // ERROR AND GAMMA FUNCTIONS
  CHECK(chai.eval<double>("erf(0.5)") == erf(0.5));
  CHECK(chai.eval<double>("erfc(0.5)") == erfc(0.5));
  CHECK(chai.eval<double>("tgamma(0.5)") == tgamma(0.5));
  CHECK(chai.eval<double>("lgamma(0.5)") == lgamma(0.5));

  // ROUNDING AND REMAINDER FUNCTIONS
  CHECK(chai.eval<double>("ceil(0.5)") == ceil(0.5));
  CHECK(chai.eval<double>("floor(0.5)") == floor(0.5));
  CHECK(chai.eval<double>("fmod(0.5, 0.5)") == fmod(0.5, 0.5));
  CHECK(chai.eval<double>("trunc(0.5)") == trunc(0.5));
  CHECK(chai.eval<double>("round(0.5)") == round(0.5));
  CHECK(chai.eval<long int>("lround(0.5)") == lround(0.5));
  CHECK(chai.eval<long long int>("llround(0.5)") == llround(0.5)); // long longs do not work
  CHECK(chai.eval<double>("rint(0.5)") == rint(0.5));
  CHECK(chai.eval<long int>("lrint(0.5)") == lrint(0.5));
  CHECK(chai.eval<long long int>("llrint(0.5)") == llrint(0.5));
  CHECK(chai.eval<double>("nearbyint(0.5)") == nearbyint(0.5));
  CHECK(chai.eval<double>("remainder(6.0, 2.5)") == remainder(6.0, 2.5));
  int quot = 0; CHECK(chai.eval<double>("var quot = 0\nremquo(6.0, 2.5, quot)") == remquo(6.0, 2.5, &quot));

  // MINIMUM, MAXIMUM, DIFFERENCE FUNCTIONS
  CHECK(chai.eval<double>("fdim(6.0, 2.5)") == fdim(6.0, 2.5));
  CHECK(chai.eval<double>("fmax(6.0, 2.5)") == fmax(6.0, 2.5));
  CHECK(chai.eval<double>("fmin(6.0, 2.5)") == fmin(6.0, 2.5));

  // OTHER FUNCTIONS
  CHECK(chai.eval<double>("fabs(-0.5)") == fabs(-0.5));
  CHECK(chai.eval<double>("abs(-0.5)") == std::abs(-0.5));
  CHECK(chai.eval<double>("fma(0.5, 0.5, 0.5)") == fma(0.5, 0.5, 0.5));

  // CLASSIFICATION FUNCTIONS
  CHECK(chai.eval<int>("fpclassify(0.5)") == std::fpclassify(0.5));
  CHECK(chai.eval<bool>("isfinite(0.5)") == std::isfinite(0.5));
  CHECK(chai.eval<bool>("isinf(0.5)") == std::isinf(0.5));
  CHECK(chai.eval<bool>("isnan(0.5)") == std::isnan(0.5));
  CHECK(chai.eval<bool>("isnormal(0.5)") == std::isnormal(0.5));
  CHECK(chai.eval<bool>("signbit(0.5)") == std::signbit(0.5));

  // COMPARISON FUNCTIONS
  CHECK(chai.eval<bool>("isgreater(1.0, 0.5)") == std::isgreater(1.0, 0.5));
  CHECK(chai.eval<bool>("isgreaterequal(1.0, 0.5)") == std::isgreaterequal(1.0, 0.5));
  CHECK(chai.eval<bool>("isless(1.0, 0.5)") == std::isless(1.0, 0.5));
  CHECK(chai.eval<bool>("islessequal(1.0, 0.5)") == std::islessequal(1.0, 0.5));
  CHECK(chai.eval<bool>("islessgreater(1.0, 0.5)") == std::islessgreater(1.0, 0.5));
  CHECK(chai.eval<bool>("isunordered(1.0, 0.5)") == std::isunordered(1.0, 0.5));
}


