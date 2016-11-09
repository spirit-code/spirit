#pragma once
#ifndef UTILITY_VECTOROPERATORS_H
#define UTILITY_VECTOROPERATORS_H

#include <vector>

#include "Core_Defines.h"

// Vector Add Into
void operator+=(std::vector<scalar> &v1, const std::vector<scalar> &v2);

void operator-=(std::vector<scalar> &v1, const std::vector<scalar> &v2);

// Vector Add
std::vector<scalar> operator+(std::vector<scalar> &v1, const std::vector<scalar> &v2);

// Vector Subtract
std::vector<scalar> operator-(std::vector<scalar> &v1, const std::vector<scalar> &v2);

// Scalar Product
scalar operator*(std::vector<scalar> &v1, const std::vector<scalar> &v2);

// Add Scalar
std::vector<scalar> operator+(std::vector<scalar> &v, const scalar &d);
std::vector<scalar> operator+(const scalar &d, std::vector<scalar> &v);

// Subtract Scalar
std::vector<scalar> operator-(std::vector<scalar> &v, const scalar &d);
std::vector<scalar> operator-(const scalar &d, std::vector<scalar> &v);

// Multiply Scalar
std::vector<scalar> operator*(std::vector<scalar> &v, const scalar &d);
std::vector<scalar> operator*(const scalar &d, std::vector<scalar> &v);

// Divide Scalar
std::vector<scalar> operator/(std::vector<scalar> &v, const scalar &d);
std::vector<scalar> operator/(const scalar &d, std::vector<scalar> &v);

// Vector Cross Product (1D, 3 Elements) ////// AT SOME POINT REPLACE WITH http://www.cplusplus.com/reference/numeric/inner_product/
std::vector<scalar> operator%(std::vector<scalar> const &v1, std::vector<scalar> const &v2);


// Sum over elements	////// EVENTUALLY REPLACE WITH http://en.cppreference.com/w/cpp/experimental/reduce
scalar sum(std::vector<scalar> v);

#endif