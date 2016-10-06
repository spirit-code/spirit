#pragma once
#ifndef UTILITY_VECTOROPERATORS_H
#define UTILITY_VECTOROPERATORS_H

#include <vector>

// Vector Add Into
void operator+=(std::vector<double> &v1, const std::vector<double> &v2);

void operator-=(std::vector<double> &v1, const std::vector<double> &v2);

// Vector Add
std::vector<double> operator+(std::vector<double> &v1, const std::vector<double> &v2);

// Vector Subtract
std::vector<double> operator-(std::vector<double> &v1, const std::vector<double> &v2);

// Scalar Product
double operator*(std::vector<double> &v1, const std::vector<double> &v2);

// Add Scalar
std::vector<double> operator+(std::vector<double> &v, const double &d);
std::vector<double> operator+(const double &d, std::vector<double> &v);

// Subtract Scalar
std::vector<double> operator-(std::vector<double> &v, const double &d);
std::vector<double> operator-(const double &d, std::vector<double> &v);

// Multiply Scalar
std::vector<double> operator*(std::vector<double> &v, const double &d);
std::vector<double> operator*(const double &d, std::vector<double> &v);

// Divide Scalar
std::vector<double> operator/(std::vector<double> &v, const double &d);
std::vector<double> operator/(const double &d, std::vector<double> &v);

// Vector Cross Product (1D, 3 Elements) ////// AT SOME POINT REPLACE WITH http://www.cplusplus.com/reference/numeric/inner_product/
std::vector<double> operator%(std::vector<double> const &v1, std::vector<double> const &v2);


// Sum over elements	////// EVENTUALLY REPLACE WITH http://en.cppreference.com/w/cpp/experimental/reduce
double sum(std::vector<double> v);

#endif