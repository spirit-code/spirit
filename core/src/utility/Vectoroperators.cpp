#include <Vectoroperators.hpp>

// Vector Add Into
void operator+=(std::vector<scalar> &v1, const std::vector<scalar> &v2) {
	for (unsigned int i = 0; i < v1.size(); ++i)
	{
		v1[i] += v2[i];
	}
}

void operator-=(std::vector<scalar> &v1, const std::vector<scalar> &v2) {
	for (unsigned int i = 0; i < v1.size(); ++i)
	{
		v1[i] -= v2[i];
	}
}

// Vector Add
std::vector<scalar> operator+(std::vector<scalar> &v1, const std::vector<scalar> &v2) {
	std::vector<scalar> ret(v1.size());
	for (unsigned int i = 0; i < v1.size(); ++i)
	{
		ret[i] = v1[i];
	}
	ret += v2;
	return ret;
}

// Vector Subtract
std::vector<scalar> operator-(std::vector<scalar> &v1, const std::vector<scalar> &v2) {
	std::vector<scalar> ret(v1.size());
	for (unsigned int i = 0; i < v1.size(); ++i)
	{
		ret[i] = v1[i];
	}
	ret -= v2;
	return ret;
}

// Scalar Product
scalar operator*(std::vector<scalar> &v1, const std::vector<scalar> &v2) {
	scalar ret = 0;
	for (unsigned int i = 0; i < v1.size(); ++i)
	{
		ret += v1[i] * v2[i];
	}
	return ret;
}

// Add Scalar
std::vector<scalar> operator+(std::vector<scalar> &v, const scalar &d) {
	std::vector<scalar> ret(v.size());
	for (unsigned int i = 0; i < v.size(); ++i)
	{
		ret[i] = v[i];
	}
	for (unsigned int i = 0; i < ret.size(); ++i)
	{
		ret[i] += d;
	}
	return ret;
}
std::vector<scalar> operator+(const scalar &d, std::vector<scalar> &v) {
	std::vector<scalar> ret(v.size());
	for (unsigned i = 0; i < v.size(); ++i)
	{
		ret[i] = v[i];
	}
	for (unsigned int i = 0; i < ret.size(); ++i)
	{
		ret[i] += d;
	}
	return ret;
}

// Subtract Scalar
std::vector<scalar> operator-(std::vector<scalar> &v, const scalar &d) {
	std::vector<scalar> ret(v.size());
	for (unsigned int i = 0; i < v.size(); ++i)
	{
		ret[i] = v[i];
	}
	for (unsigned int i = 0; i < ret.size(); ++i)
	{
		ret[i] -= d;
	}
	return ret;
}
std::vector<scalar> operator-(const scalar &d, std::vector<scalar> &v) {
	std::vector<scalar> ret(v.size());
	for (unsigned int i = 0; i < v.size(); ++i)
	{
		ret[i] = v[i];
	}
	for (unsigned int i = 0; i < ret.size(); ++i)
	{
		ret[i] -= d;
	}
	return ret;
}

// Multiply Scalar
std::vector<scalar> operator*(std::vector<scalar> &v, const scalar &d) {
	std::vector<scalar> ret(v.size());
	for (unsigned int i = 0; i < v.size(); ++i)
	{
		ret[i] = v[i];
	}
	for (unsigned int i = 0; i < ret.size(); ++i)
	{
		ret[i] *= d;
	}
	return ret;
}
std::vector<scalar> operator*(const scalar &d, std::vector<scalar> &v) {
	std::vector<scalar> ret(v.size());
	for (unsigned int i = 0; i < v.size(); ++i)
	{
		ret[i] = v[i];
	}
	for (unsigned int i = 0; i < ret.size(); ++i)
	{
		ret[i] *= d;
	}
	return ret;
}

// Divide Scalar
std::vector<scalar> operator/(std::vector<scalar> &v, const scalar &d) {
	std::vector<scalar> ret(v.size());
	for (unsigned int i = 0; i < v.size(); ++i)
	{
		ret[i] = v[i];
	}
	for (unsigned int i = 0; i < ret.size(); ++i)
	{
		ret[i] /= d;
	}
	return ret;
}
std::vector<scalar> operator/(const scalar &d, std::vector<scalar> &v) {
	std::vector<scalar> ret(v.size());
	for (unsigned int i = 0; i < v.size(); ++i)
	{
		ret[i] = v[i];
	}
	for (unsigned int i = 0; i < ret.size(); ++i)
	{
		ret[i] /= d;
	}
	return ret;
}

// Vector Cross Product (1D, 3 Elements) ////// AT SOME POINT REPLACE WITH http://www.cplusplus.com/reference/numeric/inner_product/
std::vector<scalar> operator%(std::vector<scalar> const &v1, std::vector<scalar> const &v2) {
	std::vector<scalar> ret(v1.size());
	ret[0] = v1[1] * v2[2] - v1[2] * v2[1];
	ret[1] = v1[2] * v2[0] - v1[0] * v2[2];
	ret[2] = v1[0] * v2[1] - v1[1] * v2[0];
	return ret;
}


// Sum over elements	////// EVENTUALLY REPLACE WITH http://en.cppreference.com/w/cpp/experimental/reduce
scalar sum(std::vector<scalar> v)
{
	scalar ret = 0;
	for (unsigned int i = 0; i < v.size(); ++i)
	{
		ret += v[i];
	}
	return ret;
}