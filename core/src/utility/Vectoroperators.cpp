#include <Vectoroperators.h>

// Vector Add Into
void operator+=(std::vector<double> &v1, const std::vector<double> &v2) {
	for (unsigned int i = 0; i < v1.size(); ++i)
	{
		v1[i] += v2[i];
	}
}

void operator-=(std::vector<double> &v1, const std::vector<double> &v2) {
	for (unsigned int i = 0; i < v1.size(); ++i)
	{
		v1[i] -= v2[i];
	}
}

// Vector Add
std::vector<double> operator+(std::vector<double> &v1, const std::vector<double> &v2) {
	std::vector<double> ret(v1.size());
	for (unsigned int i = 0; i < v1.size(); ++i)
	{
		ret[i] = v1[i];
	}
	ret += v2;
	return ret;
}

// Vector Subtract
std::vector<double> operator-(std::vector<double> &v1, const std::vector<double> &v2) {
	std::vector<double> ret(v1.size());
	for (unsigned int i = 0; i < v1.size(); ++i)
	{
		ret[i] = v1[i];
	}
	ret -= v2;
	return ret;
}

// Scalar Product
double operator*(std::vector<double> &v1, const std::vector<double> &v2) {
	double ret = 0;
	for (unsigned int i = 0; i < v1.size(); ++i)
	{
		ret += v1[i] * v2[i];
	}
	return ret;
}

// Add Scalar
std::vector<double> operator+(std::vector<double> &v, const double &d) {
	std::vector<double> ret(v.size());
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
std::vector<double> operator+(const double &d, std::vector<double> &v) {
	std::vector<double> ret(v.size());
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
std::vector<double> operator-(std::vector<double> &v, const double &d) {
	std::vector<double> ret(v.size());
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
std::vector<double> operator-(const double &d, std::vector<double> &v) {
	std::vector<double> ret(v.size());
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
std::vector<double> operator*(std::vector<double> &v, const double &d) {
	std::vector<double> ret(v.size());
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
std::vector<double> operator*(const double &d, std::vector<double> &v) {
	std::vector<double> ret(v.size());
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
std::vector<double> operator/(std::vector<double> &v, const double &d) {
	std::vector<double> ret(v.size());
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
std::vector<double> operator/(const double &d, std::vector<double> &v) {
	std::vector<double> ret(v.size());
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
std::vector<double> operator%(std::vector<double> const &v1, std::vector<double> const &v2) {
	std::vector<double> ret(v1.size());
	ret[0] = v1[1] * v2[2] - v1[2] * v2[1];
	ret[1] = v1[2] * v2[0] - v1[0] * v2[2];
	ret[2] = v1[0] * v2[1] - v1[1] * v2[0];
	return ret;
}


// Sum over elements	////// EVENTUALLY REPLACE WITH http://en.cppreference.com/w/cpp/experimental/reduce
double sum(std::vector<double> v)
{
	double ret = 0;
	for (unsigned int i = 0; i < v.size(); ++i)
	{
		ret += v[i];
	}
	return ret;
}