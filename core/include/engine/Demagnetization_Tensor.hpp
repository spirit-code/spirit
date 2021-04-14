#pragma once
#ifndef DEMAGNETIZATION_TENSOR_HPP
#define DEMAGNETIZATION_TENSOR_HPP

#include <utility/Constants.hpp>
#include <cmath>
#include <limits>

namespace Engine
{
namespace Demagnetization_Tensor
{
    using namespace Utility;
    namespace Exact
    {
        // Helper functions
        template<typename scalar>
        scalar kappa(const scalar & x, const scalar & y, const scalar & z, const scalar & R)
        {
            // If the argument of the log gets large, it is zeroed out by the prefactor in f or g. Therefore we just set it to zero here, to avoid division by zero.
            auto res = std::log((x + R) / (std::sqrt(y*y + z*z)));
            if(std::isnan(res) || std::isinf(res))
                res = 0;
            return res;
        }

        template<typename scalar>
        scalar delta(const scalar & x, const scalar & y, const scalar & z, const scalar & R)
        {
            auto arg = x*y / (z*R);
            auto res = std::atan(arg);

            // If the arg is infinite atan(arg) will give +- Pi/2 depending on sign
            // The std::atan function should know about this, but we do not rely on it here
            if(std::isinf(arg))
                if(arg<0)
                    return -Constants::Pi/2;
                else
                    return Constants::Pi/2;

            // If arg is nan it most likely means a division 0/0 ocurred, 
            // we just return 0 because the delta function is cancelled by prefactors in that case
            if(std::isnan(arg))
                return 0;

            return res;
        }

        // Helper function for Nxx. Symmetric in z and y.
        template<typename scalar>
        scalar f(const scalar & x, const scalar & y, const scalar & z)
        {
            scalar R = std::sqrt(x*x + y*y + z*z);
            return (
                      (y/2.0) * (z-x) * (z+x) * kappa(y, x, z, R)
                    + (z/2.0) * (y-x) * (y+x) * kappa(z, x, y, R)
                    - x*y*z   * delta(y, z, x, R)
                    + 1.0/6.0 * ((x-y)*(x+y) + (x-z)*(x+z)) * R
                );
        }

        // Helper function for Nxy. Symmetric in x and y.
        template<typename scalar>
        scalar g(const scalar & x, const scalar & y, const scalar & z)
        {
            scalar R = std::sqrt(x*x + y*y + z*z);
            return ( 
                          (x*y*z)                 * kappa(z, x, y, R)
                        + (y/6.0) * (3*z*z - y*y) * kappa(x, y, z, R)
                        + (x/6.0) * (3*z*z - x*x) * kappa(y, x, z, R)
                        - (z*z*z/6.0)             * delta(x, y, z, R)
                        - (z*y*y/2.0)             * delta(x, z, y, R)
                        - (z*x*x/2.0)             * delta(y, z, x, R)
                        - x*y*R/3.0
                    );
        }

        template<typename scalar>
        scalar gamma(const scalar & e1, const scalar & e2, const scalar & e3)
        {
            return 8.0/std::pow(-2, std::abs(e1) + std::abs(e2) + std::abs(e3));
        }

        // Exact term for the demagnetization tensor Nxx and Nxy components, see Newell 1993
        // These formulas suffer from loss of significant digits as the distance increases
        // Therefore they also return an estimate of their error, which helps to guide the use of asymptotic formulas. 
        // See Donahue "Accurate computation of thedemagnetization tensor"

        template<typename scalar>
        scalar Nxx(const scalar & X, const scalar & Y, const scalar & Z, const scalar & dx, const scalar & dy, const scalar & dz, scalar & abs_error)
        {
            scalar cur_max = 0;
            scalar res = 0;
            for (int e1=-1; e1<=1; e1++)
            {
                for (int e2=-1; e2<=1; e2++)
                {
                    for(int e3=-1; e3<=1; e3++)
                    {
                        auto tmp = gamma(e1,e2,e3) / (4*Constants::Pi * dx*dy*dz) * f(X+e1*dx, Y+e2*dy, Z+e3*dz);
                        res += tmp;
                        if(std::abs(tmp) > cur_max)
                            cur_max = std::abs(tmp);
                        if(std::abs(tmp) > cur_max)
                            cur_max = std::abs(res);
                    }
                }
            }
            // The main sources of error are temporary values with large magnitudes, while the final result of the sum is small in magnitude.
            // Therefore, we approximate the absolute error of the sum as the abosolute error of the largest summand.
            abs_error = std::abs( cur_max ) / std::pow(10, std::numeric_limits<scalar>::digits10);
            return res;
        }

        // See Nxx comment.
        template<typename scalar>
        scalar Nxy(const scalar & X, const scalar & Y, const scalar & Z, const scalar & dx, const scalar & dy, const scalar & dz, scalar & abs_error)
        {
            scalar cur_max = 0;
            scalar res = 0;
            for (int e1=-1; e1<=1; e1++)
            {
                for (int e2=-1; e2<=1; e2++)
                {
                    for(int e3=-1; e3<=1; e3++)
                    {
                        auto tmp = gamma(e1,e2,e3) / (4 * Constants::Pi * dx*dy*dz) * g(X+e1*dx, Y+e2*dy, Z+e3*dz);
                        res += tmp;
                        if(std::abs(tmp) > cur_max)
                            cur_max = std::abs(tmp);
                        if(std::abs(res) > cur_max)
                            cur_max = std::abs(res);
                    }
                }
            }
            abs_error = std::abs( cur_max ) / std::pow(10, std::numeric_limits<scalar>::digits10);
            return res;
        }
    }

    namespace Asymptote
    {
        // The exact formula can be rewritten as 
        // Nxx = 1/(4*pi*dx*dy*dz) * 2(cosh(dx * del_x)-1) * 2(cosh(dy * del_y)-1) * 2(cosh(dz * del_z)-1) * f(x,y,z)
        // where del_x is to be understood as the partial derivative wrt x etc. For Nxy replace f with g.
        // The cosh terms compute the finite difference terms, in the exact formula, via taylor series
        //     e.g: h^2 f''(x) ~ f(x+h) - 2*f(x) + f(x-h) = h^2 * 2(cosh(h * d/dx) - 1)
        // We compute the asymptotes by expanding 2(cosh(h * d/dx) - 1) up to finitely many terms
        // The first term we get is:
        //     1/(4 * pi) * dx dy dz (del_x^2 del_y^2 del_z^2 f(x,y,z)),
        // which turns out to be just the dipole approximation.
        // The following asymptotes also include the next higher terms.
        // See Donahue "Accurate computation of thedemagnetization tensor".

        // Implements del_x^2 del_y^2 del_z^2 f(x,y,z)
        template<typename scalar>
        scalar f2(const scalar & x, const scalar & y, const scalar & z, const scalar & R)
        {
            scalar R2 = R*R;
            scalar R5 = R*R*R*R*R;
            return (3.0*x*x - R*R)/R5;
        }

        // Implements del_x^4 del_y^2 del_z^2 f(x,y,z)
        template<typename scalar>
        scalar f2xx(const scalar & x, const scalar & y, const scalar & z, const scalar & R)
        {
            scalar R2 = R*R;
            scalar R5 = R*R*R*R*R;
            return (-40.0*x*x/R2 - 5.0 * (7.0*x*x/R2 - 1) * (-2.0*x*x + y*y + z*z) / R2 + 4.0) / R5;
        }

        // Implements del_x^2 del_y^4 del_z^2 f(x,y,z)
        template<typename scalar>
        scalar f2yy(const scalar & x, const scalar & y, const scalar & z, const scalar & R)
        {
            scalar R2 = R*R;
            scalar R5 = R*R*R*R*R;
            return (20.0*y*y/R2 - 5.0 * (7.0*y*y/R2 - 1.0) * (-2.0*x*x + y*y + z*z) / R2 - 2.0) / R5;
        }

        // Implements del_x^2 del_y^2 del_z^4 f(x,y,z)
        template<typename scalar>
        scalar f2zz(const scalar & x, const scalar & y, const scalar & z, const scalar & R)
        {
            return f2yy(x,z,y,R); // Swap y and z, since f(x,y,z) is symmetric in these
        }

        // Implements del_x^2 del_y^2 del_z^2 g(x,y,z)
        template<typename scalar>
        scalar g2(const scalar & x, const scalar & y, const scalar & z, const scalar & R)
        {
            scalar R2 = R*R;
            scalar R5 = R*R*R*R*R;
            return 3.0*x*y/R5;
        }

        // Implements del_x^4 del_y^2 del_z^2 g(x,y,z)
        template<typename scalar>
        scalar g2xx(const scalar & x, const scalar & y, const scalar & z, const scalar & R)
        {
            scalar R2 = R*R;
            scalar R7 = R*R*R*R*R*R*R;
            return 15.0*x*y*(7.0*x*x/R2 - 3.0)/R7;
        }

        // Implements del_x^2 del_y^4 del_z^2 g(x,y,z)
        template<typename scalar>
        scalar g2yy(const scalar & x, const scalar & y, const scalar & z, const scalar & R)
        {
            return g2xx(y,x,z,R); // Swap x and y, since g(x,y,z) is symmetric in these
        }

        // Implements del_x^2 del_y^2 del_z^4 g(x,y,z)
        template<typename scalar>
        scalar g2zz(const scalar & x, const scalar & y, const scalar & z, const scalar & R)
        {
            scalar R2 = R*R;
            scalar R7 = R*R*R*R*R*R*R;
            return 15.0 * x * y * (7.0*z*z/R2 - 1.0)/R7;
        }

        template<typename scalar>
        scalar Nxx_asym(const scalar & X, const scalar & Y, const scalar & Z, const scalar & dx, const scalar & dy, const scalar & dz)
        {
            scalar R = std::sqrt(X*X + Y*Y + Z*Z);
            return -1.0/(4.0 * Constants::Pi) * dx*dy*dz * ( f2(X,Y,Z,R) + 1.0/12.0 * ( dx*dx * f2xx(X,Y,Z,R) + dy*dy * f2yy(X,Y,Z,R) + dz*dz * f2zz(X,Y,Z,R) ) );
        }

        template<typename scalar>
        scalar Nxy_asym(const scalar & X, const scalar & Y, const scalar & Z, const scalar & dx, const scalar & dy, const scalar & dz)
        {
            scalar R = std::sqrt(X*X + Y*Y + Z*Z);
            return -1.0/(4.0 * Constants::Pi) * dx*dy*dz * ( g2(X,Y,Z,R) + 1.0/12.0 * ( dx*dx * g2xx(X,Y,Z,R) + dy*dy * g2yy(X,Y,Z,R) + dz*dz * g2zz(X,Y,Z,R) ) );
        }
    }

    namespace Automatic
    {
        // These functions implement an automatic switching between the asymptotic expression and the exact one.
        // Based on the floating point precision and the reported estimated error
        template<typename scalar>
        scalar Nxx(const scalar & X, const scalar & Y, const scalar & Z, const scalar & dx, const scalar & dy, const scalar & dz)
        {
            scalar abs_error  = 0;
            auto nxx_analytical       = Exact::Nxx<scalar>(X,Y,Z,dx,dy,dz,abs_error);
            auto nxx_asym             = Asymptote::Nxx_asym<scalar>(X,Y,Z,dx,dy,dz);

            // If the asymptote is within the error due to loss of significance we use it instead of the exact formula
            if(std::abs(nxx_analytical - nxx_asym) < 10*abs_error)
                return nxx_asym;

            return nxx_analytical;
        }

        template<typename scalar>
        scalar Nxy(const scalar & X, const scalar & Y, const scalar & Z, const scalar & dx, const scalar & dy, const scalar & dz)
        {
            scalar abs_error  = 0;
            auto nxy_analytical = Exact::Nxy<scalar>(X,Y,Z,dx,dy,dz,abs_error);
            auto nxy_asym       = Asymptote::Nxy_asym<scalar>(X,Y,Z,dx,dy,dz);

            // std::cout << "nxy_analytical " << nxy_analytical << "\n";
            // std::cout << "nxy_asym " << nxy_asym << "\n";
            // std::cout << "abs_error " << abs_error << "\n";

            // If the asymptote is within the error due to loss of significance we use it instead of the exact formula
            if(std::abs(nxy_analytical - nxy_asym) < 10*abs_error)
                return nxy_asym;

            return nxy_analytical;
        }
    }
}
}
#endif