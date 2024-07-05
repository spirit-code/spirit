#pragma once

#include <data/Geometry.hpp>
#include <engine/Backend.hpp>
#include <engine/Vectormath_Defines.hpp>

#include <Eigen/Dense>

namespace Engine
{

namespace Backend
{

namespace Functor
{

// functional identity, introduced to the standard library in C++20
struct identity
{
    template<typename T>
    SPIRIT_HOSTDEVICE constexpr T && operator()( T && t ) const noexcept
    {
        return std::forward<T>( t );
    }
};

template<typename Functor>
struct discard
{
    constexpr explicit discard( Functor f ) noexcept : functor( f ) {};

    template<typename... Args>
    SPIRIT_HOSTDEVICE void operator()( Args &&... args )
    {
        functor( std::forward<Args>( args )... );
    }

private:
    Functor functor;
};

template<typename VectorType>
struct dot
{
    [[nodiscard]] SPIRIT_HOSTDEVICE scalar operator()( const VectorType & u, const VectorType & v ) const
    {
        return u.dot( v );
    };
};

template<>
struct dot<scalar>
{
    [[nodiscard]] SPIRIT_HOSTDEVICE constexpr scalar operator()( const scalar u, const scalar v ) const
    {
        return u * v;
    };
};

template<typename VectorType>
struct cross;

template<>
struct cross<Vector3>
{
    [[nodiscard]] SPIRIT_HOSTDEVICE Vector3 operator()( const Vector3 & u, const Vector3 & v ) const
    {
        return u.cross( v );
    };
};

template<typename T>
struct scale
{
    constexpr explicit scale( T value ) noexcept : value( value ) {};

    template<typename Arg>
    [[nodiscard]] SPIRIT_HOSTDEVICE auto operator()( const Arg & arg ) const
    {
        return value * arg;
    }

private:
    T value;
};

} // namespace Functor

namespace cpu
{

using namespace Backend::Functor;

}

namespace cuda
{

using namespace Backend::Functor;

}

} // namespace Backend

} // namespace Engine
