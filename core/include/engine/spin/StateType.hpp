#pragma once

#include <engine/Vectormath_Defines.hpp>

#include <Eigen/Core>

namespace Engine
{

namespace Spin
{

enum struct Field
{
    Spin = 0,
};

template<typename T>
using quantity = T;

template<Field field, typename T>
T & get( T & q )
{
    return q;
}

template<Field field, typename T>
const T & get( const T & q )
{
    return q;
}

using StateType = vectorfield;
using StatePtr  = Vector3 *;
using StateCPtr = const Vector3 *;

}

template<typename state_type>
struct state_traits;

template<>
struct state_traits<Spin::StateType>
{
    using type            = Spin::StateType;
    using pointer         = Spin::StateType::pointer;
    using const_pointer   = Spin::StateType::const_pointer;
};

template<typename state_t>
state_t make_state( int nos );

template<>
inline Spin::StateType make_state( int nos ){
    return vectorfield( nos );
};

}
