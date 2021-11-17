#pragma once
#ifndef SPIRIT_CORE_ENGINE_BACKEND_CPU_HPP
#define SPIRIT_CORE_ENGINE_BACKEND_CPU_HPP

#include <engine/Vectormath_Defines.hpp>

namespace Engine
{
namespace Backend
{
namespace seq
{
template<typename A, typename F>
scalar reduce( const field<A> & vf1, const F f )
{
    scalar res = 0;
#pragma omp parallel for reduction( + : res )
    for( unsigned int idx = 0; idx < vf1.size(); ++idx )
    {
        res += f( vf1[idx] );
    }
    return res;
}

// result = sum_i  f( vf1[i], vf2[i] )
template<typename A, typename B, typename F>
scalar reduce( const field<A> & vf1, const field<B> & vf2, const F & f )
{
    scalar res = 0;
    for( unsigned int idx = 0; idx < vf1.size(); ++idx )
    {
        res += f( vf1[idx], vf2[idx] );
    }
    return res;
}

// vf1[i] = f( vf2[i] )
template<typename A, typename B, typename F>
void set( field<A> & vf1, const field<B> & vf2, const F & f )
{
    for( unsigned int idx = 0; idx < vf1.size(); ++idx )
    {
        vf1[idx] = f( vf2[idx] );
    }
}

// vf1[i] = f(i)
template<typename A, typename F>
void set( field<A> & vf1, const F & f )
{
    for( unsigned int idx = 0; idx < vf1.size(); ++idx )
    {
        vf1[idx] = f( idx );
    }
}

// f( vf1[idx], idx ) for all i
template<typename F>
void apply( int N, const F & f )
{
    for( unsigned int idx = 0; idx < N; ++idx )
    {
        f( idx );
    }
}

} // namespace seq
} // namespace Backend
} // namespace Engine

#endif