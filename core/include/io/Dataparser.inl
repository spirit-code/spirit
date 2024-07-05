#pragma once

#include <io/Dataparser.hpp>

namespace IO
{

namespace Spin
{

namespace State
{

template<typename T>
class Buffer
{
    static_assert( std::is_same_v<std::remove_const_t<T>, vectorfield> );
    using pointer = std::conditional_t<std::is_const_v<T>, const T *, T *>;

public:
    explicit Buffer( T & state ) : m_state( &state ) {};

    scalar * data()
    {
        static_assert( !std::is_const_v<T> );

        if( size() == 0 )
            return nullptr;
        return m_state->data()->data();
    };

    const scalar * data() const
    {
        if( size() == 0 )
            return nullptr;
        return m_state->data()->data();
    };

    std::size_t size() const
    {
        if( m_state == nullptr )
            return 0;
        return m_state->size();
    };

    friend void copy( const Buffer & buffer, vectorfield & state )
    {
        if( buffer.m_state == &state )
            return;
        if( buffer.m_state == nullptr )
            state.clear();

        std::copy( buffer.m_state->begin(), buffer.m_state->end(), state.begin() );
    }

private:
    pointer m_state;
};

} // namespace State

} // namespace Spin

} // namespace IO
