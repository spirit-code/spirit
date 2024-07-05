#ifndef SPIRIT_ENGINE_SPAN
#define SPIRIT_ENGINE_SPAN

#include <engine/Vectormath_Defines.hpp>
#include <utility/Exception.hpp>

#include <cstddef>
#include <iterator>
#include <type_traits>

namespace Engine
{

/*
 * lightweight alternative to C++20's `std::span`: a non owning view into contiguous memory.
 * should be replacable by `std::span` once this code base uses C++20.
 */
template<typename T>
class Span
{
public:
    static_assert( !std::is_void<T>::value );

    using value_type       = T;
    using const_value_type = typename std::conditional<std::is_const<T>::value, T, const T>::type;
    using pointer          = T *;
    using const_pointer    = const_value_type *;
    using reference        = T &;
    using const_reference  = const_value_type &;
    using size_type        = std::size_t;

    constexpr Span() noexcept = default;

    constexpr Span( T * data, size_type size ) noexcept : data_( data ), size_( size ) {};

    template<typename Iterator>
    constexpr Span( Iterator begin, size_type size ) noexcept : data_( std::addressof( *begin ) ), size_( size ){};

    template<typename Iterator>
    constexpr Span( Iterator begin, Iterator end ) noexcept
            : data_( std::addressof( *begin ) ), size_( static_cast<size_type>( std::distance( begin, end ) ) ){};

    SPIRIT_HOSTDEVICE constexpr pointer begin()
    {
        return data_;
    }
    SPIRIT_HOSTDEVICE constexpr pointer end()
    {
        return data_ + size_;
    }

    SPIRIT_HOSTDEVICE constexpr const_pointer begin() const
    {
        return data_;
    }
    SPIRIT_HOSTDEVICE constexpr const_pointer end() const
    {
        return data_ + size_;
    }

    SPIRIT_HOSTDEVICE constexpr size_type size() const
    {
        return size_;
    }

    SPIRIT_HOSTDEVICE constexpr reference at( std::size_t index )
    {
        if( ( std::is_signed<decltype( index )>::value && index < 0 ) || index >= size_ )
            spirit_throw(
                Utility::Exception_Classifier::Unknown_Exception, Utility::Log_Level::Error,
                "Span: index out of bounds" );
        return data_[index];
    }
    SPIRIT_HOSTDEVICE constexpr const_reference at( std::size_t index ) const
    {
        if( ( std::is_signed<decltype( index )>::value && index < 0 ) || index >= size_ )
            spirit_throw(
                Utility::Exception_Classifier::Unknown_Exception, Utility::Log_Level::Error,
                "Span: index out of bounds" );
        return data_[index];
    }

    SPIRIT_HOSTDEVICE constexpr reference operator[]( std::size_t index )
    {
        return data_[index];
    }
    SPIRIT_HOSTDEVICE constexpr const_reference operator[]( std::size_t index ) const
    {
        return data_[index];
    }

private:
    pointer data_   = nullptr;
    size_type size_ = 0;
};

// type deduction
template<typename Iterator>
struct iterator_value_type : std::remove_pointer<typename std::iterator_traits<Iterator>::pointer>
{
};

template<typename T>
Span( T * data, std::size_t size ) -> Span<T>;

template<typename Iterator>
Span( Iterator begin, std::size_t size ) -> Span<typename iterator_value_type<Iterator>::type>;

template<typename Iterator>
Span( Iterator begin, Iterator end ) -> Span<typename iterator_value_type<Iterator>::type>;

} // namespace Engine

#endif
