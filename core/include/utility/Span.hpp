#ifndef SPIRIT_UTILITY_SPAN
#define SPIRIT_UTILITY_SPAN

#include <cstddef>
#include <iterator>
#include <type_traits>

namespace Utility
{

/*
 * lightweight alternative to C++20's `std::span`: a non owning view into contiguous memory.
 * should be replacable by `std::span` once this code base uses C++20.
 */
template<typename T>
class Span
{
public:
    using value_type       = T;
    using const_value_type = typename std::conditional_t<std::is_const_v<T>, T, const T>;
    using pointer          = T *;
    using const_pointer    = const_value_type *;
    using reference        = T &;
    using const_reference  = const_value_type &;
    using size_type        = std::size_t;

    constexpr Span( T * data, size_type size ) noexcept : data_( data ), size_( size ){};

    template<typename Iterator>
    constexpr Span( Iterator begin, size_type size ) noexcept : data_( &( *begin ) ), size_( size ){};

    template<typename Iterator>
    constexpr Span( Iterator begin, Iterator end ) noexcept
            : data_( &( *begin ) ), size_( static_cast<size_type>( std::distance( begin, end ) ) ){};

    constexpr pointer begin()
    {
        return data_;
    }
    constexpr pointer end()
    {
        return data_ + size_;
    }

    constexpr const_pointer begin() const
    {
        return data_;
    }
    constexpr const_pointer end() const
    {
        return data_ + size_;
    }

    constexpr size_type size() const
    {
        return size_;
    }

    constexpr reference operator[]( std::size_t index )
    {
        return data_[index];
    }
    constexpr const_reference operator[]( std::size_t index ) const
    {
        return data_[index];
    }

private:
    pointer data_;
    size_type size_;
};

// type deduction
template<typename Iterator>
using iterator_value_type_t = std::remove_pointer_t<typename std::iterator_traits<Iterator>::pointer>;


template<typename T>
Span( T * data, std::size_t size ) -> Span<T>;

template<typename Iterator>
Span( Iterator begin, std::size_t size ) -> Span<iterator_value_type_t<Iterator>>;

template<typename Iterator>
Span( Iterator begin, Iterator end ) -> Span<iterator_value_type_t<Iterator>>;

} // namespace Utility

#endif
