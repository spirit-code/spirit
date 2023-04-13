#pragma once
#ifndef SPIRIT_CORE_UTILITY_INDICES_HPP
#define SPIRIT_CORE_UTILITY_INDICES_HPP

#include <iterator>
#include <cstdint>


namespace Utility {


//-----------------------------------------------------------------------------
/** 
 * @brief  non-modifiable range that (conceptually) contains consecutive indices
 */
class index_range
{
public:
    using value_type = std::size_t;
    using size_type  = value_type;

private:
    value_type beg_ = 0;
    value_type end_ = 0;

public:
    class iterator {
    public:
        using iterator_category = std::contiguous_iterator_tag;
        using value_type = index_range::value_type;
        using difference_type = std::int64_t;

    private:
        value_type i_ = 0;

    public:
        constexpr
        iterator () = default;

        constexpr explicit
        iterator (value_type i) noexcept : i_{i} {}

        [[nodiscard]] constexpr
        value_type operator * () const noexcept { return i_; }

        [[nodiscard]] constexpr
        value_type operator [] (difference_type idx) const noexcept { 
            return i_ + idx;
        }

        constexpr auto operator <=> (iterator const&) const noexcept = default;

        constexpr iterator& operator ++ () noexcept { ++i_; return *this; }
        constexpr iterator& operator -- () noexcept { ++i_; return *this; }

        constexpr iterator
        operator ++ (int) noexcept { 
            auto old {*this};
            ++i_;
            return old;
        }

        constexpr iterator
        operator -- (int) noexcept { 
            auto old {*this};
            --i_;
            return old;
        }

        constexpr iterator&
        operator += (difference_type offset) noexcept { 
            i_ += offset;
            return *this;
        }

        constexpr iterator&
        operator -= (difference_type offset) noexcept { 
            i_ -= offset;
            return *this;
        }

        [[nodiscard]] constexpr friend iterator
        operator + (iterator const& it, difference_type idx) noexcept { 
            return iterator{it.i_ + idx}; 
        }

        [[nodiscard]] constexpr friend iterator
        operator + (difference_type idx, iterator const& it) noexcept { 
            return iterator{it.i_ + idx}; 
        }

        [[nodiscard]] constexpr friend iterator
        operator - (iterator const& it, difference_type idx) noexcept { 
            return iterator{it.i_ - idx}; 
        }

        [[nodiscard]] constexpr friend iterator
        operator - (difference_type idx, iterator const& it) noexcept { 
            return iterator{it.i_ - idx}; 
        }

        [[nodiscard]] friend constexpr
        difference_type operator - (iterator const& a, iterator const& b) noexcept { 
            return difference_type(b.i_) - difference_type(a.i_);
        }
    };

    using const_iterator = iterator;


    constexpr
    index_range () = default;

    constexpr explicit
    index_range (value_type end) noexcept:
        beg_{0}, end_{end}
    {}

    constexpr explicit
    index_range (value_type beg, value_type end) noexcept:
        beg_{beg}, end_{end}
    {}


    [[nodiscard]] constexpr
    value_type operator [] (size_type idx) const noexcept { return beg_ + idx; }


    [[nodiscard]]
    size_type size () const noexcept { return end_ - beg_; }

    [[nodiscard]]
    bool empty () const noexcept { return end_ <= beg_; }


    [[nodiscard]] constexpr
    iterator begin () const noexcept { return iterator{beg_}; }

    [[nodiscard]] constexpr
    iterator end () const noexcept { return iterator{end_}; }
};






}  // namespace Utility

#endif


