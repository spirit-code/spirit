#pragma once
#ifndef SPIRIT_CORE_UTILITY_STDEXEC_ALGORITHMS_HPP
#define SPIRIT_CORE_UTILITY_STDEXEC_ALGORITHMS_HPP

#include <utility/Execution.hpp>

// #include <fmt/format.h>

#ifdef _OPENMP
    #include <omp.h>
#endif

#include <concepts>
#include <algorithm>
#include <execution>
#include <numeric>
#include <ranges>
#include <span>
#include <utility>


namespace Execution {


//-----------------------------------------------------------------------------
template <typename Fn>
concept IndexToValueMapping = 
    std::invocable<Fn,std::size_t>;
    // std::convertible_to<std::invoke_result_t<Fn,std::size_t>,Value>


#ifdef SPIRIT_USE_STDEXEC


//-----------------------------------------------------------------------------
/** 
 * @brief  non-modifiable range that (conceptually) contains consecutive indices
 */
class indices
{
public:
    using value_type = std::size_t;
    using size_type  = value_type;

private:
    value_type beg_ = 0;
    value_type end_ = 0;

public:
    class iterator {
        value_type i_ = 0;
    public:
        using iterator_category = std::contiguous_iterator_tag;
        using value_type = indices::value_type;
        using difference_type = std::int64_t;

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
    indices () = default;

    constexpr explicit
    indices (value_type beg, value_type end) noexcept:
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




//-----------------------------------------------------------------------------
// based on example from PR2300
[[nodiscard]] stdexec::sender auto
async_inclusive_scan (
    stdexec::scheduler auto sch,
    std::span<const double> input,
    std::span<double> output,
    double init,
    std::size_t tileCount)
{
    std::size_t const tileSize = (input.size() + tileCount - 1) / tileCount;

    std::vector<double> partials(tileCount + 1);
    partials[0] = init;

    return stdexec::transfer_just(sch, std::move(partials))
        | stdexec::bulk(tileCount,
            [=](std::size_t i, std::vector<double>&& partials) {
                auto start = i * tileSize;
                auto end   = std::min(input.size(), (i + 1) * tileSize);
                partials[i + 1] = *--std::inclusive_scan(begin(input) + start,
                                                        begin(input) + end,
                                                        begin(output) + start);
            })
        | stdexec::then(
            [](std::vector<double>&& partials) {
                std::inclusive_scan(begin(partials), end(partials),
                                    begin(partials));
                return partials;
            })
        | stdexec::bulk(tileCount,
            [=](std::size_t i, std::vector<double>&& partials) {
                auto start = i * tileSize;
                auto end   = std::min(input.size(), (i + 1) * tileSize);
                std::for_each(output.begin() + start, output.begin() + end,
                [=] (double& e) { e = partials[i] + e; }
                );
            })
        | stdexec::then(
            [](std::vector<double>&& partials) { return partials; } );
}




//-----------------------------------------------------------------------------
template <typename Fn, typename T>
concept NearestNeighborFn = 
    std::floating_point<T> &&
    std::invocable<Fn,T,T,T,T,T> &&
    std::same_as<T,std::invoke_result_t<Fn,T,T,T,T,T>>;




// [[nodiscard]] stdexec::sender auto
// transform_matrix_nearest_neigbors (
//     stdexec::scheduler auto sch,
//     span2d<double> matrix,
//     double border,
//     std::size_t stripeCount,
//     NearestNeighborFn<double> auto fn)
// {
//     auto stripeHeight = (matrix.nrows() + stripeCount - 1) / stripeCount;
//     if (stripeHeight < 2) {
//         stripeHeight = 2;
//         stripeCount = (matrix.nrows() + 1) / 2;
//     }
//
//     auto compute_stripe = [=](std::size_t stripeIdx, span2d<double> m) {
//         std::size_t rstart = stripeIdx * stripeHeight;
//         std::size_t rend   = std::min(m.nrows(), (stripeIdx+1) * stripeHeight);
//         bool const lastRow = rend == m.nrows();
//         if (lastRow) --rend;
//
//         if (rstart == 0) {
//             m(0,0) = fn(        border,
//                         border, m(0,0), m(0,1),
//                                 m(1,0) );
//
//             for (std::size_t c = 1; c < m.ncols()-1; ++c) {
//                 m(0,c) = fn(          border,
//                             m(0,c-1), m(0,c), m(0,c+1),
//                                       m(1,c) );
//             }
//             auto const c = m.ncols() - 1;
//             m(0,c) = fn(          border,
//                         m(0,c-1), m(0,c), border,
//                                   m(1,c) );
//             rstart = 1;
//         }
//         for (std::size_t r = rstart; r < rend; ++r) {
//             m(r,0) = fn(        m(r-1,0), 
//                         border, m(r  ,0), m(r,1),
//                                 m(r+1,0) );
//
//             for (std::size_t c = 1; c < m.ncols()-1; ++c) {
//                 m(r,c) = fn(          m(r-1,c), 
//                             m(r,c-1), m(r  ,c), m(r,c+1),
//                                       m(r+1,c) );
//             }
//             auto const c = m.ncols() - 1;
//             m(r,c) = fn(          m(r-1,c), 
//                         m(r,c-1), m(r  ,c), border,
//                                   m(r+1,c) );
//         }
//         if (lastRow) {
//             auto const r = m.nrows() - 1;
//             m(r,0) = fn(        m(r-1,0),
//                         border, m(r  ,0), m(r,1),
//                                 border );
//
//             for (std::size_t c = 1; c < m.ncols()-1; ++c) {
//                 m(r,c) = fn(          m(r-1,c),
//                             m(0,c-1), m(r  ,c), m(r,c+1),
//                                       border );
//             }
//             auto const c = m.ncols() - 1;
//             m(r,c) = fn(          m(r-1,c),
//                         m(r,c-1), m(r  ,c), border,
//                                   border );
//         }
//     };
//
//     return stdexec::transfer_just(sch, matrix)
//         // even-numbered stripes
//         | stdexec::bulk(stripeCount,
//             [=](std::size_t stripeIdx, span2d<double> m)
//             {
//                 if (not (stripeIdx % 2)) { compute_stripe(stripeIdx, m); }
//             })
//         // odd-numbered stripes
//         | stdexec::bulk(stripeCount,
//             [=](std::size_t stripeIdx, span2d<double> m)
//             {
//                 if (stripeIdx % 2) { compute_stripe(stripeIdx, m); }
//             });
//         // | stdexec::then([](span2d<double> out)
//         //     {
//         //         return out;
//         //     });
// }




//-----------------------------------------------------------------------------
template <stdexec::scheduler Scheduler, typename Input, typename Body>
requires 
    std::ranges::random_access_range<Input> &&
    std::ranges::sized_range<Input> &&
    std::invocable<Body,std::ranges::range_value_t<Input>>
[[nodiscard]] stdexec::sender auto
for_each (Scheduler sched, Input const& input, std::size_t tileCount, Body body)
{
    auto const size = std::ranges::size(input);
    auto const tileSize = (size + tileCount - 1) / tileCount;

    return stdexec::schedule(sched)
      | stdexec::bulk(tileCount, [=](std::size_t tileIdx)
        {
            auto const end = std::ranges::begin(input) 
                           + std::min(size, (tileIdx + 1) * tileSize);

            for (auto i = std::ranges::begin(input) + tileIdx * tileSize;
                 i != end; ++i) 
            {
                body(*i);
            }

            // auto const beg = std::ranges::begin(input) + tileIdx * tileSize;
            // auto const end = std::ranges::begin(input) 
            //                + std::min(size, (tileIdx + 1) * tileSize);
            //
            // std::for_each(std::execution::unseq, beg, end, body);
        });
}




//-----------------------------------------------------------------------------
template <typename Input, typename Body>
requires 
    std::ranges::random_access_range<Input> &&
    std::ranges::sized_range<Input> &&
    std::invocable<Body,std::ranges::range_value_t<Input>>
void for_each (Context ctx, Input const& input, Body&& body)
{
    auto const tileCount = ctx.resource_shape().threads;

    auto task = for_each(ctx.get_scheduler(), input, tileCount,
                         std::forward<Body>(body));

    stdexec::sync_wait(task).value();
}




//-----------------------------------------------------------------------------
// zip_transform (in1, in2, out, f(v1,v2)->o)  
template <
    stdexec::scheduler Scheduler,
    typename InRange1,
    typename InRange2,
    typename OutRange,
    typename Transf,
    typename Value1 = std::ranges::range_value_t<InRange1>,
    typename Value2 = std::ranges::range_value_t<InRange2>,
    typename OutValue = std::ranges::range_value_t<OutRange>
>
requires 
    std::ranges::random_access_range<InRange1> &&
    std::ranges::random_access_range<InRange2> &&
    std::ranges::random_access_range<OutRange> &&
    std::ranges::sized_range<InRange1> &&
    std::ranges::sized_range<InRange2> &&
    std::ranges::sized_range<OutRange> &&
    std::invocable<Transf,Value1,Value2> &&
    std::convertible_to<OutValue,std::invoke_result_t<Transf,Value1,Value2>>
[[nodiscard]] stdexec::sender auto
zip_transform (
    Scheduler sched,
    InRange1 const& in1,
    InRange2 const& in2,
    OutRange & out,
    std::size_t tileCount,
    Transf fn)
{
    // return stdexec::transfer_just(sched, std::span{input1}, std::span{input2}, std::span{output})
    //   | stdexec::bulk(tileCount, [&](std::size_t tileIdx, auto in1, auto in2, auto out)
    return stdexec::schedule(sched)
      | stdexec::bulk(tileCount, [&,tileCount](std::size_t tileIdx)
        {
            auto const size     = std::min(std::min(in1.size(), in2.size()), out.size());
            auto const tileSize = (size + tileCount - 1) / tileCount;
            auto const start    = tileIdx * tileSize;
            auto const end      = std::min(size, (tileIdx + 1) * tileSize);

            // fmt::print("{} {} {} {} {}\n", size, tileSize, tileIdx, start, end);

            for (std::size_t i = start; i < end; ++i) {
                out[i] = fn(in1[i], in2[i]);
            }
        });

    // return stdexec::schedule(sched)
    //     | stdexec::then([&]{
    //             auto const size = std::min(std::min(in1.size(), in2.size()),out.size());
    //             for (std::size_t i = 0; i < size; ++i) {
    //                 out[i] = fn(in1[i], in2[i]);
    //             }
    //         });
}




//-----------------------------------------------------------------------------
// zip_transform (in1, in2, out, f(v1,v2,o)->())  
template <
    stdexec::scheduler Scheduler,
    typename InRange1,
    typename InRange2,
    typename OutRange,
    typename Transf,
    typename Value1 = std::ranges::range_value_t<InRange1>,
    typename Value2 = std::ranges::range_value_t<InRange2>,
    typename OutValue = std::ranges::range_value_t<OutRange>
>
requires 
    std::ranges::random_access_range<InRange1> &&
    std::ranges::random_access_range<InRange2> &&
    std::ranges::random_access_range<OutRange> &&
    std::ranges::sized_range<InRange1> &&
    std::ranges::sized_range<InRange2> &&
    std::ranges::sized_range<OutRange> &&
    std::invocable<Transf,Value1,Value2,OutValue&>
[[nodiscard]] stdexec::sender auto
zip_transform (
    Scheduler sched,
    InRange1 const& in1,
    InRange2 const& in2,
    OutRange & out,
    std::size_t tileCount,
    Transf fn)
{
    return stdexec::schedule(sched)
      | stdexec::bulk(tileCount, [&,tileCount](std::size_t tileIdx)
        {
            auto const size = std::min(std::min(in1.size(), in2.size()), out.size());
            auto const tileSize = (size + tileCount - 1) / tileCount;
            auto const start = tileIdx * tileSize;
            auto const end   = std::min(size, (tileIdx + 1) * tileSize);
            
            for (std::size_t i = start; i < end; ++i) {
                fn(in1[i], in2[i], out[i]);
            }
        });
}




//-----------------------------------------------------------------------------
[[nodiscard]] stdexec::sender auto
with_elements (stdexec::scheduler auto sched,
               std::ranges::random_access_range auto& output,
               std::size_t tileCount)
{
    return stdexec::transfer_just(sched, std::span{output}, tileCount);
}


[[nodiscard]] stdexec::sender auto
with_elements (Context ctx,
               std::ranges::random_access_range auto& output,
               std::size_t tileCount)
{
    return stdexec::transfer_just(ctx.get_scheduler(), ctx, std::span{output}, tileCount);
}




//-----------------------------------------------------------------------------
[[nodiscard]] auto 
generate_indexed (std::size_t tileCount, IndexToValueMapping auto gen)
{
    return 
        stdexec::bulk(tileCount,
        [=]<typename T>(std::size_t tileIdx, std::span<T> out)
        {
            auto const size  = (out.size() + tileCount-1) / tileCount;
            auto const start = tileIdx * size;
            auto const end   = std::min(out.size(), (tileIdx + 1) * size);

            for (std::size_t i = start; i < end; ++i) {
                out[i] = gen(i);
            }
        })
    |   stdexec::then([]<typename T>(std::span<T> out){
            return out;
        });
}



// starts a new sender chain with scheduler and output range
template <
    stdexec::scheduler Scheduler, 
    typename OutRange, 
    IndexToValueMapping Generator
>
requires std::ranges::random_access_range<OutRange> &&
         std::ranges::sized_range<OutRange>
[[nodiscard]] stdexec::sender auto
generate_indexed (
    Scheduler sched, OutRange& output, std::size_t tileCount, Generator&& gen)
{
    auto const size = std::ranges::size(output);
    auto const tileSize = (size + tileCount - 1) / tileCount;

    return stdexec::transfer_just(sched, std::span{output})
        |  generate_indexed(tileCount, std::forward<Generator>(gen));
}



// starts a new sender chain with context and output range
template <typename OutRange, IndexToValueMapping Generator>
    requires std::ranges::random_access_range<OutRange> &&
             std::ranges::sized_range<OutRange>
void generate_indexed (Context ctx, OutRange& output, Generator gen)
{
    auto const tileCount = ctx.resource_shape().threads;
    auto const size = std::ranges::size(output);
    auto const tileSize = (size + tileCount - 1) / tileCount;

    auto task = generate_indexed(ctx.get_scheduler(),
                                 output, tileCount,
                                 std::forward<Generator>(gen));
    
    stdexec::sync_wait(task).value();
}



#else   // SPIRIT_USE_STDEXEC



//-----------------------------------------------------------------------------
template <typename OutRange, IndexToValueMapping Generator>
    requires std::ranges::random_access_range<OutRange> &&
             std::ranges::sized_range<OutRange>
void generate_indexed (Context, OutRange& output, Generator gen)
{
    #pragma omp parallel for
    for (std::size_t i = 0; i < output.size(); ++i) {
        output[i] = gen(i);
    }
}




#endif  // SPIRIT_USE_STDEXEC

}  // namespace Execution

#endif


