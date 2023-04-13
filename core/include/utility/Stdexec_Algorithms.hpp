#pragma once
#ifndef SPIRIT_CORE_UTILITY_STDEXEC_ALGORITHMS_HPP
#define SPIRIT_CORE_UTILITY_STDEXEC_ALGORITHMS_HPP

#include <utility/Execution.hpp>
#include <utility/Indices.hpp>

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
// template <typename... Ts>
// using any_sender_of =
//     typename exec::any_receiver_ref<stdexec::completion_signatures<Ts...>>::template any_sender<>;




//-----------------------------------------------------------------------------
template <typename Fn, typename Range>
concept IndexToValueMapping = 
    std::invocable<Fn,std::size_t> &&
    std::convertible_to<std::invoke_result_t<Fn,std::size_t>,
                        std::ranges::range_value_t<Range>>;



//-----------------------------------------------------------------------------
[[nodiscard]]                                         
inline stdexec::sender auto                           
schedule (Context ctx)                                
{                                                     
    return stdexec::schedule(ctx.get_scheduler());    
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
void for_each (Scheduler sched, Input const& input, std::size_t tileCount, Body body)
{
    auto const size = std::ranges::size(input);
    auto const tileSize = (size + tileCount - 1) / tileCount;

    auto task = stdexec::schedule(sched) 
        |  stdexec::bulk(tileCount, [=,&input](std::size_t tileIdx)
            {
                auto const end = std::ranges::begin(input) 
                            + std::min(size, (tileIdx + 1) * tileSize);

                for (auto i = std::ranges::begin(input) + tileIdx * tileSize;
                        i != end; ++i) 
                {
                    body(*i);
                }
            });

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
void zip_transform (
    Scheduler sched,
    InRange1 const& in1,
    InRange2 const& in2,
    OutRange & out,
    std::size_t tileCount,
    Transf fn)
{
    // return stdexec::transfer_just(sched, std::span{input1}, std::span{input2}, std::span{output})
    //   | stdexec::bulk(tileCount, [&](std::size_t tileIdx, auto in1, auto in2, auto out)
    auto task = stdexec::schedule(sched)
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

    stdexec::sync_wait(task).value();
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
void zip_transform (
    Scheduler sched,
    InRange1 const& in1,
    InRange2 const& in2,
    OutRange & out,
    std::size_t tileCount,
    Transf fn)
{
    auto task = stdexec::schedule(sched)
      | stdexec::bulk(tileCount, [&](std::size_t tileIdx)
        {
            auto const size = std::min(std::min(in1.size(), in2.size()), out.size());
            auto const tileSize = (size + tileCount - 1) / tileCount;
            auto const start = tileIdx * tileSize;
            auto const end   = std::min(size, (tileIdx + 1) * tileSize);
            
            for (std::size_t i = start; i < end; ++i) {
                fn(in1[i], in2[i], out[i]);
            }
        });

    stdexec::sync_wait(task).value();
}




//-----------------------------------------------------------------------------
template <typename Fn, typename T>
concept PairReductionOperation =
    (std::floating_point<T> || std::signed_integral<T>) &&
    std::invocable<Fn,T,T,T> &&
    std::convertible_to<T,std::invoke_result_t<Fn,T,T,T>>;

[[nodiscard]] 
double zip_reduce (
    stdexec::scheduler auto sch,
    std::span<double const> in1,
    std::span<double const> in2,
    double initValue,
    std::size_t tileCount,
    PairReductionOperation<double> auto redOp)
    // Operation redOp)
{
    using ValueT = double;

    auto const inSize = std::min(in1.size(), in2.size());
    std::size_t const tileSize = (inSize + tileCount - 1) / tileCount;

    std::vector<ValueT> partials(tileCount);

    auto task = stdexec::transfer_just(sch, std::move(partials))
        | stdexec::bulk(tileCount,
            [=](std::size_t tileIdx, std::vector<ValueT>&& part)
            {
                auto const start = tileIdx * tileSize;
                auto const end   = std::min(inSize, (tileIdx + 1) * tileSize);

                auto intermediate = ValueT(0);
                for (std::size_t i = start; i < end; ++i) {
                    part[i] = redOp(intermediate, in1[i], in2[i]);
                }
            })
        | stdexec::then(
            [=](std::vector<ValueT>&& part)
            {
                return std::reduce(begin(part), end(part), initValue);
            });

    return stdexec::sync_wait(task).value();
}




//-----------------------------------------------------------------------------
template <
    stdexec::scheduler Scheduler, 
    typename OutRange, 
    typename Generator
>
requires std::ranges::random_access_range<OutRange> &&
         std::ranges::sized_range<OutRange> &&
         IndexToValueMapping<Generator,OutRange>
void generate_indexed (
    Scheduler sched, OutRange& out, std::size_t tileCount, Generator gen)
{
    auto task = stdexec::schedule(sched)
        | stdexec::bulk(tileCount,
            [=,&out](std::size_t tileIdx)
            {
                auto const size  = (out.size() + tileCount-1) / tileCount;
                auto const start = tileIdx * size;
                auto const end   = std::min(out.size(), (tileIdx + 1) * size);

                for (std::size_t i = start; i < end; ++i) {
                    out[i] = gen(i);
                }
            });

    stdexec::sync_wait(task).value();
}



}  // namespace Execution

#endif


