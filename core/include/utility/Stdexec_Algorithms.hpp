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
    std::copy_constructible<Fn> &&
    std::invocable<Fn,std::size_t> &&
    std::convertible_to<std::invoke_result_t<Fn,std::size_t>,
                        std::ranges::range_value_t<Range>>;



// template <typename Fn, typename Range>
// concept Transformation = 
//     std::copy_constructible<Fn> &&
//     std::invocable<Fn,std::ranges::range_value_t<Range>> &&
//     std::convertible_to<std::invoke_result_t<Fn,std::ranges::range_value_t<Range>>,
//                         std::ranges::range_value_t<Range>>;



// template <typename Fn, typename Range>
// concept IndexTransformation = 
//     std::copy_constructible<Fn> &&
//     std::invocable<Fn,std::ranges::range_value_t<Range>,std::size_t> &&
//     std::convertible_to<std::invoke_result_t<Fn,std::ranges::range_value_t<Range>,std::size_t>,
//                         std::ranges::range_value_t<Range>>;



// template <typename Fn, typename T>
// concept NearestNeighborFn = 
//     std::floating_point<T> &&
//     std::invocable<Fn,T,T,T,T,T> &&
//     std::same_as<T,std::invoke_result_t<Fn,T,T,T,T,T>>;



template <typename Fn, typename T>
concept PairReductionOperation =
    (std::floating_point<T> || std::signed_integral<T>) &&
    std::invocable<Fn,T,T,T> &&
    std::convertible_to<T,std::invoke_result_t<Fn,T,T,T>>;




//-----------------------------------------------------------------------------
template <typename Input, typename Body>
requires 
    std::ranges::random_access_range<Input> &&
    std::ranges::sized_range<Input> &&
    std::copy_constructible<Body> &&
    std::invocable<Body,std::ranges::range_value_t<Input>>
void for_each (Context ctx, Input const& input, Body body)
{
    auto const size      = std::ranges::size(input);
    auto const tileCount = std::min(size, static_cast<std::size_t>(ctx.resource_shape().threads));
    auto const tileSize  = (size + tileCount - 1) / tileCount;

    auto sched = ctx.get_scheduler();

    auto task = stdexec::schedule(sched) 
    |   stdexec::bulk(tileCount, [&](std::size_t tileIdx)
        {
            auto const end = std::ranges::begin(input) 
                           + std::min(size, (tileIdx + 1) * tileSize);

            for (auto i = std::ranges::begin(input) + tileIdx * tileSize; i != end; ++i) 
            {
                body(*i);
            }
        });

    stdexec::sync_wait(task).value();
}




//-----------------------------------------------------------------------------
template <typename GridExtents, typename Body>
requires 
    std::copy_constructible<Body>
void for_each_grid_index (Context ctx, GridExtents ext, Body body)
{
    // size of collapsed index range
    std::size_t size = 1;
    for (auto x : ext) { size *= static_cast<std::size_t>(x); }
    
    auto const N = static_cast<int>(ext.size());
    auto const tileCount = std::min(size, static_cast<std::size_t>(ctx.resource_shape().threads));
    auto const tileSize  = static_cast<std::size_t>((size + tileCount - 1) / tileCount);

    auto sched = ctx.get_scheduler();

    auto task = stdexec::schedule(sched) 
    |   stdexec::bulk(tileCount, [&](std::size_t tileIdx)
        {
            // start/end of collapsed index range
            auto const start = tileIdx * tileSize;
            auto const end   = std::min(size, (tileIdx + 1) * tileSize);
            if (start >= end) return;
            // compute start index
            GridExtents idx;
            for (int i = 0; i < N; ++i) { idx[i] = 0; }

            if (start > 0) {
                std::size_t mul[N];
                mul[0] = 1;
                for (int i = 0; i < N-1; ++i) {
                    mul[i+1] = mul[i] * ext[i]; 
                }
                auto offset = start;
                for (int i = N; i > 0; --i) {
                    if (offset >= mul[i-1]) {
                        idx[i-1] += offset / mul[i-1];
                        offset = offset % mul[i-1];
                        if (offset == 0) break;
                    }
                }
            }
            // execute body on local index subrange
            for (auto ci = start; ci < end; ++ci) {
                body(idx);
                // increment index
                for (int i = 0; i < N; ++i) {
                    ++idx[i];
                    if (idx[i] < ext[i]) break;
                    idx[i] = 0;
                }
            }
        });

    stdexec::sync_wait(task).value();
}




//-----------------------------------------------------------------------------
template <typename OutRange, typename Generator>
requires std::ranges::random_access_range<OutRange> &&
         std::ranges::sized_range<OutRange> &&
         IndexToValueMapping<Generator,OutRange>
void generate_indexed (Context ctx, OutRange& out, Generator gen)
{
    auto const outSize   = static_cast<std::size_t>(std::ranges::size(out));
    auto const tileCount = std::min(outSize, static_cast<std::size_t>(ctx.resource_shape().threads));

    auto sched = ctx.get_scheduler();

    auto task = stdexec::schedule(sched)
    |   stdexec::bulk(tileCount, [&](std::size_t tileIdx)
        {
            auto const size  = (outSize + tileCount-1) / tileCount;
            auto const start = tileIdx * size;
            auto const end   = std::min(outSize, (tileIdx + 1) * size);

            for (auto i = start; i < end; ++i) {
                out[i] = gen(i);
            }
        });

    stdexec::sync_wait(task).value();
}




//-----------------------------------------------------------------------------
template <typename InRange, typename OutRange, typename Transf>
requires std::ranges::random_access_range<InRange> &&
         std::ranges::sized_range<InRange> &&
         std::ranges::random_access_range<OutRange> &&
         std::ranges::sized_range<OutRange> &&
         std::copy_constructible<Transf>
void transform (Context ctx, InRange const& in, OutRange& out, Transf fn)
{
    auto const size      = static_cast<std::size_t>(std::ranges::size(in));
    auto const tileCount = std::min(size, static_cast<std::size_t>(ctx.resource_shape().threads));
    auto const tileSize  = (size + tileCount-1) / tileCount;

    auto sched = ctx.get_scheduler();

    auto task = stdexec::schedule(sched)
    |   stdexec::bulk(tileCount, [&](std::size_t tileIdx)
        {
            auto const start = tileIdx * tileSize;
            auto const end   = std::min(size, (tileIdx + 1) * tileSize);

            for (auto i = start; i < end; ++i) {
                out[i] = fn(in[i]);
            }
        });

    stdexec::sync_wait(task).value();
}




//-----------------------------------------------------------------------------
template <typename InRange, typename OutRange, typename Transf>
requires std::ranges::random_access_range<InRange> &&
         std::ranges::sized_range<InRange> &&
         std::ranges::random_access_range<OutRange> &&
         std::ranges::sized_range<OutRange> &&
         std::copy_constructible<Transf>
void transform_indexed (Context ctx, InRange const& in, OutRange& out, Transf fn)
{
    auto const size      = static_cast<std::size_t>(std::ranges::size(in));
    auto const tileCount = std::min(size, static_cast<std::size_t>(ctx.resource_shape().threads));
    auto const tileSize  = (size + tileCount-1) / tileCount;

    auto sched = ctx.get_scheduler();

    auto task = stdexec::schedule(sched)
    |   stdexec::bulk(tileCount, [&](std::size_t tileIdx)
        {
            auto const start = tileIdx * tileSize;
            auto const end   = std::min(size, (tileIdx + 1) * tileSize);

            for (auto i = start; i < end; ++i) {
                out[i] = fn(i,in[i]);
            }
        });

    stdexec::sync_wait(task).value();
}




//-----------------------------------------------------------------------------
template <
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
    std::copy_constructible<Transf> &&
    std::invocable<Transf,Value1,Value2> &&
    std::convertible_to<OutValue,std::invoke_result_t<Transf,Value1,Value2>>
void zip_transform (
    Context ctx,
    InRange1 const& in1,
    InRange2 const& in2,
    OutRange & out,
    Transf fn)
{
    auto const size = std::min(
        std::min(std::ranges::size(in1), std::ranges::size(in2)),
        std::ranges::size(out));

    auto const tileCount = std::min(size, static_cast<std::size_t>(ctx.resource_shape().threads));

    auto const tileSize = (size + tileCount-1) / tileCount;

    auto sched = ctx.get_scheduler();

    auto task = stdexec::schedule(sched)
    |   stdexec::bulk(tileCount, [&](std::size_t tileIdx)
        {
            auto const start = tileIdx * tileSize;
            auto const end   = std::min(size, (tileIdx + 1) * tileSize);

            for (auto i = start; i != end; ++i) {
                out[i] = fn(in1[i],in2[i]);
            }
        });

    stdexec::sync_wait(task).value();
}




//-----------------------------------------------------------------------------
template <
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
    std::copy_constructible<Transf> &&
    std::invocable<Transf,Value1,Value2,OutValue&>
void zip_transform (
    Context ctx,
    InRange1 const& in1,
    InRange2 const& in2,
    OutRange & out,
    Transf fn)
{
    auto const size = std::min(
        std::min(std::ranges::size(in1), std::ranges::size(in2)),
        std::ranges::size(out));

    auto const tileCount = std::min(size, static_cast<std::size_t>(ctx.resource_shape().threads));
    auto const tileSize  = (size + tileCount - 1) / tileCount;

    auto sched = ctx.get_scheduler();

    auto task = stdexec::schedule(sched)
    |   stdexec::bulk(tileCount, [&](std::size_t tileIdx)
        {
            auto const start = tileIdx * tileSize;
            auto const end   = std::min(size, (tileIdx + 1) * tileSize);
            
            for (auto i = start; i < end; ++i) {
                fn(in1[i], in2[i], out[i]);
            }
        });

    stdexec::sync_wait(task).value();
}




//-----------------------------------------------------------------------------
[[nodiscard]] 
double zip_reduce (
    Context ctx,
    std::span<double const> in1,
    std::span<double const> in2,
    double initValue,
    PairReductionOperation<double> auto redOp)
{
    using ValueT = double;

    auto const inSize    = std::min(in1.size(), in2.size());
    auto const tileCount = std::min(inSize, static_cast<std::size_t>(ctx.resource_shape().threads));
    auto const tileSize  = (inSize + tileCount - 1) / tileCount;

    std::vector<ValueT> partials(tileCount);

    auto sched = ctx.get_scheduler();

    auto task = stdexec::transfer_just(sched, std::move(partials))
    |   stdexec::bulk(tileCount,
        [=](std::size_t tileIdx, std::vector<ValueT>&& part)
        {
            auto const start = tileIdx * tileSize;
            auto const end   = std::min(inSize, (tileIdx + 1) * tileSize);

            auto intermediate = ValueT(0);
            for (auto i = start; i < end; ++i) {
                part[i] = redOp(intermediate, in1[i], in2[i]);
            }
        })
    |   stdexec::then([=](std::vector<ValueT>&& part)
        {
            // return std::reduce(std::execution::par_unseq, begin(part), end(part), initValue);
            return std::reduce(begin(part), end(part), initValue);
        });

    return stdexec::sync_wait(task).value();
}



}  // namespace Execution

#endif


