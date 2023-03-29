#pragma once
#include <bits/ranges_base.h>
#ifndef SPIRIT_CORE_UTILITY_STDEXEC_ALGORITHMS_HPP
#define SPIRIT_CORE_UTILITY_STDEXEC_ALGORITHMS_HPP

#include <utility/Execution.hpp>

#include <fmt/format.h>


#ifdef SPIRIT_USE_STDEXEC

#include <concepts>
#include <algorithm>
#include <ranges>
#include <span>
#include <utility>


namespace Execution {




template <typename Fn>
concept IndexOperation = std::invocable<Fn,std::size_t>;




[[nodiscard]] stdexec::sender auto
parallel_for (
    stdexec::scheduler auto sched,
    std::size_t start,
    std::size_t end,
    std::size_t tileCount,
    IndexOperation auto body)
{
    assert (end >= start);

    auto const size = end - start;
    std::size_t const tileSize = (size + tileCount - 1) / tileCount;

    return stdexec::schedule(sched)
        | stdexec::bulk(tileCount, [=](std::size_t tileIdx)
            {
                auto const tileStart = tileIdx * tileSize;
                auto const tileEnd   = std::min(size, (tileIdx + 1) * tileSize);

                for (std::size_t i = tileStart; i < tileEnd; ++i) {
                    body(i);
                }
            });
}




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
    InRange1 const& input1,
    InRange2 const& input2,
    OutRange & output,
    std::size_t tileCount,
    Transf fn)
{
    auto const inSize = std::min(input1.size(), input2.size());
    std::size_t const tileSize = (inSize + tileCount - 1) / tileCount;

    return stdexec::schedule(sched)
        | stdexec::bulk(tileCount, [&](std::size_t tileIdx)
            {
                auto const start = tileIdx * tileSize;
                auto const end   = std::min(inSize, (tileIdx + 1) * tileSize);

                for (std::size_t i = start; i < end; ++i) {
                    output[i] = fn(input1[i], input2[i]);
                }
            });
}



template <
    stdexec::scheduler Scheduler, 
    typename OutRange, 
    IndexOperation Generator
>
requires std::ranges::random_access_range<OutRange> &&
         std::ranges::sized_range<OutRange>
[[nodiscard]] stdexec::sender auto
generate_enumerated (
    Scheduler sched, OutRange& output, std::size_t tileCount, Generator gen)
{
    auto const size = std::ranges::size(output);
    std::size_t const tileSize = (size + tileCount - 1) / tileCount;

    return stdexec::transfer_just(sched, std::span{output})
        | stdexec::bulk(tileCount, [=](std::size_t tileIdx, auto out)
            {
                auto const tileStart = tileIdx * tileSize;
                auto const tileEnd   = std::min(out.size(), (tileIdx + 1) * tileSize);

                auto o = out.begin() + tileStart;
                for (std::size_t i = tileStart; i < tileEnd; ++i) {
                    *o++ = gen(i);
                }
            });
}



/**
 * writes results of an invocable to an output range
 * expects an output range in the sender chain
 */
[[nodiscard]] auto 
generate_indexed (std::size_t tileCount, IndexOperation auto gen)
{
    return 
        stdexec::bulk(tileCount,
        [=]<typename T>(std::size_t tileIdx, std::span<T> out)
        {
            auto const size  = (out.size() + tileCount-1) / tileCount;
            auto const start = tileIdx * size;
            auto const end   = std::min(out.size(), (tileIdx + 1) * size);

            auto o = out.begin() + start;
            for (std::size_t i = start; i < end; ++i) {
                *o++ = gen(i);
            }
        })
    |   stdexec::then([]<typename T>(std::span<T> out){
            return out;
        });
}


/**
 * writes results of an invocable to an output range
 * sets the number of parallel tiles to the number of threads
 */
template <IndexOperation Gen>
[[nodiscard]] auto 
generate_indexed (Gen&& gen)
{
    std::size_t tileCount = 1;
    return stdexec::then(
        [&]<typename T>(Context ctx, std::span<T> out) {
            tileCount = ctx.max_concurrency();
            return out;
        })
    |   generate_indexed(tileCount, std::forward<Gen>(gen));
}




[[nodiscard]] stdexec::sender auto
with_elements (stdexec::scheduler auto sched,
               std::ranges::random_access_range auto& output)
{
    return stdexec::transfer_just(sched, std::span{output});
}


[[nodiscard]] stdexec::sender auto
with_elements (Context ctx,
               std::ranges::random_access_range auto& output)
{
    return stdexec::transfer_just(ctx.get_scheduler(), ctx, std::span{output});
}



}  // namespace Execution

#endif  // SPIRIT_USE_STDEXEC

#endif


