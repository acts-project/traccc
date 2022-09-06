/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <traccc/futhark/entry.h>

#include <cstring>
#include <numeric>
#include <traccc/futhark/context.hpp>
#include <tuple>
#include <vector>

namespace traccc::futhark {
template <typename, typename, typename>
struct wrapper {};

template <typename T, typename... IArgs, typename... OArgs>
struct wrapper<T, std::tuple<IArgs...>, std::tuple<OArgs...>> {
    using output_t = std::tuple<std::vector<typename OArgs::cpp_t>...>;

    template <std::size_t... IIdxs, std::size_t... OIdxs>
    static output_t run_helper(struct futhark_context &ctx,
                               std::vector<typename IArgs::cpp_t> &&... args,
                               std::index_sequence<IIdxs...>,
                               std::index_sequence<OIdxs...>) {
        std::tuple<typename IArgs::futhark_t *...> futhark_inputs = {
            IArgs::alloc_f(&ctx, args.data(), args.size())...};
        std::tuple<typename OArgs::futhark_t *...> futhark_outputs;

        /*
         * Make the call to the Futhark entry point.
         */
        T::entry_f(&ctx, (&std::get<OIdxs>(futhark_outputs))...,
                   std::get<IIdxs>(futhark_inputs)...);

        /*
         * Free the inputs, which are no longer needed.
         */
        (IArgs::free_f(&ctx, std::get<IIdxs>(futhark_inputs)), ...);

        /*
         * Retrieve the shapes of output vectors.
         */
        std::tuple<std::array<int64_t, OArgs::rank_v>...> output_ranks;
        std::array<const int64_t *, sizeof...(OArgs)> output_rank_ptrs;
        ((std::get<OIdxs>(output_rank_ptrs) =
              OArgs::shape_f(&ctx, std::get<OIdxs>(futhark_outputs))),
         ...);
        ((std::memcpy(std::get<OIdxs>(output_ranks).data(),
                      std::get<OIdxs>(output_rank_ptrs),
                      OArgs::rank_v * sizeof(int64_t))),
         ...);

        /*
         * Create the output vectors.
         */
        std::tuple<std::vector<typename OArgs::cpp_t>...> out(
            std::accumulate(std::get<OIdxs>(output_ranks).begin(),
                            std::get<OIdxs>(output_ranks).end(), 1,
                            std::multiplies<int64_t>())...);

        /*
         * Copy the values from the Futhark output vectors.
         */
        (OArgs::values_f(&ctx, std::get<OIdxs>(futhark_outputs),
                         std::get<OIdxs>(out).data()),
         ...);

        /*
         * Free the Futhark output vectors.
         */
        (OArgs::free_f(&ctx, std::get<OIdxs>(futhark_outputs)), ...);

        return out;
    }

    static std::tuple<std::vector<typename OArgs::cpp_t>...> run(
        std::vector<typename IArgs::cpp_t> &&... args) {
        return run_helper(
            get_context(),
            std::forward<std::vector<typename IArgs::cpp_t>>(args)...,
            std::index_sequence_for<IArgs...>{},
            std::index_sequence_for<OArgs...>{});
    }
};
}  // namespace traccc::futhark
