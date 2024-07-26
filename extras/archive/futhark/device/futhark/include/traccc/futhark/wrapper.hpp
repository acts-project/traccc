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
#include <traccc/futhark/utils.hpp>
#include <tuple>
#include <vector>

namespace traccc::futhark {
struct futhark_u64_1d_wrapper {
    using cpp_t = uint64_t;
    using futhark_t = struct futhark_u64_1d;
    static constexpr futhark_t *(*alloc_f)(struct futhark_context *,
                                           const cpp_t *,
                                           int64_t) = &futhark_new_u64_1d;
    static constexpr int (*free_f)(struct futhark_context *,
                                   futhark_t *) = &futhark_free_u64_1d;
    static constexpr const int64_t *(*shape_f)(
        struct futhark_context *, futhark_t *) = &futhark_shape_u64_1d;
    static constexpr int (*values_f)(struct futhark_context *, futhark_t *,
                                     cpp_t *) = &futhark_values_u64_1d;
    static constexpr std::size_t rank_v = 1;
};

struct futhark_i64_1d_wrapper {
    using cpp_t = int64_t;
    using futhark_t = struct futhark_i64_1d;
    static constexpr futhark_t *(*alloc_f)(struct futhark_context *,
                                           const cpp_t *,
                                           int64_t) = &futhark_new_i64_1d;
    static constexpr int (*free_f)(struct futhark_context *,
                                   futhark_t *) = &futhark_free_i64_1d;
    static constexpr const int64_t *(*shape_f)(
        struct futhark_context *, futhark_t *) = &futhark_shape_i64_1d;
    static constexpr int (*values_f)(struct futhark_context *, futhark_t *,
                                     cpp_t *) = &futhark_values_i64_1d;
    static constexpr std::size_t rank_v = 1;
};

struct futhark_f32_1d_wrapper {
    using cpp_t = float;
    using futhark_t = struct futhark_f32_1d;
    static constexpr futhark_t *(*alloc_f)(struct futhark_context *,
                                           const cpp_t *,
                                           int64_t) = &futhark_new_f32_1d;
    static constexpr int (*free_f)(struct futhark_context *,
                                   futhark_t *) = &futhark_free_f32_1d;
    static constexpr const int64_t *(*shape_f)(
        struct futhark_context *, futhark_t *) = &futhark_shape_f32_1d;
    static constexpr int (*values_f)(struct futhark_context *, futhark_t *,
                                     cpp_t *) = &futhark_values_f32_1d;
    static constexpr std::size_t rank_v = 1;
};

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

        FUTHARK_ERROR_CHECK(futhark_context_sync(&ctx));

        /*
         * Make the call to the Futhark entry point.
         */
        FUTHARK_ERROR_CHECK(T::entry_f(&ctx,
                                       (&std::get<OIdxs>(futhark_outputs))...,
                                       std::get<IIdxs>(futhark_inputs)...));

        FUTHARK_ERROR_CHECK(futhark_context_sync(&ctx));

        /*
         * Free the inputs, which are no longer needed.
         */
        (FUTHARK_ERROR_CHECK(
             IArgs::free_f(&ctx, std::get<IIdxs>(futhark_inputs))),
         ...);

        FUTHARK_ERROR_CHECK(futhark_context_sync(&ctx));

        /*
         * Retrieve the shapes of output vectors.
         */
        std::tuple<std::array<int64_t, OArgs::rank_v>...> output_ranks;
        std::array<const int64_t *, sizeof...(OArgs)> output_rank_ptrs;
        ((std::get<OIdxs>(output_rank_ptrs) =
              OArgs::shape_f(&ctx, std::get<OIdxs>(futhark_outputs))),
         ...);

        FUTHARK_ERROR_CHECK(futhark_context_sync(&ctx));

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
        (FUTHARK_ERROR_CHECK(OArgs::values_f(&ctx,
                                             std::get<OIdxs>(futhark_outputs),
                                             std::get<OIdxs>(out).data())),
         ...);

        FUTHARK_ERROR_CHECK(futhark_context_sync(&ctx));

        /*
         * Free the Futhark output vectors.
         */
        (FUTHARK_ERROR_CHECK(
             OArgs::free_f(&ctx, std::get<OIdxs>(futhark_outputs))),
         ...);

        FUTHARK_ERROR_CHECK(futhark_context_sync(&ctx));

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
