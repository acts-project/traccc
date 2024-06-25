/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::cuda {
/**
 * @brief Calculate the index of the current thread in the set of threads in
 * the warp for which a predicate evaluates to true.
 *
 * @param predicate The predicate to evaluate.
 *
 * @returns The tuple (T, I), where T is the total number of threads in the
 * warp for which the provided predicate was true, and where I is a unique,
 * consecutive index in the set of threads for which this was true.
 *
 * @example If a predicate P evaluates as `[T, F, F, T]` in a 4-lane warp, the
 * call to `warp_indexed_ballot_sync(P)` will evaluate to the values
 * `[(2, 0), (2, ?), (2, ?), (2, 1)]` where `?` indicates an undefined value.
 *
 * @warning The value of T is always well-defined, the value if I is only
 * well-defined if the given predicate was true for the given thread.
 *
 * @note This function forces thread synchronization.
 */
__device__ __forceinline__ std::pair<uint32_t, uint32_t>
warp_indexed_ballot_sync(bool predicate) {
    uint32_t mask = __ballot_sync(0xFFFFFFFFu, predicate);

    uint32_t tot = __popc(mask);
    uint32_t idx =
        __popc(mask & ~(0xFFFFFFFEu << (threadIdx.x % warpSize))) - 1u;

    return {tot, idx};
}
}  // namespace traccc::cuda
