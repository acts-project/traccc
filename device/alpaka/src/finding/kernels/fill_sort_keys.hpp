/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "../../utils/utils.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/finding/device/fill_sort_keys.hpp"

namespace traccc::alpaka {

struct FillSortKeysKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, device::fill_sort_keys_payload payload) const {

        int globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];

        device::fill_sort_keys(globalThreadIdx, payload);
    }
};

}  // namespace traccc::alpaka
