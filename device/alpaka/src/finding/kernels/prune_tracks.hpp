/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "../../utils/utils.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/finding/device/prune_tracks.hpp"

namespace traccc::alpaka {

struct PruneTracksKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  device::prune_tracks_payload payload) const {

        int globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];

        device::prune_tracks(globalThreadIdx, payload);
    }
};

}  // namespace traccc::alpaka
