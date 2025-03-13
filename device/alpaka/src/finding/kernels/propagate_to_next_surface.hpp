/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/finding/device/propagate_to_next_surface.hpp"
#include "traccc/finding/finding_config.hpp"
#include "traccc/geometry/detector.hpp"

namespace traccc::alpaka {

template <typename propagator_t, typename bfield_t>
struct PropagateToNextSurfaceKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, const finding_config cfg,
        device::propagate_to_next_surface_payload<propagator_t, bfield_t>*
            payload) const {

        device::global_index_t globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];

        device::propagate_to_next_surface<propagator_t, bfield_t>(
            globalThreadIdx, cfg, *payload);
    }
};

}  // namespace traccc::alpaka
