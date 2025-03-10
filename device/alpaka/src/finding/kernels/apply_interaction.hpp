/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/edm/track_parameters.hpp"
#include "traccc/finding/device/apply_interaction.hpp"
#include "traccc/geometry/detector.hpp"

namespace traccc::alpaka {

template <typename detector_t>
struct ApplyInteractionKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, const finding_config& cfg,
        device::apply_interaction_payload<detector_t> payload) const {

        int globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];

        device::apply_interaction<detector_t>(globalThreadIdx, cfg, payload);
    }
};

}  // namespace traccc::alpaka
