/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "../../utils/global_index.hpp"
#include "traccc/finding/device/propagate_to_next_surface.hpp"
#include "traccc/finding/finding_config.hpp"

// ─── Cooperative-groups ────────────────────────────────────────────────────────
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace traccc::cuda::kernels {

template <typename propagator_t, typename bfield_t>
__global__ void propagate_stage1(
    const finding_config cfg,
    device::propagate_to_next_surface_payload<propagator_t, bfield_t> payload) {

    // (1) coarse stepping  ── regs 60–80, memory-bound
    device::propagate_stage1<propagator_t, bfield_t>(
        details::global_index1(), cfg, payload);

    // (2) grid-wide barrier so EVERY block finishes before we touch covariance
    cg::this_grid().sync();

    // (3) high-accuracy covariance update  ── regs 90-110, math-bound
    device::propagate_stage2<propagator_t, bfield_t>(
        details::global_index1(), cfg, payload);
}

}  // namespace traccc::cuda::kernels