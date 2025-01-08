/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "apply_interaction_src.cuh"
#include "types.hpp"

namespace traccc::cuda::kernels {

template __global__ void apply_interaction<traccc::default_detector::device>(
    const finding_config,
    device::apply_interaction_payload<traccc::default_detector::device>);

}
