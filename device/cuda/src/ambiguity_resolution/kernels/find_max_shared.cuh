/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/ambiguity_resolution/device/find_max_shared.hpp"

namespace traccc::cuda::kernels {

__global__ void find_max_shared(device::find_max_shared_payload payload);
}
