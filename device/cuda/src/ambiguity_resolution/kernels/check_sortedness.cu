/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../utils/global_index.hpp"
#include "check_sortedness.cuh"

namespace traccc::cuda::kernels {

__global__ void check_sortedness(device::check_sortedness_payload payload) {

    device::check_sortedness(details::global_index1(), payload);
}
}  // namespace traccc::cuda::kernels
