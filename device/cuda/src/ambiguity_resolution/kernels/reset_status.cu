/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../utils/global_index.hpp"
#include "reset_status.cuh"

namespace traccc::cuda::kernels {

__global__ void reset_status(device::reset_status_payload payload) {

    device::reset_status(details::global_index1(), payload);
}

}  // namespace traccc::cuda::kernels
