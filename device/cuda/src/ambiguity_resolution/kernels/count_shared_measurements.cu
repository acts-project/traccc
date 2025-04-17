/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../utils/global_index.hpp"
#include "count_shared_measurements.cuh"

namespace traccc::cuda::kernels {

__global__ void count_shared_measurements(
    device::count_shared_measurements_payload payload) {

    device::count_shared_measurements(details::global_index1(), payload);
}
}  // namespace traccc::cuda::kernels
