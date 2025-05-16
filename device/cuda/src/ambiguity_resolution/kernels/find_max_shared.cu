/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../utils/barrier.hpp"
#include "../../utils/global_index.hpp"
#include "find_max_shared.cuh"

namespace traccc::cuda::kernels {

__global__ void find_max_shared(device::find_max_shared_payload payload) {

    cuda::barrier barrier;

    device::find_max_shared(details::global_index1(), barrier, payload);
}

}  // namespace traccc::cuda::kernels
