/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../utils/barrier.hpp"
#include "../../utils/global_index.hpp"
#include "block_inclusive_scan.cuh"

namespace traccc::cuda::kernels {

__global__ void block_inclusive_scan(
    device::block_inclusive_scan_payload payload) {

    cuda::barrier barrier;
    extern __shared__ int shared_temp[];

    device::block_inclusive_scan(details::global_index1(), blockDim.x,
                                 blockIdx.x, threadIdx.x, barrier, payload,
                                 shared_temp);
}

}  // namespace traccc::cuda::kernels
