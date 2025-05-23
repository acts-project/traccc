/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../utils/barrier.hpp"
#include "../../utils/global_index.hpp"
#include "scan_block_offsets.cuh"

namespace traccc::cuda::kernels {

__global__ void scan_block_offsets(device::scan_block_offsets_payload payload) {

    cuda::barrier barrier;
    extern __shared__ int shared_temp[];

    device::scan_block_offsets(details::global_index1(), blockDim.x,
                               threadIdx.x, barrier, payload, shared_temp);
}

}  // namespace traccc::cuda::kernels
