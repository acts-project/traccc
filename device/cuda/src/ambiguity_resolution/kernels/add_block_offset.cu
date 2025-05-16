/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../utils/barrier.hpp"
#include "../../utils/global_index.hpp"
#include "add_block_offset.cuh"

namespace traccc::cuda::kernels {

__global__ void add_block_offset(device::add_block_offset_payload payload) {

    device::add_block_offset(details::global_index1(), payload);
}

}  // namespace traccc::cuda::kernels
