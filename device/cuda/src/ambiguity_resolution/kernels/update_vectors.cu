/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../utils/barrier.hpp"
#include "../../utils/global_index.hpp"
#include "update_vectors.cuh"

namespace traccc::cuda::kernels {

__global__ void update_vectors(device::update_vectors_payload payload) {

    cuda::barrier barrier;
    __shared__ unsigned int shared_tids[1024];
    __shared__ std::size_t shared_meas_ids[1024];

    device::update_vectors(details::global_index1(), threadIdx.x, barrier,
                           payload, shared_tids, shared_meas_ids);
}

}  // namespace traccc::cuda::kernels
