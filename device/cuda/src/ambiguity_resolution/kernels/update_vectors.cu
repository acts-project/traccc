/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../utils/global_index.hpp"
#include "update_vectors.cuh"

namespace traccc::cuda::kernels {

__global__ void update_vectors(
    device::update_vectors_payload payload) {

    device::update_vectors(details::global_index1(), payload);
}

}  // namespace traccc::cuda::kernels
