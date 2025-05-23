/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../utils/global_index.hpp"
#include "fill_inverted_ids.cuh"

namespace traccc::cuda::kernels {

__global__ void fill_inverted_ids(device::fill_inverted_ids_payload payload) {

    device::fill_inverted_ids(details::global_index1(), payload);
}

}  // namespace traccc::cuda::kernels
