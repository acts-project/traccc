/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../utils/global_index.hpp"
#include "fill_sort_keys.cuh"

// Project include(s).
#include "traccc/finding/device/fill_sort_keys.hpp"

namespace traccc::cuda::kernels {

__global__ void fill_sort_keys(device::fill_sort_keys_payload payload) {

    device::fill_sort_keys(details::global_index1(), payload);
}

}  // namespace traccc::cuda::kernels
