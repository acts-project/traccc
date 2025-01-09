/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../utils/global_index.hpp"
#include "make_barcode_sequence.cuh"

// Project include(s).
#include "traccc/finding/device/make_barcode_sequence.hpp"

namespace traccc::cuda::kernels {

__global__ void make_barcode_sequence(
    device::make_barcode_sequence_payload payload) {

    device::make_barcode_sequence(details::global_index1(), payload);
}

}  // namespace traccc::cuda::kernels
