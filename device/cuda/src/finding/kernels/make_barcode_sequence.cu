/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "make_barcode_sequence.cuh"
#include "traccc/edm/measurement.hpp"
#include "traccc/finding/device/make_barcode_sequence.hpp"

namespace traccc::cuda::kernels {

__global__ void make_barcode_sequence(
    device::make_barcode_sequence_payload payload) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    device::make_barcode_sequence(gid, payload);
}
}  // namespace traccc::cuda::kernels
