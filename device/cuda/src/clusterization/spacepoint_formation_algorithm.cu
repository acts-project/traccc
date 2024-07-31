/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../utils/cuda_error_handling.hpp"
#include "../utils/utils.hpp"
#include "traccc/cuda/clusterization/spacepoint_formation_algorithm.hpp"

// Project include(s)
#include "traccc/clusterization/device/form_spacepoints.hpp"

namespace traccc::cuda {
namespace kernels {

__global__ void form_spacepoints(
    measurement_collection_types::const_view measurements_view,
    const detector_description::const_view det_descr_view,
    const unsigned int measurement_count,
    spacepoint_collection_types::view spacepoints_view) {

    device::form_spacepoints(threadIdx.x + blockIdx.x * blockDim.x,
                             measurements_view, det_descr_view,
                             measurement_count, spacepoints_view);
}

}  // namespace kernels

spacepoint_formation_algorithm::spacepoint_formation_algorithm(
    const traccc::memory_resource& mr, vecmem::copy& copy, stream& str)
    : m_mr(mr), m_copy(copy), m_stream(str) {}

spacepoint_formation_algorithm::output_type
spacepoint_formation_algorithm::operator()(
    const measurement_collection_types::const_view& measurements,
    const detector_description::const_view& det_descr) const {

    // Get the number of measurements.
    const measurement_collection_types::const_view::size_type num_measurements =
        m_copy.get().get_size(measurements);

    // Create the result buffer.
    spacepoint_collection_types::buffer spacepoints(num_measurements,
                                                    m_mr.main);
    m_copy.get().setup(spacepoints)->ignore();

    // If there are no measurements, we can conclude here.
    if (num_measurements == 0) {
        return spacepoints;
    }

    // Get a convenience variable for the stream that we'll be using.
    cudaStream_t stream = details::get_stream(m_stream);

    // Launch parameters for the kernel.
    const unsigned int blockSize = 1024;
    const unsigned int nBlocks = (num_measurements + blockSize - 1) / blockSize;

    // Launch the spacepoint formation kernel.
    kernels::form_spacepoints<<<nBlocks, blockSize, 0, stream>>>(
        measurements, det_descr, num_measurements, spacepoints);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    // Return the reconstructed spacepoints.
    return spacepoints;
}

}  // namespace traccc::cuda
