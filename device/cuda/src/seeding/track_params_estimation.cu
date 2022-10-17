/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/cuda/seeding/track_params_estimation.hpp"
#include "traccc/cuda/utils/definitions.hpp"
#include "traccc/seeding/device/estimate_track_params.hpp"

// VecMem include(s).
#include <vecmem/utils/cuda/copy.hpp>

namespace traccc {
namespace cuda {

namespace kernels {
/// CUDA kernel for running @c traccc::device::estimate_track_params
__global__ void estimate_track_params(
    spacepoint_container_types::const_view spacepoints_view,
    seed_collection_types::const_view seed_view,
    bound_track_parameters_collection_types::view params_view) {

    device::estimate_track_params(threadIdx.x + blockIdx.x * blockDim.x,
                                  spacepoints_view, seed_view, params_view);
}
}  // namespace kernels

track_params_estimation::track_params_estimation(
    const traccc::memory_resource& mr)
    : m_mr(mr) {

    // Initialize m_copy ptr based on memory resources that were given
    if (mr.host) {
        m_copy = std::make_unique<vecmem::cuda::copy>();
    } else {
        m_copy = std::make_unique<vecmem::copy>();
    }
}

bound_track_parameters_collection_types::host
track_params_estimation::operator()(
    const spacepoint_container_types::const_view& spacepoints_view,
    const seed_collection_types::const_view& seeds_view) const {

    // Get the size of the seeds view
    const std::size_t seeds_size = m_copy->get_size(seeds_view);

    // Create output host container
    bound_track_parameters_collection_types::host params(
        seeds_size, (m_mr.host ? m_mr.host : &(m_mr.main)));

    // Check if anything needs to be done.
    if (seeds_size == 0) {
        return params;
    }

    // Create device buffer for the parameters
    bound_track_parameters_collection_types::buffer params_buffer(seeds_size,
                                                                  m_mr.main);
    m_copy->setup(params_buffer);

    // -- Num threads
    // The dimension of block is the integer multiple of WARP_SIZE (=32)
    unsigned int num_threads = WARP_SIZE * 2;

    // -- Num blocks
    // The dimension of grid is (number_of_seeds + num_threads - 1) /
    // num_threads + 1
    unsigned int num_blocks = (seeds_size + num_threads - 1) / num_threads;

    // run the kernel
    kernels::estimate_track_params<<<num_blocks, num_threads>>>(
        spacepoints_view, seeds_view, params_buffer);

    // cuda error check
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Copy the results back to the host
    (*m_copy)(params_buffer, params);

    return params;
}

}  // namespace cuda
}  // namespace traccc
