/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/cuda/seeding/track_params_estimation.hpp"
#include "traccc/cuda/utils/definitions.hpp"

// VecMem include(s).
#include <vecmem/utils/cuda/copy.hpp>

namespace traccc {
namespace cuda {

/// Forward declaration of track parameter estimating kernel
/// The bound track parameters at the bottom spacepoints are obtained
///
/// @param seeds_view seeds found by seed finding
/// @param params_view vector of bound track parameters at the bottom
/// spacepoints
__global__ void track_params_estimating_kernel(
    spacepoint_container_types::const_view spacepoints_view,
    vecmem::data::vector_view<const seed> seeds_view,
    vecmem::data::vector_view<bound_track_parameters> params_view);

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

host_bound_track_parameters_collection track_params_estimation::operator()(
    const spacepoint_container_types::const_view& spacepoints_view,
    const vecmem::data::vector_view<const seed>& seeds_view) const {

    // Get the size of the seeds view
    const std::size_t seeds_size = m_copy->get_size(seeds_view);

    // Create output host container
    host_bound_track_parameters_collection params(
        seeds_size, (m_mr.host ? m_mr.host : &(m_mr.main)));

    // Check if anything needs to be done.
    if (seeds_size == 0) {
        return params;
    }

    // Create device buffer for the parameters
    vecmem::data::vector_buffer<bound_track_parameters> params_buffer(
        seeds_size, m_mr.main);
    m_copy->setup(params_buffer);

    // -- Num threads
    // The dimension of block is the integer multiple of WARP_SIZE (=32)
    unsigned int num_threads = WARP_SIZE * 2;

    // -- Num blocks
    // The dimension of grid is number_of_seeds / num_threads + 1
    unsigned int num_blocks = seeds_size / num_threads + 1;

    // run the kernel
    track_params_estimating_kernel<<<num_blocks, num_threads>>>(
        spacepoints_view, seeds_view, params_buffer);

    // cuda error check
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Copy the results back to the host
    (*m_copy)(params_buffer, params);

    return params;
}

__global__ void track_params_estimating_kernel(
    spacepoint_container_types::const_view spacepoints_view,
    vecmem::data::vector_view<const seed> seeds_view,
    vecmem::data::vector_view<bound_track_parameters> params_view) {

    // Get device container for input parameters
    const spacepoint_container_types::const_device spacepoints_device(
        spacepoints_view);
    vecmem::device_vector<const seed> seeds_device(seeds_view);
    device_bound_track_parameters_collection params_device(params_view);

    // vector index for threads
    unsigned int gid = threadIdx.x + blockIdx.x * blockDim.x;

    // prevent overflow
    if (gid >= seeds_device.size()) {
        return;
    }

    // convenient assumption on bfield and mass
    // TODO: make use of bfield extension for the future
    vector3 bfield = {0, 0, 2};

    const auto& seed = seeds_device.at(gid);
    auto& param = params_device[gid].vector();

    // Get bound track parameter
    param =
        seed_to_bound_vector(spacepoints_device, seed, bfield, PION_MASS_MEV);
}

}  // namespace cuda
}  // namespace traccc
