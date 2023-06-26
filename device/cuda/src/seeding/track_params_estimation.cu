/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../utils/utils.hpp"
#include "traccc/cuda/seeding/track_params_estimation.hpp"
#include "traccc/cuda/utils/definitions.hpp"

// Project include(s).
#include "traccc/seeding/device/estimate_track_params.hpp"

// VecMem include(s).
#include <vecmem/utils/cuda/copy.hpp>

namespace traccc {
namespace cuda {

namespace kernels {
/// CUDA kernel for running @c traccc::device::estimate_track_params
__global__ void estimate_track_params(
    spacepoint_collection_types::const_view spacepoints_view,
    seed_collection_types::const_view seed_view, const vector3 bfield,
    bound_track_parameters_collection_types::view params_view) {

    device::estimate_track_params(threadIdx.x + blockIdx.x * blockDim.x,
                                  spacepoints_view, seed_view, bfield,
                                  params_view);
}
}  // namespace kernels

track_params_estimation::track_params_estimation(
    const traccc::memory_resource& mr, vecmem::copy& copy, stream& str)
    : m_mr(mr), m_copy(copy), m_stream(str) {}

track_params_estimation::output_type track_params_estimation::operator()(
    const spacepoint_collection_types::const_view& spacepoints_view,
    const seed_collection_types::const_view& seeds_view,
    const vector3& bfield) const {

    // Get a convenience variable for the stream that we'll be using.
    cudaStream_t stream = details::get_stream(m_stream);

    // Get the size of the seeds view
    const std::size_t seeds_size = m_copy.get_size(seeds_view);

    // Create device buffer for the parameters
    bound_track_parameters_collection_types::buffer params_buffer(seeds_size,
                                                                  m_mr.main);
    m_copy.setup(params_buffer);

    // Check if anything needs to be done.
    if (seeds_size == 0) {
        return params_buffer;
    }

    // -- Num threads
    // The dimension of block is the integer multiple of WARP_SIZE (=32)
    unsigned int num_threads = WARP_SIZE * 2;

    // -- Num blocks
    // The dimension of grid is (number_of_seeds + num_threads - 1) /
    // num_threads + 1
    unsigned int num_blocks = (seeds_size + num_threads - 1) / num_threads;

    // run the kernel
    kernels::estimate_track_params<<<num_blocks, num_threads, 0, stream>>>(
        spacepoints_view, seeds_view, bfield, params_buffer);
    CUDA_ERROR_CHECK(cudaGetLastError());

    return params_buffer;
}

}  // namespace cuda
}  // namespace traccc
