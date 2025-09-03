/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../utils/cuda_error_handling.hpp"
#include "../utils/get_size.hpp"
#include "../utils/global_index.hpp"
#include "../utils/utils.hpp"
#include "traccc/cuda/seeding/track_params_estimation.hpp"

// Project include(s).
#include "traccc/seeding/device/estimate_track_params.hpp"

// VecMem include(s).
#include <vecmem/utils/cuda/copy.hpp>

namespace traccc::cuda {

namespace kernels {
/// CUDA kernel for running @c traccc::device::estimate_track_params
__global__ void estimate_track_params(
    measurement_collection_types::const_view measurements_view,
    edm::spacepoint_collection::const_view spacepoints_view,
    edm::seed_collection::const_view seed_view, const vector3 bfield,
    const std::array<traccc::scalar, traccc::e_bound_size> stddev,
    bound_track_parameters_collection_types::view params_view) {

    device::estimate_track_params(details::global_index1(), measurements_view,
                                  spacepoints_view, seed_view, bfield, stddev,
                                  params_view);
}
}  // namespace kernels

track_params_estimation::track_params_estimation(
    const traccc::memory_resource& mr, vecmem::copy& copy, stream& str,
    std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)),
      m_mr(mr),
      m_copy(copy),
      m_stream(str),
      m_warp_size(details::get_warp_size(str.device())) {}

track_params_estimation::output_type track_params_estimation::operator()(
    const measurement_collection_types::const_view& measurements_view,
    const edm::spacepoint_collection::const_view& spacepoints_view,
    const edm::seed_collection::const_view& seeds_view, const vector3& bfield,
    const std::array<traccc::scalar, traccc::e_bound_size>& stddev) const {

    // Get a convenience variable for the stream that we'll be using.
    cudaStream_t stream = details::get_stream(m_stream);

    // Staging area for copying sizes from device to host
    vecmem::unique_alloc_ptr<unsigned int> size_staging_ptr =
        vecmem::make_unique_alloc<unsigned int>(*(m_mr.host));

    // Get the size of the seeds view
    const unsigned int seeds_size =
        get_size(seeds_view, size_staging_ptr.get(), stream);

    // Create device buffer for the parameters
    bound_track_parameters_collection_types::buffer params_buffer(seeds_size,
                                                                  m_mr.main);
    m_copy.setup(params_buffer)->ignore();

    // Check if anything needs to be done.
    if (seeds_size == 0) {
        return params_buffer;
    }

    // -- Num threads
    // The dimension of block is the integer multiple of warp size (=32)
    unsigned int num_threads = m_warp_size * 2;

    // -- Num blocks
    // The dimension of grid is (number_of_seeds + num_threads - 1) /
    // num_threads + 1
    unsigned int num_blocks = (seeds_size + num_threads - 1) / num_threads;

    // run the kernel
    kernels::estimate_track_params<<<num_blocks, num_threads, 0, stream>>>(
        measurements_view, spacepoints_view, seeds_view, bfield, stddev,
        params_buffer);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    return params_buffer;
}

}  // namespace traccc::cuda
