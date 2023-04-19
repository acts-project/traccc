/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/alpaka/seeding/track_params_estimation.hpp"
#include "traccc/seeding/device/estimate_track_params.hpp"
#include "traccc/alpaka/utils/definitions.hpp"

// VecMem include(s).
#include <vecmem/utils/cuda/copy.hpp>

namespace traccc::alpaka {

struct EstimateTrackParamsKernel {
    template <typename Acc>
    ALPAKA_FN_ACC void operator()(
        Acc const& acc,
        spacepoint_collection_types::const_view spacepoints_view,
        seed_collection_types::const_view seed_view,
        bound_track_parameters_collection_types::view params_view
    ) const
    {
        auto const globalThreadIdx = ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0u];
        device::estimate_track_params(globalThreadIdx, spacepoints_view, seed_view, params_view);
    }
};

track_params_estimation::track_params_estimation(
    const traccc::memory_resource& mr, vecmem::copy& copy)
    : m_mr(mr),
      m_copy(copy) {}

track_params_estimation::output_type track_params_estimation::operator()(
    const spacepoint_collection_types::const_view& spacepoints_view,
    const seed_collection_types::const_view& seeds_view) const {

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
    // // The dimension of block is the integer multiple of WARP_SIZE (=32)
    // unsigned int num_threads = WARP_SIZE * 2;

    // -- Num blocks
    // The dimension of grid is (number_of_seeds + num_threads - 1) /
    // num_threads + 1
    // unsigned int num_blocks = (seeds_size + num_threads - 1) / num_threads;

    // run the kernel
    // kernels::estimate_track_params<<<num_blocks, num_threads>>>(
    //     spacepoints_view, seeds_view, params_buffer);

    return params_buffer;
}

}  // namespace traccc::alpaka
