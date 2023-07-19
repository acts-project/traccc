/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
//
// Local include(s).
#include "traccc/alpaka/utils/definitions.hpp"

// Project include(s).
#include "traccc/alpaka/seeding/track_params_estimation.hpp"
#include "traccc/seeding/device/estimate_track_params.hpp"
#include "traccc/alpaka/utils/definitions.hpp"

namespace traccc::alpaka {

struct EstimateTrackParamsKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        spacepoint_collection_types::const_view spacepoints_view,
        seed_collection_types::const_view seed_view, const vector3 bfield,
        bound_track_parameters_collection_types::view params_view
    ) const
    {
        auto const globalThreadIdx = ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0u];
        device::estimate_track_params(globalThreadIdx, spacepoints_view, seed_view, bfield, params_view);
    }
};

track_params_estimation::track_params_estimation(
    const traccc::memory_resource& mr, vecmem::copy& copy)
    : m_mr(mr),
      m_copy(copy) {}

track_params_estimation::output_type track_params_estimation::operator()(
    const spacepoint_collection_types::const_view& spacepoints_view,
    const seed_collection_types::const_view& seeds_view,
    const vector3& bfield) const {

    // Get the size of the seeds view
    auto seeds_size = m_copy.get_size(seeds_view);

    // Create device buffer for the parameters
    bound_track_parameters_collection_types::buffer params_buffer(seeds_size,
                                                                  m_mr.main);
    m_copy.setup(params_buffer);

    // Check if anything needs to be done.
    if (seeds_size == 0) {
        return params_buffer;
    }

    auto devAcc = ::alpaka::getDevByIdx<Acc>(0u);
    auto queue = Queue{devAcc};
    auto const deviceProperties = ::alpaka::getAccDevProps<Acc>(devAcc);
    auto const maxThreadsPerBlock = deviceProperties.m_blockThreadExtentMax[0];
    auto const threadsPerBlock = maxThreadsPerBlock;

    auto blocksPerGrid = (seeds_size + threadsPerBlock - 1) / threadsPerBlock;
    auto elementsPerThread = 1u;
    auto workDiv = WorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};

    // run the kernel
    ::alpaka::exec<Acc>(
        queue, workDiv,
        EstimateTrackParamsKernel{},
        spacepoints_view,
        seeds_view,
        bfield,
        vecmem::get_data(params_buffer)
    );
    ::alpaka::wait(queue);


    return params_buffer;
}

}  // namespace traccc::alpaka
