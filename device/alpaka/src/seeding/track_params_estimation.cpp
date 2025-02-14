/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
//
// Local include(s).
#include "traccc/alpaka/seeding/track_params_estimation.hpp"

#include "../utils/utils.hpp"

// Project include(s).
#include "traccc/seeding/device/estimate_track_params.hpp"

namespace traccc::alpaka {

struct EstimateTrackParamsKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        const measurement_collection_types::const_view& measurements_view,
        edm::spacepoint_collection::const_view spacepoints_view,
        edm::seed_collection::const_view seed_view, const vector3 bfield,
        const std::array<traccc::scalar, traccc::e_bound_size> stddev,
        bound_track_parameters_collection_types::view params_view) const {
        auto const globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0u];

        device::estimate_track_params(globalThreadIdx, measurements_view,
                                      spacepoints_view, seed_view, bfield,
                                      stddev, params_view);
    }
};

track_params_estimation::track_params_estimation(
    const traccc::memory_resource& mr, vecmem::copy& copy,
    std::unique_ptr<const Logger> ilogger)
    : messaging(std::move(ilogger)), m_mr(mr), m_copy(copy) {}

track_params_estimation::output_type track_params_estimation::operator()(
    const measurement_collection_types::const_view& measurements_view,
    const edm::spacepoint_collection::const_view& spacepoints_view,
    const edm::seed_collection::const_view& seeds_view, const vector3& bfield,
    const std::array<traccc::scalar, traccc::e_bound_size>& stddev) const {

    // Get the size of the seeds view
    auto seeds_size = m_copy.get_size(seeds_view);

    // Create device buffer for the parameters
    bound_track_parameters_collection_types::buffer params_buffer(seeds_size,
                                                                  m_mr.main);
    m_copy.setup(params_buffer)->ignore();

    // Check if anything needs to be done.
    if (seeds_size == 0) {
        return params_buffer;
    }

    auto devAcc = ::alpaka::getDevByIdx(::alpaka::Platform<Acc>{}, 0u);
    auto queue = Queue{devAcc};
    auto const deviceProperties = ::alpaka::getAccDevProps<Acc>(devAcc);
    auto const threadsPerBlock = deviceProperties.m_blockThreadExtentMax[0];

    auto blocksPerGrid = (seeds_size + threadsPerBlock - 1) / threadsPerBlock;
    auto workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

    // Run the kernel
    ::alpaka::exec<Acc>(queue, workDiv, EstimateTrackParamsKernel{},
                        measurements_view, spacepoints_view, seeds_view, bfield,
                        stddev, ::vecmem::get_data(params_buffer));
    ::alpaka::wait(queue);

    return params_buffer;
}

}  // namespace traccc::alpaka
