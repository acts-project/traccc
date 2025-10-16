/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
//
// Local include(s).
#include "traccc/alpaka/seeding/track_params_estimation.hpp"

#include "../utils/get_queue.hpp"
#include "../utils/utils.hpp"

// Project include(s).
#include "traccc/seeding/device/estimate_track_params.hpp"

namespace traccc::alpaka {

struct EstimateTrackParamsKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, const track_params_estimation_config config,
        const edm::measurement_collection<default_algebra>::const_view&
            measurements_view,
        edm::spacepoint_collection::const_view spacepoints_view,
        edm::seed_collection::const_view seed_view, const vector3 bfield,
        bound_track_parameters_collection_types::view params_view) const {
        auto const globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0u];

        device::estimate_track_params(globalThreadIdx, config,
                                      measurements_view, spacepoints_view,
                                      seed_view, bfield, params_view);
    }
};

track_params_estimation::track_params_estimation(
    const track_params_estimation_config& config,
    const traccc::memory_resource& mr, vecmem::copy& copy, queue& q,
    std::unique_ptr<const Logger> ilogger)
    : messaging(std::move(ilogger)),
      m_config(config),
      m_mr(mr),
      m_copy(copy),
      m_queue(q) {}

track_params_estimation::output_type track_params_estimation::operator()(
    const edm::measurement_collection<default_algebra>::const_view&
        measurements_view,
    const edm::spacepoint_collection::const_view& spacepoints_view,
    const edm::seed_collection::const_view& seeds_view,
    const vector3& bfield) const {

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

    auto queue = details::get_queue(m_queue);
    const Idx threadsPerBlock = getWarpSize<Acc>() * 2;

    Idx blocksPerGrid = (seeds_size + threadsPerBlock - 1) / threadsPerBlock;
    auto workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

    // Run the kernel
    ::alpaka::exec<Acc>(queue, workDiv, EstimateTrackParamsKernel{}, m_config,
                        measurements_view, spacepoints_view, seeds_view, bfield,
                        ::vecmem::get_data(params_buffer));

    return params_buffer;
}

}  // namespace traccc::alpaka
