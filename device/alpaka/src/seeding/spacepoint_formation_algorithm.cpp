/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/alpaka/seeding/spacepoint_formation_algorithm.hpp"

#include "../utils/barrier.hpp"
#include "../utils/get_queue.hpp"
#include "../utils/utils.hpp"

// Project include(s).
#include "traccc/geometry/detector.hpp"
#include "traccc/seeding/device/form_spacepoints.hpp"

namespace traccc::alpaka {

template <typename detector_t>
struct FormSpacepointsKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, typename detector_t::view det_view,
        measurement_collection_types::const_view measurements_view,
        edm::spacepoint_collection::view spacepoints_view) const {

        auto const globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0u];

        device::form_spacepoints<detector_t>(
            globalThreadIdx, det_view, measurements_view, spacepoints_view);
    }
};

spacepoint_formation_algorithm::spacepoint_formation_algorithm(
    const traccc::memory_resource& mr, vecmem::copy& copy, queue& q,
    std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)), m_mr(mr), m_copy(copy), m_queue(q) {}

edm::spacepoint_collection::buffer spacepoint_formation_algorithm::operator()(
    const detector_buffer& det,
    const measurement_collection_types::const_view& measurements_view) const {

    // Get a convenience variable for the queue that we'll be using.
    auto queue = details::get_queue(m_queue);

    // Get the number of measurements.
    const measurement_collection_types::const_view::size_type num_measurements =
        m_copy.get().get_size(measurements_view);

    // Create the result buffer.
    edm::spacepoint_collection::buffer spacepoints(
        num_measurements, m_mr.main, vecmem::data::buffer_type::resizable);
    m_copy.get().setup(spacepoints)->ignore();
    edm::spacepoint_collection::view spacepoints_view{spacepoints};

    // If there are no measurements, we can conclude here.
    if (num_measurements == 0) {
        return spacepoints;
    }

    // Launch parameters for the kernel.
    const unsigned int blockSize = 1024;
    const unsigned int nBlocks = (num_measurements + blockSize - 1) / blockSize;
    auto workDiv = makeWorkDiv<Acc>(blockSize, nBlocks);

    detector_buffer_visitor<detector_type_list>(
        det, [&]<typename detector_traits_t>(
                 const typename detector_traits_t::view& det_view) {
            // Launch the spacepoint formation kernel.
            ::alpaka::exec<Acc>(queue, workDiv,
                                FormSpacepointsKernel<detector_traits_t>{},
                                det_view, measurements_view, spacepoints_view);
        });

    // Return the reconstructed spacepoints.
    return spacepoints;
}

}  // namespace traccc::alpaka
