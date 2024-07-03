/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/alpaka/clusterization/spacepoint_formation_algorithm.hpp"

#include "../utils/barrier.hpp"
#include "../utils/utils.hpp"

// Project include(s)
#include "traccc/clusterization/device/form_spacepoints.hpp"

namespace traccc::alpaka {

struct FormSpacepointsKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        measurement_collection_types::const_view measurements_view,
        cell_module_collection_types::const_view modules_view,
        const unsigned int measurement_count,
        spacepoint_collection_types::view spacepoints_view) const {

        auto const globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0u];

        device::form_spacepoints(globalThreadIdx, measurements_view,
                                 modules_view, measurement_count,
                                 spacepoints_view);
    }
};

spacepoint_formation_algorithm::spacepoint_formation_algorithm(
    const traccc::memory_resource& mr, vecmem::copy& copy)
    : m_mr(mr), m_copy(copy) {}

spacepoint_formation_algorithm::output_type
spacepoint_formation_algorithm::operator()(
    const measurement_collection_types::const_view& measurements_view,
    const cell_module_collection_types::const_view& modules_view) const {

    // Setup alpaka
    auto devAcc = ::alpaka::getDevByIdx(::alpaka::Platform<Acc>{}, 0u);
    auto queue = Queue{devAcc};

    // Get the number of measurements.
    const measurement_collection_types::const_view::size_type num_measurements =
        m_copy.get().get_size(measurements_view);

    // Create the result buffer.
    spacepoint_collection_types::buffer spacepoints(num_measurements,
                                                    m_mr.main);
    m_copy.get().setup(spacepoints)->ignore();

    // If there are no measurements, we can conclude here.
    if (num_measurements == 0) {
        return spacepoints;
    }

    // Launch parameters for the kernel.
    const unsigned int blockSize = 1024;
    const unsigned int nBlocks = (num_measurements + blockSize - 1) / blockSize;
    auto workDiv = makeWorkDiv<Acc>(blockSize, nBlocks);

    // Launch the spacepoint formation kernel.
    ::alpaka::exec<Acc>(queue, workDiv, FormSpacepointsKernel{},
                        measurements_view, modules_view, num_measurements,
                        vecmem::get_data(spacepoints));
    ::alpaka::wait(queue);

    // Return the reconstructed spacepoints.
    return spacepoints;
}

}  // namespace traccc::alpaka
