/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/clusterization/spacepoint_formation_algorithm.hpp"

#include "traccc/clusterization/details/spacepoint_formation.hpp"

namespace traccc::host {

spacepoint_formation_algorithm::spacepoint_formation_algorithm(
    vecmem::memory_resource& mr)
    : m_mr(mr) {}

spacepoint_formation_algorithm::output_type
spacepoint_formation_algorithm::operator()(
    const measurement_collection_types::const_view& measurements_view,
    const cell_module_collection_types::const_view& modules_view) const {

    // Create device containers for the inputs.
    const measurement_collection_types::const_device measurements{
        measurements_view};
    const cell_module_collection_types::const_device modules{modules_view};

    // Create the result container.
    output_type result(measurements.size(), &(m_mr.get()));

    // Set up each spacepoint in the result container.
    for (measurement_collection_types::const_device::size_type i = 0;
         i < measurements.size(); ++i) {

        const measurement& meas = measurements.at(i);
        details::fill_spacepoint(result[i], meas, modules.at(meas.module_link));
    }

    // Return the created container.
    return result;
}

}  // namespace traccc::host
