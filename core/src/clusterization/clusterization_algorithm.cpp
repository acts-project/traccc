/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/clusterization/clusterization_algorithm.hpp"

namespace traccc {

clusterization_algorithm::clusterization_algorithm(vecmem::memory_resource& mr)
    : m_cc(mr), m_mc(mr), m_mr(mr) {}

clusterization_algorithm::output_type clusterization_algorithm::operator()(
    const cell_container_types::host& cells) const {

    // Create the result container.
    output_type result(&(m_mr.get()));
    result.reserve(cells.size());

    // Loop over all of the detector modules.
    for (std::size_t i = 0; i < cells.size(); ++i) {

        // Get the cells for the current module.
        cell_module module = cells.at(i).header;
        cell_container_types::host::item_vector::const_reference
            cells_per_module = cells.at(i).items;

        // Reconstruct all measurements for the current module.
        traccc::cluster_container_types::host clusters =
            m_cc(cells_per_module, module);
        for (cluster_id& cl_id : clusters.get_headers()) {
            cl_id.pixel = module.pixel;
        }
        traccc::host_measurement_collection measurements =
            m_mc(clusters, module);

        // Save the measurements into the event-wide container.
        result.push_back(module, std::move(measurements));
    }

    // Return the measurements for all detector modules.
    return result;
}

}  // namespace traccc
