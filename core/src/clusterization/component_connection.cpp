/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/clusterization/component_connection.hpp"

#include "traccc/clusterization/detail/sparse_ccl.hpp"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/vector.hpp>

namespace traccc {

component_connection::output_type component_connection::operator()(
    const cell_collection_types::host& cells, const cell_module& module) const {

    // Create the result container.
    output_type result(&(m_mr.get()));

    // Set up a device collection on top of the host collection.
    const cell_collection_types::const_view cells_view =
        vecmem::get_data(cells);
    const cell_collection_types::const_device cells_device(cells_view);

    // Set up the index vector.
    vecmem::vector<unsigned int> connected_cells(cells.size(), &(m_mr.get()));
    vecmem::device_vector<unsigned int> connected_cells_device(
        vecmem::get_data(connected_cells));

    // Run the algorithm
    unsigned int num_clusters = 0;
    detail::sparse_ccl(cells_device, connected_cells_device, num_clusters);

    result.resize(num_clusters);
    for (auto& cl_id : result.get_headers()) {
        cl_id.module = module.module;
        cl_id.placement = module.placement;
        cl_id.pixel = module.pixel;
    }

    auto& cluster_items = result.get_items();
    unsigned int icell = 0;
    for (auto cell_label : connected_cells) {
        auto cindex = static_cast<unsigned int>(cell_label - 1);
        if (cindex < cluster_items.size()) {
            cluster_items[cindex].push_back(cells[icell++]);
        }
    }

    // Return the cluster container.
    return result;
}

}  // namespace traccc
