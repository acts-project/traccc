/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/clusterization/sparse_ccl_algorithm.hpp"

#include "traccc/clusterization/details/sparse_ccl.hpp"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/vector.hpp>

namespace traccc::host {

sparse_ccl_algorithm::sparse_ccl_algorithm(vecmem::memory_resource& mr)
    : m_mr(mr) {}

sparse_ccl_algorithm::output_type sparse_ccl_algorithm::operator()(
    const cell_collection_types::const_view& cells_view) const {

    // Run SparseCCL to fill CCL indices.
    const cell_collection_types::const_device cells{cells_view};
    vecmem::vector<unsigned int> cluster_indices{cells.size(), &(m_mr.get())};
    vecmem::device_vector<unsigned int> cluster_indices_device{
        vecmem::get_data(cluster_indices)};
    const unsigned int num_clusters =
        details::sparse_ccl(cells, cluster_indices_device);

    // Create the result container.
    output_type clusters(num_clusters, &(m_mr.get()));

    // Add cells to their clusters.
    for (std::size_t i = 0; i < cluster_indices.size(); ++i) {
        clusters.get_items()[cluster_indices[i]].push_back(cells[i]);
    }

    // Return the clusters.
    return clusters;
}

}  // namespace traccc::host
