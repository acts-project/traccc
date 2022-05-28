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
    const cell_container_types::host& cells) const {

    std::vector<std::size_t> num_clusters(cells.size(), 0);
    std::vector<std::vector<unsigned int>> CCL_indices(cells.size());

    for (std::size_t i = 0; i < cells.size(); i++) {
        const auto& cells_per_module = cells.get_items()[i];

        CCL_indices[i] = std::vector<unsigned int>(cells_per_module.size());

        // Run SparseCCL to fill CCL indices
        num_clusters[i] = detail::sparse_ccl(cells_per_module, CCL_indices[i]);
    }

    // Get total number of clusters
    const std::size_t N =
        std::accumulate(num_clusters.begin(), num_clusters.end(), 0);

    // Create the result container.
    output_type result(N, &(m_mr.get()));

    std::size_t stack = 0;
    for (std::size_t i = 0; i < cells.size(); i++) {

        auto& cells_per_module = cells.get_items()[i];

        // Fill the module link
        std::fill(result.get_headers().begin() + stack,
                  result.get_headers().begin() + stack + num_clusters[i], i);

        // Full the cluster cells
        for (std::size_t j = 0; j < CCL_indices[i].size(); j++) {

            auto cindex = static_cast<unsigned int>(CCL_indices[i][j] - 1);

            result.get_items()[stack + cindex].push_back(cells_per_module[j]);
        }

        stack += num_clusters[i];
    }

    return result;
}

}  // namespace traccc
