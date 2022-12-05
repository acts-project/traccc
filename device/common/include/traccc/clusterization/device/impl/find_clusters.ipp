/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void find_clusters(
    std::size_t globalIndex, const cell_container_types::const_view& cells_view,
    vecmem::data::jagged_vector_view<unsigned int> sparse_ccl_indices_view,
    vecmem::data::vector_view<std::size_t> clusters_per_module_view) {

    // Initialize the device container for cells
    cell_container_types::const_device cells_device(cells_view);

    // Ignore if idx is out of range
    if (globalIndex >= cells_device.size())
        return;

    // Get the cells for the current module
    const auto& cells = cells_device.at(globalIndex).items;

    // Vectors used for cluster indices found by sparse CCL
    vecmem::jagged_device_vector<unsigned int> device_sparse_ccl_indices(
        sparse_ccl_indices_view);
    auto cluster_indices = device_sparse_ccl_indices[globalIndex];

    // Run the sparse CCL algorithm
    unsigned int n_clusters = detail::sparse_ccl(cells, cluster_indices);

    // Fill the "number of clusters per
    // module" vector
    vecmem::device_vector<std::size_t> device_clusters_per_module(
        clusters_per_module_view);
    device_clusters_per_module[globalIndex] = n_clusters;
}

}  // namespace traccc::device