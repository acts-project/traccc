/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// VecMem include(s).
#include <vecmem/memory/device_atomic_ref.hpp>

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void count_cluster_cells(
    std::size_t globalIndex,
    vecmem::data::jagged_vector_view<unsigned int> sparse_ccl_indices_view,
    vecmem::data::vector_view<std::size_t> cluster_prefix_sum_view,
    vecmem::data::vector_view<const device::prefix_sum_element_t>
        cells_prefix_sum_view,
    vecmem::data::vector_view<unsigned int> cluster_sizes_view) {

    // Get the device vector of the cell prefix sum
    vecmem::device_vector<const device::prefix_sum_element_t> cells_prefix_sum(
        cells_prefix_sum_view);

    // Ignore if idx is out of range
    if (globalIndex >= cells_prefix_sum.size())
        return;

    // Get the indices for the module and the cell in this
    // module, from the prefix sum
    auto module_idx = cells_prefix_sum[globalIndex].first;
    auto cell_idx = cells_prefix_sum[globalIndex].second;

    // Vectors used for cluster indices found by sparse CCL
    vecmem::jagged_device_vector<unsigned int> device_sparse_ccl_indices(
        sparse_ccl_indices_view);
    const auto& cluster_indices = device_sparse_ccl_indices[module_idx];

    // Get the cluster prefix sum at this module_idx to know
    // where to write current clusters in the
    // cluster container
    vecmem::device_vector<std::size_t> device_cluster_prefix_sum(
        cluster_prefix_sum_view);
    const std::size_t prefix_sum =
        (module_idx == 0 ? 0 : device_cluster_prefix_sum[module_idx - 1]);

    // Calculate the number of clusters found for this module from the prefix
    // sums
    const unsigned int n_clusters =
        (module_idx == 0 ? device_cluster_prefix_sum[0]
                         : device_cluster_prefix_sum[module_idx] - prefix_sum);

    // Vector to fill in with the sizes of each cluster
    vecmem::device_vector<unsigned int> device_cluster_sizes(
        cluster_sizes_view);

    // Count the cluster sizes for each position
    unsigned int cindex = cluster_indices[cell_idx] - 1;
    if (cindex < n_clusters) {
        vecmem::device_atomic_ref<unsigned int>(
            device_cluster_sizes[prefix_sum + cindex])
            .fetch_add(1);
    }
}

}  // namespace traccc::device