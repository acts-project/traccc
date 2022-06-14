/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "component_connection.hpp"
#include "traccc/cuda/utils/definitions.hpp"

namespace traccc::cuda {

namespace kernel {
__global__ void component_connection(
    cell_container_types::const_view cells_view,
    vecmem::data::vector_view<std::size_t> cluster_prefix_sum_view,
    cluster_container_types::view clusters_view,
    vecmem::data::jagged_vector_view<unsigned int> sparse_ccl_indices_view,
    vecmem::data::vector_view<const device::prefix_sum_element_t>
        cells_prefix_sum_view) {
    auto idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Get device vector of the cells prefix sum
    vecmem::device_vector<const device::prefix_sum_element_t> cells_prefix_sum(
        cells_prefix_sum_view);
    // Ignore if idx is out of range
    if (idx >= cells_prefix_sum.size())
        return;

    // Get the indices for the module idx and the cell idx
    auto module_idx = cells_prefix_sum[idx].first;
    auto cell_idx = cells_prefix_sum[idx].second;

    // Initialize the device containers for cells and clusters
    cell_container_types::const_device cells_device(cells_view);
    cluster_container_types::device clusters_device(clusters_view);

    // Get the cells for the current module idx
    const auto& cells = cells_device.at(module_idx).items;

    // Vectors used for cluster indices found by sparse CCL
    vecmem::jagged_device_vector<unsigned int> device_sparse_ccl_indices(
        sparse_ccl_indices_view);
    const auto& cluster_indices = device_sparse_ccl_indices.at(module_idx);

    // Number of clusters found for this module
    const auto num_clusters = cluster_indices.back();

    // Get the cluster prefix sum for this module idx
    vecmem::device_vector<std::size_t> device_cluster_prefix_sum(
        cluster_prefix_sum_view);
    const auto prefix_sum = device_cluster_prefix_sum[module_idx];

    // Push back the cells to the correct item vector indicated
    // by the cluster prefix sum
    unsigned int cindex = cluster_indices[cell_idx] - 1;
    if (cindex < num_clusters) {
        // Push back the header and items
        clusters_device[prefix_sum + cindex].header = module_idx;
        clusters_device[prefix_sum + cindex].items.push_back(cells[cell_idx]);
    }
}
}  // namespace kernel

void component_connection(
    cluster_container_types::view clusters_view,
    const cell_container_types::const_view& cells_view,
    vecmem::data::jagged_vector_view<unsigned int> sparse_ccl_indices_view,
    vecmem::data::vector_view<std::size_t> cluster_prefix_sum_view,
    vecmem::data::vector_view<const device::prefix_sum_element_t>
        cells_prefix_sum_view) {

    auto n_cells = cells_prefix_sum_view.size();
    auto nComponentConThreads = 64;
    auto nComponentConBlocks =
        (n_cells + nComponentConThreads - 1) / nComponentConThreads;

    kernel::component_connection<<<nComponentConBlocks, nComponentConThreads>>>(
        cells_view, cluster_prefix_sum_view, clusters_view,
        sparse_ccl_indices_view, cells_prefix_sum_view);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

}  // namespace traccc::cuda