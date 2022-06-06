/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "cluster_counting.hpp"
#include "traccc/cuda/utils/definitions.hpp"

namespace traccc::cuda {
namespace kernels {
__global__ void cluster_counting(vecmem::data::jagged_vector_view<unsigned int> sparse_ccl_indices_view,
    vecmem::data::vector_view<unsigned int> cluster_sizes_view,
    vecmem::data::vector_view<std::size_t> cluster_prefix_sum_view,
    vecmem::data::vector_view<const device::prefix_sum_element_t> cells_prefix_sum_view)
{   
    //auto id_x = item.get_global_id(0);
    std::size_t  idx = threadIdx.x + blockIdx.x * blockDim.x; 

    // Get the device vector of the cell prefix sum
    vecmem::device_vector<const device::prefix_sum_element_t>
        cells_prefix_sum(cells_prefix_sum_view);

    // Ignore if id_x is out of range
    if (idx >= cells_prefix_sum.size())
        return;

    // Get the indices for the module and the cell in this
    // module, from the prefix sum
    auto module_idx = cells_prefix_sum[idx].first;
    auto cell_idx = cells_prefix_sum[idx].second;

    // Vectors used for cluster indices found by sparse CCL
    vecmem::jagged_device_vector<unsigned int>
        device_sparse_ccl_indices(sparse_ccl_indices_view);
    const auto& cluster_indices =
        device_sparse_ccl_indices[module_idx];

    // Number of clusters that sparce_ccl found for this module
    const unsigned int n_clusters = cluster_indices.back();

    // Get the cluster prefix sum at this module_idx to know
    // where to write current clusters in the
    // cluster container
    vecmem::device_vector<std::size_t>
        device_cluster_prefix_sum(cluster_prefix_sum_view);
    const std::size_t prefix_sum =
        device_cluster_prefix_sum[module_idx];

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

__global__ void set_zero(vecmem::data::vector_view<unsigned int> cluster_sizes_view){

    std::size_t  idx = threadIdx.x + blockIdx.x * blockDim.x ;

    vecmem::device_vector<unsigned int> device_cluster_sizes(
    cluster_sizes_view);

    if (idx >= device_cluster_sizes.size())
        return;
    device_cluster_sizes[idx] = 0;
}
} //namespace kernels
void cluster_counting(
    vecmem::data::jagged_vector_view<unsigned int> sparse_ccl_indices_view,
    vecmem::data::vector_view<unsigned int> cluster_sizes_view,
    vecmem::data::vector_view<std::size_t> cluster_prefix_sum_view,
    vecmem::data::vector_view<const device::prefix_sum_element_t>
        cells_prefix_sum_view)
    {
    
    auto n_cells = cells_prefix_sum_view.size();



    //CUDA kernel dementions
    auto nClusterCountingThreads = 64;
    auto nClusterCountingBlocks = (n_cells + nClusterCountingThreads - 1) / nClusterCountingThreads;
    
    kernels::set_zero<<<nClusterCountingBlocks,nClusterCountingThreads>>>(
        cluster_sizes_view
    );
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());  

    kernels::cluster_counting<<<nClusterCountingBlocks,nClusterCountingThreads>>>(
        sparse_ccl_indices_view,cluster_sizes_view,cluster_prefix_sum_view,
        cells_prefix_sum_view);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());      
}

}  // namespace traccc::cuda
