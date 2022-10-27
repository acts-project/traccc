/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// CUDA Library include(s).
#include "traccc/cuda/clusterization/clusterization_algorithm.hpp"

// Project include(s)
#include "traccc/clusterization/device/connect_components.hpp"
#include "traccc/clusterization/device/count_cluster_cells.hpp"
#include "traccc/clusterization/device/create_measurements.hpp"
#include "traccc/clusterization/device/find_clusters.hpp"
#include "traccc/clusterization/device/form_spacepoints.hpp"
#include "traccc/cuda/utils/make_prefix_sum_buff.hpp"
#include "traccc/device/fill_prefix_sum.hpp"

// Vecmem include(s).
#include <vecmem/utils/copy.hpp>
#include <vecmem/utils/cuda/copy.hpp>

// System include(s).
#include <algorithm>

// Local include(s)
#include "traccc/cuda/utils/definitions.hpp"

namespace traccc::cuda {
namespace kernels {

__global__ void find_clusters(
    const cell_container_types::const_view cells_view,
    vecmem::data::jagged_vector_view<unsigned int> sparse_ccl_indices_view,
    vecmem::data::vector_view<std::size_t> clusters_per_module_view) {

    device::find_clusters(threadIdx.x + blockIdx.x * blockDim.x, cells_view,
                          sparse_ccl_indices_view, clusters_per_module_view);
}

__global__ void count_cluster_cells(
    vecmem::data::jagged_vector_view<unsigned int> sparse_ccl_indices_view,
    vecmem::data::vector_view<std::size_t> cluster_prefix_sum_view,
    vecmem::data::vector_view<const device::prefix_sum_element_t>
        cells_prefix_sum_view,
    vecmem::data::vector_view<unsigned int> cluster_sizes_view) {

    device::count_cluster_cells(
        threadIdx.x + blockIdx.x * blockDim.x, sparse_ccl_indices_view,
        cluster_prefix_sum_view, cells_prefix_sum_view, cluster_sizes_view);
}

__global__ void connect_components(
    const cell_container_types::const_view cells_view,
    vecmem::data::jagged_vector_view<unsigned int> sparse_ccl_indices_view,
    vecmem::data::vector_view<std::size_t> cluster_prefix_sum_view,
    vecmem::data::vector_view<const device::prefix_sum_element_t>
        cells_prefix_sum_view,
    cluster_container_types::view clusters_view) {

    device::connect_components(threadIdx.x + blockIdx.x * blockDim.x,
                               cells_view, sparse_ccl_indices_view,
                               cluster_prefix_sum_view, cells_prefix_sum_view,
                               clusters_view);
}
__global__ void create_measurements(
    const cell_container_types::const_view cells_view,
    cluster_container_types::const_view clusters_view,
    measurement_container_types::view measurements_view) {

    device::create_measurements(threadIdx.x + blockIdx.x * blockDim.x,
                                clusters_view, cells_view, measurements_view);
}

__global__ void form_spacepoints(
    measurement_container_types::const_view measurements_view,
    vecmem::data::vector_view<const device::prefix_sum_element_t>
        measurements_prefix_sum_view,
    spacepoint_container_types::view spacepoints_view) {

    device::form_spacepoints(threadIdx.x + blockIdx.x * blockDim.x,
                             measurements_view, measurements_prefix_sum_view,
                             spacepoints_view);
}

}  // namespace kernels

clusterization_algorithm::clusterization_algorithm(
    const traccc::memory_resource& mr)
    : m_mr(mr) {

    // Initialize m_copy ptr based on memory resources that were given
    if (mr.host) {
        m_copy = std::make_unique<vecmem::cuda::copy>();
    } else {
        m_copy = std::make_unique<vecmem::copy>();
    }
}

clusterization_algorithm::output_type clusterization_algorithm::operator()(
    const cell_container_types::host& cells_per_event) const {

    // Vecmem copy object for moving the data between host and device
    vecmem::copy copy;

    // Number of modules
    unsigned int num_modules = cells_per_event.size();

    // Work block size for kernel execution
    std::size_t threadsPerBlock = 64;

    // Get the view of the cells container
    auto cells_data =
        get_data(cells_per_event, (m_mr.host ? m_mr.host : &(m_mr.main)));

    // Get the sizes of the cells in each module
    auto cell_sizes = copy.get_sizes(cells_data.items);

    /*
     * Helper container for sparse CCL calculations.
     * Each inner vector corresponds to 1 module.
     * The indices in a particular inner vector will be filled by sparse ccl
     * and will indicate to which cluster, a particular cell in the module
     * belongs to.
     */
    vecmem::data::jagged_vector_buffer<unsigned int> sparse_ccl_indices_buff(
        std::vector<std::size_t>(cell_sizes.begin(), cell_sizes.end()),
        m_mr.main, m_mr.host);
    m_copy->setup(sparse_ccl_indices_buff);

    /*
     * cl_per_module_prefix_buff is a vector buffer with numbers of found
     * clusters in each module. Later it will be transformed into prefix sum
     * vector (hence the name). The logic is the following. After
     * cluster_finding_kernel, the buffer will contain cluster sizes e.i.
     *
     * cluster sizes: | 1 | 12 | 5 | 102 | 42 | ... - cl_per_module_prefix_buff
     * module index:  | 0 |  1 | 2 |  3  |  4 | ...
     *
     * Now, we copy those cluster sizes to the host and make a duplicate vector
     * of them. So, we are left with cl_per_module_prefix_host, and
     * clusters_per_module_host - which are the same. Now, we procede to
     * modifying the cl_per_module_prefix_host to actually resemble its name
     * i.e.
     *
     * We do std::inclusive_scan on it, which will result in a prefix sum
     * vector:
     *
     * cl_per_module_prefix_host: | 1 | 13 | 18 | 120 | 162 | ...
     *
     * Then, we copy this vector into the previous cl_per_module_prefix_buff.
     * In this way, we don't need to allocate the memory on the device twice.
     *
     * Now, the monotonic prefix sum buffer - cl_per_module_prefix_buff, will
     * allow us to insert the clusters at the correct position inside the
     * kernel. The remaining host vector - clusters_per_module_host, will be
     * needed to allocate memory for other buffers later in the code.
     */
    vecmem::data::vector_buffer<std::size_t> cl_per_module_prefix_buff(
        num_modules, m_mr.main);
    m_copy->setup(cl_per_module_prefix_buff);

    // Create views to pass to cluster finding kernel
    const cell_container_types::const_view cells_view(cells_data);

    // Calculating grid size for cluster finding kernel
    std::size_t blocksPerGrid =
        (num_modules + threadsPerBlock - 1) / threadsPerBlock;

    // Invoke find clusters that will call cluster finding kernel
    kernels::find_clusters<<<blocksPerGrid, threadsPerBlock>>>(
        cells_view, sparse_ccl_indices_buff, cl_per_module_prefix_buff);

    // Create prefix sum buffer
    vecmem::data::vector_buffer cells_prefix_sum_buff =
        make_prefix_sum_buff(cell_sizes, *m_copy, m_mr);

    // Wait here for the find_clusters kernel to finish
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Copy the sizes of clusters per module to the host
    // and create a copy of "clusters per module" vector
    vecmem::vector<std::size_t> cl_per_module_prefix_host(
        m_mr.host ? m_mr.host : &(m_mr.main));
    (*m_copy)(cl_per_module_prefix_buff, cl_per_module_prefix_host,
              vecmem::copy::type::copy_type::device_to_host);
    std::vector<std::size_t> clusters_per_module_host(
        cl_per_module_prefix_host.begin(), cl_per_module_prefix_host.end());

    // Perform the inclusive scan operation
    std::inclusive_scan(cl_per_module_prefix_host.begin(),
                        cl_per_module_prefix_host.end(),
                        cl_per_module_prefix_host.begin());

    unsigned int total_clusters = cl_per_module_prefix_host.back();

    // Copy the prefix sum back to its device container
    (*m_copy)(vecmem::get_data(cl_per_module_prefix_host),
              cl_per_module_prefix_buff,
              vecmem::copy::type::copy_type::host_to_device);

    // Vector of the exact cluster sizes, will be filled in cluster counting
    vecmem::data::vector_buffer<unsigned int> cluster_sizes_buffer(
        total_clusters, m_mr.main);
    m_copy->setup(cluster_sizes_buffer);
    m_copy->memset(cluster_sizes_buffer, 0);

    // Calclating grid size for cluster counting kernel (block size 64)
    blocksPerGrid =
        (cells_prefix_sum_buff.size() + threadsPerBlock - 1) / threadsPerBlock;
    // Invoke cluster counting will call count cluster cells kernel
    kernels::count_cluster_cells<<<blocksPerGrid, threadsPerBlock>>>(
        sparse_ccl_indices_buff, cl_per_module_prefix_buff,
        cells_prefix_sum_buff, cluster_sizes_buffer);
    // Check for kernel launch errors and Wait for the cluster_counting kernel
    // to finish
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Copy cluster sizes back to the host
    std::vector<unsigned int> cluster_sizes;
    (*m_copy)(cluster_sizes_buffer, cluster_sizes,
              vecmem::copy::type::copy_type::device_to_host);

    // Cluster container buffer for the clusters and headers (cluster ids)
    cluster_container_types::buffer clusters_buffer{
        {total_clusters, m_mr.main},
        {std::vector<std::size_t>(total_clusters, 0),
         std::vector<std::size_t>(cluster_sizes.begin(), cluster_sizes.end()),
         m_mr.main, m_mr.host}};
    m_copy->setup(clusters_buffer.headers);
    m_copy->setup(clusters_buffer.items);

    // Using previous block size and thread size (64)
    // Invoke connect components will call connect components kernel
    kernels::connect_components<<<blocksPerGrid, threadsPerBlock>>>(
        cells_view, sparse_ccl_indices_buff, cl_per_module_prefix_buff,
        cells_prefix_sum_buff, clusters_buffer);

    // Resizable buffer for the measurements
    measurement_container_types::buffer measurements_buffer{
        {num_modules, m_mr.main},
        {std::vector<std::size_t>(num_modules, 0), clusters_per_module_host,
         m_mr.main, m_mr.host}};
    m_copy->setup(measurements_buffer.headers);
    m_copy->setup(measurements_buffer.items);

    // Spacepoint container buffer to fill inside the spacepoint formation
    // kernel
    spacepoint_container_types::buffer spacepoints_buffer{
        {num_modules, m_mr.main},
        {std::vector<std::size_t>(num_modules, 0), clusters_per_module_host,
         m_mr.main, m_mr.host}};
    m_copy->setup(spacepoints_buffer.headers);
    m_copy->setup(spacepoints_buffer.items);

    // Calculating grid size for measurements creation kernel (block size 64)
    blocksPerGrid = (clusters_buffer.headers.size() - 1 + threadsPerBlock) /
                    threadsPerBlock;

    // Wait here for the connect_components kernel to finish
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Invoke measurements creation will call create measurements kernel
    kernels::create_measurements<<<blocksPerGrid, threadsPerBlock>>>(
        cells_view, clusters_buffer, measurements_buffer);

    // Check for kernel launch errors and Wait here for the measurements
    // creation kernel to finish
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Create prefix sum buffer
    vecmem::data::vector_buffer meas_prefix_sum_buff = make_prefix_sum_buff(
        m_copy->get_sizes(measurements_buffer.items), *m_copy, m_mr);

    // Using the same grid size as before
    // Invoke spacepoint formation will call form_spacepoints kernel
    kernels::form_spacepoints<<<blocksPerGrid, threadsPerBlock>>>(
        measurements_buffer, meas_prefix_sum_buff, spacepoints_buffer);
    // Check for kernel launch errors and Wait for the spacepoint formation
    // kernel to finish
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    return spacepoints_buffer;
}

}  // namespace traccc::cuda