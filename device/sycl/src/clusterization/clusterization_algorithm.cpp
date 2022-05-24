/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "traccc/sycl/clusterization/clusterization_algorithm.hpp"

#include "traccc/device/get_prefix_sum.hpp"

// SYCL library include(s).
#include "cluster_counting.hpp"
#include "clusters_sum.hpp"
#include "component_connection.hpp"
#include "measurement_creation.hpp"
#include "spacepoint_formation.hpp"

// Vecmem include(s).
#include <vecmem/utils/sycl/copy.hpp>

// System include(s).
#include <algorithm>

namespace traccc::sycl {

clusterization_algorithm::clusterization_algorithm(
    vecmem::memory_resource &mr, vecmem::memory_resource &device_mr,
    queue_wrapper queue)
    : m_mr(mr), m_device_mr(device_mr), m_queue(queue) {}

clusterization_algorithm::output_type clusterization_algorithm::operator()(
    const cell_container_types::host &cells_per_event) const {

    // Number of modules
    unsigned int num_modules = cells_per_event.size();

    // Vecmem copy object for moving the data between host and device
    vecmem::sycl::copy copy(m_queue.queue());

    // Get the view of the cells container
    auto cells_data = get_data(cells_per_event, &m_mr.get());

    // Get the sizes of the cells in each module
    auto cell_sizes = copy.get_sizes(cells_data.items);

    // Get the cell sizes with +1 in each entry for sparse ccl indices buffer
    // The +1 is needed to store the number of found clusters at the end of
    // the vector in each module
    std::vector<std::size_t> cell_sizes_plus(num_modules);
    std::transform(cell_sizes.begin(), cell_sizes.end(),
                   cell_sizes_plus.begin(),
                   [](std::size_t x) { return x + 1; });

    // Helper container for sparse CCL calculations
    vecmem::data::jagged_vector_buffer<unsigned int> sparse_ccl_indices(
        cell_sizes_plus, m_device_mr.get(), &m_mr.get());
    copy.setup(sparse_ccl_indices);

    // Vector buffer for prefix sums for proper indexing, used only on device.
    // Vector with "clusters per module" is needed for measurement creation
    // buffer
    vecmem::data::vector_buffer<std::size_t> cluster_prefix_sum(
        num_modules, m_device_mr.get());
    vecmem::data::vector_buffer<std::size_t> clusters_per_module(
        num_modules, m_device_mr.get());
    copy.setup(cluster_prefix_sum);
    copy.setup(clusters_per_module);

    // Invoke the reduction kernel that gives the total number of clusters which
    // will be found (and also computes the prefix sums and clusters per module)
    auto total_clusters = vecmem::make_unique_alloc<unsigned int>(m_mr.get());
    *total_clusters = 0;

    // Get the prefix sum of the cells and copy it to the device buffer
    const device::prefix_sum_t cells_prefix_sum =
        device::get_prefix_sum(cell_sizes, m_mr.get());
    vecmem::data::vector_buffer<device::prefix_sum_element_t>
        cells_prefix_sum_buff(cells_prefix_sum.size(), m_device_mr.get());
    copy.setup(cells_prefix_sum_buff);
    copy(vecmem::get_data(cells_prefix_sum), cells_prefix_sum_buff);

    // Clusters sum kernel
    traccc::sycl::clusters_sum(cells_data, sparse_ccl_indices, *total_clusters,
                               cluster_prefix_sum, clusters_per_module,
                               m_device_mr.get(), m_queue);

    // Vector of the exact cluster sizes, will be filled in cluster counting
    vecmem::data::vector_buffer<unsigned int> cluster_sizes_buffer(
        *total_clusters, m_device_mr.get());
    copy.setup(cluster_sizes_buffer);

    // Cluster counting kernel
    traccc::sycl::cluster_counting(sparse_ccl_indices, cluster_sizes_buffer,
                                   cluster_prefix_sum, cells_prefix_sum_buff,
                                   m_queue);

    std::vector<unsigned int> cluster_sizes;
    copy(cluster_sizes_buffer, cluster_sizes);

    // Cluster container buffer for the clusters and headers (cluster ids)
    cluster_container_types::buffer clusters_buffer{
        {*total_clusters, m_device_mr.get()},
        {std::vector<std::size_t>(*total_clusters, 0),
         std::vector<std::size_t>(cluster_sizes.begin(), cluster_sizes.end()),
         m_device_mr.get(), &m_mr.get()}};
    copy.setup(clusters_buffer.headers);
    copy.setup(clusters_buffer.items);

    // Component connection kernel
    traccc::sycl::component_connection(clusters_buffer, cells_data,
                                       sparse_ccl_indices, cluster_prefix_sum,
                                       cells_prefix_sum_buff, m_queue);

    // Copy the sizes of clusters per each module to the std vector for
    // initialization of measurements buffer
    std::vector<std::size_t> clusters_per_module_host;
    copy(clusters_per_module, clusters_per_module_host);

    // Resizable buffer for the measurements
    measurement_container_types::buffer measurements_buffer{
        {num_modules, m_device_mr.get()},
        {std::vector<std::size_t>(num_modules, 0), clusters_per_module_host,
         m_device_mr.get(), &m_mr.get()}};
    copy.setup(measurements_buffer.headers);
    copy.setup(measurements_buffer.items);

    // Measurement creation kernel
    traccc::sycl::measurement_creation(measurements_buffer, clusters_buffer,
                                       cells_data, m_queue);

    // Spacepoint container buffer to fill in spacepoint formation
    spacepoint_container_types::buffer spacepoints_buffer{
        {num_modules, m_device_mr.get()},
        {std::vector<std::size_t>(num_modules, 0), clusters_per_module_host,
         m_device_mr.get(), &m_mr.get()}};
    copy.setup(spacepoints_buffer.headers);
    copy.setup(spacepoints_buffer.items);

    // Get the prefix sum of the measurements and copy it to the device buffer
    const device::prefix_sum_t meas_prefix_sum = device::get_prefix_sum(
        copy.get_sizes(measurements_buffer.items), m_mr.get());
    vecmem::data::vector_buffer<device::prefix_sum_element_t>
        meas_prefix_sum_buff(meas_prefix_sum.size(), m_device_mr.get());
    copy.setup(meas_prefix_sum_buff);
    copy(vecmem::get_data(meas_prefix_sum), meas_prefix_sum_buff);

    // Spacepoint formation kernel
    traccc::sycl::spacepoint_formation(spacepoints_buffer, measurements_buffer,
                                       meas_prefix_sum_buff, m_queue);

    return spacepoints_buffer;
}

}  // namespace traccc::sycl