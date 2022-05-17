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
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <algorithm>

namespace traccc::sycl {

clusterization_algorithm::clusterization_algorithm(vecmem::memory_resource &mr,
                                                   queue_wrapper queue)
    : m_mr(mr), m_queue(queue) {}

host_spacepoint_container clusterization_algorithm::operator()(
    const cell_container_types::host &cells_per_event) const {

    // Number of modules
    unsigned int num_modules = cells_per_event.size();

    // Vecmem copy object for moving the data between host and device
    vecmem::copy copy;

    // Get the view of the cells container
    auto cells_data = get_data(cells_per_event, &m_mr.get());
    cell_container_types::const_view cells_view(cells_data);

    // Get the sizes of the cells in each module
    auto cell_sizes = copy.get_sizes(cells_view.items);

    // Get the cell sizes with +1 in each entry for sparse ccl indices buffer
    // The +1 is needed to store the number of found clusters at the end of
    // the vector in each module
    std::vector<std::size_t> cell_sizes_plus(num_modules);
    std::transform(cell_sizes.begin(), cell_sizes.end(),
                   cell_sizes_plus.begin(),
                   [](std::size_t x) { return x + 1; });

    // Helper container for sparse CCL calculations
    vecmem::data::jagged_vector_buffer<unsigned int> sparse_ccl_indices(
        cell_sizes_plus, m_mr.get());
    copy.setup(sparse_ccl_indices);

    // Vector buffer for prefix sums for proper indexing, used only on device.
    // Vector with "clusters per module" is needed for measurement creation
    // buffer
    vecmem::data::vector_buffer<std::size_t> cluster_prefix_sum(num_modules,
                                                                m_mr.get());
    vecmem::data::vector_buffer<std::size_t> clusters_per_module(num_modules,
                                                                 m_mr.get());
    copy.setup(cluster_prefix_sum);
    copy.setup(clusters_per_module);

    // Invoke the reduction kernel that gives the total number of clusters which
    // will be found (and also computes the prefix sums and clusters per module)
    auto total_clusters = vecmem::make_unique_alloc<unsigned int>(m_mr.get());
    *total_clusters = 0;

    // Clusters sum kernel
    traccc::sycl::clusters_sum(cells_view, sparse_ccl_indices, *total_clusters,
                               cluster_prefix_sum, clusters_per_module,
                               m_queue);

    // Get the prefix sum of the cells
    const device::prefix_sum_t cells_prefix_sum =
        device::get_prefix_sum(cell_sizes, m_mr.get());

    // Vector of the exact cluster sizes, will be filled in cluster counting
    vecmem::vector<unsigned int> cluster_sizes(*total_clusters, 0, &m_mr.get());

    // Cluster counting kernel
    traccc::sycl::cluster_counting(
        sparse_ccl_indices, vecmem::get_data(cluster_sizes), cluster_prefix_sum,
        vecmem::get_data(cells_prefix_sum), m_queue);

    // Cluster container buffer for the clusters and headers (cluster ids)
    cluster_container_types::buffer clusters_buffer{
        {*total_clusters, m_mr.get()},
        {std::vector<std::size_t>(*total_clusters, 0),
         std::vector<std::size_t>(cluster_sizes.begin(), cluster_sizes.end()),
         m_mr.get()}};
    copy.setup(clusters_buffer.headers);
    copy.setup(clusters_buffer.items);

    // Component connection kernel
    traccc::sycl::component_connection(
        clusters_buffer, cells_view, sparse_ccl_indices, cluster_prefix_sum,
        vecmem::get_data(cells_prefix_sum), m_queue);

    // Copy the sizes of clusters per each module to the std vector for
    // initialization of measurements buffer
    std::vector<std::size_t> clusters_per_module_host;
    copy(clusters_per_module, clusters_per_module_host);

    // Resizable buffer for the measurements
    measurement_container_buffer measurements_buffer{
        {num_modules, m_mr.get()},
        {std::vector<std::size_t>(num_modules, 0), clusters_per_module_host,
         m_mr.get()}};
    copy.setup(measurements_buffer.headers);
    copy.setup(measurements_buffer.items);

    // Measurement creation kernel
    traccc::sycl::measurement_creation(measurements_buffer, clusters_buffer,
                                       cells_view, m_queue);

    // Spacepoint container buffer to fill in spacepoint formation
    spacepoint_container_buffer spacepoints_buffer{
        {num_modules, m_mr.get()},
        {std::vector<std::size_t>(num_modules, 0), clusters_per_module_host,
         m_mr.get()}};
    copy.setup(spacepoints_buffer.headers);
    copy.setup(spacepoints_buffer.items);

    // Get the prefix sum of the measurements.
    const device::prefix_sum_t measurements_prefix_sum = device::get_prefix_sum(
        copy.get_sizes(measurements_buffer.items), m_mr.get());

    // Spacepoint formation kernel
    traccc::sycl::spacepoint_formation(
        spacepoints_buffer, measurements_buffer,
        vecmem::get_data(measurements_prefix_sum), m_queue);

    // Copy the results back to the host
    host_spacepoint_container spacepoints_per_event(&m_mr.get());
    copy(spacepoints_buffer.headers, spacepoints_per_event.get_headers());
    copy(spacepoints_buffer.items, spacepoints_per_event.get_items());

    return spacepoints_per_event;
}

}  // namespace traccc::sycl