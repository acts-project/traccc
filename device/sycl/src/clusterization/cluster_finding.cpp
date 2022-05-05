/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// SYCL library include(s).
#include "traccc/sycl/clusterization/cluster_finding.hpp"

// SYCL library include(s).
#include "component_connection.hpp"
#include "measurement_creation.hpp"
#include "cluster_counting.hpp"
#include "clusters_sum.hpp"

#include <numeric>

// Vecmem include(s).
#include <vecmem/utils/sycl/copy.hpp>

namespace traccc::sycl {

cluster_finding::cluster_finding(vecmem::memory_resource &mr,
                                 queue_wrapper queue)
    : m_mr(mr), m_queue(queue) {}

host_measurement_container cluster_finding::operator()(
    const cell_container_types::host &cells_per_event) const {

    // Number of modules
    unsigned int num_modules = cells_per_event.size();

    // Vecmem copy object for moving the data between host and device
    vecmem::copy copy;

    // Get the sizes of the cells in each module
    std::vector<std::size_t> cell_sizes(num_modules, 0);
    for (unsigned long j = 0; j < num_modules; ++j) {
        cell_sizes[j] = cells_per_event.get_items().at(j).size() + 1;
    }

    // Helper container for sparse CCL calculations
    vecmem::data::jagged_vector_buffer<unsigned int> sparse_ccl_indices{cell_sizes,
                                                                     m_mr.get()};
    copy.setup(sparse_ccl_indices);

    // Number of all clusters that will be found
    // cluster_prefix_sum will contain just sizes of clusters per each module after this kernel
    // prefix sum will be created on the host side
    vecmem::vector<std::size_t> cluster_prefix_sum(num_modules, &m_mr.get());
    traccc::sycl::clusters_sum(cells_per_event, sparse_ccl_indices, vecmem::get_data(cluster_prefix_sum), m_mr.get(), m_queue);

    // save the sizes of clusters per each module found to the std vector for measurement
    // buffer initialization
    std::vector<std::size_t> clusters_per_module(num_modules);
    for (std::size_t i = 0; i < num_modules; ++i) {
        clusters_per_module[i] = cluster_prefix_sum[i];
    }

    // Create the prefix sums
    std::inclusive_scan(cluster_prefix_sum.begin(), cluster_prefix_sum.end(), cluster_prefix_sum.begin());
    unsigned int total_clusters = cluster_prefix_sum.back();

    // Vector with cluster sizes
    vecmem::vector<std::size_t> cluster_sizes(total_clusters, 0, &m_mr.get()); 
    traccc::sycl::cluster_counting(cells_per_event, sparse_ccl_indices, vecmem::get_data(cluster_sizes),
                                                                    vecmem::get_data(cluster_prefix_sum), m_mr.get(), m_queue);

    // copy the sizes tot he std::vector to construct the buffer for component connection
    std::vector<std::size_t> cluster_sizes_host(total_clusters);
    for (std::size_t i = 0; i < total_clusters; ++i){
        cluster_sizes_host[i] = cluster_sizes[i];
    }
    // Cluster container buffer for the clusters and headers (cluster ids)
    cluster_container_buffer clusters_buffer{
        {total_clusters, m_mr.get()},
        {std::vector<std::size_t>(total_clusters, 0), 
        cluster_sizes_host,
         m_mr.get()}};

    copy.setup(clusters_buffer.headers);
    copy.setup(clusters_buffer.items);

    traccc::sycl::component_connection(clusters_buffer, cells_per_event, sparse_ccl_indices, 
                                        vecmem::get_data(cluster_prefix_sum),
                                       m_mr.get(), m_queue);

    // Resizable buffer for the measurements
    measurement_container_buffer measurement_buffer{
        {num_modules, m_mr.get()},
        {std::vector<std::size_t>(num_modules, 0), clusters_per_module,
         m_mr.get()}};
    copy.setup(measurement_buffer.items);

    traccc::sycl::measurement_creation(measurement_buffer, clusters_buffer,
                                       total_clusters, m_queue);

    host_measurement_container measurements(&m_mr.get());
    copy(measurement_buffer.items, measurements.get_items());

    assert(cells_per_event.get_headers().size() ==
           measurements.get_items().size());
    for (const cell_module &cm : cells_per_event.get_headers()) {
        measurements.get_headers().push_back(cm);
    }

    return measurements;
}

}  // namespace traccc::sycl