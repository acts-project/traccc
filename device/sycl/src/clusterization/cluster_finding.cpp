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

// Vecmem include(s).
#include <vecmem/utils/sycl/copy.hpp>

namespace traccc::sycl {

cluster_finding::cluster_finding(vecmem::memory_resource &mr,
                                 queue_wrapper queue)
    : m_mr(mr), m_queue(queue) {}

host_measurement_container cluster_finding::operator()(
    const host_cell_container &cells_per_event) const {

    // Number of modules
    unsigned int num_modules = cells_per_event.size();

    // Number of cells from all the modules
    unsigned int cells_total_size = cells_per_event.total_size();

    // Vecmem copy object for moving the data between host and device
    vecmem::sycl::copy copy{m_queue.queue()};

    // Find the largest cell collection
    // to estimate max. size of cluster (assuming that this cell collection
    // would constitute to the whole cluster)
    unsigned int max_cluster_size =
        std::max_element(
            cells_per_event.get_items().begin(),
            cells_per_event.get_items().end(),
            [](auto &c1, auto &c2) { return c1.size() < c2.size(); })
            ->size();

    // Cluster container buffer for the clusters and headers (cluster ids)
    // assuming the worst case where 1 cell = 1 cluster
    // and max. number of cells per module is max. number of cells per cluster
    cluster_container_buffer clusters_buffer{
        {cells_total_size, m_mr.get()},
        {std::vector<std::size_t>(cells_total_size, 0),
         std::vector<std::size_t>(cells_total_size, max_cluster_size),
         m_mr.get()}};
    copy.setup(clusters_buffer.headers);
    copy.setup(clusters_buffer.items);

    // Vector for counts of clusters per each module
    vecmem::vector<unsigned int> cluster_sizes(num_modules, 1, &m_mr.get());

    // Atomic count of all clusters (needed inside component connection kernel
    // but also here)
    auto total_clusters = vecmem::make_unique_alloc<unsigned int>(m_mr.get());
    *total_clusters = 0;

    traccc::sycl::component_connection(clusters_buffer, cells_per_event,
                                       vecmem::get_data(cluster_sizes),
                                       total_clusters, m_mr.get(), m_queue);

    // Copy the vecmem vector of cluster sizes to the std vector for measurement
    // buffer initialization
    std::vector<std::size_t> cluster_sizes_host(num_modules);
    for (std::size_t i = 0; i < num_modules; ++i) {
        cluster_sizes_host[i] = cluster_sizes[i];
    }

    // Resizable buffer for the measurements
    measurement_container_buffer measurement_buffer{
        {num_modules, m_mr.get()},
        {std::vector<std::size_t>(num_modules, 0), cluster_sizes_host,
         m_mr.get()}};
    copy.setup(measurement_buffer.items);

    // range of kernel execution
    unsigned int range = *total_clusters;

    traccc::sycl::measurement_creation(measurement_buffer, clusters_buffer,
                                       range, m_queue);

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