/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/alpaka/clusterization/clusterization_algorithm.hpp"

#include "../utils/barrier.hpp"
#include "../utils/get_queue.hpp"
#include "../utils/thread_id.hpp"
#include "../utils/utils.hpp"

// Project include(s)
#include "traccc/clusterization/clustering_config.hpp"
#include "traccc/clusterization/device/ccl_kernel.hpp"
#include "traccc/clusterization/device/reify_cluster_data.hpp"
#include "traccc/utils/projections.hpp"
#include "traccc/utils/relations.hpp"

// System include(s).
#include <algorithm>
#include <mutex>

namespace traccc::alpaka {
namespace kernels {

/// Alpaka kernel functor for @c traccc::device::ccl_kernel
struct ccl_kernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, const clustering_config cfg,
        const edm::silicon_cell_collection::const_view cells_view,
        const silicon_detector_description::const_view det_descr_view,
        vecmem::data::vector_view<device::details::index_t> f_backup_view,
        vecmem::data::vector_view<device::details::index_t> gf_backup_view,
        vecmem::data::vector_view<unsigned char> adjc_backup_view,
        vecmem::data::vector_view<device::details::index_t> adjv_backup_view,
        uint32_t* backup_mutex_ptr,
        vecmem::data::vector_view<unsigned int> disjoint_set_view,
        vecmem::data::vector_view<unsigned int> cluster_size_view,
        measurement_collection_types::view measurements_view,
        vecmem::data::vector_view<unsigned int> cell_links) const {

        details::thread_id1 thread_id(acc);

        auto& partition_start =
            ::alpaka::declareSharedVar<std::size_t, __COUNTER__>(acc);
        auto& partition_end =
            ::alpaka::declareSharedVar<std::size_t, __COUNTER__>(acc);
        auto& outi = ::alpaka::declareSharedVar<std::size_t, __COUNTER__>(acc);

        device::details::index_t* const shared_v =
            ::alpaka::getDynSharedMem<device::details::index_t>(acc);
        vecmem::data::vector_view<device::details::index_t> f_view{
            cfg.max_partition_size(), shared_v};
        vecmem::data::vector_view<device::details::index_t> gf_view{
            cfg.max_partition_size(), shared_v + cfg.max_partition_size()};

        vecmem::device_atomic_ref<uint32_t> backup_mutex(*backup_mutex_ptr);

        alpaka::barrier<TAcc> barry_r(&acc);

        device::ccl_kernel(
            cfg, thread_id, cells_view, det_descr_view, partition_start,
            partition_end, outi, f_view, gf_view, f_backup_view, gf_backup_view,
            adjc_backup_view, adjv_backup_view, backup_mutex, disjoint_set_view,
            cluster_size_view, barry_r, measurements_view, cell_links);
    }
};

/// Alpaka kernel functor for @c traccc::device::reify_cluster_data
struct reify_cluster_data {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, vecmem::data::vector_view<unsigned int> disjoint_set,
        edm::silicon_cluster_collection::view clusters) const {

        device::reify_cluster_data(details::thread_id1{acc}.getGlobalThreadId(),
                                   disjoint_set, clusters);
    }
};

}  // namespace kernels

clusterization_algorithm::clusterization_algorithm(
    const traccc::memory_resource& mr, vecmem::copy& copy, queue& q,
    const config_type& config, std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)),
      m_config(config),
      m_mr(mr),
      m_copy(copy),
      m_queue(q),
      m_f_backup(m_config.backup_size(), m_mr.main),
      m_gf_backup(m_config.backup_size(), m_mr.main),
      m_adjc_backup(m_config.backup_size(), m_mr.main),
      m_adjv_backup(m_config.backup_size() * 8, m_mr.main),
      m_backup_mutex(vecmem::make_unique_alloc<unsigned int>(m_mr.main)) {

    m_copy.get().setup(m_f_backup)->wait();
    m_copy.get().setup(m_gf_backup)->wait();
    m_copy.get().setup(m_adjc_backup)->wait();
    m_copy.get().setup(m_adjv_backup)->wait();

    m_copy.get()
        .memset(
            vecmem::data::vector_view<unsigned int>{1u, m_backup_mutex.get()},
            0)
        ->wait();
}

clusterization_algorithm::output_type clusterization_algorithm::operator()(
    const edm::silicon_cell_collection::const_view& cells,
    const silicon_detector_description::const_view& det_descr,
    bool reconstruct_clusters) const {

    // Get the Alpaka queue.
    Queue& queue = details::get_queue(m_queue);

    // Number of cells
    const edm::silicon_cell_collection::view::size_type num_cells =
        m_copy.get().get_size(cells);

    // If there are no cells, return right away.
    if (num_cells == 0) {
        return {
            .measurements = {},
            .clusters =
                (reconstruct_clusters
                     ? std::optional<
                           edm::silicon_cluster_collection::
                               buffer>{edm::silicon_cluster_collection::
                                           buffer{}}
                     : std::optional<edm::silicon_cluster_collection::buffer>{
                           std::nullopt})};
    }

    // Create the result object, overestimating the number of measurements.
    output_type result{
        {num_cells, m_mr.main, vecmem::data::buffer_type::resizable},
        std::nullopt};
    m_copy.get().setup(result.measurements)->ignore();

    // Create buffer for linking cells to their measurements.
    vecmem::data::vector_buffer<unsigned int> cell_links(num_cells, m_mr.main);
    m_copy.get().setup(cell_links)->ignore();

    // Ensure that the chosen maximum cell count is compatible with the maximum
    // stack size.
    assert(m_config.max_cells_per_thread <=
           device::details::CELLS_PER_THREAD_STACK_LIMIT);

    // If we are keeping the disjoint set data structure, allocate space for it.
    vecmem::data::vector_buffer<unsigned int> disjoint_set;
    vecmem::data::vector_buffer<unsigned int> cluster_sizes;
    if (reconstruct_clusters) {
        disjoint_set = {num_cells, m_mr.main};
        cluster_sizes = {num_cells, m_mr.main};
    }

    // Launch ccl kernel. Each thread will handle a single cell.
    Idx num_blocks = (num_cells + (m_config.target_partition_size()) - 1) /
                     m_config.target_partition_size();
    static_assert(::alpaka::isMultiThreadAcc<Acc>,
                  "Clustering algorithm must be compiled for an accelerator "
                  "with support for multi-thread blocks.");
    auto workDiv = makeWorkDiv<Acc>(num_blocks, m_config.threads_per_partition);

    ::alpaka::exec<Acc>(
        queue, workDiv, kernels::ccl_kernel{}, m_config, cells, det_descr,
        vecmem::get_data(m_f_backup), vecmem::get_data(m_gf_backup),
        vecmem::get_data(m_adjc_backup), vecmem::get_data(m_adjv_backup),
        m_backup_mutex.get(), vecmem::get_data(disjoint_set),
        vecmem::get_data(cluster_sizes), vecmem::get_data(result.measurements),
        vecmem::get_data(cell_links));
    ::alpaka::wait(queue);

    if (reconstruct_clusters) {
        assert(m_mr.host != nullptr);

        auto num_measurements = m_copy.get().get_size(result.measurements);

        // This could be further optimized by only copying the number of
        // elements necessary. But since cluster making is mainly meant for
        // performance measurements, on first order this should be good enough.
        vecmem::vector<unsigned int> cluster_sizes_host{m_mr.host};
        m_copy.get()(cluster_sizes, cluster_sizes_host)->wait();
        cluster_sizes_host.resize(num_measurements);

        result.clusters.emplace(cluster_sizes_host, m_mr.main, m_mr.host,
                                vecmem::data::buffer_type::resizable);
        m_copy.get().setup(*(result.clusters))->ignore();

        constexpr unsigned int num_threads = 512;
        num_blocks = (num_cells + num_threads - 1) / num_threads;
        workDiv = makeWorkDiv<Acc>(num_blocks, num_threads);

        ::alpaka::exec<Acc>(queue, workDiv, kernels::reify_cluster_data{},
                            vecmem::get_data(disjoint_set),
                            vecmem::get_data(*(result.clusters)));
        ::alpaka::wait(queue);
    }

    // Return the reconstructed measurements/clusters.
    return result;
}

}  // namespace traccc::alpaka

// Define the required trait needed for Dynamic shared memory allocation.
namespace alpaka::trait {

template <typename TAcc>
struct BlockSharedMemDynSizeBytes<traccc::alpaka::kernels::ccl_kernel, TAcc> {
    template <typename TVec, typename... TArgs>
    ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
        traccc::alpaka::kernels::ccl_kernel const& /* kernel */,
        TVec const& /* blockThreadExtent */, TVec const& /* threadElemExtent */,
        const traccc::clustering_config config, TArgs const&... /* args */
        ) -> std::size_t {
        return static_cast<std::size_t>(2 * config.max_partition_size() *
                                        sizeof(unsigned short));
    }
};

}  // namespace alpaka::trait
