/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../utils/barrier.hpp"
#include "../utils/hip_error_handling.hpp"
#include "../utils/thread_id.hpp"
#include "../utils/utils.hpp"
#include "./kernels/ccl_kernel.hpp"
#include "./kernels/reify_cluster_data.hpp"
#include "traccc/hip/clusterization/clusterization_algorithm.hpp"

// Project include(s).
#include "traccc/clusterization/device/ccl_kernel_definitions.hpp"

// Vecmem include(s).
#include <cassert>
#include <cstring>

// HIP include(s).
#include <hip/hip_runtime.h>

namespace traccc::hip {

clusterization_algorithm::clusterization_algorithm(
    const traccc::memory_resource& mr, vecmem::copy& copy, stream& str,
    const config_type& config, std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)),
      m_mr(mr),
      m_copy(copy),
      m_stream(str),
      m_config(config),
      m_f_backup(m_config.backup_size(), m_mr.main),
      m_gf_backup(m_config.backup_size(), m_mr.main),
      m_backup_mutex(vecmem::make_unique_alloc<unsigned int>(m_mr.main)),
      m_adjc_backup(m_config.backup_size(), m_mr.main),
      m_adjv_backup(m_config.backup_size() * 8, m_mr.main) {

    m_copy.get().setup(m_f_backup)->wait();
    m_copy.get().setup(m_gf_backup)->wait();
    m_copy.get().setup(m_adjc_backup)->wait();
    m_copy.get().setup(m_adjv_backup)->wait();
    TRACCC_HIP_ERROR_CHECK(hipMemset(
        m_backup_mutex.get(), 0,
        sizeof(std::remove_extent_t<decltype(m_backup_mutex)::element_type>)));
}

edm::measurement_collection<default_algebra>::buffer
clusterization_algorithm::operator()(
    const edm::silicon_cell_collection::const_view& cells,
    const silicon_detector_description::const_view& det_descr) const {

    return this->operator()(cells, det_descr,
                            device::clustering_discard_disjoint_set{});
}

edm::measurement_collection<default_algebra>::buffer
clusterization_algorithm::operator()(
    const edm::silicon_cell_collection::const_view& cells,
    const silicon_detector_description::const_view& det_descr,
    device::clustering_discard_disjoint_set&&) const {

    auto [res, djs] = this->execute_impl(cells, det_descr, false);
    assert(!djs.has_value());
    return std::move(res);
}

std::pair<edm::measurement_collection<default_algebra>::buffer,
          edm::silicon_cluster_collection::buffer>
clusterization_algorithm::operator()(
    const edm::silicon_cell_collection::const_view& cells,
    const silicon_detector_description::const_view& det_descr,
    device::clustering_keep_disjoint_set&&) const {

    auto [res, djs] = this->execute_impl(cells, det_descr, true);
    assert(djs.has_value());
    return {std::move(res), std::move(*djs)};
}

std::pair<edm::measurement_collection<default_algebra>::buffer,
          std::optional<edm::silicon_cluster_collection::buffer>>
clusterization_algorithm::execute_impl(
    const edm::silicon_cell_collection::const_view& cells,
    const silicon_detector_description::const_view& det_descr,
    bool keep_disjoint_set) const {

    // Get a convenience variable for the stream that we'll be using.
    hipStream_t stream = details::get_stream(m_stream);

    // Get the number of cells
    const edm::silicon_cell_collection::const_view::size_type num_cells =
        m_copy.get().get_size(cells);

    // Create the result object, overestimating the number of measurements.
    edm::measurement_collection<default_algebra>::buffer measurements{
        num_cells, m_mr.main, vecmem::data::buffer_type::resizable};
    m_copy.get().setup(measurements)->ignore();

    // If there are no cells, return right away.
    if (num_cells == 0) {
        return {std::move(measurements),
                traccc::edm::silicon_cluster_collection::buffer{}};
    }

    // Create buffer for linking cells to their measurements.
    //
    // @todo Construct cell clusters on demand in a member function for
    // debugging.
    //
    vecmem::data::vector_buffer<unsigned int> cell_links(num_cells, m_mr.main);
    m_copy.get().setup(cell_links)->ignore();

    // Launch ccl kernel. Each thread will handle a single cell.
    const unsigned int num_blocks_ccl =
        (num_cells + (m_config.target_partition_size()) - 1) /
        m_config.target_partition_size();

    // Ensure that the chosen maximum cell count is compatible with the maximum
    // stack size.
    assert(m_config.max_cells_per_thread <=
           device::details::CELLS_PER_THREAD_STACK_LIMIT);

    vecmem::data::vector_buffer<unsigned int> disjoint_set;
    vecmem::data::vector_buffer<unsigned int> cluster_sizes;

    // If we are keeping the disjoint set data structure, allocate space for it.
    if (keep_disjoint_set) {
        disjoint_set = {num_cells, m_mr.main};
        cluster_sizes = {num_cells, m_mr.main};
    }

    hipLaunchKernelGGL(
        kernels::ccl_kernel, num_blocks_ccl,
        dim3(m_config.threads_per_partition),
        2 * m_config.max_partition_size() * sizeof(device::details::index_t),
        stream, m_config, cells, det_descr, measurements, cell_links,
        m_f_backup, m_gf_backup, m_adjc_backup, m_adjv_backup,
        m_backup_mutex.get(), disjoint_set, cluster_sizes);
    TRACCC_HIP_ERROR_CHECK(hipGetLastError());

    std::optional<traccc::edm::silicon_cluster_collection::buffer>
        cluster_data = std::nullopt;

    if (keep_disjoint_set) {

        auto num_measurements = m_copy.get().get_size(measurements);

        // This could be further optimized by only copying the number of
        // elements necessary. But since cluster making is mainly meant for
        // performance measurements, on first order this should be good enough.
        vecmem::vector<unsigned int> cluster_sizes_host{
            m_mr.host != nullptr ? m_mr.host : &(m_mr.main)};
        m_copy.get()(cluster_sizes, cluster_sizes_host)->wait();
        cluster_sizes_host.resize(num_measurements);

        cluster_data.emplace(cluster_sizes_host, m_mr.main, m_mr.host,
                             vecmem::data::buffer_type::resizable);
        m_copy.get().setup(*cluster_data)->ignore();

        constexpr unsigned int num_threads_reify = 512;
        const unsigned int num_blocks_reify =
            (num_cells + num_threads_reify - 1) / num_threads_reify;

        hipLaunchKernelGGL(kernels::reify_cluster_data, num_blocks_reify,
                           num_threads_reify, 0, stream, disjoint_set,
                           *cluster_data);
        TRACCC_HIP_ERROR_CHECK(hipGetLastError());
    }

    // Return the reconstructed measurements.
    return {std::move(measurements), std::move(cluster_data)};
}

}  // namespace traccc::hip
