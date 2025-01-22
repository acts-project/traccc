/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// CUDA Library include(s).
#include "../sanity/contiguous_on.cuh"
#include "../sanity/ordered_on.cuh"
#include "../utils/barrier.hpp"
#include "../utils/cuda_error_handling.hpp"
#include "../utils/thread_id.hpp"
#include "../utils/utils.hpp"
#include "./kernels/ccl_kernel.cuh"
#include "traccc/clusterization/clustering_config.hpp"
#include "traccc/clusterization/device/ccl_kernel_definitions.hpp"
#include "traccc/cuda/clusterization/clusterization_algorithm.hpp"
#include "traccc/utils/projections.hpp"
#include "traccc/utils/relations.hpp"

// Vecmem include(s).
#include <cstring>
#include <vecmem/utils/copy.hpp>

namespace traccc::cuda {

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
      m_adjc_backup(m_config.backup_size(), m_mr.main),
      m_adjv_backup(m_config.backup_size() * 8, m_mr.main),
      m_backup_mutex(vecmem::make_unique_alloc<unsigned int>(m_mr.main)) {
    m_copy.get().setup(m_f_backup)->wait();
    m_copy.get().setup(m_gf_backup)->wait();
    m_copy.get().setup(m_adjc_backup)->wait();
    m_copy.get().setup(m_adjv_backup)->wait();
    TRACCC_CUDA_ERROR_CHECK(cudaMemset(
        m_backup_mutex.get(), 0,
        sizeof(std::remove_extent_t<decltype(m_backup_mutex)::element_type>)));
}

clusterization_algorithm::output_type clusterization_algorithm::operator()(
    const edm::silicon_cell_collection::const_view& cells,
    const silicon_detector_description::const_view& det_descr) const {

    // Get a convenience variable for the stream that we'll be using.
    cudaStream_t stream = details::get_stream(m_stream);

    // Get the number of cells
    const edm::silicon_cell_collection::const_view::size_type num_cells =
        m_copy.get().get_size(cells);

    // Create the result object, overestimating the number of measurements.
    measurement_collection_types::buffer measurements{
        num_cells, m_mr.main, vecmem::data::buffer_type::resizable};
    m_copy.get().setup(measurements)->ignore();

    // If there are no cells, return right away.
    if (num_cells == 0) {
        return measurements;
    }

    assert(is_contiguous_on<edm::silicon_cell_collection::const_device>(
        cell_module_projection(), m_mr.main, m_copy, m_stream, cells));
    assert(is_ordered_on<edm::silicon_cell_collection::const_device>(
        channel0_major_cell_order_relation(), m_mr.main, m_copy, m_stream,
        cells));

    // Create buffer for linking cells to their measurements.
    //
    // @todo Construct cell clusters on demand in a member function for
    // debugging.
    //
    vecmem::data::vector_buffer<unsigned int> cell_links(num_cells, m_mr.main);
    m_copy.get().setup(cell_links)->ignore();

    // Launch ccl kernel. Each thread will handle a single cell.
    unsigned int num_blocks =
        (num_cells + (m_config.target_partition_size()) - 1) /
        m_config.target_partition_size();

    // Ensure that the chosen maximum cell count is compatible with the maximum
    // stack size.
    assert(m_config.max_cells_per_thread <=
           device::details::CELLS_PER_THREAD_STACK_LIMIT);

    kernels::ccl_kernel<<<num_blocks, m_config.threads_per_partition,
                          2 * m_config.max_partition_size() *
                              sizeof(device::details::index_t),
                          stream>>>(
        m_config, cells, det_descr, measurements, cell_links, m_f_backup,
        m_gf_backup, m_adjc_backup, m_adjv_backup, m_backup_mutex.get());
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    // Return the reconstructed measurements.
    return measurements;
}

}  // namespace traccc::cuda
