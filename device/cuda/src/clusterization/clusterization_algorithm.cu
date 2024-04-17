/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// CUDA Library include(s).
#include "../utils/barrier.hpp"
#include "../utils/cuda_error_handling.hpp"
#include "../utils/utils.hpp"
#include "traccc/cuda/clusterization/clusterization_algorithm.hpp"

// Project include(s)
#include "traccc/clusterization/device/ccl_kernel.hpp"

// Vecmem include(s).
#include <vecmem/utils/copy.hpp>

namespace traccc::cuda {

namespace kernels {

/// CUDA kernel for running @c traccc::device::ccl_kernel
__global__ void ccl_kernel(
    const cell_collection_types::const_view cells_view,
    const cell_module_collection_types::const_view modules_view,
    const device::details::index_t max_cells_per_partition,
    const device::details::index_t target_cells_per_partition,
    measurement_collection_types::view measurements_view,
    vecmem::data::vector_view<unsigned int> cell_links) {

    __shared__ unsigned int partition_start, partition_end;
    __shared__ unsigned int outi;
    extern __shared__ device::details::index_t shared_v[];
    device::details::index_t* f = &shared_v[0];
    device::details::index_t* f_next = &shared_v[max_cells_per_partition];
    traccc::cuda::barrier barry_r;

    device::ccl_kernel(
        threadIdx.x, blockDim.x, blockIdx.x, cells_view, modules_view,
        max_cells_per_partition, target_cells_per_partition, partition_start,
        partition_end, outi, f, f_next, barry_r, measurements_view, cell_links);
}

}  // namespace kernels

clusterization_algorithm::clusterization_algorithm(
    const traccc::memory_resource& mr, vecmem::copy& copy, stream& str,
    const unsigned short target_cells_per_partition)
    : m_mr(mr),
      m_copy(copy),
      m_stream(str),
      m_target_cells_per_partition(target_cells_per_partition) {}

clusterization_algorithm::output_type clusterization_algorithm::operator()(
    const cell_collection_types::const_view& cells,
    const cell_module_collection_types::const_view& modules) const {

    // Get a convenience variable for the stream that we'll be using.
    cudaStream_t stream = details::get_stream(m_stream);

    // Get the number of cells
    const cell_collection_types::view::size_type num_cells =
        m_copy.get().get_size(cells);

    // Create the result object, overestimating the number of measurements.
    measurement_collection_types::buffer measurements{
        num_cells, m_mr.main, vecmem::data::buffer_type::resizable};
    m_copy.get().setup(measurements);

    // If there are no cells, return right away.
    if (num_cells == 0) {
        return measurements;
    }

    // Create buffer for linking cells to their measurements.
    //
    // @todo Construct cell clusters on demand in a member function for
    // debugging.
    //
    vecmem::data::vector_buffer<unsigned int> cell_links(num_cells, m_mr.main);
    m_copy.get().setup(cell_links);

    // Launch ccl kernel. Each thread will handle a single cell.
    const device::details::ccl_kernel_helper helper{
        m_target_cells_per_partition, num_cells};
    kernels::ccl_kernel<<<helper.num_partitions, helper.threads_per_partition,
                          2 * helper.max_cells_per_partition *
                              sizeof(device::details::index_t),
                          stream>>>(
        cells, modules, helper.max_cells_per_partition,
        m_target_cells_per_partition, measurements, cell_links);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    // Return the reconstructed measurements.
    return measurements;
}

}  // namespace traccc::cuda
