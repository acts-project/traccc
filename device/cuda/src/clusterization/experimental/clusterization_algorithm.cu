/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// CUDA Library include(s).
#include "../../utils/utils.hpp"
#include "traccc/cuda/clusterization/experimental/clusterization_algorithm.hpp"
#include "traccc/cuda/utils/barrier.hpp"
#include "traccc/cuda/utils/definitions.hpp"

// Project include(s)
#include "traccc/clusterization/device/aggregate_cluster.hpp"
#include "traccc/clusterization/device/ccl_kernel.hpp"
#include "traccc/clusterization/device/reduce_problem_cell.hpp"

// Vecmem include(s).
#include <vecmem/utils/copy.hpp>

// Thrust include(s).
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace traccc::cuda::experimental {

namespace {
/// These indices in clusterization will only range from 0 to
/// max_cells_per_partition, so we only need a short.
using index_t = unsigned short;

static constexpr int TARGET_CELLS_PER_THREAD = 8;
static constexpr int MAX_CELLS_PER_THREAD = 12;
}  // namespace

namespace kernels {

/// CUDA kernel for running @c traccc::device::ccl_kernel
__global__ void ccl_kernel(
    const cell_collection_types::const_view cells_view,
    const cell_module_collection_types::const_view modules_view,
    const index_t max_cells_per_partition,
    const index_t target_cells_per_partition,
    measurement_collection_types::view measurements_view,
    unsigned int& measurement_count,
    vecmem::data::vector_view<unsigned int> cell_links) {
    __shared__ unsigned int partition_start, partition_end;
    __shared__ unsigned int outi;
    extern __shared__ index_t shared_v[];
    index_t* f = &shared_v[0];
    index_t* f_next = &shared_v[max_cells_per_partition];
    traccc::cuda::barrier barry_r;

    device::ccl_kernel(threadIdx.x, blockDim.x, blockIdx.x, cells_view,
                       modules_view, max_cells_per_partition,
                       target_cells_per_partition, partition_start,
                       partition_end, outi, f, f_next, barry_r,
                       measurements_view, measurement_count, cell_links);
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

    // Number of cells
    const cell_collection_types::view::size_type num_cells =
        m_copy.get_size(cells);

    if (num_cells == 0) {
        return {0, m_mr.main};
    }

    // Create result object for the CCL kernel with size overestimation
    measurement_collection_types::buffer measurements_buffer(num_cells,
                                                             m_mr.main);
    m_copy.setup(measurements_buffer);

    // Counter for number of measurements
    vecmem::unique_alloc_ptr<unsigned int> num_measurements_device =
        vecmem::make_unique_alloc<unsigned int>(m_mr.main);
    CUDA_ERROR_CHECK(cudaMemsetAsync(num_measurements_device.get(), 0,
                                     sizeof(unsigned int), stream));

    const unsigned short max_cells_per_partition =
        (m_target_cells_per_partition * MAX_CELLS_PER_THREAD +
         TARGET_CELLS_PER_THREAD - 1) /
        TARGET_CELLS_PER_THREAD;
    const unsigned int threads_per_partition =
        (m_target_cells_per_partition + TARGET_CELLS_PER_THREAD - 1) /
        TARGET_CELLS_PER_THREAD;
    const unsigned int num_partitions =
        (num_cells + m_target_cells_per_partition - 1) /
        m_target_cells_per_partition;

    // Create buffer for linking cells to their spacepoints.
    vecmem::data::vector_buffer<unsigned int> cell_links(num_cells, m_mr.main);
    m_copy.setup(cell_links);

    // Launch ccl kernel. Each thread will handle a single cell.
    kernels::
        ccl_kernel<<<num_partitions, threads_per_partition,
                     2 * max_cells_per_partition * sizeof(index_t), stream>>>(
            cells, modules, max_cells_per_partition,
            m_target_cells_per_partition, measurements_buffer,
            *num_measurements_device, cell_links);

    CUDA_ERROR_CHECK(cudaGetLastError());

    // Copy number of measurements to host
    vecmem::unique_alloc_ptr<unsigned int> num_measurements_host =
        vecmem::make_unique_alloc<unsigned int>(
            (m_mr.host != nullptr) ? *(m_mr.host) : m_mr.main);
    CUDA_ERROR_CHECK(cudaMemcpyAsync(
        num_measurements_host.get(), num_measurements_device.get(),
        sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
    m_stream.synchronize();

    // Create a new measurement buffer with a right size
    measurement_collection_types::buffer new_measurements_buffer(
        *num_measurements_host, m_mr.main);
    m_copy.setup(new_measurements_buffer);

    vecmem::device_vector<measurement> measurements_device(measurements_buffer);
    vecmem::device_vector<measurement> new_measurements_device(
        new_measurements_buffer);

    CUDA_ERROR_CHECK(cudaMemcpyAsync(
        new_measurements_device.begin(), measurements_device.begin(),
        sizeof(measurement) * (*num_measurements_host),
        cudaMemcpyDeviceToDevice, stream));

    m_stream.synchronize();

    // Sort the measurements w.r.t geometry barcode
    thrust::sort(thrust::cuda::par.on(stream), new_measurements_device.begin(),
                 new_measurements_device.end(), measurement_sort_comp());

    return new_measurements_buffer;
}

}  // namespace traccc::cuda::experimental
