/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// CUDA Library include(s).
#include "../utils/utils.hpp"
#include "traccc/cuda/clusterization/clusterization_algorithm.hpp"
#include "traccc/cuda/utils/definitions.hpp"

// Project include(s)
#include "traccc/clusterization/device/aggregate_cluster.hpp"
#include "traccc/clusterization/device/form_spacepoints.hpp"
#include "traccc/clusterization/device/reduce_problem_cell.hpp"

// Vecmem include(s).
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <algorithm>

namespace traccc::cuda {

namespace {
/// These indices in clusterization will only range from 0 to
/// MAX_CELLS_PER_PARTITION, so we only need a short
using index_t = unsigned short;
}  // namespace

namespace kernels {

/// Implementation of a FastSV algorithm with the following steps:
///   1) mix of stochastic and aggressive hooking
///   2) shortcutting
///
/// The implementation corresponds to an adapted versiion of Algorithm 3 of
/// the following paper:
/// https://www.sciencedirect.com/science/article/pii/S0743731520302689
///
/// @param[inout] f     array holding the parent cell ID for the current
/// iteration.
/// @param[inout] gf    array holding grandparent cell ID from the previous
/// iteration.
///                     This array only gets updated at the end of the iteration
///                     to prevent race conditions.
/// @param[in] adjc     The number of adjacent cells
/// @param[in] adjv     Vector of adjacent cells
/// @param[in] tid      The thread index
///
__device__ void fast_sv_1(index_t* f, index_t* gf, unsigned char adjc,
                          index_t adjv[8], index_t tid) {
    /*
     * The algorithm finishes if an iteration leaves the arrays unchanged.
     * This varible will be set if a change is made, and dictates if another
     * loop is necessary.
     */
    bool gf_changed;

    do {
        /*
         * Reset the end-parameter to false, so we can set it to true if we
         * make a change to the gf array.
         */
        gf_changed = false;

        /*
         * The algorithm executes in a loop of three distinct parallel
         * stages. In this first one, a mix of stochastic and aggressive
         * hooking, we examine adjacent cells and copy their grand parents
         * cluster ID if it is lower than ours, essentially merging the two
         * together.
         */
        __builtin_assume(adjc <= 8);
        for (unsigned char k = 0; k < adjc; ++k) {
            index_t q = gf[adjv[k]];

            if (gf[tid] > q) {
                f[f[tid]] = q;
                f[tid] = q;
            }
        }

        /*
         * Each stage in this algorithm must be preceded by a
         * synchronization barrier!
         */
        __syncthreads();

        /*
         * The second stage is shortcutting, which is an optimisation that
         * allows us to look at any shortcuts in the cluster IDs that we
         * can merge without adjacency information.
         */
        if (f[tid] > gf[tid]) {
            f[tid] = gf[tid];
        }

        /*
         * Synchronize before the final stage.
         */
        __syncthreads();

        /*
         * Update the array for the next generation, keeping track of any
         * changes we make.
         */
        if (gf[tid] != f[f[tid]]) {
            gf[tid] = f[f[tid]];
            gf_changed = true;
        }

        /*
         * To determine whether we need another iteration, we use block
         * voting mechanics. Each thread checks if it has made any changes
         * to the arrays, and votes. If any thread votes true, all threads
         * will return a true value and go to the next iteration. Only if
         * all threads return false will the loop exit.
         */
    } while (__syncthreads_or(gf_changed));
}

__global__ void ccl_kernel(
    const alt_cell_collection_types::const_view cells_view,
    const cell_module_collection_types::const_view modules_view,
    const partition_collection_types::const_view partitions_view,
    const unsigned short max_cells_per_partition,
    alt_measurement_collection_types::view measurements_view,
    unsigned int& measurement_count) {

    const index_t tid = threadIdx.x;

    const partition_collection_types::const_device partitions_device(
        partitions_view);
    assert(blockIdx.x < partitions_device.size());

    // Get partition for this thread group
    const partition start = partitions_device[blockIdx.x];
    const partition end = partitions_device[blockIdx.x + 1];
    assert(end - start <= max_cells_per_partition);

    // Check if any work needs to be done
    if (tid >= end - start) {
        return;
    }

    const alt_cell_collection_types::const_device cells_device(cells_view);
    const cell_module_collection_types::const_device modules_device(
        modules_view);

    alt_measurement_collection_types::device measurements_device(
        measurements_view);

    // Vector of indices of the adjacent cells
    index_t adjv[8];
    /*
     * The number of adjacent cells for each cell must start at zero, to
     * avoid uninitialized memory. adjv does not need to be zeroed, as
     * we will only access those values if adjc indicates that the value
     * is set.
     */
    // Number of adjacent cells
    unsigned char adjc = 0;

    /*
     * Look for adjacent cells to the current one.
     */
    device::reduce_problem_cell(cells_device, tid, start, end, adjc, adjv);

    /*
     * These arrays are the meat of the pudding of this algorithm, and we
     * will constantly be writing and reading from them which is why we
     * declare them to be in the fast shared memory. Note that this places a
     * limit on the maximum contiguous activations per module, as the amount of
     * shared memory is limited. These could always be moved to global memory,
     * but the algorithm would be decidedly slower in that case.
     */
    extern __shared__ index_t shared_v[];
    index_t* f = &shared_v[0];
    index_t* f_next = &shared_v[max_cells_per_partition];

    /*
     * At the start, the values of f and f_next should be equal to the
     * ID of the cell.
     */
    f[tid] = tid;
    f_next[tid] = tid;

    /*
     * Now that the data has initialized, we synchronize again before we
     * move onto the actual processing part.
     */
    __syncthreads();

    /*
     * Run FastSV algorithm, which will update the father index to that of the
     * cell belonging to the same cluster with the lowest index.
     */
    fast_sv_1(f, f_next, adjc, adjv, tid);

    /*
     * This variable will be used to write to the output later.
     */
    __shared__ unsigned int outi;

    /*
     * Initialize the counter of clusters per thread block
     */
    if (tid == 0) {
        outi = 0;
    }

    __syncthreads();

    /*
     * Count the number of clusters by checking how many cells have
     * themself assigned as a parent.
     */
    if (f[tid] == tid) {
        atomicAdd(&outi, 1);
    }

    __syncthreads();

    /*
     * Add the number of clusters of each thread block to the total
     * number of clusters. At the same time, a cluster id is retrieved
     * for the next data processing step.
     * Note that this might be not the same cluster as has been treated
     * previously. However, since each thread block spawns a the maximum
     * amount of threads per block, this has no sever implications.
     */
    if (tid == 0) {
        outi = atomicAdd(&measurement_count, outi);
    }

    __syncthreads();

    /*
     * Get the position to fill the measurements found in this thread group.
     */
    const unsigned int groupPos = outi;

    __syncthreads();

    if (tid == 0) {
        outi = 0;
    }

    __syncthreads();

    vecmem::data::vector_view<index_t> f_view(max_cells_per_partition, f);
    vecmem::data::vector_view<index_t> f_next_view(max_cells_per_partition,
                                                   f_next);

    if (f[tid] == tid) {
        /*
         * If we are a cluster owner, atomically claim a position in the
         * output array which we can write to.
         */
        const unsigned int id = atomicAdd(&outi, 1);
        device::aggregate_cluster(cells_device, modules_device, f_view, start,
                                  end, tid, measurements_device[groupPos + id]);
    }
}

__global__ void form_spacepoints(
    alt_measurement_collection_types::const_view measurements_view,
    cell_module_collection_types::const_view modules_view,
    const unsigned int measurement_count,
    spacepoint_collection_types::view spacepoints_view) {

    device::form_spacepoints(threadIdx.x + blockIdx.x * blockDim.x,
                             measurements_view, modules_view, measurement_count,
                             spacepoints_view);
}

}  // namespace kernels

clusterization_algorithm::clusterization_algorithm(
    const traccc::memory_resource& mr, vecmem::copy& copy, stream& str,
    const unsigned short max_cells_per_partition)
    : m_mr(mr),
      m_copy(copy),
      m_stream(str),
      m_max_cells_per_partition(max_cells_per_partition) {}

clusterization_algorithm::output_type clusterization_algorithm::operator()(
    const alt_cell_collection_types::const_view& cells,
    const cell_module_collection_types::const_view& modules,
    const partition_collection_types::const_view& partitions) const {

    // Get a convenience variable for the stream that we'll be using.
    cudaStream_t stream = details::get_stream(m_stream);

    // Number of cells
    const alt_cell_collection_types::view::size_type num_cells =
        m_copy.get_size(cells);
    // Number of cell partitions
    const partition_collection_types::view::size_type num_partitions =
        m_copy.get_size(partitions) - 1;

    // Create result object for the CCL kernel with size overestimation
    alt_measurement_collection_types::buffer measurements_buffer(num_cells,
                                                                 m_mr.main);

    // Counter for number of measurements
    vecmem::unique_alloc_ptr<unsigned int> num_measurements_device =
        vecmem::make_unique_alloc<unsigned int>(m_mr.main);
    CUDA_ERROR_CHECK(
        cudaMemset(num_measurements_device.get(), 0, sizeof(unsigned int)));

    // Launch ccl kernel. Each thread will handle a single cell.
    kernels::
        ccl_kernel<<<num_partitions, m_max_cells_per_partition,
                     2 * m_max_cells_per_partition * sizeof(index_t), stream>>>(
            cells, modules, partitions, m_max_cells_per_partition,
            measurements_buffer, *num_measurements_device);

    CUDA_ERROR_CHECK(cudaGetLastError());
    m_stream.synchronize();

    // Copy number of measurements to host
    vecmem::unique_alloc_ptr<unsigned int> num_measurements_host =
        vecmem::make_unique_alloc<unsigned int>(*(m_mr.host));
    CUDA_ERROR_CHECK(cudaMemcpyAsync(
        num_measurements_host.get(), num_measurements_device.get(),
        sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
    m_stream.synchronize();

    spacepoint_collection_types::buffer spacepoints_buffer(
        *num_measurements_host, m_mr.main);

    // For the following kernel, we can now use whatever the desired number of
    // threads
    const unsigned int num_blocks =
        (*num_measurements_host + m_max_cells_per_partition - 1) /
        m_max_cells_per_partition;

    // Turn 2D measurements into 3D spacepoints
    kernels::
        form_spacepoints<<<num_blocks, m_max_cells_per_partition, 0, stream>>>(
            measurements_buffer, modules, *num_measurements_host,
            spacepoints_buffer);

    CUDA_ERROR_CHECK(cudaGetLastError());
    m_stream.synchronize();

    return spacepoints_buffer;
}

}  // namespace traccc::cuda