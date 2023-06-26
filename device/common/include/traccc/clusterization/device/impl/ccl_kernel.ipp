/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/clusterization/device/aggregate_cluster.hpp"
#include "traccc/clusterization/device/reduce_problem_cell.hpp"

namespace traccc::device {

/// Implementation of a FastSV algorithm with the following steps:
///   1) mix of stochastic and aggressive hooking
///   2) shortcutting
///
/// The implementation corresponds to an adapted versiion of Algorithm 3 of
/// the following paper:
/// https://www.sciencedirect.com/science/article/pii/S0743731520302689
///
///                     This array only gets updated at the end of the iteration
///                     to prevent race conditions.
/// @param[in] adjc     The number of adjacent cells
/// @param[in] adjv     Vector of adjacent cells
/// @param[in] tid      The thread index
/// @param[in] blckDim  The block size
/// @param[inout] f     array holding the parent cell ID for the current
/// iteration.
/// @param[inout] gf    array holding grandparent cell ID from the previous
/// iteration.
/// @param[in] barrier  A generic object for block-wide synchronisation
///
template <typename barrier_t>
TRACCC_DEVICE void fast_sv_1(index_t* f, index_t* gf,
                             unsigned char adjc[MAX_CELLS_PER_THREAD],
                             index_t adjv[MAX_CELLS_PER_THREAD][8],
                             const index_t tid, const index_t blckDim,
                             barrier_t& barrier) {
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
        for (index_t tst = 0; tst < MAX_CELLS_PER_THREAD; ++tst) {
            const index_t cid = tst * blckDim + tid;

            __builtin_assume(adjc[tst] <= 8);
            for (unsigned char k = 0; k < adjc[tst]; ++k) {
                index_t q = gf[adjv[tst][k]];

                if (gf[cid] > q) {
                    f[f[cid]] = q;
                    f[cid] = q;
                }
            }
        }

        /*
         * Each stage in this algorithm must be preceded by a
         * synchronization barrier!
         */
        barrier.blockBarrier();

#pragma unroll
        for (index_t tst = 0; tst < MAX_CELLS_PER_THREAD; ++tst) {
            const index_t cid = tst * blckDim + tid;
            /*
             * The second stage is shortcutting, which is an optimisation that
             * allows us to look at any shortcuts in the cluster IDs that we
             * can merge without adjacency information.
             */
            if (f[cid] > gf[cid]) {
                f[cid] = gf[cid];
            }
        }

        /*
         * Synchronize before the final stage.
         */
        barrier.blockBarrier();

#pragma unroll
        for (index_t tst = 0; tst < MAX_CELLS_PER_THREAD; ++tst) {
            const index_t cid = tst * blckDim + tid;
            /*
             * Update the array for the next generation, keeping track of any
             * changes we make.
             */
            if (gf[cid] != f[f[cid]]) {
                gf[cid] = f[f[cid]];
                gf_changed = true;
            }
        }

        /*
         * To determine whether we need another iteration, we use block
         * voting mechanics. Each thread checks if it has made any changes
         * to the arrays, and votes. If any thread votes true, all threads
         * will return a true value and go to the next iteration. Only if
         * all threads return false will the loop exit.
         */
    } while (barrier.blockOr(gf_changed));
}

template <typename barrier_t>
TRACCC_DEVICE inline void ccl_kernel(
    const index_t threadId, const index_t blckDim, const unsigned int blockId,
    const cell_collection_types::const_view cells_view,
    const cell_module_collection_types::const_view modules_view,
    const index_t max_cells_per_partition,
    const index_t target_cells_per_partition, unsigned int& partition_start,
    unsigned int& partition_end, unsigned int& outi, index_t* f, index_t* gf,
    barrier_t& barrier,
    alt_measurement_collection_types::view measurements_view,
    unsigned int& measurement_count,
    vecmem::data::vector_view<unsigned int> cell_links) {

    // Get device copy of input parameters
    const cell_collection_types::const_device cells_device(cells_view);
    const cell_module_collection_types::const_device modules_device(
        modules_view);
    alt_measurement_collection_types::device measurements_device(
        measurements_view);

    const unsigned int num_cells = cells_device.size();

    /*
     * First, we determine the exact range of cells that is to be examined
     * by this block of threads. We start from an initial range determined
     * by the block index multiplied by the target number of cells per
     * block. We then shift both the start and the end of the block forward
     * (to a later point in the array); start and end may be moved different
     * amounts.
     */
    if (threadId == 0) {
        unsigned int start = blockId * target_cells_per_partition;
        assert(start < num_cells);
        unsigned int end =
            std::min(num_cells, start + target_cells_per_partition);
        outi = 0;

        /*
         * Next, shift the starting point to a position further in the
         * array; the purpose of this is to ensure that we are not operating
         * on any cells that have been claimed by the previous block (if
         * any).
         */
        while (start != 0 &&
               cells_device[start - 1].module_link ==
                   cells_device[start].module_link &&
               cells_device[start].channel1 <=
                   cells_device[start - 1].channel1 + 1) {
            ++start;
        }

        /*
         * Then, claim as many cells as we need past the naive end of the
         * current block to ensure that we do not end our partition on a
         * cell that is not a possible boundary!
         */
        while (end < num_cells &&
               cells_device[end - 1].module_link ==
                   cells_device[end].module_link &&
               cells_device[end].channel1 <=
                   cells_device[end - 1].channel1 + 1) {
            ++end;
        }
        partition_start = start;
        partition_end = end;
    }

    barrier.blockBarrier();

    // Vector of indices of the adjacent cells
    index_t adjv[MAX_CELLS_PER_THREAD][8];
    /*
     * The number of adjacent cells for each cell must start at zero, to
     * avoid uninitialized memory. adjv does not need to be zeroed, as
     * we will only access those values if adjc indicates that the value
     * is set.
     */
    // Number of adjacent cells
    unsigned char adjc[MAX_CELLS_PER_THREAD];

    // It seems that sycl runs into undefined behaviour when calling
    // group synchronisation functions when some threads have already run
    // into a return. As such, we cannot use returns in this kernel.

    // Get partition for this thread group
    const index_t size = partition_end - partition_start;
    assert(size <= max_cells_per_partition);

#pragma unroll
    for (index_t tst = 0; tst < MAX_CELLS_PER_THREAD; ++tst) {
        adjc[tst] = 0;
    }

    for (index_t tst = 0, cid; (cid = tst * blckDim + threadId) < size; ++tst) {
        /*
         * Look for adjacent cells to the current one.
         */
        device::reduce_problem_cell(cells_device, cid, partition_start,
                                    partition_end, adjc[tst], adjv[tst]);
    }

#pragma unroll
    for (index_t tst = 0; tst < MAX_CELLS_PER_THREAD; ++tst) {
        const index_t cid = tst * blckDim + threadId;
        /*
         * At the start, the values of f and gf should be equal to the
         * ID of the cell.
         */
        f[cid] = cid;
        gf[cid] = cid;
    }

    /*
     * Now that the data has initialized, we synchronize again before we
     * move onto the actual processing part.
     */
    barrier.blockBarrier();

    /*
     * Run FastSV algorithm, which will update the father index to that of
     * the cell belonging to the same cluster with the lowest index.
     */
    fast_sv_1(&f[0], &gf[0], adjc, adjv, threadId, blckDim, barrier);

    barrier.blockBarrier();

    /*
     * Count the number of clusters by checking how many cells have
     * themself assigned as a parent.
     */
    for (index_t tst = 0, cid; (cid = tst * blckDim + threadId) < size; ++tst) {

        if (f[cid] == cid) {
            // Increment the summary values in the header object.
            vecmem::device_atomic_ref<unsigned int,
                                      vecmem::device_address_space::local>
                atom(outi);
            atom.fetch_add(1);
        }
    }

    barrier.blockBarrier();

    /*
     * Add the number of clusters of each thread block to the total
     * number of clusters. At the same time, a cluster id is retrieved
     * for the next data processing step.
     * Note that this might be not the same cluster as has been treated
     * previously. However, since each thread block spawns a the maximum
     * amount of threads per block, this has no sever implications.
     */
    if (threadId == 0) {
        vecmem::device_atomic_ref<unsigned int,
                                  vecmem::device_address_space::global>
            atom(measurement_count);
        outi = atom.fetch_add(outi);
    }

    barrier.blockBarrier();

    /*
     * Get the position to fill the measurements found in this thread group.
     */
    const unsigned int groupPos = outi;

    barrier.blockBarrier();

    if (threadId == 0) {
        outi = 0;
    }

    barrier.blockBarrier();

    const vecmem::data::vector_view<unsigned short> f_view(
        max_cells_per_partition, &f[0]);

    for (index_t tst = 0, cid; (cid = tst * blckDim + threadId) < size; ++tst) {
        if (f[cid] == cid) {
            /*
             * If we are a cluster owner, atomically claim a position in the
             * output array which we can write to.
             */
            vecmem::device_atomic_ref<unsigned int,
                                      vecmem::device_address_space::local>
                atom(outi);
            const unsigned int id = atom.fetch_add(1);

            device::aggregate_cluster(cells_device, modules_device, f_view,
                                      partition_start, partition_end, cid,
                                      measurements_device[groupPos + id],
                                      cell_links, groupPos + id);
        }
    }
}

}  // namespace traccc::device