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
///                     iteration.
/// @param[inout] gf    array holding grandparent cell ID from the previous
///                     iteration.
/// @param[in] barrier  A generic object for block-wide synchronisation
///
template <typename barrier_t>
TRACCC_DEVICE void fast_sv_1(
    vecmem::device_vector<details::index_t>& f,
    vecmem::device_vector<details::index_t>& gf,
    unsigned char adjc[details::MAX_CELLS_PER_THREAD],
    details::index_t adjv[details::MAX_CELLS_PER_THREAD][8],
    const details::index_t tid, const details::index_t blckDim,
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
        for (details::index_t tst = 0; tst < details::MAX_CELLS_PER_THREAD;
             ++tst) {
            const details::index_t cid = tst * blckDim + tid;

            TRACCC_ASSUME(adjc[tst] <= 8);
            for (unsigned char k = 0; k < adjc[tst]; ++k) {
                details::index_t q = gf.at(adjv[tst][k]);

                if (gf.at(cid) > q) {
                    f.at(f.at(cid)) = q;
                    f.at(cid) = q;
                }
            }
        }

        /*
         * Each stage in this algorithm must be preceded by a
         * synchronization barrier!
         */
        barrier.blockBarrier();

        TRACCC_PRAGMA_UNROLL
        for (details::index_t tst = 0; tst < details::MAX_CELLS_PER_THREAD;
             ++tst) {
            const details::index_t cid = tst * blckDim + tid;
            /*
             * The second stage is shortcutting, which is an optimisation that
             * allows us to look at any shortcuts in the cluster IDs that we
             * can merge without adjacency information.
             */
            if (f.at(cid) > gf.at(cid)) {
                f.at(cid) = gf.at(cid);
            }
        }

        /*
         * Synchronize before the final stage.
         */
        barrier.blockBarrier();

        TRACCC_PRAGMA_UNROLL
        for (details::index_t tst = 0; tst < details::MAX_CELLS_PER_THREAD;
             ++tst) {
            const details::index_t cid = tst * blckDim + tid;
            /*
             * Update the array for the next generation, keeping track of any
             * changes we make.
             */
            if (gf.at(cid) != f.at(f.at(cid))) {
                gf.at(cid) = f.at(f.at(cid));
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
    const details::index_t threadId, const details::index_t blckDim,
    const unsigned int blockId,
    const cell_collection_types::const_view cells_view,
    const cell_module_collection_types::const_view modules_view,
    [[maybe_unused]] const details::index_t max_cells_per_partition,
    const details::index_t target_cells_per_partition,
    unsigned int& partition_start, unsigned int& partition_end,
    unsigned int& outi, vecmem::data::vector_view<details::index_t> f_view,
    vecmem::data::vector_view<details::index_t> gf_view, barrier_t& barrier,
    measurement_collection_types::view measurements_view,
    vecmem::data::vector_view<unsigned int> cell_links) {

    // Construct device containers around the views.
    const cell_collection_types::const_device cells_device(cells_view);
    const cell_module_collection_types::const_device modules_device(
        modules_view);
    measurement_collection_types::device measurements_device(measurements_view);
    vecmem::device_vector<details::index_t> f(f_view);
    vecmem::device_vector<details::index_t> gf(gf_view);

    const cell_collection_types::const_device::size_type num_cells =
        cells_device.size();

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
        assert(partition_start <= partition_end);
    }

    barrier.blockBarrier();

    // Vector of indices of the adjacent cells
    details::index_t adjv[details::MAX_CELLS_PER_THREAD][8];
    /*
     * The number of adjacent cells for each cell must start at zero, to
     * avoid uninitialized memory. adjv does not need to be zeroed, as
     * we will only access those values if adjc indicates that the value
     * is set.
     */
    // Number of adjacent cells
    unsigned char adjc[details::MAX_CELLS_PER_THREAD];

    // It seems that sycl runs into undefined behaviour when calling
    // group synchronisation functions when some threads have already run
    // into a return. As such, we cannot use returns in this kernel.

    // Get partition for this thread group
    const details::index_t size = partition_end - partition_start;
    assert(size <= max_cells_per_partition);

    TRACCC_PRAGMA_UNROLL
    for (details::index_t tst = 0; tst < details::MAX_CELLS_PER_THREAD; ++tst) {
        adjc[tst] = 0;
    }

    for (details::index_t tst = 0, cid; (cid = tst * blckDim + threadId) < size;
         ++tst) {
        /*
         * Look for adjacent cells to the current one.
         */
        assert(tst < details::MAX_CELLS_PER_THREAD);
        reduce_problem_cell(cells_device, cid, partition_start, partition_end,
                            adjc[tst], adjv[tst]);
    }

    TRACCC_PRAGMA_UNROLL
    for (details::index_t tst = 0; tst < details::MAX_CELLS_PER_THREAD; ++tst) {
        const details::index_t cid = tst * blckDim + threadId;
        /*
         * At the start, the values of f and gf should be equal to the
         * ID of the cell.
         */
        f.at(cid) = cid;
        gf.at(cid) = cid;
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
    fast_sv_1(f, gf, adjc, adjv, threadId, blckDim, barrier);

    barrier.blockBarrier();

    for (details::index_t tst = 0, cid; (cid = tst * blckDim + threadId) < size;
         ++tst) {
        if (f.at(cid) == cid) {
            // Add a new measurement to the output buffer. Remembering its
            // position inside of the container.
            const measurement_collection_types::device::size_type meas_pos =
                measurements_device.push_back({});
            // Set up the measurement under the appropriate index.
            aggregate_cluster(cells_device, modules_device, f_view,
                              partition_start, partition_end, cid,
                              measurements_device.at(meas_pos), cell_links,
                              meas_pos);
        }
    }
}

}  // namespace traccc::device
