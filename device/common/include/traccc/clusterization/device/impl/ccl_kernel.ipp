/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <mutex>

#include "traccc/clusterization/clustering_config.hpp"
#include "traccc/clusterization/device/aggregate_cluster.hpp"
#include "traccc/clusterization/device/ccl_kernel.hpp"
#include "traccc/clusterization/device/ccl_kernel_definitions.hpp"
#include "traccc/clusterization/device/reduce_problem_cell.hpp"
#include "traccc/device/concepts/barrier.hpp"
#include "traccc/device/concepts/thread_id.hpp"
#include "traccc/device/mutex.hpp"
#include "traccc/device/sort.hpp"
#include "traccc/device/unique_lock.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/measurement.hpp"
#include "vecmem/memory/device_atomic_ref.hpp"

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
template <device::concepts::barrier barrier_t,
          device::concepts::thread_id1 thread_id_t>
TRACCC_DEVICE void fast_sv_1(const thread_id_t& thread_id,
                             vecmem::device_vector<details::index_t>& f,
                             vecmem::device_vector<details::index_t>& gf,
                             unsigned char* adjc, details::index_t* adjv,
                             details::index_t thread_cell_count,
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
        for (details::index_t tst = 0; tst < thread_cell_count; ++tst) {
            const details::index_t cid =
                tst * thread_id.getBlockDimX() + thread_id.getLocalThreadIdX();

            TRACCC_ASSUME(adjc[tst] <= 8);
            for (unsigned char k = 0; k < adjc[tst]; ++k) {
                details::index_t q = gf.at(adjv[8 * tst + k]);

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

        for (details::index_t tst = 0; tst < thread_cell_count; ++tst) {
            const details::index_t cid =
                tst * thread_id.getBlockDimX() + thread_id.getLocalThreadIdX();
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

        for (details::index_t tst = 0; tst < thread_cell_count; ++tst) {
            const details::index_t cid =
                tst * thread_id.getBlockDimX() + thread_id.getLocalThreadIdX();
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

template <device::concepts::barrier barrier_t,
          device::concepts::thread_id1 thread_id_t>
TRACCC_DEVICE inline void ccl_core(
    const thread_id_t& thread_id, details::ccl_kernel_static_smem_parcel& smem,
    vecmem::device_vector<details::index_t> f,
    vecmem::device_vector<details::index_t> gf,
    vecmem::data::vector_view<unsigned int> cell_links, details::index_t* adjv,
    unsigned char* adjc, const cell_collection_types::const_device cells_device,
    const cell_module_collection_types::const_device modules_device,
    measurement_collection_types::device measurements_device,
    barrier_t& barrier) {
    const details::index_t size = smem.partition_end - smem.partition_start;

    assert(size <= f.size());
    assert(size <= gf.size());

    details::index_t thread_cell_count =
        (size - thread_id.getLocalThreadIdX() + thread_id.getBlockDimX() - 1) /
        thread_id.getBlockDimX();

    for (details::index_t tst = 0; tst < thread_cell_count; ++tst) {
        /*
         * Look for adjacent cells to the current one.
         */
        const details::index_t cid =
            tst * thread_id.getBlockDimX() + thread_id.getLocalThreadIdX();
        adjc[tst] = 0;
        reduce_problem_cell(cells_device, cid, smem.partition_start,
                            smem.partition_end, adjc[tst], &adjv[8 * tst]);
    }

    for (details::index_t tst = 0; tst < thread_cell_count; ++tst) {
        const details::index_t cid =
            tst * thread_id.getBlockDimX() + thread_id.getLocalThreadIdX();
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
    fast_sv_1(thread_id, f, gf, adjc, adjv, thread_cell_count, barrier);

    barrier.blockBarrier();

    /*
     * We will now repurpose the `gf` shared vector to store the cells which
     * are parents. For the time being, `outi` will be the size of `gf`.
     */
    if (thread_id.getLocalThreadIdX() == 0) {
        smem.outi = 0;
    }

    barrier.blockBarrier();

    /*
     * We now collect the parents into the `gf` array.
     */
    for (details::index_t tst = 0; tst < thread_cell_count; ++tst) {
        const details::index_t cid =
            tst * thread_id.getBlockDimX() + thread_id.getLocalThreadIdX();

        if (f.at(cid) == cid) {
            vecmem::device_atomic_ref<unsigned int,
                                      vecmem::device_address_space::local>
                atom(smem.outi);
            gf.at(atom.fetch_add(1)) = cid;
        }
    }

    barrier.blockBarrier();

    uint32_t n_measurements = smem.outi;

    barrier.blockBarrier();

    /*
     * The lead threat resizes the measurement vector to contain the total
     * number of measurements. The `outi` variable is now repurposed to contain
     * the start of this block's region in the output array.
     */
    if (thread_id.getLocalThreadIdX() == 0) {
        smem.outi = measurements_device.bulk_append_implicit(n_measurements);
    }

    barrier.blockBarrier();

    /*
     * Sort the measurements so that the modules are ordered, which guarantees
     * that they will be contiguous.
     *
     * TODO: Can this be done more cleverly? Perhaps while collecting the
     * elements into `gf`.
     */
    blockOddEvenSort(
        thread_id, barrier, gf.data(), n_measurements,
        [&](const unsigned short lhs, const unsigned short rhs) {
            return cells_device.at(smem.partition_start + f.at(lhs))
                       .module_link <
                   cells_device.at(smem.partition_start + f.at(rhs))
                       .module_link;
        });

    barrier.blockBarrier();

    /*
     * Aggregate the clusters and write them to global memory.
     */
    for (uint32_t i = thread_id.getLocalThreadIdX(); i < n_measurements;
         i += thread_id.getBlockDimX()) {
        uint32_t meas_pos = smem.outi + i;
        // Set up the measurement under the appropriate index.
        aggregate_cluster(cells_device, modules_device, f, smem.partition_start,
                          smem.partition_end, gf.at(i),
                          measurements_device.at(meas_pos), cell_links,
                          meas_pos);
    }
}

TRACCC_DEVICE inline std::size_t ccl_partition_find_helper(
    const cell_collection_types::const_device cells, std::size_t begin,
    std::size_t target_end) {
    bool found_good = false;
    std::size_t last_good;

    for (std::size_t i = begin; i < target_end || !found_good; ++i) {
        if (i == cells.size() ||
            cells.at(i - 1).module_link != cells.at(i).module_link) {
            found_good = true;
            last_good = i;
        }
    }

    assert(found_good);
    return last_good;
}

template <device::concepts::barrier barrier_t,
          device::concepts::thread_id1 thread_id_t>
TRACCC_DEVICE inline void ccl_kernel(
    const clustering_config cfg, const thread_id_t& thread_id,
    const cell_collection_types::const_view cells_view,
    const cell_module_collection_types::const_view modules_view,
    details::ccl_kernel_static_smem_parcel& smem,
    vecmem::data::vector_view<details::index_t> f_view,
    vecmem::data::vector_view<details::index_t> gf_view,
    vecmem::data::vector_view<details::index_t> f_backup_view,
    vecmem::data::vector_view<details::index_t> gf_backup_view,
    vecmem::data::vector_view<unsigned char> adjc_backup_view,
    vecmem::data::vector_view<details::index_t> adjv_backup_view,
    vecmem::device_atomic_ref<uint32_t> backup_mutex, barrier_t& barrier,
    measurement_collection_types::view measurements_view,
    vecmem::data::vector_view<unsigned int> cell_links) {
    // Construct device containers around the views.
    const cell_collection_types::const_device cells_device(cells_view);
    const cell_module_collection_types::const_device modules_device(
        modules_view);
    measurement_collection_types::device measurements_device(measurements_view);
    vecmem::device_vector<details::index_t> f_primary(f_view);
    vecmem::device_vector<details::index_t> gf_primary(gf_view);
    vecmem::device_vector<details::index_t> f_backup(f_backup_view);
    vecmem::device_vector<details::index_t> gf_backup(gf_backup_view);
    vecmem::device_vector<unsigned char> adjc_backup(adjc_backup_view);
    vecmem::device_vector<details::index_t> adjv_backup(adjv_backup_view);

    mutex<uint32_t> mutex(backup_mutex);
    unique_lock lock(mutex, std::defer_lock);

    /*
     * First, we determine the exact range of cells that is to be examined
     * by this block of threads. We start from an initial range determined
     * by the block index multiplied by the target number of cells per
     * block. We then shift both the start and the end of the block forward
     * (to a later point in the array); start and end may be moved different
     * amounts.
     */
    if (thread_id.getLocalThreadIdX() == 0) {
        const std::size_t num_cells = cells_device.size();

        smem.outi = 0;

        std::size_t boundary_beg =
            std::max(1lu, std::min(num_cells, thread_id.getBlockIdX() *
                                                  cfg.target_partition_size()));
        std::size_t boundary_mid =
            std::min(num_cells, (thread_id.getBlockIdX() + 1) *
                                    cfg.target_partition_size());
        std::size_t boundary_end =
            std::min(num_cells, (thread_id.getBlockIdX() + 2) *
                                    cfg.target_partition_size());

        /*
         * Next, shift the starting point to a position further in the
         * array; the purpose of this is to ensure that we are not operating
         * on any cells that have been claimed by the previous block (if
         * any).
         */
        if (thread_id.getBlockIdX() == 0) {
            smem.partition_start = 0;
        } else {
            smem.partition_start = ccl_partition_find_helper(
                cells_device, boundary_beg, boundary_mid);
        }

        /*
         * Then, claim as many cells as we need past the naive end of the
         * current block to ensure that we do not end our partition on a
         * cell that is not a possible boundary!
         */
        smem.partition_end =
            ccl_partition_find_helper(cells_device, boundary_mid, boundary_end);

        assert(smem.partition_start <= smem.partition_end);
    }

    barrier.blockBarrier();

    // Vector of indices of the adjacent cells
    details::index_t _adjv[details::CELLS_PER_THREAD_STACK_LIMIT * 8];

    /*
     * The number of adjacent cells for each cell must start at zero, to
     * avoid uninitialized memory. adjv does not need to be zeroed, as
     * we will only access those values if adjc indicates that the value
     * is set.
     */
    unsigned char _adjc[details::CELLS_PER_THREAD_STACK_LIMIT];

    // It seems that sycl runs into undefined behaviour when calling
    // group synchronisation functions when some threads have already run
    // into a return. As such, we cannot use returns in this kernel.

    // Get partition for this thread group
    const details::index_t size = smem.partition_end - smem.partition_start;

    // If the size is zero, we can just retire the whole block.
    if (size == 0) {
        return;
    }

    details::index_t* adjv;
    unsigned char* adjc;
    bool use_scratch;

    /*
     * If our partition is too large, we need to handle this specific edge
     * case. The first thread of the block will attempt to enter a critical
     * section by obtaining a lock on a mutex in global memory. When this is
     * obtained, we can use some memory in global memory instead of the shared
     * memory. This can be done more efficiently, but this should be a very
     * rare edge case.
     */
    if (size > cfg.max_partition_size()) {
        if (thread_id.getLocalThreadIdX() == 0) {
            lock.lock();
        }

        barrier.blockBarrier();

        adjc = adjc_backup.data() +
               (thread_id.getLocalThreadIdX() * cfg.max_cells_per_thread *
                cfg.backup_size_multiplier);
        adjv = adjv_backup.data() +
               (thread_id.getLocalThreadIdX() * 8 * cfg.max_cells_per_thread *
                cfg.backup_size_multiplier);
        use_scratch = true;
    } else {
        adjc = _adjc;
        adjv = _adjv;
        use_scratch = false;
    }

    ccl_core(thread_id, smem, use_scratch ? f_backup : f_primary,
             use_scratch ? gf_backup : gf_primary, cell_links, adjv, adjc,
             cells_device, modules_device, measurements_device, barrier);

    barrier.blockBarrier();
}
}  // namespace traccc::device
