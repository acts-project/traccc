/**
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cstdint>

#include "traccc/definitions/qualifiers.hpp"

namespace traccc {
/**
 * @brief Configuration type for massively parallel clustering algorithms.
 */
struct clustering_config {
    /**
     * @brief The desired number of threads per partition.
     *
     * This directly correlates to the block size on most algorithms, so don't
     * set this too low (which will reduce occupancy due to available thread
     * slots) or too high (which may not be supported on a device).
     */
    unsigned int threads_per_partition;

    /**
     * @brief The maximum number of cells per thread.
     *
     * This sets the maximum thread coarsening factor for the CCA algorithm.
     * Increasing this value increases shared memory usage and may decrease
     * occupancy. If this is too low, scratch space will need to be used which
     * may slow the algorithm down.
     */
    unsigned int max_cells_per_thread;

    /**
     * @brief The desired number of cells per thread.
     *
     * This sets the desired thread coarsening factor for the CCA algorithm.
     * Decreasing this may decrease occupancy. Increasing this increases the
     * probability that scratch space will need to be used.
     */
    unsigned int target_cells_per_thread;

    /**
     * @brief The upscaling factor for the scratch space.
     *
     * The scratch space will be large enough to support partitions this number
     * of times larger than the maximum partition size determined by
     * `threads_per_partition` and `max_cells_per_thread`
     */
    unsigned int backup_size_multiplier;

    /**
     * @brief The maximum number of cells per partition.
     */
    TRACCC_HOST_DEVICE constexpr std::size_t max_partition_size() const {
        return threads_per_partition * max_cells_per_thread;
    }

    /**
     * @brief The target number of cells per partition.
     */
    TRACCC_HOST_DEVICE constexpr std::size_t target_partition_size() const {
        return threads_per_partition * target_cells_per_thread;
    }

    /**
     * @brief The total size of the scratch space, in number of cells.
     */
    TRACCC_HOST_DEVICE constexpr std::size_t backup_size() const {
        return max_partition_size() * backup_size_multiplier;
    }
};
}  // namespace traccc
