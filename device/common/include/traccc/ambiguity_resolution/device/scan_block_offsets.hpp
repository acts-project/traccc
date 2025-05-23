/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/device/concepts/barrier.hpp"
#include "traccc/device/global_index.hpp"

// Project include(s)
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/containers/device_vector.hpp>
namespace traccc::device {

/// (Event Data) Payload for the @c
/// traccc::device::scan_block_offsets function
struct scan_block_offsets_payload {

    /**
     * @brief Whether to terminate the calculation
     */
    int* terminate;

    /**
     * @brief The number of accepted tracks
     */
    unsigned int* n_accepted;

    /**
     * @brief The number of updated tracks
     */
    unsigned int* n_updated_tracks;

    /**
     * @brief View object to the block_offset vector
     */
    vecmem::data::vector_view<const int> block_offsets_view;

    /**
     * @brief View object to the scanned block_offset vector
     */
    vecmem::data::vector_view<int> scanned_block_offsets_view;
};

/// Function used for updating vectors
///
/// @param[in] globalIndex   The index of the current thread
/// @param[in] barrier            A block-wide barrier
/// @param[inout] payload      The function call payload
///
template <concepts::barrier barrier_t>
TRACCC_DEVICE inline void scan_block_offsets(
    global_index_t globalIndex, const unsigned int blockSize,
    const unsigned int threadIndex, const barrier_t& barrier,
    const scan_block_offsets_payload& payload, int* shared_temp);

}  // namespace traccc::device

// Include the implementation.
#include "./impl/scan_block_offsets.ipp"
