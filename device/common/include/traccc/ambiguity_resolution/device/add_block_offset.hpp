/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/device/global_index.hpp"

// Project include(s)
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/containers/device_vector.hpp>
namespace traccc::device {

/// (Event Data) Payload for the @c
/// traccc::device::add_block_offset function
struct add_block_offset_payload {

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
     * @brief View object to the prefix_sum vector
     */
    vecmem::data::vector_view<int> prefix_sums_view;
};

/// Function used for updating vectors
///
/// @param[in] globalIndex   The index of the current thread
/// @param[inout] payload      The function call payload
///
TRACCC_DEVICE inline void add_block_offset(
    global_index_t globalIndex, const add_block_offset_payload& payload);

}  // namespace traccc::device

// Include the implementation.
#include "./impl/add_block_offset.ipp"
