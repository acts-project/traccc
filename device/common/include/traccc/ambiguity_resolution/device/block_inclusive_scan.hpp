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
#include "traccc/edm/device/update_result.hpp"

// Project include(s)
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/containers/device_vector.hpp>
namespace traccc::device {

/// (Event Data) Payload for the @c
/// traccc::device::block_inclusive_scan function
struct block_inclusive_scan_payload {

    /**
     * @brief Update result
     */
    update_result* update_res;

    /**
     * @brief View object to the whether track id is updated
     */
    vecmem::data::vector_view<const int> is_updated_view;

    /**
     * @brief View object to the block offset vector
     */
    vecmem::data::vector_view<int> block_offsets_view;

    /**
     * @brief View object to the prefix_sum vector
     */
    vecmem::data::vector_view<int> prefix_sums_view;
};

/// Function used for updating vectors
///
/// @param[in] globalIndex   The index of the current thread
/// @param[in] barrier            A block-wide barrier
/// @param[inout] payload      The function call payload
///
template <concepts::barrier barrier_t>
TRACCC_DEVICE inline void block_inclusive_scan(
    global_index_t globalIndex, const barrier_t& barrier,
    const block_inclusive_scan_payload& payload);

}  // namespace traccc::device

// Include the implementation.
#include "./impl/block_inclusive_scan.ipp"
