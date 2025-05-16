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
/// traccc::device::find_max_shared function
struct find_max_shared_payload {

    /**
     * @brief View object to the vector of sorted track ids
     */
    vecmem::data::vector_view<const unsigned int> sorted_ids_view;

    /**
     * @brief The number of accepted tracks
     */
    unsigned int* n_accepted;

    /**
     * @brief View object to the vector of number of shared measurements
     */
    vecmem::data::vector_view<const unsigned int> n_shared_view;

    /**
     * @brief Update result
     */
    update_result* update_res;
};

/// Function used for updating vectors
///
/// @param[in] globalIndex   The index of the current thread
/// @param[in] barrier            A block-wide barrier
/// @param[inout] payload      The function call payload
///
template <concepts::barrier barrier_t>
TRACCC_DEVICE inline void find_max_shared(global_index_t globalIndex,
                                         const barrier_t& barrier,
                                         const find_max_shared_payload& payload);

}  // namespace traccc::device

// Include the implementation.
#include "./impl/find_max_shared.ipp"
