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
#include <vecmem/containers/data/jagged_vector_view.hpp>
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>

namespace traccc::device {

/// (Event Data) Payload for the @c
/// traccc::device::check_sortedness function
struct check_sortedness_payload {

    /**
     * @brief View object to the number of accepted tracks per measurement
     */
    vecmem::data::vector_view<const unsigned int>
        sorted_ids_view;

    /**
     * @brief View object to the vector of relative number of sharedmeasurements
     */
    vecmem::data::vector_view<const traccc::scalar> rel_shared_view;

    /**
     * @brief View object to the vector of pvalues
     */
    vecmem::data::vector_view<const traccc::scalar> pvals_view;

    /**
     * @brief The number of updated tracks
     */
    unsigned int n_updated_tracks;

    /**
     * @brief View object to the vector of affected track IDs
     */
    vecmem::data::vector_view<const unsigned int> updated_tracks_view;

    /**
     * @brief Whether to do sort or not after the kernel
     */
    bool* do_sort;
};

/// Function used for updating vectors
///
/// @param[in] globalIndex   The index of the current thread
/// @param[inout] payload      The function call payload
///
TRACCC_HOST_DEVICE inline void check_sortedness(
    global_index_t globalIndex, const check_sortedness_payload& payload);

}  // namespace traccc::device

// Include the implementation.
#include "./impl/check_sortedness.ipp"
