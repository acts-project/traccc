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
/// traccc::device::rearrange_tracks function
struct rearrange_tracks_payload {

    /**
     * @brief View object to the sorted track
     */
    vecmem::data::vector_view<const unsigned int> sorted_ids_view;

    /**
     * @brief View object to the inverted ids
     */
    vecmem::data::vector_view<const unsigned int> inverted_ids_view;

    /**
     * @brief View object to the vector of relative number of shared
     * measurements
     */
    vecmem::data::vector_view<const traccc::scalar> rel_shared_view;

    /**
     * @brief View object to the vector of relative number of pvalues
     */
    vecmem::data::vector_view<const traccc::scalar> pvals_view;

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
     * @brief View object to the updated track
     */
    vecmem::data::vector_view<const unsigned int> updated_tracks_view;

    /**
     * @brief View object to the whether track id is updated
     */
    vecmem::data::vector_view<const int> is_updated_view;

    /**
     * @brief View object to the prefix sum vector
     */
    vecmem::data::vector_view<const int> prefix_sums_view;

    /**
     * @brief View object to the temporary sorted track
     */
    vecmem::data::vector_view<unsigned int> temp_sorted_ids_view;
};

/// Function used for updating vectors
///
/// @param[in] globalIndex   The index of the current thread
/// @param[in] barrier            A block-wide barrier
/// @param[inout] payload      The function call payload
///
template <concepts::barrier barrier_t>
TRACCC_DEVICE inline void rearrange_tracks(
    global_index_t globalIndex, const barrier_t& barrier,
    const rearrange_tracks_payload& payload);

}  // namespace traccc::device

// Include the implementation.
#include "./impl/rearrange_tracks.ipp"
