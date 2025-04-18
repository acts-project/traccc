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
#include "traccc/definitions/qualifiers.hpp"

// VecMem include(s).
#include <vecmem/containers/data/jagged_vector_view.hpp>
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>

namespace traccc::device {

/// (Event Data) Payload for the @c traccc::device::fill_tracks_per_measurement
/// function
struct fill_tracks_per_measurement_payload {

    /**
     * @brief View object to the accepted ids
     */
    vecmem::data::vector_view<const unsigned int> accepted_ids_view;

    /**
     * @brief View object to the vector of measured ids per track
     */
    vecmem::data::jagged_vector_view<const std::size_t> meas_ids_view;

    /**
     * @brief View object to the unique measurement ids
     */
    vecmem::data::vector_view<const std::size_t> unique_meas_view;

    /**
     * @brief View object to the tracks per measurement
     */
    vecmem::data::jagged_vector_view<std::size_t> tracks_per_measurement_view;

    /**
     * @brief View object to the track status per measurement
     */
    vecmem::data::jagged_vector_view<int> track_status_per_measurement_view;

    /**
     * @brief View object to the number of accepted tracks per measurement
     */
    vecmem::data::vector_view<unsigned int>
        n_accepted_tracks_per_measurement_view;
};

/// Function used for fill_vectors
///
/// @param[in] globalIndex   The index of the current thread
/// @param[inout] payload      The function call payload
///
TRACCC_HOST_DEVICE inline void fill_tracks_per_measurement(
    global_index_t globalIndex,
    const fill_tracks_per_measurement_payload& payload);

}  // namespace traccc::device

// Include the implementation.
#include "./impl/fill_tracks_per_measurement.ipp"
