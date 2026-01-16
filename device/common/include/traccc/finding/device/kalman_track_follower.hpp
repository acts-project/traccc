/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/device/global_index.hpp"

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/track_container.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/finding/finding_config.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// (Event Data) Payload for the @c traccc::device::kalman_track_follower
/// function
template <typename propagator_t>
struct kalman_track_follower_payload {
    using algebra_t = typename propagator_t::detector_type::algebra_type;

    /**
     * @brief View object to the tracking detector description
     */
    typename propagator_t::detector_type::const_view_type det_data;

    /**
     * @brief View object to the magnetic field
     */
    typename propagator_t::stepper_type::magnetic_field_type field_data;

    /**
     * @brief View object to the vector of track parameters
     */
    bound_track_parameters_collection_types::view seeds_view;

    /**
     * @brief View object to the vector of bound track parameters
     *
     * @warning Measurements on the same surface must be adjacent
     */
    edm::measurement_collection<algebra_t>::const_view measurements_view;

    /**
     * @brief View object to the vector of measurement index ranges per surface
     */
    vecmem::data::vector_view<unsigned int> measurement_ranges_view;

    /**
     * @brief View object to the vector of track candidates
     */
    edm::track_container<algebra_t>::view tracks_view;
};

/// Function that run Kalman filter based track following
///
/// @param[in] globalIndex        The index of the current thread
/// @param[in] cfg                Track finding config object
/// @param[inout] payload         The function call payload
///
template <typename propagator_t>
TRACCC_HOST_DEVICE inline void kalman_track_follower(
    global_index_t globalIndex, const finding_config& cfg,
    const kalman_track_follower_payload<propagator_t>& payload);

}  // namespace traccc::device

// Include the implementation.
#include "./impl/kalman_track_follower.ipp"
