/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/device/global_index.hpp"

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/track_candidate.hpp"
#include "traccc/edm/track_state.hpp"

namespace traccc::device {

// Payload for the fitting algorithm
template <typename fitter_t>
struct fit_payload {
    /**
     * @brief View object to the detector description
     */
    typename fitter_t::detector_type::view_type det_data;

    /**
     * @brief View object to the magnetic field description
     */
    typename fitter_t::bfield_type field_data;

    /**
     * @brief View object to the input track candidates
     */
    track_candidate_container_types::const_view track_candidates_view;

    /**
     * @brief View object to the input track parameters
     */
    vecmem::data::vector_view<const unsigned int> param_ids_view;

    /**
     * @brief View object to the output track states
     */
    track_state_container_types::view track_states_view;

    /**
     * @brief View object to the output barcode sequence
     */
    vecmem::data::jagged_vector_view<detray::geometry::barcode> barcodes_view;
};

/// Function used for fitting a track for a given track candidates
///
/// @param[in] globalIndex   The index of the current thread
/// @param[in] det_data      Detector view object
/// @param[in] track_candidates_view The input track candidates
/// @param[out] track_states_view The output of fitted track states
/// @param[out] barcodes_view The barcode sequence for backward filter
///
template <typename fitter_t>
TRACCC_HOST_DEVICE inline void fit(global_index_t globalIndex,
                                   const typename fitter_t::config_type cfg,
                                   const fit_payload<fitter_t>&);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/fitting/device/impl/fit.ipp"
