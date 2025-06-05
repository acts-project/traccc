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
#include "traccc/edm/track_state.hpp"

// VecMem include(s).
#include <vecmem/containers/data/jagged_vector_view.hpp>
#include <vecmem/containers/data/vector_view.hpp>

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
     * @brief View object to the input track parameters
     */
    vecmem::data::vector_view<const unsigned int> param_ids_view;

    /**
     * @brief View object to the vector of parameter liveness
     */
    vecmem::data::vector_view<unsigned int> param_liveness_view;

    /**
     * @brief View object to the output track states
     */
    track_state_container_types::view track_states_view;

    /**
     * @brief View object to the output barcode sequence
     */
    vecmem::data::jagged_vector_view<detray::geometry::barcode> barcodes_view;
};

}  // namespace traccc::device
