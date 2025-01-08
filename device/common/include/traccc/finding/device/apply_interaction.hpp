/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/device/global_index.hpp"

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/finding/finding_config.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// (Event Data) Payload for the @c traccc::device::apply_interaction function
template <typename detector_t>
struct apply_interaction_payload {
    /**
     * @brief View object describing the tracking detector
     */
    typename detector_t::view_type det_data;

    /**
     * @brief Total number of input parameters (including non-live ones)
     */
    unsigned int n_params;

    /**
     * @brief View object to the vector of bound track parameters
     */
    bound_track_parameters_collection_types::view params_view;

    /**
     * @brief View object to the vector of boolean-like integers describing
     * whether each parameter is live. Has the same size as \ref params_view
     */
    vecmem::data::vector_view<const unsigned int> params_liveness_view;
};

/// Function applying the material interaction to tracks spawned described by
/// bound track parameters
///
/// @param[in] globalIndex     The index of the current thread
/// @param[in] cfg             Track finding config object
/// @param[inout] payload      The function call payload
///
template <typename detector_t>
TRACCC_DEVICE inline void apply_interaction(
    global_index_t globalIndex, const finding_config& cfg,
    const apply_interaction_payload<detector_t>& payload);

}  // namespace traccc::device

// Include the implementation.
#include "./impl/apply_interaction.ipp"
