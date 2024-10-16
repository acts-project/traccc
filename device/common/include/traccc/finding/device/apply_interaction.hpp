/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "detray/navigation/navigator.hpp"
#include "detray/propagator/actors/pointwise_material_interactor.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/finding/finding_config.hpp"
#include "traccc/utils/particle.hpp"

namespace traccc::device {
template <typename detector_t>
struct apply_interaction_payload {
    typename detector_t::view_type det_data;
    const int n_params;
    bound_track_parameters_collection_types::view params_view;
    vecmem::data::vector_view<const unsigned int> params_liveness_view;
};

/// Function applying the Pre material interaction to tracks spawned by bound
/// track parameters
///
/// @param[in] globalIndex     The index of the current thread
/// @param[in] cfg             Track finding config object
/// @param[inout] payload      The function call payload
template <typename detector_t>
TRACCC_DEVICE inline void apply_interaction(
    std::size_t globalIndex, const finding_config& cfg,
    const apply_interaction_payload<detector_t>& payload);
}  // namespace traccc::device

#include "./impl/apply_interaction.ipp"
