/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/track_candidate.hpp"
#include "traccc/edm/track_state.hpp"

namespace traccc::host::details {

/// Templated implementation of the track fitting algorithm.
///
/// Concrete track fitting algorithms can use this function with the appropriate
/// specializations, to fit tracks on top of a specific detector type, magnetic
/// field type, and track fitting configuration.
///
/// @tparam fitter_t The fitter type used for the track fitting
///
/// @param det               The detector object
/// @param field             The magnetic field object
/// @param track_candidates  All track candidates to fit
/// @param config            The track fitting configuration
///
/// @return A container of the fitted track states
///
template <typename fitter_t>
track_state_container_types::host fit_tracks(
    const typename fitter_t::detector_type& det,
    const typename fitter_t::bfield_type& field,
    const track_candidate_container_types::const_view& track_candidates_view,
    const typename fitter_t::config_type& config) {

    // Create the fitter object.
    fitter_t fitter(det, field, config);

    // Output container.
    track_state_container_types::host output_states;

    // Iterate over the tracks,
    const track_candidate_container_types::const_device track_candidates{
        track_candidates_view};
    for (track_candidate_container_types::const_device::size_type i = 0;
         i < track_candidates.size(); ++i) {

        // Make a vector of track states for this track.
        vecmem::vector<track_state<typename fitter_t::algebra_type> >
            input_states;
        input_states.reserve(track_candidates.get_items()[i].size());
        for (auto& measurement : track_candidates.get_items()[i]) {
            input_states.emplace_back(measurement);
        }

        // Make a fitter state
        typename fitter_t::state fitter_state(std::move(input_states));

        // Run the fitter.
        fitter.fit(track_candidates.get_headers()[i], fitter_state);

        // Save the results into the output container.
        output_states.push_back(
            std::move(fitter_state.m_fit_res),
            std::move(fitter_state.m_fit_actor_state.m_track_states));
    }

    // Return the fitted track states.
    return output_states;
}

}  // namespace traccc::host::details
