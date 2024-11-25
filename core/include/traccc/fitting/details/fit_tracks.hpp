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

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

namespace traccc::host::details {

/// Templated implementation of the track fitting algorithm.
///
/// Concrete track fitting algorithms can use this function with the appropriate
/// specializations, to fit tracks on top of a specific detector type, magnetic
/// field type, and track fitting configuration.
///
/// @note The memory resource received by this function is not used thoroughly
///       for the setup of the output container. Inner vectors in the output's
///       jagged vector are created using the default memory resource.
///
/// @tparam fitter_t The fitter type used for the track fitting
///
/// @param[in] fitter           The fitter object to use on the track candidates
/// @param[in] track_candidates All track candidates to fit
/// @param[in] mr               Memory resource to use for the output container
///
/// @return A container of the fitted track states
///
template <typename fitter_t>
track_state_container_types::host fit_tracks(
    fitter_t& fitter,
    const track_candidate_container_types::const_view& track_candidates_view,
    vecmem::memory_resource& mr) {

    // Create the output container.
    track_state_container_types::host result{&mr};

    // Iterate over the tracks,
    const track_candidate_container_types::const_device track_candidates{
        track_candidates_view};
    for (track_candidate_container_types::const_device::size_type i = 0;
         i < track_candidates.size(); ++i) {

        // Make a vector of track states for this track.
        vecmem::vector<track_state<typename fitter_t::algebra_type> >
            input_states{&mr};
        input_states.reserve(track_candidates.get_items()[i].size());
        for (auto& measurement : track_candidates.get_items()[i]) {
            input_states.emplace_back(measurement);
        }

        // Make a fitter state
        typename fitter_t::state fitter_state(std::move(input_states));

        // Run the fitter.
        fitter.fit(track_candidates.get_headers()[i], fitter_state);

        // Save the results into the output container.
        result.push_back(
            std::move(fitter_state.m_fit_res),
            std::move(fitter_state.m_fit_actor_state.m_track_states));
    }

    // Return the fitted track states.
    return result;
}

}  // namespace traccc::host::details
