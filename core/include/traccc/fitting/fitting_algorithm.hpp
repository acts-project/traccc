/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/track_candidate.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/fitting/fitting_config.hpp"
#include "traccc/fitting/kalman_filter/kalman_fitter.hpp"
#include "traccc/utils/algorithm.hpp"

namespace traccc {

/// Fitting algorithm for a set of tracks
template <typename fitter_t>
class fitting_algorithm
    : public algorithm<track_state_container_types::host(
          const typename fitter_t::detector_type&,
          const typename track_candidate_container_types::host&)> {

    public:
    using transform3_type = typename fitter_t::transform3_type;
    /// Configuration type
    using config_type = typename fitter_t::config_type;

    /// Constructor for the fitting algorithm
    ///
    /// @param cfg  Configuration object
    fitting_algorithm(const config_type& cfg) : m_cfg(cfg) {}

    /// Run the algorithm
    ///
    /// @param track_candidates the candidate measurements from track finding
    /// @return the container of the fitted track parameters
    track_state_container_types::host operator()(
        const typename fitter_t::detector_type& det,
        const typename track_candidate_container_types::host& track_candidates)
        const override {

        fitter_t fitter(det, m_cfg);

        track_state_container_types::host output_states;

        // The number of tracks
        std::size_t n_tracks = track_candidates.size();

        // Iterate over tracks
        for (std::size_t i = 0; i < n_tracks; i++) {

            // Seed parameter
            const auto& seed_param = track_candidates[i].header;

            // Make a vector of track state
            auto& cands = track_candidates[i].items;
            vecmem::vector<track_state<transform3_type>> input_states;
            input_states.reserve(cands.size());
            for (auto& cand : cands) {
                input_states.emplace_back(cand);
            }

            // Make a fitter state
            typename fitter_t::state fitter_state(std::move(input_states));

            // Run fitter
            fitter.fit(seed_param, fitter_state);

            output_states.push_back(
                std::move(fitter_state.m_fit_info),
                std::move(fitter_state.m_fit_actor_state.m_track_states));
        }

        return output_states;
    }

    /// Config object
    config_type m_cfg;
};

}  // namespace traccc