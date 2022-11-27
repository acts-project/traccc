/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/track_candidate.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/fitting/kalman_filter/kalman_fitter.hpp"
#include "traccc/utils/algorithm.hpp"

namespace traccc {

/// Fitting algorithm for a set of tracks
template <typename fitter_t>
class fitting_algorithm
    : public algorithm<track_state_container_types::host(
          const typename track_candidate_container_types::host&)> {

    public:
    using detector_type = typename fitter_t::detector_type;
    using transform3_type = typename fitter_t::transform3_type;

    /// Constructor with a detector
    fitting_algorithm(const detector_type& det)
        : m_detector(std::make_unique<detector_type>(det)) {}

    /// Run the algorithm
    ///
    /// @param track_candidates the candidate measurements from track finding
    /// @return the container of the fitted track parameters
    track_state_container_types::host operator()(
        const typename track_candidate_container_types::host& track_candidates)
        const override {

        fitter_t fitter(*m_detector.get());

        track_state_container_types::host trk_states;

        // The number of tracks
        std::size_t n_tracks = track_candidates.size();

        // Iterate over tracks
        for (std::size_t i = 0; i < n_tracks; i++) {

            // Seed parameter
            const auto& seed_param = track_candidates[i].header;

            // Make a vector of track state
            auto& cands = track_candidates[i].items;
            vecmem::vector<track_state<transform3_type>> track_states;
            for (auto& cand : cands) {
                track_states.emplace_back(cand);
            }

            // Run fitter
            fitter.fit(seed_param, std::move(track_states));

            trk_states.push_back(std::move(fitter.get_fitter_info()),
                                 std::move(fitter.get_track_states()));
        }

        return trk_states;
    }

    private:
    std::unique_ptr<detector_type> m_detector;
};

}  // namespace traccc