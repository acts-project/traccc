/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/ambiguity_resolution/resolution_config.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/messaging.hpp"

namespace traccc {

/// Evicts tracks that seem to be duplicates or fakes. This algorithm takes a
/// greedy approach in the sense that it will remove the track which looks "most
/// duplicate/fake"
class greedy_ambiguity_resolution_algorithm
    : public algorithm<track_candidate_container_types::host(
          const typename track_candidate_container_types::host&)>,
      public messaging {

    public:
    using config_type = resolution_config;

    /// Constructor for the greedy ambiguity resolution algorithm
    ///
    /// @param cfg  Configuration object
    greedy_ambiguity_resolution_algorithm(
        const config_type& cfg,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone())
        : messaging(std::move(logger)), m_config{cfg} {}

    /// Run the algorithm
    ///
    /// @param track_candidates the container of found patterns
    /// @return the container without ambiguous tracks
    track_candidate_container_types::host operator()(
        const typename track_candidate_container_types::host& track_candidates)
        const override;

    /// Algorithm configuration
    config_type m_config;
};

}  // namespace traccc
