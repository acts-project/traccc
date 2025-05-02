/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/ambiguity_resolution/ambiguity_resolution_config.hpp"
#include "traccc/edm/track_candidate.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/messaging.hpp"

namespace traccc::host {

/// Evicts tracks that seem to be duplicates or fakes. This algorithm takes a
/// greedy approach in the sense that it will remove the track which looks "most
/// duplicate/fake"
class greedy_ambiguity_resolution_algorithm
    : public algorithm<track_candidate_container_types::host(
          const typename track_candidate_container_types::const_view&)>,
      public messaging {

    public:
    using config_type = ambiguity_resolution_config;

    /// Constructor for the greedy ambiguity resolution algorithm
    ///
    /// @param cfg  Configuration object
    greedy_ambiguity_resolution_algorithm(
        const config_type& cfg, traccc::memory_resource& mr,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone())
        : messaging(std::move(logger)), m_config{cfg}, m_mr{mr} {}

    /// Run the algorithm
    ///
    /// @param track_candidates the container of found patterns
    /// @return the container without ambiguous tracks
    track_candidate_container_types::host operator()(
        const track_candidate_container_types::const_view&
            track_candidates_view) const override;

    /// Get configuration
    config_type& get_config() { return m_config; }

    private:
    /// Algorithm configuration
    config_type m_config;
    /// The memory resource to use
    traccc::memory_resource m_mr;
};

}  // namespace traccc::host
