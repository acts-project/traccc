/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/finding/finding_config.hpp"
#include "traccc/options/details/config_provider.hpp"
#include "traccc/options/details/interface.hpp"
#include "traccc/options/details/value_array.hpp"

// detray include(s).
#include "detray/definitions/units.hpp"

// System include(s).
#include <limits>

namespace traccc::opts {

/// Configuration for track finding
class track_finding : public interface, public config_provider<finding_config> {

    public:
    /// Constructor
    track_finding();

    /// Configuration conversion operators
    operator finding_config() const override;

    private:
    /// @name Options
    /// @{
    /// Max number of branches per seed
    unsigned int max_num_branches_per_seed = 10;
    /// Max number of branches per surface
    unsigned int max_num_branches_per_surface = 10;
    /// Number of track candidates per seed
    opts::value_array<unsigned int, 2> track_candidates_range{3, 100};
    /// Minimum step length that track should make to reach the next surface. It
    /// should be set higher than the overstep tolerance not to make it stay on
    /// the same surface
    float min_step_length_for_next_surface = 0.5f * detray::unit<float>::mm;
    /// Maximum step counts that track can make to reach the next surface
    unsigned int max_step_counts_for_next_surface = 100;
    /// Maximum chi2 for a measurement to be included in the track
    float chi2_max = 10.f;
    /// Maximum number of branches which each initial seed can have at a step
    unsigned int nmax_per_seed = 10;
    /// Maximum allowed number of skipped steps per candidate
    unsigned int max_num_skipping_per_cand = 3;
    /// PDG number for particle hypothesis (Default: muon)
    int pdg_number = 13;
    /// @}

    /// Print the specific options of this class
    std::ostream& print_impl(std::ostream& out) const override;

};  // class track_finding

}  // namespace traccc::opts
