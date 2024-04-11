/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/options/details/interface.hpp"
#include "traccc/options/details/value_array.hpp"

// System include(s).
#include <limits>

namespace traccc::opts {

/// Configuration for track finding
class track_finding : public interface {

    public:
    /// @name Options
    /// @{

    /// Number of track candidates per seed
    opts::value_array<unsigned int, 2> track_candidates_range{3, 100};
    /// Maximum chi2 for a measurement to be included in the track
    float chi2_max = 30.f;
    /// Maximum number of branches which each initial seed can have at a step
    unsigned int nmax_per_seed = std::numeric_limits<unsigned int>::max();

    /// @}

    /// Constructor
    track_finding();

    private:
    /// Print the specific options of this class
    std::ostream& print_impl(std::ostream& out) const override;

};  // class track_finding

}  // namespace traccc::opts
