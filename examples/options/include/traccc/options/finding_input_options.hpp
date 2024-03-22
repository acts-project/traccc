/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/options/options.hpp"

// Boost include(s).
#include <boost/program_options.hpp>

// System include(s).
#include <iosfwd>
#include <limits>

namespace traccc {

/// Configuration for track finding
struct finding_input_options {

    /// Number of track candidates per seed
    Reals<unsigned int, 2> track_candidates_range{3, 100};
    /// Maximum chi2 for a measurement to be included in the track
    float chi2_max = 30.f;
    /// Maximum number of branches which each initial seed can have at a step
    unsigned int nmax_per_seed = std::numeric_limits<unsigned int>::max();

    /// Constructor on top of a common @c program_options object
    ///
    /// @param desc The program options to add to
    ///
    finding_input_options(boost::program_options::options_description& desc);

    /// Read/process the command line options
    ///
    /// @param vm The command line options to interpret/read
    ///
    void read(const boost::program_options::variables_map& vm);

};  // struct finding_input_options

/// Printout helper for @c traccc::finding_input_options
std::ostream& operator<<(std::ostream& out, const finding_input_options& opt);

}  // namespace traccc
