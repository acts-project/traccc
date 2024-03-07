/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Detray include(s).
#include "detray/propagator/propagation_config.hpp"

// Boost include(s).
#include <boost/program_options.hpp>

// System include(s).
#include <iosfwd>

namespace traccc {

/// Command line options used in the propagation tests
struct propagation_options {

    /// Propagation configuration object
    detray::propagation::config<float> propagation;

    /// Constructor on top of a common @c program_options object
    ///
    /// @param desc The program options to add to
    ///
    propagation_options(boost::program_options::options_description& desc);

    /// Read/process the command line options
    ///
    /// @param vm The command line options to interpret/read
    ///
    void read(const boost::program_options::variables_map& vm);

};  // struct propagation_options

/// Printout helper for @c traccc::propagation_options
std::ostream& operator<<(std::ostream& out, const propagation_options& opt);

}  // namespace traccc
