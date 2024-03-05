/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Boost include(s).
#include <boost/program_options.hpp>

// System include(s).
#include <iosfwd>
#include <string>

namespace traccc {

/// Configuration for a full tracking chain
struct full_tracking_input_options {

    /// The digitization configuration file
    std::string digitization_config_file =
        "tml_detector/default-geometric-config-generic.json";

    /// Constructor on top of a common @c program_options object
    ///
    /// @param desc The program options to add to
    ///
    full_tracking_input_options(
        boost::program_options::options_description& desc);

    /// Read/process the command line options
    ///
    /// @param vm The command line options to interpret/read
    ///
    void read(const boost::program_options::variables_map& vm);

};  // struct full_tracking_input_config

/// Printout helper for @c traccc::full_tracking_input_options
std::ostream& operator<<(std::ostream& out,
                         const full_tracking_input_options& opt);

}  // namespace traccc
