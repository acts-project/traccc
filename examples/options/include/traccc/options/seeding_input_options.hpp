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

namespace traccc {

/// Command line options used in the seeding input tests
struct seeding_input_options {

    /// Constructor on top of a common @c program_options object
    ///
    /// @param desc The program options to add to
    ///
    seeding_input_options(boost::program_options::options_description& desc);

    /// Read/process the command line options
    ///
    /// @param vm The command line options to interpret/read
    ///
    void read(const boost::program_options::variables_map& vm);

};  // struct seeding_input_options

/// Printout helper for @c traccc::seeding_input_options
std::ostream& operator<<(std::ostream& out, const seeding_input_options& opt);

}  // namespace traccc
