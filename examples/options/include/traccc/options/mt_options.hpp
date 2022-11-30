/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Boost include(s).
#include <boost/program_options.hpp>

// System include(s).
#include <cstddef>
#include <iosfwd>

namespace traccc {

/// Options for multi-threaded code execution
struct mt_options {

    /// The number of threads to use for the data processing
    std::size_t threads = 1;

    /// Constructor on top of a common @c program_options object
    ///
    /// @param desc The program options to add to
    ///
    mt_options(boost::program_options::options_description& desc);

    /// Read the command line options
    ///
    /// @param vm The command line options to interpret/read
    ///
    void read(const boost::program_options::variables_map& vm);

};  // struct mt_options

/// Printout helper for @c traccc::mt_options
std::ostream& operator<<(std::ostream& out, const mt_options& opt);

}  // namespace traccc
