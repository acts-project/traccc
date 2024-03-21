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
#include <cstddef>
#include <iosfwd>

namespace traccc::opts {

/// Option(s) for multi-threaded code execution
class threading {

    public:
    /// @name Options
    /// @{

    /// The number of threads to use for the data processing
    std::size_t threads = 1;

    /// @}

    /// Constructor on top of a common @c program_options object
    ///
    /// @param desc The program options to add to
    ///
    threading(boost::program_options::options_description& desc);

    /// Read/process the command line options
    ///
    /// @param vm The command line options to interpret/read
    ///
    void read(const boost::program_options::variables_map& vm);

    private:
    /// Description of this program option group
    boost::program_options::options_description m_desc;

};  // struct threading

/// Printout helper for @c traccc::opts::threading
std::ostream& operator<<(std::ostream& out, const threading& opt);

}  // namespace traccc::opts
