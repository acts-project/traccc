/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Boost include(s).
#include <boost/program_options.hpp>

// System include(s).
#include <iosfwd>

namespace traccc::opts {

/// Command line options used to configure performance measurements
class performance {

    public:
    /// @name Options
    /// @{

    /// Whether to run performance checks
    bool run = false;

    /// @}

    /// Constructor on top of a common @c program_options object
    ///
    /// @param desc The program options to add to
    ///
    performance(boost::program_options::options_description& desc);

    /// Read/process the command line options
    ///
    /// @param vm The command line options to interpret/read
    ///
    void read(const boost::program_options::variables_map& vm);

    private:
    /// Description of this program option group
    boost::program_options::options_description m_desc;

};  // struct performance

/// Printout helper for @c traccc::opts::performance
std::ostream& operator<<(std::ostream& out, const performance& opt);

}  // namespace traccc::opts
