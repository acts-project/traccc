/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Boost include(s).
#include <boost/program_options.hpp>

// System include(s).
#include <iosfwd>

namespace traccc::opts {

/// Configuration for track ambiguity resulution
class track_resolution {

    public:
    /// @name Options
    /// @{

    /// Whether to perform ambiguity resolution
    bool run = true;

    /// @}

    /// Constructor on top of a common @c program_options object
    ///
    /// @param desc The program options to add to
    ///
    track_resolution(boost::program_options::options_description& desc);

    /// Read/process the command line options
    ///
    /// @param vm The command line options to interpret/read
    ///
    void read(const boost::program_options::variables_map& vm);

    private:
    /// Description of this program option group
    boost::program_options::options_description m_desc;

};  // class track_resolution

/// Printout helper for @c traccc::opts::track_resolution
std::ostream& operator<<(std::ostream& out, const track_resolution& opt);

}  // namespace traccc::opts
