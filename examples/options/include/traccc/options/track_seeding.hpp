/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/seeding/detail/seeding_config.hpp"

// Boost include(s).
#include <boost/program_options.hpp>

// System include(s).
#include <iosfwd>

namespace traccc::opts {

/// Command line options used to configure track seeding
class track_seeding {

    public:
    /// @name Options
    /// @{

    /// Configuration for the seed-finding
    traccc::seedfinder_config seedfinder;
    /// Configuration for the seed filtering
    traccc::seedfilter_config seedfilter;

    /// @}

    /// Constructor on top of a common @c program_options object
    ///
    /// @param desc The program options to add to
    ///
    track_seeding(boost::program_options::options_description& desc);

    /// Read/process the command line options
    ///
    /// @param vm The command line options to interpret/read
    ///
    void read(const boost::program_options::variables_map& vm);

    private:
    /// Description of this program option group
    boost::program_options::options_description m_desc;

};  // struct track_seeding

/// Printout helper for @c traccc::opts::track_seeding
std::ostream& operator<<(std::ostream& out, const track_seeding& opt);

}  // namespace traccc::opts
