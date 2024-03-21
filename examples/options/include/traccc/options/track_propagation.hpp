/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/options/details/value_array.hpp"

// Detray include(s).
#include <detray/propagator/propagation_config.hpp>

// Boost include(s).
#include <boost/program_options.hpp>

// System include(s).
#include <iosfwd>

namespace traccc::opts {

/// Command line options used in the propagation tests
class track_propagation {

    public:
    /// @name Options
    /// @{

    /// Propagation configuration object
    detray::propagation::config<float> config;

    /// @}

    /// Constructor on top of a common @c program_options object
    ///
    /// @param desc The program options to add to
    ///
    track_propagation(boost::program_options::options_description& desc);

    /// Read/process the command line options
    ///
    /// @param vm The command line options to interpret/read
    ///
    void read(const boost::program_options::variables_map& vm);

    private:
    /// Description of this program option group
    boost::program_options::options_description m_desc;
    /// Search window
    value_array<unsigned int, 2> m_search_window = {0u, 0u};

};  // class track_propagation

/// Printout helper for @c traccc::opts::track_propagation
std::ostream& operator<<(std::ostream& out, const track_propagation& opt);

}  // namespace traccc::opts
