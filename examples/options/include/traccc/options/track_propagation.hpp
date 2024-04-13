/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/options/details/interface.hpp"
#include "traccc/options/details/value_array.hpp"

// Detray include(s).
#include <detray/propagator/propagation_config.hpp>

namespace traccc::opts {

/// Command line options used in the propagation tests
class track_propagation : public interface {

    public:
    /// @name Options
    /// @{

    /// Propagation configuration object
    detray::propagation::config<float> config;

    /// @}

    /// Constructor
    track_propagation();

    /// Read/process the command line options
    ///
    /// @param vm The command line options to interpret/read
    ///
    void read(const boost::program_options::variables_map& vm) override;

    private:
    /// Print the specific options of this class
    std::ostream& print_impl(std::ostream& out) const override;

    /// Search window (helper variable)
    value_array<unsigned int, 2> m_search_window = {3u, 3u};

};  // class track_propagation

}  // namespace traccc::opts
