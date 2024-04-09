/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/options/details/interface.hpp"

// Boost include(s).
#include <boost/program_options.hpp>

namespace traccc::opts {

/// Configuration for track ambiguity resulution
class track_resolution : public interface {

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

    private:
    /// Print the specific options of this class
    std::ostream& print_impl(std::ostream& out) const override;

};  // class track_resolution

}  // namespace traccc::opts
