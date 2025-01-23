/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/options/details/interface.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"

// System include(s).
#include <iosfwd>

namespace traccc::opts {

/// Command line options used to configure track seeding
class track_seeding : public interface {

    public:
    /// @name Options
    /// @{

    /// Configuration for the seed-finding
    traccc::seedfinder_config seedfinder;
    /// Configuration for the seed filtering
    traccc::seedfilter_config seedfilter;

    /// @}

    /// Constructor
    track_seeding();

    std::unique_ptr<configuration_printable> as_printable() const override;
};  // struct track_seeding

}  // namespace traccc::opts
