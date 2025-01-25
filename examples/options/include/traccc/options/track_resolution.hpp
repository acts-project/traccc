/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/options/details/interface.hpp"

namespace traccc::opts {

/// Configuration for track ambiguity resulution
class track_resolution : public interface {

    public:
    /// @name Options
    /// @{

    /// Whether to perform ambiguity resolution
    bool run = true;

    /// @}

    /// Constructor
    track_resolution();

    std::unique_ptr<configuration_printable> as_printable() const override;
};  // class track_resolution

}  // namespace traccc::opts
