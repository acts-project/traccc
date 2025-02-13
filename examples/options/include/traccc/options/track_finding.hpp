/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/finding/finding_config.hpp"
#include "traccc/options/details/config_provider.hpp"
#include "traccc/options/details/interface.hpp"
#include "traccc/options/details/value_array.hpp"

// System include(s).
#include <limits>

namespace traccc::opts {

/// Configuration for track finding
class track_finding : public interface, public config_provider<finding_config> {

    public:
    /// Constructor
    track_finding();

    /// Configuration conversion operators
    operator finding_config() const override;

    std::unique_ptr<configuration_printable> as_printable() const override;

    private:
    /// The internal configuration
    finding_config m_config;
    /// Additional variables which we cannot simply store in the config
    opts::value_array<unsigned int, 2> m_track_candidates_range{0, 0};
    int m_pdg_number = 0;
};  // class track_finding

}  // namespace traccc::opts
