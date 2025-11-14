/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/edm/MultiTrajectory.hpp"
#include "traccc/edm/TrackContainerBackend.hpp"

// Acts include(s).
#include "Acts/EventData/TrackContainer.hpp"

namespace traccc::edm {

/// @c Acts::TrackContainer specialisation for traccc produced tracks
using TrackContainer =
    Acts::TrackContainer<TrackContainerBackend, MultiTrajectory,
                         Acts::detail::ValueHolder>;

}  // namespace traccc::edm
