/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/edm/DeviceMultiTrajectory.hpp"
#include "traccc/edm/DeviceTrackBackend.hpp"

// Acts include(s).
#include <Acts/EventData/TrackContainer.hpp>

namespace traccc::edm {

/// @c Acts::TrackContainer specialisation for traccc produced tracks
using DeviceTrackContainer =
    Acts::TrackContainer<DeviceTrackBackend, DeviceMultiTrajectory,
                         Acts::detail::ValueHolder>;

}  // namespace traccc::edm
