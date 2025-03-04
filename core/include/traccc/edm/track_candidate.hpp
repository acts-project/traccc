/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/edm/track_quality.hpp"

namespace traccc {

/// Finding result per track
struct finding_result {

    /// Seed track parameter
    traccc::bound_track_parameters<> seed_params;

    /// Track summary
    traccc::track_quality trk_quality;
};

/// Equality operator for finding result
TRACCC_HOST_DEVICE
inline bool operator==(const finding_result& lhs, const finding_result& rhs) {

    return (lhs.seed_params == rhs.seed_params) &&
           (lhs.trk_quality == rhs.trk_quality);
}

/// Declare a finding results collection types
using finding_result_collection_types = collection_types<finding_result>;

/// Track candidate is the measurement
using track_candidate = measurement;

/// Declare a track candidates collection types
using track_candidate_collection_types = collection_types<track_candidate>;
/// Declare a track candidates container type
using track_candidate_container_types =
    container_types<finding_result, track_candidate>;

}  // namespace traccc
