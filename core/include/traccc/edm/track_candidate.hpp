/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/fitting_result.hpp"

// Detray include(s).
#include "detray/geometry/barcode.hpp"

namespace traccc {

struct track_summary {

    /// (Mandatory) Seed track parameter
    bound_track_parameters seed;

    /// (Optional) Fitting result
    fitting_result fit_res;
}; 

/// Track candidate is the measurement
using track_candidate = measurement;

/// Declare a track candidates collection types
using track_candidate_collection_types = collection_types<track_candidate>;
/// Declare a track candidates container type
using track_candidate_container_types =
    container_types<track_summary, track_candidate>;

}  // namespace traccc