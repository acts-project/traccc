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

// Detray include(s).
#include "detray/geometry/barcode.hpp"

namespace traccc {

/// Finding result per track
template <typename algebra_t>
struct finding_result {
    using scalar_type = detray::dscalar<algebra_t>;

    /// Seed track parameter
    detray::bound_track_parameters<algebra_t> seed_params;

    /// Number of degree of freedoms of fitted track
    scalar_type ndf{0};

    /// Chi square of fitted track
    scalar_type chi2{0};

    // The number of holes (The number of sensitive surfaces which do not have a
    // measurement for the track pattern)
    unsigned int n_holes{0u};
};


/// Track candidate is the measurement
using track_candidate = measurement;

/// Declare a track candidates collection types
using track_candidate_collection_types = collection_types<track_candidate>;
/// Declare a track candidates container type
using track_candidate_container_types =
    container_types<finding_result<default_algebra>, track_candidate>;

}  // namespace traccc