/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// traccc include
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/definitions/track_parametrization.hpp"
#include "traccc/edm/container.hpp"

// detray include(s).
#include "detray/tracks/tracks.hpp"

namespace traccc {

using free_track_parameters = detray::free_track_parameters<transform3>;
using bound_track_parameters = detray::bound_track_parameters<transform3>;
using free_vector = free_track_parameters::vector_type;
using free_covariance = free_track_parameters::covariance_type;
using bound_vector = bound_track_parameters::vector_type;
using bound_covariance = bound_track_parameters::covariance_type;

/// Declare all track_parameters collection types
using bound_track_parameters_collection_types =
    collection_types<bound_track_parameters>;

}  // namespace traccc
