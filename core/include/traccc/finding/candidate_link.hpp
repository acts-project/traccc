/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/measurement.hpp"

// Detray include(s).
#include "detray/geometry/barcode.hpp"

// Thrust include(s).
#include <thrust/pair.h>

namespace traccc {

// A link that contains the index of corresponding measurement and the index of
// a link from a previous step of track finding
struct candidate_link {

    // Type of index
    using link_index_type = thrust::pair<unsigned int, unsigned int>;

    // Index of link from the previous step
    link_index_type previous;

    // Measurement link
    typename measurement_container_types::host::link_type meas_link;

    // Surface ID (Geometry ID)
    detray::geometry::barcode surface_link;
};

}  // namespace traccc