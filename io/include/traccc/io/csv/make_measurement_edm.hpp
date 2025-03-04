/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/measurement.hpp"

// Local include(s).
#include "traccc/io/csv/measurement.hpp"

#pragma once

namespace traccc::io::csv {

/// Make measurement EDM from csv measurement
///
/// @param[in] csv_meas input csv measurement
/// @param[in] acts_to_detray_id Map for acts-to-detray geometry ID converision
///
traccc::measurement make_measurement_edm(
    const traccc::io::csv::measurement& csv_meas,
    const std::map<geometry_id, geometry_id>* acts_to_detray_id);

}  // namespace traccc::io::csv
