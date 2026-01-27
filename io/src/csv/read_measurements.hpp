/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/geometry/host_detector.hpp"
#include "traccc/geometry/silicon_detector_description.hpp"

// System include(s).
#include <string_view>
#include <vector>

namespace traccc::io::csv {

/// Read measurement information from a specific CSV file
///
/// @param[out] measurements The collection to fill with the measurement data
/// @param[in]  filename     The file to read the measurement data from
/// @param[in]  detector  detray detector
/// @param[in]  do_sort      Whether to sort the measurements or not
///
std::vector<measurement_id_type> read_measurements(
    edm::measurement_collection<default_algebra>::host& measurements,
    std::string_view filename, const traccc::host_detector* detector = nullptr,
    const traccc::silicon_detector_description::host* detector_description =
        nullptr,
    const bool do_sort = true);

}  // namespace traccc::io::csv
