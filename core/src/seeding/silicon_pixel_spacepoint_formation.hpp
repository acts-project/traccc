/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/seeding/detail/spacepoint_formation.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

namespace traccc::host::details {

/// Common implementation for the spacepoint formation algorithm's execute
/// functions
///
/// @tparam detector_t The detector type to use
///
/// @param det               The detector object
/// @param measurements_view The view of the measurements to process
/// @param mr                The memory resource to create the output with
/// @return A container of the created spacepoints
///
template <typename detector_t>
spacepoint_collection_types::host silicon_pixel_spacepoint_formation(
    const detector_t& det,
    const measurement_collection_types::const_view& measurements_view,
    vecmem::memory_resource& mr) {

    // Create a device container for the input.
    const measurement_collection_types::const_device measurements{
        measurements_view};

    // Create the result container.
    spacepoint_collection_types::host result(&mr);
    result.reserve(measurements.size());

    // Set up each spacepoint in the result container.
    for (const auto& meas : measurements) {
        if (traccc::details::is_valid_measurement(meas)) {
            result.push_back(traccc::details::create_spacepoint(det, meas));
        }
    }

    // Return the created container.
    return result;
}

}  // namespace traccc::host::details
