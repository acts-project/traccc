/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Detray include(s).
#include "detray/geometry/tracking_surface.hpp"

namespace traccc::device::experimental {

/// Function for creating 3D spacepoints out of 2D measurements
///
template <typename detector_t>
TRACCC_HOST_DEVICE inline void form_spacepoints(
    const std::size_t globalIndex, typename detector_t::view_type det_data,
    measurement_collection_types::const_view measurements_view,
    spacepoint_collection_types::view spacepoints_view) {

    // Detector
    detector_t det(det_data);

    // Get device copy of input measurements
    const measurement_collection_types::const_device measurements(
        measurements_view);

    // Get device copy of output measurements
    spacepoint_collection_types::device spacepoints(spacepoints_view);

    if (globalIndex >= measurements.size()) {
        return;
    }

    // Access the measurements of the current module.
    const measurement& ms = measurements.at(globalIndex);

    const detray::tracking_surface sf{det, ms.surface_link};

    // This local to global transformation only works for 2D planar measurement
    // (e.g. barrel pixel and endcap pixel detector)
    const auto global = sf.bound_to_global({}, ms.local, {});

    spacepoints.push_back({global, ms});
}

}  // namespace traccc::device::experimental
