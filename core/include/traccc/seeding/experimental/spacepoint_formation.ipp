/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Detray include(s).
#include "detray/geometry/tracking_surface.hpp"

namespace traccc::experimental {

template <typename detector_t>
spacepoint_formation<detector_t>::spacepoint_formation(
    vecmem::memory_resource& mr)
    : m_mr(mr) {}

template <typename detector_t>
spacepoint_collection_types::host spacepoint_formation<detector_t>::operator()(
    const detector_t& det,
    const measurement_collection_types::host& measurements) const {

    // Create the result container.
    spacepoint_collection_types::host result(&(m_mr.get()));

    // Iterate over the measurements.
    for (std::size_t i = 0; i < measurements.size(); ++i) {

        // Access the measurements of the current module.
        const measurement& ms = measurements.at(i);

        const detray::tracking_surface sf{det, ms.surface_link};

        // This local to global transformation only works for 2D planar
        // measurement
        // (e.g. barrel pixel and endcap pixel detector)
        const auto global = sf.bound_to_global({}, ms.local, {});

        // Fill result with this spacepoint
        result.push_back({global, ms});
    }

    // Return the created container.
    return result;
}

}  // namespace traccc::experimental
