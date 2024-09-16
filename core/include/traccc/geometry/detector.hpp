/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Detray include(s).
#include <detray/core/detector.hpp>
#include <detray/detectors/telescope_metadata.hpp>
#include <detray/detectors/toy_metadata.hpp>

namespace traccc {

/// Base struct for the different detector types supported by the project.
template <typename metadata_t>
struct detector {

    /// Metadata type of the detector.
    using metadata_type = metadata_t;

    /// Host type of the detector.
    using host = detray::detector<metadata_type, detray::host_container_types>;
    /// Device type of the detector.
    using device =
        detray::detector<metadata_type, detray::device_container_types>;

    /// Non-const view of the detector.
    using view = typename host::view_type;
    /// Const view of the detector.
    using const_view = typename host::const_view_type;

    /// Buffer for a detector's data.
    using buffer = typename host::buffer_type;

};  // struct default_detector

/// Default detector (also used for ODD)
using default_detector = detector<detray::default_metadata>;

/// Telescope detector
using telescope_detector = detector<detray::telescope_metadata<> >;

/// Toy detector
using toy_detector = detector<detray::toy_metadata>;

}  // namespace traccc
