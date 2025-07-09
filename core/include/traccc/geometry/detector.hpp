/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/primitives.hpp"

// Detray include(s).
#include <any>
#include <detray/core/detector.hpp>
#include <detray/detectors/default_metadata.hpp>
#include <detray/detectors/telescope_metadata.hpp>
#include <detray/detectors/toy_metadata.hpp>

namespace traccc {

/// Base struct for the different detector types supported by the project.
template <typename metadata_t>
struct detector_traits {

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

template <typename T>
concept is_detector_traits = requires {
    typename T::metadata_type;
    typename T::host;
    typename T::device;
    typename T::view;
    typename T::const_view;
    typename T::buffer;
};

/// Default detector (also used for ODD)
using default_detector =
    detector_traits<detray::default_metadata<traccc::default_algebra>>;

/// Telescope detector
using telescope_detector = detector_traits<
    detray::telescope_metadata<traccc::default_algebra, detray::rectangle2D>>;

/// Toy detector
using toy_detector =
    detector_traits<detray::toy_metadata<traccc::default_algebra>>;

using detector_type_list = std::tuple<default_detector, telescope_detector>;
}  // namespace traccc
