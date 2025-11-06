/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"

// Detray include(s).
#include <detray/geometry/barcode.hpp>

namespace traccc::device {

/// Helper struct to compare surface descriptors by their barcodes
///
/// This is used in the CFK to find the measurement ranges for each surface
/// efficiently.
///
struct barcode_surface_comparator {

    template <typename sf_descriptor_t>
    TRACCC_HOST_DEVICE bool operator()(const sf_descriptor_t sf_desc,
                                       const detray::geometry::barcode& bc) {
        return sf_desc.barcode() < bc;
    }

    template <typename sf_descriptor_t>
    TRACCC_HOST_DEVICE bool operator()(const detray::geometry::barcode& bc,
                                       const sf_descriptor_t sf_desc) {
        return bc < sf_desc.barcode();
    }

};  // struct barcode_surface_comparator

}  // namespace traccc::device
