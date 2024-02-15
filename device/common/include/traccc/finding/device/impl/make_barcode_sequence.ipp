/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// System include(s).
#include <cassert>

namespace traccc::device {

TRACCC_DEVICE inline void make_barcode_sequence(
    std::size_t globalIndex,
    measurement_collection_types::const_view uniques_view,
    vecmem::data::vector_view<detray::geometry::barcode> barcodes_view) {

    measurement_collection_types::const_device uniques(uniques_view);
    vecmem::device_vector barcodes(barcodes_view);
    assert(uniques.size() >= barcodes.size());

    if (globalIndex >= barcodes.size()) {
        return;
    }

    // Assign barcode
    barcodes.at(globalIndex) = uniques.at(globalIndex).surface_link;
}

}  // namespace traccc::device