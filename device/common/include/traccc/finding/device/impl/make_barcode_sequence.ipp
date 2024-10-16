/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/measurement.hpp"

namespace traccc::device {

TRACCC_DEVICE inline void make_barcode_sequence(
    std::size_t globalIndex, const make_barcode_sequence_payload& payload) {

    measurement_collection_types::const_device uniques(payload.uniques_view);
    vecmem::device_vector barcodes(payload.barcodes_view);
    assert(uniques.size() >= barcodes.size());

    if (globalIndex >= barcodes.size()) {
        return;
    }

    // Assign barcode
    barcodes.at(globalIndex) = uniques.at(globalIndex).surface_link;
}

}  // namespace traccc::device
