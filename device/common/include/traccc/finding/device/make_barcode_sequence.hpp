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
struct make_barcode_sequence_payload {
    /**
     * @brief View object to the vector of unique measurement indices
     */
    measurement_collection_types::const_view uniques_view;

    /**
     * @brief View object to the output vector of barcodes
     */
    vecmem::data::vector_view<detray::geometry::barcode> barcodes_view;
};

/// Function filling the barcode sequence
///
/// @param[in] globalIndex   The index of the current thread
/// @param[inout] payload      The function call payload
TRACCC_DEVICE inline void make_barcode_sequence(
    std::size_t globalIndex, const make_barcode_sequence_payload& payload);
}  // namespace traccc::device

#include "./impl/make_barcode_sequence.ipp"
