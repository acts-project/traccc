/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/device/global_index.hpp"

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/measurement.hpp"

// Detray include(s).
#include <detray/geometry/barcode.hpp>

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// (Event Data) Payload for the @c traccc::device::make_barcode_sequence
/// function
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
///
TRACCC_DEVICE inline void make_barcode_sequence(
    global_index_t globalIndex, const make_barcode_sequence_payload& payload);

}  // namespace traccc::device

// Include the implementation.
#include "./impl/make_barcode_sequence.ipp"
