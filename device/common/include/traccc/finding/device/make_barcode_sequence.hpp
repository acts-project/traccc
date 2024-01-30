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

/// Function filling the barcode sequence
///
/// @param[in] globalIndex   The index of the current thread
/// @param[in] uniques_view   Measurement container view object
/// @param[out] barcodes_view   Unsorted module map of <module ID, header ID>
///
TRACCC_DEVICE inline void make_barcode_sequence(
    std::size_t globalIndex,
    measurement_collection_types::const_view uniques_view,
    vecmem::data::vector_view<detray::geometry::barcode> barcodes_view);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/finding/device/impl/make_barcode_sequence.ipp"
